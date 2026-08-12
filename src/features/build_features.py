# -*- coding: utf-8 -*-
"""
Construccion de la tabla de variables para los modelos que aprenden.

El problema de recomendar diez productos se reformula como una pregunta
binaria hecha muchas veces: dado un par (usuario, producto) del historial de
esa persona, .lo va a comprar en su proxima orden? El modelo devuelve una
probabilidad por par, se ordena de mayor a menor y se corta en K.

Decisiones que definen esta tabla
---------------------------------
1. Candidatos: solo los productos que el usuario ya compro alguna vez. Es
   coherente con la decision de sprint de rankear el historial en vez de
   generar productos nuevos, y le pone al modelo un techo conocido de 0.5537
   de Recall@10 macro.

2. Etiqueta: 1 si el par aparece en la orden `train` del usuario, 0 si no.
   La orden `train` no se usa para ninguna otra cosa.

3. Particion: por USUARIO entero, nunca por fila. Si las filas de una misma
   persona cayeran de los dos lados, el modelo aprenderia de una parte de su
   orden objetivo y se lo evaluaria con la otra parte de esa misma orden. Es
   fuga de informacion y no se ve en las metricas.

   La particion queda estratificada por segmento, para que validacion tenga
   la misma mezcla de nuevo/medio/heavy que entrenamiento.

Consecuencia a tener presente
-----------------------------
Los baselines estan medidos sobre los 131.209 usuarios evaluables. El modelo
se va a medir sobre los usuarios de validacion, que son menos. Para que la
tabla de la Demo compare lo comparable, hay que volver a evaluar los
baselines restringidos a esos mismos usuarios. La columna `split` de esta
tabla es lo que permite hacerlo.

Uso:
    python src/features/build_features.py                # completo
    python src/features/build_features.py --usuarios 5000  # prueba piloto
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import duckdb

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
SALIDA = PROCESSED_DIR / "features.parquet"

# Fijo para que la particion sea siempre la misma. Cambiarlo invalida la
# comparacion con cualquier resultado medido antes.
SEMILLA = 42
PROPORCION_TRAIN = 0.80


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--usuarios", type=int, default=None,
                   help="Limita la cantidad de usuarios, para probar rapido.")
    p.add_argument("--salida", type=Path, default=SALIDA)
    return p.parse_args()


def sql_path(path: Path) -> str:
    return str(path).replace("\\", "/")


def construir(con: duckdb.DuckDBPyConnection, limite: int | None) -> None:
    """Deja creada la vista `features` con una fila por par usuario-producto."""
    filtro = f"USING SAMPLE {limite} ROWS (reservoir, {SEMILLA})" if limite else ""

    con.execute(f"""
        CREATE OR REPLACE TABLE usuarios_evaluables AS
        SELECT DISTINCT t.user_id, u.segmento_usuario
        FROM '{sql_path(PROCESSED_DIR / "targets_train.parquet")}' AS t
        INNER JOIN '{sql_path(PROCESSED_DIR / "usuarios.parquet")}' AS u
            USING (user_id)
        {filtro};
    """)

    # Particion por usuario, estratificada por segmento. El hash sobre
    # (user_id, semilla) la hace determinista: la misma corrida da siempre
    # los mismos usuarios de cada lado.
    con.execute(f"""
        CREATE OR REPLACE TABLE particion AS
        WITH ordenados AS (
            SELECT user_id, segmento_usuario,
                   ROW_NUMBER() OVER (
                       PARTITION BY segmento_usuario
                       ORDER BY hash(CAST(user_id AS BIGINT) * 1000003
                                     + {SEMILLA})
                   ) AS puesto,
                   COUNT(*) OVER (PARTITION BY segmento_usuario) AS del_segmento
            FROM usuarios_evaluables
        )
        SELECT user_id, segmento_usuario,
               CASE WHEN puesto <= del_segmento * {PROPORCION_TRAIN}
                    THEN 'train' ELSE 'valid' END AS split
        FROM ordenados;
    """)

    con.execute(f"""
        CREATE OR REPLACE VIEW etiquetas AS
        SELECT DISTINCT user_id, product_id, 1 AS etiqueta
        FROM '{sql_path(PROCESSED_DIR / "targets_train.parquet")}';
    """)

    con.execute(f"""
        CREATE OR REPLACE VIEW features AS
        SELECT
            i.user_id,
            i.product_id,
            p.split,
            p.segmento_usuario,

            -- ---------------- del par usuario-producto ----------------
            i.freq_usuario_producto,
            i.recencia_usuario_producto,
            i.dias_registrados_desde_ultima_compra,
            i.add_to_cart_order_promedio        AS posicion_par,
            i.ultima_order_dow                  AS dow_ultima_compra,
            i.ultima_order_hour                 AS hora_ultima_compra,

            -- ---------------- del usuario ----------------
            u.cantidad_ordenes_historicas,
            u.productos_distintos,
            u.reorder_rate_usuario,
            u.posicion_media_carrito,
            u.mediana_dias_entre_ordenes_historicas,

            -- ---------------- del producto ----------------
            pr.cantidad_compras                 AS popularidad_producto,
            pr.cantidad_usuarios                AS usuarios_del_producto,
            pr.reorder_rate_producto,

            -- ---------------- derivadas ----------------
            -- La senal del baseline: en que proporcion de sus ordenes aparece.
            -- El modelo tiene que superar a esta columna sola.
            i.freq_usuario_producto * 1.0
                / u.cantidad_ordenes_historicas AS ratio_usuario_producto,

            -- Cuantos ciclos de compra de ESA persona pasaron desde la ultima
            -- vez que llevo el producto. Un mes es mucho para quien compra
            -- semanal y poco para quien compra cada dos meses.
            i.dias_registrados_desde_ultima_compra * 1.0
                / NULLIF(u.mediana_dias_entre_ordenes_historicas, 0)
                                                AS ciclos_desde_ultima_compra,

            -- Recencia en ordenes, normalizada por el largo del historial.
            i.recencia_usuario_producto * 1.0
                / u.cantidad_ordenes_historicas AS recencia_relativa,

            -- Si lo pone antes o despues que su promedio: proxy de prioridad.
            i.add_to_cart_order_promedio
                / NULLIF(u.posicion_media_carrito, 0) AS posicion_relativa,

            -- Cuantos de los productos distintos del usuario compiten por los
            -- diez lugares.
            u.productos_distintos               AS competencia_por_slot,

            COALESCE(e.etiqueta, 0)             AS etiqueta

        FROM '{sql_path(PROCESSED_DIR / "interacciones.parquet")}' AS i
        INNER JOIN particion AS p              USING (user_id)
        INNER JOIN '{sql_path(PROCESSED_DIR / "usuarios.parquet")}' AS u
            ON u.user_id = i.user_id
        INNER JOIN '{sql_path(PROCESSED_DIR / "productos.parquet")}' AS pr
            ON pr.product_id = i.product_id
        LEFT JOIN etiquetas AS e
            ON e.user_id = i.user_id AND e.product_id = i.product_id;
    """)


def controlar(con: duckdb.DuckDBPyConnection) -> None:
    """Controles que tienen que pasar para que la tabla sirva."""
    print("\nControles")

    cruzados = con.execute("""
        SELECT COUNT(*) FROM (
            SELECT user_id FROM particion WHERE split = 'train'
            INTERSECT
            SELECT user_id FROM particion WHERE split = 'valid')
    """).fetchone()[0]
    estado = "OK" if cruzados == 0 else "FALLA"
    print(f"[{estado}] usuarios en los dos lados de la particion: {cruzados:,}")

    huerfanos = con.execute("""
        SELECT COUNT(*) FROM features WHERE etiqueta IS NULL
    """).fetchone()[0]
    estado = "OK" if huerfanos == 0 else "FALLA"
    print(f"[{estado}] filas sin etiqueta: {huerfanos:,}")

    # Los positivos de la tabla tienen que ser exactamente los objetivos que
    # el usuario ya habia comprado: es el techo del enfoque.
    positivos = con.execute(
        "SELECT SUM(etiqueta) FROM features").fetchone()[0]
    print(f"[OK] positivos (objetivos que estaban en el historial): {positivos:,}")

    nulos = con.execute("""
        SELECT
            SUM(CASE WHEN ratio_usuario_producto IS NULL THEN 1 ELSE 0 END),
            SUM(CASE WHEN ciclos_desde_ultima_compra IS NULL THEN 1 ELSE 0 END),
            SUM(CASE WHEN posicion_relativa IS NULL THEN 1 ELSE 0 END)
        FROM features
    """).fetchone()
    print(f"[{'OK' if nulos[0] == 0 else 'FALLA'}] nulos en ratio_usuario_producto: {nulos[0]:,}")
    print(f"[{'OK' if nulos[2] == 0 else 'FALLA'}] nulos en posicion_relativa: {nulos[2]:,}")

    # ciclos_desde_ultima_compra queda indefinida a proposito para los
    # usuarios cuya mediana de dias entre ordenes es 0: hacen varios pedidos
    # el mismo dia, asi que su "ciclo de compra" mide cero y no se puede
    # dividir por el. Son 106 usuarios sobre 206.209. Se deja NULL en vez de
    # imputar un numero inventado; LightGBM los maneja nativamente y para la
    # regresion logistica hay que imputar en el paso de modelado.
    afectados = con.execute("""
        SELECT COUNT(DISTINCT user_id) FROM features
        WHERE ciclos_desde_ultima_compra IS NULL
    """).fetchone()[0]
    print(f"[OK] ciclos_desde_ultima_compra sin definir: {nulos[1]:,} filas "
          f"de {afectados:,} usuarios con ciclo de compra cero")


def resumir(con: duckdb.DuckDBPyConnection) -> None:
    print("\nReparto de la particion")
    print(con.execute("""
        SELECT split, segmento_usuario, COUNT(*) AS usuarios
        FROM particion GROUP BY 1, 2 ORDER BY 1, 2
    """).df().to_string(index=False))

    print("\nFilas y etiquetas")
    print(con.execute("""
        SELECT split,
               COUNT(*) AS filas,
               COUNT(DISTINCT user_id) AS usuarios,
               SUM(etiqueta) AS positivos,
               ROUND(AVG(etiqueta), 4) AS tasa_positivos,
               ROUND(COUNT(*) * 1.0 / COUNT(DISTINCT user_id), 1) AS filas_por_usuario
        FROM features GROUP BY 1 ORDER BY 1
    """).df().to_string(index=False))


def main() -> None:
    args = parse_args()
    con = duckdb.connect(config={"memory_limit": "4GB", "threads": 4})

    t0 = time.time()
    print("Construyendo la tabla de variables...")
    construir(con, args.usuarios)

    controlar(con)
    resumir(con)

    # El ORDER BY no es cosmetico. Sin el, el orden de las filas lo decide el
    # LEFT JOIN de etiquetas, y ese orden FILTRA LA RESPUESTA: medido sobre
    # las filas empatadas, las de etiqueta 1 caen en el puesto relativo 0.39 y
    # las de etiqueta 0 en el 0.65, cuando ambas deberian dar 0.50.
    #
    # Cualquier codigo aguas abajo que ordene por un criterio con empates
    # —el baseline de recompra, sin ir mas lejos— hereda ese orden como
    # desempate y se lleva un premio que no gano. Escribir ordenado por
    # (user_id, product_id) deja el archivo neutral respecto de la etiqueta.
    con.execute(
        f"COPY (SELECT * FROM features ORDER BY user_id, product_id) "
        f"TO '{sql_path(args.salida)}' (FORMAT PARQUET)")
    mb = args.salida.stat().st_size / 1024 ** 2
    print(f"\n{args.salida.name}: {mb:,.1f} MiB en {time.time() - t0:.0f} s")
    print(f"Ruta: {args.salida}")


if __name__ == "__main__":
    main()
