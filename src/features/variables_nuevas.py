# -*- coding: utf-8 -*-
"""
Prueba seis variables nuevas, para ver si mueven lo que el modelo no pudo.

Tres intentos anteriores dieron cero: ajustar hiperparametros, cambiar el
objetivo a lambdarank y cuadruplicar los datos. La conclusion fue que el
limite lo ponen las variables. Este script pone esa conclusion a prueba.

Que le falta al modelo hoy
--------------------------
De las 25 variables actuales, ninguna mira DOS cosas:

1. La ventana reciente. `ratio_usuario_producto` es el ratio de toda la vida
   del cliente: no distingue entre un producto que compraba mucho y dejo, y
   uno que empezo a comprar el mes pasado. Los dos pueden dar el mismo ratio
   y significan cosas opuestas.

2. El pasillo. El modelo sabe cuanto compra la persona y cuanto se vende el
   producto, pero no si a esa persona le interesa esa CATEGORIA. Alguien que
   compra mucha verdura tiene mas chance de repetir una verdura que nunca
   compro dos veces, aunque su ratio para ese producto sea igual al de otro.

Las seis variables
------------------
  comprado_en_ultima    si estaba en el ultimo pedido del historial
  ratio_ultimas_3       en cuantos de sus ultimos 3 pedidos aparece
  ratio_ultimas_5       lo mismo sobre 5
  tendencia             ratio reciente menos ratio de toda la vida. Positivo
                        es un producto que esta adoptando, negativo uno que
                        esta abandonando
  afinidad_pasillo      que proporcion de todo lo que compra el usuario sale
                        del pasillo de este producto
  antiguedad_par        hace cuantos pedidos lo compro por primera vez, sobre
                        el total de pedidos del usuario. Distingue un producto
                        de siempre de uno recien descubierto

Sin fuga: todo sale de `prior`. La orden objetivo no entra en ningun calculo.

Uso:
    python src/features/variables_nuevas.py            # calcula y evalua
    python src/features/variables_nuevas.py --solo-calcular
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import duckdb
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"

SALIDA = PROCESSED_DIR / "variables_nuevas.parquet"

NUEVAS = [
    "comprado_en_ultima",
    "ratio_ultimas_3",
    "ratio_ultimas_5",
    "tendencia",
    "afinidad_pasillo",
    "antiguedad_par",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--solo-calcular", action="store_true")
    p.add_argument("--memory-limit", default="4GB")
    p.add_argument("--threads", type=int, default=4)
    return p.parse_args()


def ruta(p: Path) -> str:
    return p.resolve().as_posix().replace("'", "''")


def calcular(args) -> None:
    con = duckdb.connect()
    con.execute(f"SET memory_limit = '{args.memory_limit}'")
    con.execute(f"SET threads = {args.threads}")

    print("Calculando variables nuevas...")
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE pedidos AS
        SELECT user_id, order_id, order_number,
               -- 1 es el pedido MAS reciente del historial. Contar desde el
               -- final y no desde el principio es lo que hace que la ventana
               -- signifique lo mismo para un cliente de 5 pedidos y uno de 80.
               row_number() OVER (PARTITION BY user_id
                                  ORDER BY order_number DESC) AS desde_el_final,
               count(*) OVER (PARTITION BY user_id) AS total_pedidos
        FROM read_csv_auto('{ruta(RAW_DIR / "orders.csv")}')
        WHERE eval_set = 'prior';

        CREATE OR REPLACE TEMP TABLE compras AS
        SELECT p.user_id, p.order_id, p.order_number, p.desde_el_final,
               p.total_pedidos, op.product_id
        FROM pedidos p
        JOIN read_csv_auto('{ruta(RAW_DIR / "order_products__prior.csv")}') op
          USING (order_id);
    """)

    print("  ventana reciente y antiguedad...")
    con.execute("""
        CREATE OR REPLACE TEMP TABLE por_par AS
        SELECT
            user_id,
            product_id,
            any_value(total_pedidos) AS total_pedidos,
            count(DISTINCT order_id) AS pedidos_con,
            max(CASE WHEN desde_el_final = 1 THEN 1 ELSE 0 END)
                AS comprado_en_ultima,
            count(DISTINCT CASE WHEN desde_el_final <= 3 THEN order_id END)
                AS en_ultimas_3,
            count(DISTINCT CASE WHEN desde_el_final <= 5 THEN order_id END)
                AS en_ultimas_5,
            max(desde_el_final) AS primera_compra_desde_el_final
        FROM compras
        GROUP BY user_id, product_id;
    """)

    print("  afinidad de pasillo...")
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE catalogo AS
        SELECT product_id, aisle_id
        FROM read_parquet('{ruta(PROCESSED_DIR / "catalogo.parquet")}');

        CREATE OR REPLACE TEMP TABLE afinidad AS
        WITH compras_pasillo AS (
            SELECT c.user_id, cat.aisle_id, count(*) AS compras_en_pasillo
            FROM compras c
            JOIN catalogo cat USING (product_id)
            GROUP BY 1, 2
        ), total AS (
            SELECT user_id, sum(compras_en_pasillo) AS compras_totales
            FROM compras_pasillo GROUP BY user_id
        )
        SELECT cp.user_id, cp.aisle_id,
               cp.compras_en_pasillo::DOUBLE / t.compras_totales AS afinidad_pasillo
        FROM compras_pasillo cp JOIN total t USING (user_id);
    """)

    print("  armando la tabla final...")
    con.execute(f"""
        COPY (
            SELECT
                p.user_id,
                p.product_id,
                p.comprado_en_ultima,
                -- El denominador es el minimo entre la ventana y los pedidos
                -- que el cliente tiene. Sin eso, alguien con 2 pedidos nunca
                -- podria pasar de 0,66 en la ventana de 3, y quedaria
                -- penalizado por ser nuevo y no por su comportamiento.
                p.en_ultimas_3::DOUBLE / least(3, p.total_pedidos) AS ratio_ultimas_3,
                p.en_ultimas_5::DOUBLE / least(5, p.total_pedidos) AS ratio_ultimas_5,
                (p.en_ultimas_5::DOUBLE / least(5, p.total_pedidos))
                    - (p.pedidos_con::DOUBLE / p.total_pedidos) AS tendencia,
                COALESCE(a.afinidad_pasillo, 0.0) AS afinidad_pasillo,
                p.primera_compra_desde_el_final::DOUBLE / p.total_pedidos
                    AS antiguedad_par
            FROM por_par p
            LEFT JOIN catalogo c USING (product_id)
            LEFT JOIN afinidad a ON a.user_id = p.user_id
                                AND a.aisle_id = c.aisle_id
        ) TO '{ruta(SALIDA)}' (FORMAT PARQUET);
    """)
    n = con.execute(f"SELECT count(*) FROM read_parquet('{ruta(SALIDA)}')").fetchone()[0]
    con.close()
    print(f"  {n:,} pares guardados en {SALIDA.relative_to(PROJECT_ROOT)}")


def evaluar_aporte() -> None:
    from src.evaluation.metrics import comparar
    from src.models import boosting
    from src.models.baselines import cargar_verdad
    from src.models.regresion_logistica import VARIABLES, recomendar

    print("\nLeyendo features y uniendo las nuevas...")
    columnas = ["user_id", "product_id", "split", "segmento_usuario",
                "etiqueta"] + VARIABLES
    f = pd.read_parquet(PROCESSED_DIR / "features.parquet", columns=columnas)
    nuevas = pd.read_parquet(SALIDA)
    f = f.merge(nuevas, on=["user_id", "product_id"], how="left")

    faltantes = f[NUEVAS].isna().sum().sum()
    if faltantes:
        print(f"  aviso: {faltantes:,} valores sin par en las nuevas, van a 0")
        f[NUEVAS] = f[NUEVAS].fillna(0.0)

    train = f[f["split"] == "train"]
    valid = f[f["split"] == "valid"]
    usuarios = set(valid["user_id"].unique())
    verdad = cargar_verdad(usuarios)
    print(f"  validacion: {len(verdad):,} usuarios")

    # Se entrenan los dos con exactamente el mismo codigo y los mismos
    # parametros. Lo unico que cambia es la lista de variables.
    import src.models.regresion_logistica as rl

    print("\nModelo actual, 25 variables...")
    t = time.time()
    p_base, _ = boosting.entrenar(train, valid)
    print(f"  {time.time() - t:.0f} s")

    print(f"\nCon las {len(NUEVAS)} nuevas, {len(VARIABLES) + len(NUEVAS)} variables...")
    original = list(rl.VARIABLES)
    try:
        rl.VARIABLES[:] = original + NUEVAS
        boosting.VARIABLES = rl.VARIABLES
        t = time.time()
        p_nuevo, modelo = boosting.entrenar(train, valid)
        print(f"  {time.time() - t:.0f} s")
        importancias = pd.DataFrame({
            "variable": rl.VARIABLES,
            "gain": modelo.booster_.feature_importance("gain"),
        })
    finally:
        rl.VARIABLES[:] = original
        boosting.VARIABLES = rl.VARIABLES

    modelos = {"actual_25_variables": recomendar(valid, p_base, 10),
               "con_variables_nuevas": recomendar(valid, p_nuevo, 10)}
    tabla = pd.DataFrame(comparar(modelos, verdad, k=10,
                                  referencia="actual_25_variables"))

    print("\n" + "=" * 66)
    print("APORTE DE LAS VARIABLES NUEVAS")
    print("=" * 66)
    print(tabla[["modelo", "precision", "recall", "hit_rate", "cobertura"]]
          .round(4).to_string(index=False))

    base = tabla.iloc[0]
    nuevo = tabla.iloc[1]
    d_recall = nuevo["recall"] - base["recall"]
    d_hit = nuevo["hit_rate"] - base["hit_rate"]
    print(f"\n  recall   {d_recall:+.4f}")
    print(f"  hit rate {d_hit:+.4f}")
    print("\n  Vara: los tres intentos anteriores dieron +0,0007, -0,0003 y +0,0005.")
    if d_recall > 0.005:
        print("  ESTA es la palanca. Conviene integrarlas al pipeline.")
    elif d_recall > 0.001:
        print("  Aportan, moderado. Vale integrarlas.")
    else:
        print("  Tampoco aportan. El techo no esta en estas variables.")

    importancias["pct_gain"] = importancias["gain"] / importancias["gain"].sum()
    importancias = importancias.sort_values("gain", ascending=False)
    print("\n  Donde quedaron las nuevas, por importancia:")
    for i, (_, fila) in enumerate(importancias.iterrows(), 1):
        if fila["variable"] in NUEVAS:
            print(f"    puesto {i:>2} de {len(importancias)}: "
                  f"{fila['variable']:<22} {fila['pct_gain']:.2%}")

    REPORTS_DIR.mkdir(exist_ok=True)
    tabla.to_csv(REPORTS_DIR / "aporte_variables_nuevas.csv", index=False)
    importancias.to_csv(REPORTS_DIR / "importancia_con_nuevas.csv", index=False)
    print("\nGuardado en reports/aporte_variables_nuevas.csv")


def main() -> None:
    args = parse_args()
    t0 = time.time()
    calcular(args)
    if not args.solo_calcular:
        evaluar_aporte()
    print(f"Total {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
