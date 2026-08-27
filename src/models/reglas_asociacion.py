# -*- coding: utf-8 -*-
"""
Reglas de asociacion sobre las canastas historicas: si compro A, que compra con A.

Responde el centro que dejo Ariel en la Demo 1. Hasta ahora el sistema solo
predice RECOMPRA: mira el historial de la persona y anticipa lo que va a
volver a comprar. Eso no hace crecer la canasta, la reconstruye. Las reglas
de asociacion apuntan al otro lado: sugerir un producto COMPLEMENTARIO que
la persona no lleva, a partir de lo que compra el resto.

Tres decisiones que vale la pena entender antes de leer los numeros:

1. Se poda por soporte antes de cruzar. 49.677 productos son 1.234 millones
   de pares posibles. Cruzar todo no entra en memoria y ademas no sirve: un
   par que aparece tres veces en 3,2 M de ordenes es ruido, no una regla.

2. Se descartan las VARIANTES del mismo producto por dos vias: mismo
   pasillo, y nombres que comparten palabras. "Banana -> Banana organica"
   o "Yerba Mate Naranja -> Yerba Mate Limon" tienen lift altisimo y son
   inutiles: son sustitutos, la persona elige uno. Filtrar solo por pasillo
   no alcanza, porque dos sabores del mismo producto a veces quedan
   clasificados en pasillos distintos.

3. El lift se mide contra la probabilidad base del producto B. Sin eso, las
   reglas quedarian dominadas por los productos mas vendidos: "compro X ->
   compro banana" es cierto para casi todo X, y no aporta nada.

Uso:
    python src/models/reglas_asociacion.py
    python src/models/reglas_asociacion.py --min-ordenes 500 --min-pares 200
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import duckdb

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--min-ordenes", type=int, default=500,
                   help="Un producto entra al analisis si aparece en al menos "
                        "esta cantidad de ordenes.")
    p.add_argument("--min-pares", type=int, default=100,
                   help="Un par entra si aparece junto al menos esta cantidad "
                        "de veces.")
    p.add_argument("--min-confianza", type=float, default=0.05,
                   help="Confianza minima para reportar una regla.")
    p.add_argument("--top", type=int, default=40,
                   help="Cuantas reglas mostrar por bloque.")
    p.add_argument("--memory-limit", default="4GB")
    p.add_argument("--threads", type=int, default=4)
    return p.parse_args()


def ruta(p: Path) -> str:
    return p.resolve().as_posix().replace("'", "''")


def construir(con: duckdb.DuckDBPyConnection, args) -> None:
    prior = ruta(RAW_DIR / "order_products__prior.csv")
    catalogo = ruta(PROCESSED_DIR / "catalogo.parquet")

    print("Leyendo canastas historicas...")
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE canastas AS
        SELECT order_id, product_id
        FROM read_csv_auto('{prior}');

        CREATE OR REPLACE TEMP TABLE catalogo AS
        SELECT product_id, product_name, aisle, department
        FROM read_parquet('{catalogo}');

        CREATE OR REPLACE TEMP TABLE total AS
        SELECT count(DISTINCT order_id) AS ordenes FROM canastas;
    """)

    # Soporte individual: en cuantas ordenes aparece cada producto. Es el
    # denominador de la confianza y la probabilidad base del lift.
    print("Contando soporte por producto...")
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE soporte AS
        SELECT product_id, count(*) AS ordenes_con
        FROM canastas
        GROUP BY product_id
        HAVING count(*) >= {args.min_ordenes};
    """)

    # La poda va ANTES del cruce, no despues: es lo que hace que el self join
    # entre en memoria.
    print("Podando y cruzando pares dentro de cada orden...")
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE canastas_podadas AS
        SELECT c.order_id, c.product_id
        FROM canastas c
        SEMI JOIN soporte s USING (product_id);

        CREATE OR REPLACE TEMP TABLE pares AS
        SELECT a.product_id AS producto_a,
               b.product_id AS producto_b,
               count(*) AS ordenes_juntos
        FROM canastas_podadas a
        JOIN canastas_podadas b
          ON a.order_id = b.order_id
         AND a.product_id < b.product_id
        GROUP BY 1, 2
        HAVING count(*) >= {args.min_pares};
    """)


def reglas(con: duckdb.DuckDBPyConnection, args):
    """
    Cada par genera DOS reglas, A->B y B->A, porque la confianza no es
    simetrica: que el 40% de quienes compran pasta lleven salsa no implica
    que el 40% de quienes compran salsa lleven pasta.
    """
    return con.execute(f"""
        WITH n AS (SELECT ordenes FROM total),
        dirigidas AS (
            SELECT producto_a AS antecedente, producto_b AS consecuente, ordenes_juntos
            FROM pares
            UNION ALL
            SELECT producto_b, producto_a, ordenes_juntos FROM pares
        )
        SELECT
            d.antecedente AS producto_a_id,
            d.consecuente AS producto_b_id,
            ca.product_name AS si_compra,
            cb.product_name AS tambien_lleva,
            ca.aisle AS pasillo_a,
            cb.aisle AS pasillo_b,
            (ca.aisle = cb.aisle) AS mismo_pasillo,
            d.ordenes_juntos,
            round(d.ordenes_juntos::DOUBLE / n.ordenes, 5) AS soporte,
            round(d.ordenes_juntos::DOUBLE / sa.ordenes_con, 4) AS confianza,
            round((d.ordenes_juntos::DOUBLE / sa.ordenes_con)
                  / (sb.ordenes_con::DOUBLE / n.ordenes), 2) AS lift
        FROM dirigidas d
        CROSS JOIN n
        JOIN soporte sa ON sa.product_id = d.antecedente
        JOIN soporte sb ON sb.product_id = d.consecuente
        JOIN catalogo ca ON ca.product_id = d.antecedente
        JOIN catalogo cb ON cb.product_id = d.consecuente
        WHERE d.ordenes_juntos::DOUBLE / sa.ordenes_con >= {args.min_confianza}
    """).fetchdf()


# Palabras que aparecen en miles de nombres y no identifican al producto.
# Si no se sacan, "Organic Milk" y "Organic Bread" parecerian variantes.
GENERICAS = {
    "organic", "original", "free", "natural", "fresh", "the", "and", "with",
    "of", "no", "all", "premium", "classic", "style", "flavor", "flavored",
    "unsweetened", "sweetened", "low", "reduced", "fat", "gluten", "non",
    "gmo", "large", "small", "mini", "value", "pack", "count", "oz", "size",
}


def es_variante(nombre_a: str, nombre_b: str, minimo: int = 2) -> bool:
    """
    True si los dos nombres comparten suficientes palabras propias como para
    ser el mismo producto en otra version.

    Es una heuristica, no una clasificacion. Prefiere descartar de mas: una
    regla complementaria perdida no rompe nada, una regla que en realidad es
    "el mismo yogur en otro sabor" sugerida al cliente si.
    """
    def tokens(s):
        return {w for w in "".join(
            c.lower() if c.isalnum() else " " for c in s).split()
            if len(w) > 2 and w not in GENERICAS}

    ta, tb = tokens(nombre_a), tokens(nombre_b)
    if not ta or not tb:
        return False
    comunes = ta & tb
    # Comparten varias palabras propias, o una de las dos esta contenida
    # casi entera en la otra (caso "Yogur Frutilla" vs "Yogur Frutilla Light").
    return len(comunes) >= minimo or len(comunes) >= min(len(ta), len(tb))


def reglas_por_pasillo(con: duckdb.DuckDBPyConnection, args):
    """
    Las mismas reglas pero entre pasillos en vez de entre productos.

    Es la version que sirve para negocio: 134 pasillos en lugar de 8.290
    productos, asi que cada regla se apoya en muchisimas mas ordenes y no
    sufre el problema de las variantes. "Si compra pasta, lleva salsa" es
    una decision de negocio; "si compra ESTA pasta lleva ESTA salsa" es una
    decision de recomendador.
    """
    return con.execute(f"""
        WITH n AS (SELECT ordenes FROM total),
        canastas_pasillo AS (
            SELECT DISTINCT c.order_id, cat.aisle
            FROM canastas c
            JOIN catalogo cat USING (product_id)
        ),
        soporte_pasillo AS (
            SELECT aisle, count(*) AS ordenes_con
            FROM canastas_pasillo GROUP BY aisle
        ),
        pares_pasillo AS (
            SELECT a.aisle AS pasillo_a, b.aisle AS pasillo_b,
                   count(*) AS ordenes_juntos
            FROM canastas_pasillo a
            JOIN canastas_pasillo b
              ON a.order_id = b.order_id AND a.aisle < b.aisle
            GROUP BY 1, 2
        ),
        dirigidas AS (
            SELECT pasillo_a AS antecedente, pasillo_b AS consecuente, ordenes_juntos
            FROM pares_pasillo
            UNION ALL
            SELECT pasillo_b, pasillo_a, ordenes_juntos FROM pares_pasillo
        )
        SELECT
            d.antecedente AS si_compra_de,
            d.consecuente AS tambien_lleva_de,
            d.ordenes_juntos,
            round(d.ordenes_juntos::DOUBLE / n.ordenes, 4) AS soporte,
            round(d.ordenes_juntos::DOUBLE / sa.ordenes_con, 4) AS confianza,
            round((d.ordenes_juntos::DOUBLE / sa.ordenes_con)
                  / (sb.ordenes_con::DOUBLE / n.ordenes), 2) AS lift
        FROM dirigidas d
        CROSS JOIN n
        JOIN soporte_pasillo sa ON sa.aisle = d.antecedente
        JOIN soporte_pasillo sb ON sb.aisle = d.consecuente
        WHERE d.ordenes_juntos >= {args.min_pares}
        ORDER BY lift DESC
    """).fetchdf()


def main() -> None:
    args = parse_args()
    t0 = time.time()

    with duckdb.connect() as con:
        con.execute(f"SET memory_limit = '{args.memory_limit}'")
        con.execute(f"SET threads = {args.threads}")
        construir(con, args)

        prod = con.execute("SELECT count(*) FROM soporte").fetchone()[0]
        pares = con.execute("SELECT count(*) FROM pares").fetchone()[0]
        ordenes = con.execute("SELECT ordenes FROM total").fetchone()[0]
        print(f"\n  ordenes analizadas: {ordenes:,}")
        print(f"  productos que pasan el corte: {prod:,}")
        print(f"  pares que pasan el corte: {pares:,}")

        df = reglas(con, args)
        pasillos = reglas_por_pasillo(con, args)

    df["es_variante"] = [
        es_variante(a, b) for a, b in zip(df["si_compra"], df["tambien_lleva"])
    ]
    df["descartada"] = df["mismo_pasillo"] | df["es_variante"]
    df = df.sort_values("lift", ascending=False)

    complementarias = df[~df["descartada"]]
    descartadas = df[df["descartada"]]

    cols = ["si_compra", "tambien_lleva", "confianza", "lift", "ordenes_juntos"]

    print("\n" + "=" * 78)
    print("PASILLO -> PASILLO. La lectura de negocio.")
    print("=" * 78)
    print(pasillos.head(args.top)[
        ["si_compra_de", "tambien_lleva_de", "confianza", "lift",
         "ordenes_juntos"]].to_string(index=False))

    print("\n" + "=" * 78)
    print("PRODUCTO -> PRODUCTO complementario, sin variantes ni mismo pasillo.")
    print("=" * 78)
    print(complementarias.head(args.top)[cols].to_string(index=False))

    print("\n" + "=" * 78)
    print("DESCARTADAS - variantes o sustitutos. Se muestran para control.")
    print("=" * 78)
    print(descartadas.head(8)[cols].to_string(index=False))

    print("\nReglas de producto: {:,} totales | {:,} complementarias | "
          "{:,} descartadas por variante o pasillo"
          .format(len(df), len(complementarias), len(descartadas)))
    print("Reglas de pasillo: {:,}".format(len(pasillos)))

    REPORTS_DIR.mkdir(exist_ok=True)
    s1 = REPORTS_DIR / "reglas_asociacion.csv"
    s2 = REPORTS_DIR / "reglas_asociacion_pasillos.csv"
    complementarias.to_csv(s1, index=False)
    pasillos.to_csv(s2, index=False)
    print(f"Guardado en {s1.relative_to(PROJECT_ROOT)} y "
          f"{s2.relative_to(PROJECT_ROOT)}")
    print(f"Total {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
