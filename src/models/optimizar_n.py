# -*- coding: utf-8 -*-
"""
Mide el sistema a distintos tamanos de Top-N para decidir si 10 es el numero.

El 10 se eligio al principio del Sprint 1 porque es el default de la
literatura, no porque lo dijeran estos datos. Este script responde la
pregunta con evidencia: que gana y que pierde el sistema en cada N.

Entrena LightGBM UNA sola vez y despues corta la misma lista rankeada en
cada N. Reentrenar por cada N no cambiaria el modelo -- el ranking es el
mismo -- y multiplicaria el tiempo por nada.

Ademas del rendimiento, calcula el techo para cada N. Eso importa porque
el techo se mueve con N: con mas lugares se puede capturar mas de un
carrito grande. Sin eso, comparar el recall entre distintos N enganaria.

Uso:
    python src/models/optimizar_n.py
    python src/models/optimizar_n.py --valores 5 10 15 20
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"

from src.evaluation.metrics import evaluar  # noqa: E402
from src.models import boosting  # noqa: E402
from src.models.baselines import cargar_verdad  # noqa: E402
from src.models.regresion_logistica import VARIABLES, recomendar  # noqa: E402

VALORES_N = [5, 8, 10, 12, 15, 20, 25, 30]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--valores", type=int, nargs="+", default=VALORES_N,
                   help="Valores de N a evaluar.")
    return p.parse_args()


def techos_por_n(usuarios: set[int], valores: list[int],
                 segmentos: dict[int, str]) -> pd.DataFrame:
    """
    Techo de recall alcanzable para cada N, global y por segmento.

    Para un usuario, lo maximo que puede acertar un sistema que solo
    recomienda desde el historial es min(repetidos, N) / items. El
    min() es el que hace que el techo dependa de N: si la persona repite
    15 productos y N es 10, cinco quedan afuera por falta de lugar.
    """
    t = pd.read_parquet(PROCESSED_DIR / "targets_train.parquet",
                        columns=["user_id", "reordered"])
    t = t[t["user_id"].isin(usuarios)]
    g = t.groupby("user_id")["reordered"].agg(["size", "sum"])
    g.columns = ["items", "repetidos"]
    g["segmento"] = pd.Series(segmentos)

    filas = []
    for n in valores:
        techo_usuario = g["repetidos"].clip(upper=n) / g["items"]
        fila = {"n": n, "techo": techo_usuario.mean()}
        for seg, sub in techo_usuario.groupby(g["segmento"]):
            fila[f"techo_{seg}"] = sub.mean()
        filas.append(fila)
    return pd.DataFrame(filas).set_index("n")


def main() -> None:
    args = parse_args()
    valores = sorted(set(args.valores))
    t0 = time.time()

    print("Leyendo features.parquet...")
    columnas = ["user_id", "product_id", "split", "segmento_usuario",
                "etiqueta"] + VARIABLES
    f = pd.read_parquet(PROCESSED_DIR / "features.parquet", columns=columnas)
    train = f[f["split"] == "train"]
    valid = f[f["split"] == "valid"]

    usuarios = set(valid["user_id"].unique())
    verdad = cargar_verdad(usuarios)
    segmentos = (valid.drop_duplicates("user_id")
                 .set_index("user_id")["segmento_usuario"].to_dict())
    print(f"  validacion: {len(verdad):,} usuarios")

    print("\nEntrenando LightGBM (una sola vez)...")
    t = time.time()
    p_lgb, modelo = boosting.entrenar(train, valid)
    print(f"  listo en {time.time() - t:.0f} s "
          f"con {modelo.best_iteration_} arboles")

    techos = techos_por_n(usuarios, valores, segmentos)

    print("\nEvaluando a cada N...")
    filas = []
    for n in valores:
        recs = recomendar(valid, p_lgb, n)
        det = evaluar(recs, verdad, k=n, segmentos=segmentos)
        fila = {
            "n": n,
            "hit_rate": det["hit_rate"],
            "recall": det["recall"],
            "precision": det["precision"],
            "cobertura": det["cobertura"],
            "techo": techos.loc[n, "techo"],
        }
        fila["pct_del_techo"] = fila["recall"] / fila["techo"]
        # F1 es la que decide. Recall solo siempre sube con N y precision
        # solo siempre baja: mirar cualquiera de las dos por separado
        # llevaria a un extremo (N enorme o N=1).
        p_, r_ = fila["precision"], fila["recall"]
        fila["f1"] = 2 * p_ * r_ / (p_ + r_) if (p_ + r_) else 0.0
        for seg, m in det["segmentos"].items():
            fila[f"recall_{seg}"] = m["recall"]
            fila[f"pct_techo_{seg}"] = m["recall"] / techos.loc[n, f"techo_{seg}"]
            # F1 por segmento: es lo unico que puede decidir si a un segmento
            # le conviene un carrito distinto. El % del techo sube siempre con
            # N, igual que el recall, asi que solo mirarlo llevaria a darle el
            # carrito mas grande a todos.
            p_, r_ = m["precision"], m["recall"]
            fila[f"f1_{seg}"] = 2 * p_ * r_ / (p_ + r_) if (p_ + r_) else 0.0
        filas.append(fila)
        print(f"  N={n:>2} listo")

    tabla = pd.DataFrame(filas)

    print("\n" + "=" * 66)
    print("RENDIMIENTO POR TAMANO DE TOP-N")
    print("=" * 66)
    print(tabla[["n", "hit_rate", "recall", "precision", "f1", "cobertura",
                 "techo", "pct_del_techo"]].round(4).to_string(index=False))

    mejor = tabla.loc[tabla["f1"].idxmax()]
    print(f"\nMejor F1: N={int(mejor['n'])} (F1={mejor['f1']:.4f})")

    # Lo que se gana y lo que se paga al agrandar el carrito. Es la lectura
    # que decide: el recall siempre sube con N, asi que mirarlo solo llevaria
    # a elegir el N mas grande posible.
    print("\nQue gana y que paga cada escalon")
    d = tabla.set_index("n")
    filas_delta = []
    for antes, despues in zip(valores, valores[1:]):
        filas_delta.append({
            "escalon": f"{antes} -> {despues}",
            "gana_recall": round(d.loc[despues, "recall"] - d.loc[antes, "recall"], 4),
            "gana_hit_rate": round(d.loc[despues, "hit_rate"] - d.loc[antes, "hit_rate"], 4),
            "paga_precision": round(d.loc[despues, "precision"] - d.loc[antes, "precision"], 4),
        })
    print(pd.DataFrame(filas_delta).to_string(index=False))

    print("\nPorcentaje del techo capturado, por segmento")
    cols = ["n"] + [c for c in tabla.columns if c.startswith("pct_techo_")]
    print(tabla[cols].round(4).to_string(index=False))

    # ----------------------------------------------------------------------
    # La pregunta que decide si conviene un carrito distinto por segmento.
    # El % del techo sube siempre con N: mirarlo solo llevaria a darle 30
    # lugares a todo el mundo. El F1 por segmento es el que puede tener su
    # maximo en un N distinto para cada uno.
    # ----------------------------------------------------------------------
    print("\n" + "=" * 66)
    print("EL N OPTIMO ES EL MISMO PARA TODOS LOS SEGMENTOS?")
    print("=" * 66)
    cols_f1 = [c for c in tabla.columns if c.startswith("f1_")]
    print(tabla[["n"] + cols_f1].round(4).to_string(index=False))

    print()
    for col in cols_f1:
        seg = col[3:]
        mejor = tabla.loc[tabla[col].idxmax()]
        n_mejor = int(mejor["n"])
        # Cuanto se pierde en ese segmento por usar 10 en vez de su optimo.
        en_diez = tabla.loc[tabla["n"] == 10, col]
        perdida = (mejor[col] - en_diez.iloc[0]) if len(en_diez) else float("nan")
        print(f"  {seg:<10} mejor N = {n_mejor:>2}  (F1 {mejor[col]:.4f})"
              f"   contra N=10 gana {perdida:+.4f}")

    REPORTS_DIR.mkdir(exist_ok=True)
    salida = REPORTS_DIR / "optimizacion_n.csv"
    tabla.to_csv(salida, index=False)
    print(f"\nGuardado en {salida.relative_to(PROJECT_ROOT)}")
    print(f"Total {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
