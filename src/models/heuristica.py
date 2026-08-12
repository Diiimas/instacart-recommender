# -*- coding: utf-8 -*-
"""
Heuristica: una mezcla de senales con pesos elegidos a mano.

Es el tercero de los cinco sistemas del plan y el ultimo que no aprende. Queda
entre el baseline de recompra —que mira una sola senal— y los modelos, que
ajustan los pesos con los datos.

Para que sirve, ahora que los dos modelos ya estan medidos
---------------------------------------------------------
Su papel original era de red de seguridad: garantizar que existiera un
recomendador personalizado aunque la regresion y el boosting no llegaran. Eso
ya no hace falta.

El papel que si cumple es de control: aisla cuanto del avance viene de
combinar varias senales y cuanto de APRENDER con que peso combinarlas. Si la
heuristica queda cerca de la regresion logistica, aprender los pesos aporta
poco y hay que decirlo. Si queda lejos, la eleccion del modelo tiene evidencia
detras en vez de ser un supuesto.

Por que no incluye co-ocurrencia
--------------------------------
El plan original la definia mezclando tambien "que productos suelen ir junto
con los que el usuario compra", lo que le permitiria proponer productos
nuevos. Esa idea ya se probo por separado: es el generador de candidatos, y su
techo sobre productos genuinamente nuevos quedo en 5,2%. Se deja afuera con
ese resultado como fundamento, no por falta de tiempo.

Uso:
    python src/models/heuristica.py
    python src/models/heuristica.py --explorar    # sensibilidad, sobre train
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"

from src.evaluation.metrics import comparar, evaluar  # noqa: E402
from src.models.baselines import (  # noqa: E402
    cargar_verdad, recomendaciones_popularidad, recomendaciones_recompra,
)
from src.models.regresion_logistica import recomendar  # noqa: E402

K = 10
TECHOS = {"heavy": 0.6596, "medio": 0.5491, "nuevo": 0.4443}

# Los pesos, elegidos a mano y no buscados. El razonamiento:
#
#   0.50 al ratio      es la senal que ya funciona: el baseline entero es
#                      esto solo, y saca 0.3271.
#   0.35 a la recencia es lo que le falta al baseline, y es el error concreto
#                      que se ve en los datos: pone arriba productos que la
#                      persona compra seguido pero abandono hace nueve
#                      ordenes.
#   0.15 al producto   una pizca de "esto en general se recompra", que ayuda a
#                      desempatar. Poco peso a proposito: es informacion del
#                      catalogo, no de la persona.
PESOS = {"ratio": 0.50, "recencia": 0.35, "producto": 0.15}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--explorar", action="store_true",
                   help="Mide varias combinaciones de pesos SOBRE TRAIN.")
    p.add_argument("--k", type=int, default=K)
    return p.parse_args()


def puntuar(df: pd.DataFrame, pesos: dict[str, float]) -> np.ndarray:
    """
    Combina tres senales llevadas todas al rango 0-1, para que los pesos
    signifiquen lo que parecen.

    La recencia viene en cantidad de ordenes transcurridas, donde 0 es "lo
    llevo en su ultima compra". Se convierte con 1/(1+r), que da 1.00 si lo
    compro recien, 0.50 una orden atras y 0.10 nueve ordenes atras. Esa caida
    rapida es deliberada: lo que abandono hace mucho casi no deberia competir.
    """
    ratio = df["ratio_usuario_producto"].to_numpy(dtype="float32")
    recencia = 1.0 / (1.0 + df["recencia_usuario_producto"].to_numpy(dtype="float32"))
    producto = df["reorder_rate_producto"].to_numpy(dtype="float32")
    return (pesos["ratio"] * ratio
            + pesos["recencia"] * recencia
            + pesos["producto"] * producto)


def evaluar_pesos(df: pd.DataFrame, verdad: dict, pesos: dict, k: int) -> dict:
    recs = recomendar(df, puntuar(df, pesos), k)
    return evaluar(recs, verdad, k=k)


def main() -> None:
    args = parse_args()
    t0 = time.time()

    columnas = ["user_id", "product_id", "split", "segmento_usuario",
                "ratio_usuario_producto", "recencia_usuario_producto",
                "reorder_rate_producto"]
    f = pd.read_parquet(PROCESSED_DIR / "features.parquet", columns=columnas)
    train = f[f["split"] == "train"]
    valid = f[f["split"] == "valid"]

    if args.explorar:
        # La exploracion va SOBRE TRAIN. Elegir los pesos mirando validacion
        # seria ajustar contra el conjunto con el que despues se reporta, y el
        # numero final quedaria optimista. Es la misma regla que se aplico al
        # early stopping de LightGBM.
        verdad_train = cargar_verdad(set(train["user_id"].unique()))
        combinaciones = [
            {"ratio": 1.00, "recencia": 0.00, "producto": 0.00},
            {"ratio": 0.70, "recencia": 0.30, "producto": 0.00},
            {"ratio": 0.50, "recencia": 0.50, "producto": 0.00},
            {"ratio": 0.50, "recencia": 0.35, "producto": 0.15},
            {"ratio": 0.35, "recencia": 0.50, "producto": 0.15},
            {"ratio": 0.30, "recencia": 0.30, "producto": 0.40},
        ]
        print("Sensibilidad a los pesos, medida sobre TRAIN\n")
        filas = []
        for p in combinaciones:
            m = evaluar_pesos(train, verdad_train, p, args.k)
            filas.append({**p, "recall": m["recall"], "hit_rate": m["hit_rate"]})
        print(pd.DataFrame(filas).round(4).to_string(index=False))
        print(f"\n{time.time() - t0:.0f} s")
        return

    usuarios = set(valid["user_id"].unique())
    verdad = cargar_verdad(usuarios)
    recs_heur = recomendar(valid, puntuar(valid, PESOS), args.k)

    modelos = {
        "popularidad": recomendaciones_popularidad(verdad, args.k),
        "recompra_personal": recomendaciones_recompra(usuarios, args.k),
        "heuristica": recs_heur,
    }
    tabla = pd.DataFrame(
        comparar(modelos, verdad, k=args.k, referencia="popularidad"))
    print(f"Heuristica con pesos {PESOS}\n")
    print("Comparacion sobre validacion")
    print(tabla[["modelo", "precision", "recall", "recall_micro",
                 "hit_rate", "cobertura", "lift"]]
          .round(4).to_string(index=False))

    segmentos = (valid.drop_duplicates("user_id")
                 .set_index("user_id")["segmento_usuario"].to_dict())
    det = evaluar(recs_heur, verdad, k=args.k, segmentos=segmentos)
    por_seg = pd.DataFrame(det["segmentos"]).T
    por_seg["techo"] = pd.Series(TECHOS)
    por_seg["pct_del_techo"] = por_seg["recall"] / por_seg["techo"]
    print("\nHeuristica por segmento")
    print(por_seg[["n_usuarios", "recall", "techo", "pct_del_techo",
                   "hit_rate"]].round(4).to_string())

    REPORTS_DIR.mkdir(exist_ok=True)
    tabla.to_csv(REPORTS_DIR / "comparacion_heuristica.csv", index=False)
    print(f"\nGuardado en reports/. Total {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
