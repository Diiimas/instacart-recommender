# -*- coding: utf-8 -*-
"""
Corre los cinco sistemas y emite la tabla de comparacion de la Demo.

Existe para que ese numero salga de UNA corrida y no de cinco pegadas a mano.
Cuatro tablas copiadas de cuatro corridas distintas se desincronizan en cuanto
alguien cambia algo y no rehace todas, y el error no se nota: la tabla sigue
teniendo numeros verosimiles.

Cada sistema se pide a su propio modulo. Este script no reimplementa ninguno,
solo los ordena y los mide. Si manana se ajustan los parametros de LightGBM, la
tabla los toma sin tocar nada aca.

Los cinco se miden sobre EXACTAMENTE los mismos usuarios de validacion y con la
misma funcion evaluar(). Eso es lo que hace que la comparacion signifique algo.

Uso:
    python src/models/comparar_todos.py
    python src/models/comparar_todos.py --k 5
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

from src.evaluation.metrics import comparar, evaluar  # noqa: E402
from src.models import boosting, heuristica, regresion_logistica  # noqa: E402
from src.models.baselines import (  # noqa: E402
    cargar_verdad, recomendaciones_popularidad, recomendaciones_recompra,
)
from src.models.regresion_logistica import VARIABLES, recomendar  # noqa: E402

K = 10
TECHOS = {"heavy": 0.6596, "medio": 0.5491, "nuevo": 0.4443}

# Orden de menor a mayor complejidad. No es cosmetico: la tabla se lee como una
# escalera y el salto entre filas contiguas es lo que justifica cada escalon.
ORDEN = ["popularidad", "recompra_personal", "heuristica",
         "regresion_logistica", "lightgbm"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--k", type=int, default=K)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    t0 = time.time()

    print("Leyendo features.parquet...")
    columnas = ["user_id", "product_id", "split", "segmento_usuario",
                "etiqueta"] + VARIABLES
    f = pd.read_parquet(PROCESSED_DIR / "features.parquet", columns=columnas)
    train = f[f["split"] == "train"]
    valid = f[f["split"] == "valid"]

    usuarios = set(valid["user_id"].unique())
    verdad = cargar_verdad(usuarios)
    print(f"  validacion: {len(verdad):,} usuarios, "
          f"{sum(len(v) for v in verdad.values()):,} productos objetivo")

    modelos: dict[str, dict] = {}

    print("\nLos que no aprenden")
    t = time.time()
    modelos["popularidad"] = recomendaciones_popularidad(verdad, args.k)
    modelos["recompra_personal"] = recomendaciones_recompra(usuarios, args.k)
    modelos["heuristica"] = recomendar(
        valid, heuristica.puntuar(valid, heuristica.PESOS), args.k)
    print(f"  popularidad, recompra y heuristica listas en {time.time() - t:.0f} s")

    print("\nLos que aprenden")
    t = time.time()
    p_log, _, _ = regresion_logistica.entrenar(train, valid)
    modelos["regresion_logistica"] = recomendar(valid, p_log, args.k)
    print(f"  regresion logistica en {time.time() - t:.0f} s")

    t = time.time()
    p_lgb, modelo_lgb = boosting.entrenar(train, valid)
    modelos["lightgbm"] = recomendar(valid, p_lgb, args.k)
    print(f"  lightgbm en {time.time() - t:.0f} s "
          f"con {modelo_lgb.best_iteration_} arboles")

    ordenados = {nombre: modelos[nombre] for nombre in ORDEN}
    tabla = pd.DataFrame(
        comparar(ordenados, verdad, k=args.k, referencia="popularidad"))

    print("\n" + "=" * 62)
    print(f"TABLA DE COMPARACION · K={args.k} · {len(verdad):,} usuarios de validacion")
    print("=" * 62)
    print(tabla[["modelo", "precision", "recall", "recall_micro",
                 "hit_rate", "cobertura", "lift"]]
          .round(4).to_string(index=False))

    # De donde sale cada punto de mejora. Es la lectura que justifica haber
    # construido cinco sistemas y no dos.
    print("\nDe donde sale cada punto de Recall")
    recalls = dict(zip(tabla["modelo"], tabla["recall"]))
    escalones = [
        ("recompra_personal", "heuristica", "combinar senales a mano"),
        ("heuristica", "regresion_logistica", "aprender los pesos"),
        ("regresion_logistica", "lightgbm", "no linealidad e interacciones"),
    ]
    total = recalls["lightgbm"] - recalls["recompra_personal"]
    filas = []
    for desde, hasta, que in escalones:
        gana = recalls[hasta] - recalls[desde]
        filas.append({
            "escalon": f"{desde} -> {hasta}",
            "gana": round(gana, 4),
            "del_total": f"{gana / total:.0%}" if total else "-",
            "que_agrega": que,
        })
    print(pd.DataFrame(filas).to_string(index=False))

    print("\nPor segmento, contra el techo alcanzable")
    segmentos = (valid.drop_duplicates("user_id")
                 .set_index("user_id")["segmento_usuario"].to_dict())
    for nombre in ORDEN[1:]:
        det = evaluar(modelos[nombre], verdad, k=args.k, segmentos=segmentos)
        por_seg = pd.DataFrame(det["segmentos"]).T
        por_seg["techo"] = pd.Series(TECHOS)
        por_seg["pct_del_techo"] = por_seg["recall"] / por_seg["techo"]
        print(f"\n  {nombre}")
        print(por_seg[["n_usuarios", "recall", "techo", "pct_del_techo",
                       "hit_rate"]].round(4).to_string())

    REPORTS_DIR.mkdir(exist_ok=True)
    salida = REPORTS_DIR / "comparacion_final.csv"
    tabla.to_csv(salida, index=False)
    print(f"\nGuardado en {salida.relative_to(PROJECT_ROOT)}")
    print(f"Total {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
