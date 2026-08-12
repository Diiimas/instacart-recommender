# -*- coding: utf-8 -*-
"""
LightGBM: el candidato a modelo final.

Mismo planteo que la regresion logistica —una probabilidad por par
(usuario, producto), ordenar y cortar en K— pero con arboles impulsados en
lugar de una combinacion lineal. La diferencia practica es que puede aprender
relaciones no lineales y, sobre todo, interacciones: por ejemplo que "hace 20
dias que no lo compra" signifique cosas distintas segun cada cuanto compra esa
persona. La regresion logistica solo puede sumar pesos.

Dos cosas que este modelo NO necesita, y la logistica si:

- Normalizar las variables. Los arboles parten por umbrales, asi que la escala
  de cada columna les da igual.
- Imputar los nulos de ciclos_desde_ultima_compra. LightGBM los manda a un
  lado del corte y aprende cual conviene.

Usa exactamente la misma lista de variables que la regresion logistica, que se
importa de ahi en vez de repetirse. Si los dos modelos vieran columnas
distintas, la comparacion hablaria de los datos y no del algoritmo.

Uso:
    python src/models/boosting.py
    python src/models/boosting.py --muestra 500000   # prueba rapida
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import lightgbm as lgb
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"

from src.evaluation.metrics import comparar, evaluar  # noqa: E402
from src.models.baselines import (  # noqa: E402
    cargar_verdad, recomendaciones_popularidad, recomendaciones_recompra,
)
from src.models.regresion_logistica import (  # noqa: E402
    VARIABLES, recomendar,
)

K = 10
SEMILLA = 42
TECHOS = {"heavy": 0.6596, "medio": 0.5491, "nuevo": 0.4443}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--muestra", type=int, default=None)
    p.add_argument("--arboles", type=int, default=600)
    p.add_argument("--k", type=int, default=K)
    return p.parse_args()


def cortar_para_early_stopping(train: pd.DataFrame, proporcion: float = 0.10):
    """
    Aparta un pedazo de ENTRENAMIENTO para decidir cuando parar de agregar
    arboles.

    Tiene que salir de entrenamiento y no de validacion. Si se usara el split
    de validacion para elegir la cantidad de arboles, esa decision estaria
    tomada mirando el conjunto con el que despues se reporta el resultado, y el
    numero final quedaria optimista.

    El corte es por usuario entero, por la misma razon que la particion
    original.
    """
    usuarios = train["user_id"].drop_duplicates().sort_values()
    corte = int(len(usuarios) * (1 - proporcion))
    mezclados = usuarios.sample(frac=1.0, random_state=SEMILLA)
    de_ajuste = set(mezclados.iloc[:corte])
    es_ajuste = train["user_id"].isin(de_ajuste)
    return train[es_ajuste], train[~es_ajuste]


def main() -> None:
    args = parse_args()
    t0 = time.time()

    print("Leyendo features.parquet...")
    columnas = ["user_id", "product_id", "split", "segmento_usuario",
                "etiqueta"] + VARIABLES
    f = pd.read_parquet(PROCESSED_DIR / "features.parquet", columns=columnas)

    train = f[f["split"] == "train"]
    valid = f[f["split"] == "valid"]
    if args.muestra:
        train = train.sample(n=min(args.muestra, len(train)),
                             random_state=SEMILLA)

    ajuste, freno = cortar_para_early_stopping(train)
    print(f"  ajuste     {len(ajuste):,} filas de {ajuste['user_id'].nunique():,} usuarios")
    print(f"  freno      {len(freno):,} filas de {freno['user_id'].nunique():,} usuarios")
    print(f"  validacion {len(valid):,} filas de {valid['user_id'].nunique():,} usuarios")

    modelo = lgb.LGBMClassifier(
        n_estimators=args.arboles,
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=200,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        random_state=SEMILLA,
        n_jobs=4,
        verbose=-1,
    )

    print("\nEntrenando...")
    t1 = time.time()
    modelo.fit(
        ajuste[VARIABLES], ajuste["etiqueta"],
        eval_set=[(freno[VARIABLES], freno["etiqueta"])],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(50, verbose=False),
                   lgb.log_evaluation(0)],
    )
    print(f"  listo en {time.time() - t1:.0f} s "
          f"con {modelo.best_iteration_} arboles de {args.arboles}")

    p_valid = modelo.predict_proba(valid[VARIABLES])[:, 1]
    recs_modelo = recomendar(valid, p_valid, args.k)

    usuarios = set(valid["user_id"].unique())
    verdad = cargar_verdad(usuarios)

    modelos = {
        "popularidad": recomendaciones_popularidad(verdad, args.k),
        "recompra_personal": recomendaciones_recompra(usuarios, args.k),
        "lightgbm": recs_modelo,
    }

    tabla = pd.DataFrame(
        comparar(modelos, verdad, k=args.k, referencia="popularidad"))
    print("\nComparacion sobre validacion")
    print(tabla[["modelo", "precision", "recall", "recall_micro",
                 "hit_rate", "cobertura", "lift"]]
          .round(4).to_string(index=False))

    segmentos = (valid.drop_duplicates("user_id")
                 .set_index("user_id")["segmento_usuario"].to_dict())
    det = evaluar(recs_modelo, verdad, k=args.k, segmentos=segmentos)
    por_seg = pd.DataFrame(det["segmentos"]).T
    por_seg["techo"] = pd.Series(TECHOS)
    por_seg["pct_del_techo"] = por_seg["recall"] / por_seg["techo"]
    print("\nLightGBM por segmento")
    print(por_seg[["n_usuarios", "recall", "techo", "pct_del_techo",
                   "hit_rate"]].round(4).to_string())

    # Dos formas de medir importancia, y no dicen lo mismo:
    #
    #   split: cuantas veces el arbol partio por esa variable. Es la que
    #          devuelve LightGBM por defecto, y favorece a las variables con
    #          muchos valores distintos, que ofrecen mas puntos de corte
    #          aunque cada corte aporte poco.
    #   gain:  cuanto bajo la funcion de perdida gracias a esos cortes. Es la
    #          que responde "que variable sirve", y es la que va a la Demo.
    importancia = pd.DataFrame({
        "variable": VARIABLES,
        "gain": modelo.booster_.feature_importance(importance_type="gain"),
        "split": modelo.booster_.feature_importance(importance_type="split"),
    })
    for col in ("gain", "split"):
        importancia[f"pct_{col}"] = (importancia[col]
                                     / importancia[col].sum())
    importancia = importancia.sort_values("gain", ascending=False)
    print("\nImportancia de variables, por ganancia")
    print(importancia.head(10)[["variable", "pct_gain", "pct_split"]]
          .round(4).to_string(index=False))

    REPORTS_DIR.mkdir(exist_ok=True)
    tabla.to_csv(REPORTS_DIR / "comparacion_lightgbm.csv", index=False)
    importancia.to_csv(REPORTS_DIR / "importancia_lightgbm.csv", index=False)
    print(f"\nGuardado en reports/. Total {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
