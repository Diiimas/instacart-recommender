# -*- coding: utf-8 -*-
"""
LightGBM con objetivo de ranking (lambdarank), contra el clasificador binario.

Por que existe este experimento
-------------------------------
El modelo que usamos hoy entrena como CLASIFICADOR: por cada par
usuario-producto se pregunta "va a estar en el proximo pedido, si o no", y
optimiza acertar esa respuesta fila por fila. Despues, para recomendar,
ordenamos por probabilidad y cortamos en diez.

Pero lo que nos importa no es acertar la probabilidad de cada fila: es que el
ORDEN dentro de cada cliente sea bueno. Que un producto tenga probabilidad
0,30 o 0,45 da igual mientras quede por delante de los que la persona no va a
comprar.

Un clasificador no sabe eso. Trata todas las filas del dataset como
independientes y le pesa igual equivocarse en un cliente que en otro.
LambdaRank si: agrupa las filas por usuario y optimiza directamente la
calidad del orden dentro de cada grupo.

Es la continuacion natural de la escalera que ya contamos en la Demo 1:

    combinar senales a mano  ->  aprender los pesos  ->  no linealidad

y ahora, optimizar el objetivo correcto.

Que se mantiene igual, para que la comparacion valga
----------------------------------------------------
Las mismas variables, el mismo split temporal, los mismos usuarios de
validacion, el mismo early stopping sobre un pedazo de ENTRENAMIENTO y la
misma funcion evaluar(). Lo unico que cambia es el objetivo.

Uso:
    python src/models/ranking.py
    python src/models/ranking.py --arboles 800
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"

from src.evaluation.metrics import comparar  # noqa: E402
from src.models import boosting  # noqa: E402
from src.models.baselines import cargar_verdad  # noqa: E402
from src.models.boosting import SEMILLA, cortar_para_early_stopping  # noqa: E402
from src.models.regresion_logistica import VARIABLES, recomendar  # noqa: E402

K = 10


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arboles", type=int, default=600)
    p.add_argument("--k", type=int, default=K)
    return p.parse_args()


def por_grupos(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Ordena por usuario y devuelve el tamano de cada grupo.

    LambdaRank necesita saber que filas pertenecen al mismo cliente, y lo
    recibe como una lista de tamanos que asume las filas ya contiguas. Si el
    orden no coincide con los tamanos, entrena mezclando clientes sin avisar
    y el resultado sale mal sin ningun error.

    El orden es por user_id ascendente, explicito, y no el que traiga el
    dataframe: es la misma leccion que dejo la fuga del Sprint 1.
    """
    ordenado = df.sort_values("user_id", kind="stable")
    tamanos = ordenado.groupby("user_id", sort=True).size().to_numpy()
    return ordenado, tamanos


def entrenar(train: pd.DataFrame, valid: pd.DataFrame, arboles: int = 600,
             k: int = K, params: dict | None = None):
    """
    Ajusta el ranker y devuelve el puntaje de cada fila de `valid`.

    Devuelve un puntaje sin escala, no una probabilidad: lambdarank solo
    garantiza el orden. Para nosotros alcanza, porque recomendar() ordena y
    corta, pero conviene tenerlo presente si alguna vez se quiere mostrar un
    numero al usuario.
    """
    if params is None:
        params, origen = boosting.cargar_parametros(True)
        print(f"  parametros: {origen}")

    ajuste, freno = cortar_para_early_stopping(train)
    ajuste, g_ajuste = por_grupos(ajuste)
    freno, g_freno = por_grupos(freno)

    modelo = lgb.LGBMRanker(
        objective="lambdarank",
        n_estimators=arboles,
        subsample_freq=1,
        random_state=SEMILLA,
        n_jobs=4,
        verbose=-1,
        # Solo importa el orden de los primeros K: es donde se juega la
        # recomendacion. Sin esto, lambdarank reparte esfuerzo en ordenar
        # bien el fondo de la lista, que nunca se muestra.
        lambdarank_truncation_level=k,
        **params,
    )
    modelo.fit(
        ajuste[VARIABLES], ajuste["etiqueta"], group=g_ajuste,
        eval_set=[(freno[VARIABLES], freno["etiqueta"])],
        eval_group=[g_freno],
        eval_at=[k],
        callbacks=[lgb.early_stopping(50, verbose=False),
                   lgb.log_evaluation(0)],
    )
    return modelo.predict(valid[VARIABLES]), modelo


def main() -> None:
    args = parse_args()
    k = args.k
    t0 = time.time()

    print("Leyendo features.parquet...")
    columnas = ["user_id", "product_id", "split", "segmento_usuario",
                "etiqueta"] + VARIABLES
    f = pd.read_parquet(PROCESSED_DIR / "features.parquet", columns=columnas)
    train = f[f["split"] == "train"]
    valid = f[f["split"] == "valid"]

    usuarios = set(valid["user_id"].unique())
    verdad = cargar_verdad(usuarios)
    print(f"  validacion: {len(verdad):,} usuarios")

    print("\nClasificador binario, el modelo actual...")
    t = time.time()
    p_clf, m_clf = boosting.entrenar(train, valid, args.arboles)
    print(f"  {time.time() - t:.0f} s con {m_clf.best_iteration_} arboles")

    print("\nLambdaRank...")
    t = time.time()
    p_rank, m_rank = entrenar(train, valid, args.arboles, k)
    print(f"  {time.time() - t:.0f} s con {m_rank.best_iteration_} arboles")

    modelos = {
        "lightgbm_clasificador": recomendar(valid, p_clf, k),
        "lightgbm_lambdarank": recomendar(valid, p_rank, k),
    }
    tabla = pd.DataFrame(comparar(modelos, verdad, k=k,
                                  referencia="lightgbm_clasificador"))

    print("\n" + "=" * 66)
    print(f"CLASIFICADOR vs LAMBDARANK · K={k} · {len(verdad):,} usuarios")
    print("=" * 66)
    print(tabla[["modelo", "precision", "recall", "hit_rate", "cobertura"]]
          .round(4).to_string(index=False))

    clf = tabla[tabla["modelo"] == "lightgbm_clasificador"].iloc[0]
    rnk = tabla[tabla["modelo"] == "lightgbm_lambdarank"].iloc[0]
    d_recall = rnk["recall"] - clf["recall"]
    d_hit = rnk["hit_rate"] - clf["hit_rate"]

    print(f"\n  recall   {d_recall:+.4f}")
    print(f"  hit rate {d_hit:+.4f}")

    # El tuning de hiperparametros dio +0,0007 de recall sobre el split de
    # tuning. Sirve de vara: si cambiar el objetivo no supera eso, tampoco
    # es la palanca.
    print("\n  Vara de comparacion: el tuning de hiperparametros dio +0,0007.")
    if d_recall > 0.005:
        print("  El objetivo SI es una palanca. Vale integrarlo.")
    elif d_recall > 0:
        print("  Mejora, pero poco. Mismo orden que el tuning.")
    else:
        print("  No mejora. El objetivo tampoco es la palanca.")

    REPORTS_DIR.mkdir(exist_ok=True)
    salida = REPORTS_DIR / "comparacion_ranking.csv"
    tabla.to_csv(salida, index=False)
    print(f"\nGuardado en {salida.relative_to(PROJECT_ROOT)}")
    print(f"Total {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
