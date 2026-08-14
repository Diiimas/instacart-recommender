# -*- coding: utf-8 -*-
"""
Busqueda de hiperparametros de LightGBM con Optuna.

Los parametros con los que corre hoy el modelo se eligieron razonables sin
buscar nada. Este script los busca.

La regla que hace que el resultado siga siendo honesto
------------------------------------------------------
La busqueda NUNCA mira el split de validacion. Si probaramos cincuenta
combinaciones y nos quedaramos con la que mejor da ahi, el numero que
reportamos dejaria de ser una medicion y pasaria a ser el maximo de cincuenta
intentos sobre el mismo examen. Siempre da mejor, y siempre miente.

Entonces se parte ENTRENAMIENTO en tres, por usuario entero:

    ajuste (80%)  los arboles se construyen con esto
    freno  (10%)  decide cuando dejar de agregar arboles
    tuning (10%)  es lo que Optuna mira para elegir parametros

Validacion queda intacta para el numero final.

Que optimiza
------------
Recall@10 macro, que es la metrica que reportamos, y no AUC. Podria usarse AUC
porque es mas barata, pero mide otra cosa: la calidad del ordenamiento fila por
fila, no cuanto del pedido entra en una lista de diez. Optimizar lo que se
reporta evita la sorpresa de mejorar una cosa y empeorar la otra.

Uso:
    python src/models/tuning.py --trials 25
    python src/models/tuning.py --trials 40 --muestra 2000000
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import optuna
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
SALIDA_PARAMS = REPORTS_DIR / "mejores_parametros_lightgbm.json"

from src.evaluation.metrics import evaluar  # noqa: E402
from src.models.baselines import cargar_verdad  # noqa: E402
from src.models.regresion_logistica import VARIABLES, recomendar  # noqa: E402

K = 10
SEMILLA = 42

# Los que usa hoy boosting.py. Se corren como primer intento para tener el
# punto de partida dentro del mismo experimento y no comparar contra un numero
# medido en otro lado.
PARAMS_ACTUALES = {
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_child_samples": 200,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    # boosting.py no pasa reg_lambda, asi que usa el default de LightGBM, que
    # es 0. Aca va el minimo del rango porque la escala es logaritmica y no
    # admite cero; a efectos practicos es lo mismo, no regulariza nada.
    "reg_lambda": 1e-3,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trials", type=int, default=25)
    p.add_argument("--muestra", type=int, default=None,
                   help="Filas de ajuste a usar, para que cada intento sea mas rapido.")
    p.add_argument("--k", type=int, default=K)
    return p.parse_args()


def partir_en_tres(train: pd.DataFrame):
    """
    Parte entrenamiento en ajuste / freno / tuning, por usuario entero.

    Por usuario y no por fila, por la misma razon que la particion original: si
    las filas de una persona cayeran en dos lados, el modelo aprenderia de una
    parte de su orden objetivo y se lo evaluaria con la otra.
    """
    usuarios = (train["user_id"].drop_duplicates()
                .sample(frac=1.0, random_state=SEMILLA).to_numpy())
    n = len(usuarios)
    corte_1, corte_2 = int(n * 0.80), int(n * 0.90)

    de_ajuste = set(usuarios[:corte_1])
    de_freno = set(usuarios[corte_1:corte_2])
    de_tuning = set(usuarios[corte_2:])

    return (train[train["user_id"].isin(de_ajuste)],
            train[train["user_id"].isin(de_freno)],
            train[train["user_id"].isin(de_tuning)])


def main() -> None:
    args = parse_args()
    t0 = time.time()

    print("Leyendo features.parquet...")
    columnas = ["user_id", "product_id", "split", "etiqueta"] + VARIABLES
    f = pd.read_parquet(PROCESSED_DIR / "features.parquet", columns=columnas)
    train = f[f["split"] == "train"]
    del f

    ajuste, freno, tuning = partir_en_tres(train)
    if args.muestra:
        ajuste = ajuste.sample(n=min(args.muestra, len(ajuste)),
                               random_state=SEMILLA)

    print(f"  ajuste {len(ajuste):,} filas de {ajuste['user_id'].nunique():,} usuarios")
    print(f"  freno  {len(freno):,} filas de {freno['user_id'].nunique():,} usuarios")
    print(f"  tuning {len(tuning):,} filas de {tuning['user_id'].nunique():,} usuarios")
    print("  validacion: NO se toca")

    verdad_tuning = cargar_verdad(set(tuning["user_id"].unique()))

    def objetivo(trial: optuna.Trial) -> float:
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 31, 255, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 50, 500, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        }
        modelo = lgb.LGBMClassifier(
            n_estimators=2000, subsample_freq=1, random_state=SEMILLA,
            n_jobs=4, verbose=-1, **params,
        )
        modelo.fit(
            ajuste[VARIABLES], ajuste["etiqueta"],
            eval_set=[(freno[VARIABLES], freno["etiqueta"])],
            eval_metric="auc",
            callbacks=[lgb.early_stopping(50, verbose=False),
                       lgb.log_evaluation(0)],
        )
        p = modelo.predict_proba(tuning[VARIABLES])[:, 1]
        recall = evaluar(recomendar(tuning, p, args.k),
                         verdad_tuning, k=args.k)["recall"]
        trial.set_user_attr("arboles", modelo.best_iteration_)
        return recall

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    estudio = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEMILLA),
    )
    # Primero los parametros actuales, para tener la referencia medida dentro
    # del mismo experimento.
    estudio.enqueue_trial(PARAMS_ACTUALES)

    print(f"\nBuscando, {args.trials} intentos...")

    def avisar(estudio, trial):
        marca = " <- mejor" if trial.value >= estudio.best_value else ""
        print(f"  intento {trial.number + 1:>3}  recall {trial.value:.4f}"
              f"  ({trial.user_attrs.get('arboles', '?')} arboles){marca}")

    estudio.optimize(objetivo, n_trials=args.trials, callbacks=[avisar])

    base = estudio.trials[0].value
    print("\n" + "=" * 56)
    print(f"Referencia (parametros actuales): {base:.4f}")
    print(f"Mejor encontrado:                 {estudio.best_value:.4f}"
          f"   {estudio.best_value - base:+.4f}")
    print("=" * 56)
    print("\nMejores parametros")
    for clave, valor in estudio.best_params.items():
        print(f"  {clave:20s} {valor}")

    REPORTS_DIR.mkdir(exist_ok=True)
    SALIDA_PARAMS.write_text(json.dumps({
        "params": estudio.best_params,
        "recall_tuning": estudio.best_value,
        "recall_tuning_referencia": base,
        "arboles": estudio.best_trial.user_attrs.get("arboles"),
        "intentos": args.trials,
        "medido_sobre": "split de tuning, 10% de entrenamiento",
        "nota": ("Estos numeros son del split de tuning, NO de validacion. "
                 "El resultado final se mide corriendo comparar_todos.py."),
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nGuardado en {SALIDA_PARAMS.relative_to(PROJECT_ROOT)}")
    print(f"Total {(time.time() - t0) / 60:.1f} min")
    print("\nOjo: esos recall son del split de tuning. Para el numero real hay")
    print("que pasar los parametros a boosting.py y correr comparar_todos.py.")


if __name__ == "__main__":
    main()
