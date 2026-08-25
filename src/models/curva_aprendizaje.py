# -*- coding: utf-8 -*-
"""
Curva de aprendizaje: el modelo esta limitado por los datos o por las variables?

Cierra la pregunta que dejaron abierta los dos intentos anteriores. Ajustar
hiperparametros dio +0,0007 y cambiar el objetivo a lambdarank dio -0,0003:
los dos, cero. Falta saber si el problema es que le faltan DATOS o que le
faltan VARIABLES, porque la respuesta cambia por completo lo que hay que hacer
despues.

    Si la curva todavia sube  ->  con mas clientes seguiria mejorando, y
                                  conviene buscar mas datos.
    Si la curva esta plana    ->  el modelo ya aprendio todo lo que estas
                                  variables tienen para decir, y lo unico que
                                  queda es darle senal nueva.

Como se lee
-----------
Se entrena con porciones crecientes de los usuarios de ENTRENAMIENTO y se
evalua siempre sobre los MISMOS usuarios de validacion. Si el conjunto de
evaluacion cambiara entre puntos, la curva mediria dos cosas a la vez y no se
podria leer.

El muestreo es por usuario entero, igual que todas las particiones del
proyecto: cortar por filas partiria el historial de una persona entre dentro y
fuera del entrenamiento.

Uso:
    python src/models/curva_aprendizaje.py
    python src/models/curva_aprendizaje.py --fracciones 0.1 0.5 1.0
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
from src.models.boosting import SEMILLA  # noqa: E402
from src.models.regresion_logistica import VARIABLES, recomendar  # noqa: E402

K = 10
FRACCIONES = [0.25, 0.50, 0.75, 1.00]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fracciones", type=float, nargs="+", default=FRACCIONES)
    p.add_argument("--k", type=int, default=K)
    return p.parse_args()


def submuestra(train: pd.DataFrame, fraccion: float) -> pd.DataFrame:
    """
    Se queda con una fraccion de los USUARIOS de entrenamiento.

    Los subconjuntos son anidados -la muestra del 25% esta contenida en la del
    50%- porque se toma siempre el mismo orden barajado. Asi, si la curva sube,
    es porque se agregaron clientes y no porque tocaron otros mas faciles.
    """
    if fraccion >= 1.0:
        return train
    usuarios = (train["user_id"].drop_duplicates().sort_values()
                .sample(frac=1.0, random_state=SEMILLA))
    cuantos = int(len(usuarios) * fraccion)
    return train[train["user_id"].isin(set(usuarios.iloc[:cuantos]))]


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
    print(f"  entrenamiento: {train['user_id'].nunique():,} usuarios")
    print(f"  validacion:    {len(verdad):,} usuarios (fija en toda la curva)")

    filas = []
    for fraccion in sorted(args.fracciones):
        sub = submuestra(train, fraccion)
        n_usuarios = sub["user_id"].nunique()
        print(f"\n{fraccion:.0%} de los usuarios ({n_usuarios:,})...")
        t = time.time()
        p, modelo = boosting.entrenar(sub, valid)
        m = evaluar(recomendar(valid, p, k), verdad, k=k)
        filas.append({
            "fraccion": fraccion,
            "usuarios_entrenamiento": n_usuarios,
            "filas_entrenamiento": len(sub),
            "arboles": modelo.best_iteration_,
            "recall": m["recall"],
            "hit_rate": m["hit_rate"],
            "precision": m["precision"],
        })
        print(f"  recall {m['recall']:.4f} | hit rate {m['hit_rate']:.4f} "
              f"| {time.time() - t:.0f} s")

    tabla = pd.DataFrame(filas)

    print("\n" + "=" * 66)
    print("CURVA DE APRENDIZAJE")
    print("=" * 66)
    print(tabla[["fraccion", "usuarios_entrenamiento", "arboles",
                 "recall", "hit_rate"]].round(4).to_string(index=False))

    print("\nQue aporta cada tramo")
    for antes, despues in zip(filas, filas[1:]):
        d = despues["recall"] - antes["recall"]
        print(f"  {antes['fraccion']:.0%} -> {despues['fraccion']:.0%}: "
              f"recall {d:+.4f}")

    # El ultimo tramo es el que responde la pregunta: si duplicar los datos
    # al final ya casi no mueve la aguja, sumar mas clientes tampoco lo hara.
    ultimo = filas[-1]["recall"] - filas[-2]["recall"]
    print(f"\n  Vara: el tuning de hiperparametros dio +0,0007.")
    if ultimo > 0.005:
        print("  La curva todavia sube: con mas clientes seguiria mejorando.")
    else:
        print("  La curva esta PLANA. Mas datos de los mismos clientes no")
        print("  aportan: el limite lo ponen las variables, no el volumen.")

    REPORTS_DIR.mkdir(exist_ok=True)
    salida = REPORTS_DIR / "curva_aprendizaje.csv"
    tabla.to_csv(salida, index=False)
    print(f"\nGuardado en {salida.relative_to(PROJECT_ROOT)}")
    print(f"Total {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
