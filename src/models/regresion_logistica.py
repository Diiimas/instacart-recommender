# -*- coding: utf-8 -*-
"""
Regresion logistica: el primer modelo del proyecto que aprende.

Los dos baselines son reglas escritas a mano. Este ajusta un peso por variable
a partir de los datos: para cada par (usuario, producto) del historial estima
la probabilidad de que aparezca en la proxima orden, se ordena por esa
probabilidad y se corta en K.

Se entrena sobre el split 'train' de features.parquet y se mide sobre 'valid',
con la misma funcion evaluar() que midio los baselines y sobre exactamente los
mismos usuarios. Sin eso la comparacion no valdria.

Uso:
    python src/models/regresion_logistica.py
    python src/models/regresion_logistica.py --muestra 500000   # prueba rapida
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"

from src.evaluation.metrics import comparar, evaluar  # noqa: E402
from src.models.baselines import (  # noqa: E402
    cargar_verdad, recomendaciones_popularidad, recomendaciones_recompra,
)

K = 10
SEMILLA = 42

# Las columnas que ve el modelo. Se listan a mano y no se toman "todas las
# numericas" a proposito: si manana el pipeline agrega una columna, entra al
# modelo solo si alguien lo decide.
VARIABLES = [
    # del par usuario-producto
    "freq_usuario_producto",
    "recencia_usuario_producto",
    "dias_registrados_desde_ultima_compra",
    "posicion_par",
    "dow_ultima_compra",
    "hora_ultima_compra",
    # del usuario
    "cantidad_ordenes_historicas",
    "productos_distintos",
    "reorder_rate_usuario",
    "posicion_media_carrito",
    "mediana_dias_entre_ordenes_historicas",
    # del producto
    "popularidad_producto",
    "usuarios_del_producto",
    "reorder_rate_producto",
    # derivadas
    "ratio_usuario_producto",
    "ciclos_desde_ultima_compra",
    "recencia_relativa",
    "posicion_relativa",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--muestra", type=int, default=None,
                   help="Filas de entrenamiento a usar, para probar rapido.")
    p.add_argument("--k", type=int, default=K)
    return p.parse_args()


def preparar(df: pd.DataFrame, mediana_ciclos: float | None = None):
    """
    Deja la matriz de variables lista para sklearn.

    `ciclos_desde_ultima_compra` viene NULL para los usuarios cuya mediana de
    dias entre ordenes es 0 (hacen varios pedidos el mismo dia). La regresion
    logistica no acepta nulos, asi que se imputa con la mediana y se agrega una
    columna que avisa que fue imputado: si esos usuarios se comportan distinto,
    el modelo puede aprenderlo en vez de tragarse un valor inventado como si
    fuera real.

    La mediana se calcula SOBRE ENTRENAMIENTO y se reutiliza en validacion.
    Calcularla sobre validacion seria dejar que el conjunto de prueba influya
    en el preprocesamiento.
    """
    X = df[VARIABLES].astype("float32").copy()
    indefinido = X["ciclos_desde_ultima_compra"].isna()
    if mediana_ciclos is None:
        mediana_ciclos = float(X.loc[~indefinido, "ciclos_desde_ultima_compra"].median())
    X["ciclos_desde_ultima_compra"] = X["ciclos_desde_ultima_compra"].fillna(mediana_ciclos)
    X["ciclo_indefinido"] = indefinido.astype("float32")
    return X, mediana_ciclos


def entrenar(train: pd.DataFrame, valid: pd.DataFrame):
    """
    Ajusta el modelo con `train` y devuelve la probabilidad de cada fila de
    `valid`, junto con el modelo y los nombres de las columnas usadas.

    Vive separado de main() para que el script de comparacion pueda entrenar
    exactamente este modelo, sin copiar el procedimiento. Si estuviera
    duplicado, los dos podrian irse separando y la tabla de la Demo dejaria de
    corresponderse con lo que reporta este modulo.
    """
    X_train, mediana = preparar(train)
    y_train = train["etiqueta"].to_numpy()
    X_valid, _ = preparar(valid, mediana)

    # La regresion logistica es sensible a la escala: sin normalizar,
    # popularidad_producto (cientos de miles) aplasta a ratio (entre 0 y 1) y
    # el optimizador no converge. El escalador se ajusta SOLO con
    # entrenamiento y se aplica a validacion.
    escalador = StandardScaler().fit(X_train)

    modelo = LogisticRegression(max_iter=1000, C=1.0, random_state=SEMILLA)
    modelo.fit(escalador.transform(X_train), y_train)

    probabilidad = modelo.predict_proba(escalador.transform(X_valid))[:, 1]
    return probabilidad, modelo, list(X_train.columns)


def recomendar(df: pd.DataFrame, probabilidad: np.ndarray, k: int) -> dict:
    """
    Convierte una probabilidad por fila en una lista de k productos por usuario.

    El desempate por product_id no es decorativo: es la misma leccion que dejo
    la fuga del baseline. Si dos filas empatan, tiene que decidir un criterio
    explicito y no el orden en que quedaron las filas.
    """
    orden = pd.DataFrame({
        "user_id": df["user_id"].to_numpy(),
        "product_id": df["product_id"].to_numpy(),
        "p": probabilidad,
    })
    orden = orden.sort_values(["user_id", "p", "product_id"],
                              ascending=[True, False, True])
    top = orden.groupby("user_id").head(k)
    return top.groupby("user_id")["product_id"].agg(list).to_dict()


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

    print(f"  entrenamiento {len(train):,} filas de {train['user_id'].nunique():,} usuarios")
    print(f"  validacion    {len(valid):,} filas de {valid['user_id'].nunique():,} usuarios")

    print("\nEntrenando...")
    t1 = time.time()
    p_valid, modelo, columnas = entrenar(train, valid)
    print(f"  listo en {time.time() - t1:.0f} s")

    recs_modelo = recomendar(valid, p_valid, args.k)

    usuarios = set(valid["user_id"].unique())
    verdad = cargar_verdad(usuarios)

    modelos = {
        "popularidad": recomendaciones_popularidad(verdad, args.k),
        "recompra_personal": recomendaciones_recompra(usuarios, args.k),
        "regresion_logistica": recs_modelo,
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
    techos = {"heavy": 0.6596, "medio": 0.5491, "nuevo": 0.4443}
    por_seg = pd.DataFrame(det["segmentos"]).T
    por_seg["techo"] = pd.Series(techos)
    por_seg["pct_del_techo"] = por_seg["recall"] / por_seg["techo"]
    print("\nRegresion logistica por segmento")
    print(por_seg[["n_usuarios", "recall", "techo", "pct_del_techo",
                   "hit_rate"]].round(4).to_string())

    pesos = (pd.DataFrame({"variable": columnas,
                           "peso": modelo.coef_[0]})
             .assign(magnitud=lambda d: d["peso"].abs())
             .sort_values("magnitud", ascending=False))
    print("\nPesos del modelo (variables ya normalizadas, comparables entre si)")
    print(pesos[["variable", "peso"]].round(4).to_string(index=False))

    REPORTS_DIR.mkdir(exist_ok=True)
    tabla.to_csv(REPORTS_DIR / "comparacion_validacion.csv", index=False)
    pesos.to_csv(REPORTS_DIR / "pesos_regresion_logistica.csv", index=False)
    print(f"\nGuardado en reports/. Total {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
