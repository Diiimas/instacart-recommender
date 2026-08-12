# -*- coding: utf-8 -*-
"""
Los dos baselines del proyecto, reusables y restringibles a un split.

Estaban escritos dentro de notebooks/03_baseline.ipynb, que los midio sobre
los 131.209 usuarios evaluables. Para comparar contra los modelos que
aprenden hace falta medirlos sobre EXACTAMENTE los mismos usuarios de
validacion, asi que aca quedan como funciones que reciben el conjunto de
usuarios.

Ninguno de los dos aprende nada: son reglas de conteo. Se llaman baselines
porque son el punto de comparacion, no porque sean modelos simples.

Uso:
    python src/models/baselines.py --split valid
    python src/models/baselines.py --split todos
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

from src.evaluation.metrics import comparar, evaluar  # noqa: E402

K = 10


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split", choices=["train", "valid", "todos"],
                   default="valid",
                   help="Sobre que usuarios medir. Por defecto validacion.")
    p.add_argument("--k", type=int, default=K)
    return p.parse_args()


def cargar_verdad(usuarios: set[int] | None = None) -> dict[int, set[int]]:
    """La orden objetivo de cada usuario: {user_id: {product_id, ...}}."""
    t = pd.read_parquet(PROCESSED_DIR / "targets_train.parquet",
                        columns=["user_id", "product_id"])
    if usuarios is not None:
        t = t[t["user_id"].isin(usuarios)]
    return t.groupby("user_id")["product_id"].agg(set).to_dict()


def usuarios_del_split(split: str) -> set[int] | None:
    """Los usuarios de un lado de la particion, segun features.parquet."""
    if split == "todos":
        return None
    f = pd.read_parquet(PROCESSED_DIR / "features.parquet",
                        columns=["user_id", "split"])
    return set(f.loc[f["split"] == split, "user_id"].unique())


def recomendaciones_popularidad(usuarios, k: int = K) -> dict[int, list[int]]:
    """
    Los k productos mas vendidos del catalogo, iguales para todo el mundo.

    La popularidad se calcula sobre todo el historial `prior`, no solo sobre
    los usuarios de entrenamiento. No es fuga: `prior` es informacion
    disponible al momento de recomendar. Lo que nunca se toca es `train`.
    """
    productos = pd.read_parquet(PROCESSED_DIR / "productos.parquet",
                                columns=["product_id", "cantidad_compras"])
    top = productos.nlargest(k, "cantidad_compras")["product_id"].tolist()
    return {u: top for u in usuarios}


def recomendaciones_recompra(usuarios, k: int = K) -> dict[int, list[int]]:
    """
    Los k productos que cada usuario compra en mayor proporcion de sus ordenes,
    desempatados por el comprado mas recientemente y, si sigue el empate, por
    product_id ascendente.

    El tercer criterio parece un detalle y no lo es. Dos tercios de las filas
    empatan en (ratio, recencia): son productos comprados la misma cantidad de
    veces y con la misma recencia. Sin un desempate explicito, el orden lo
    decide como quedaron las filas en el archivo, y ese orden no es neutral
    —el LEFT JOIN de etiquetas deja las filas de etiqueta 1 mas arriba—, con
    lo cual el baseline se lleva 1,3 puntos de Recall que no gano.

    Se elige product_id y no un criterio con mas sentido (popularidad da
    0,3314 y reorder_rate del producto 0,3327) por dos razones: reproduce el
    0,3298 ya comunicado al equipo, y mantiene al baseline sin senal de
    producto. Cualquier mejora que venga de saber que unos productos se
    recompran mas que otros le corresponde al modelo, no a la vara con la que
    se lo mide: subir el piso achica el margen que hay que demostrar.
    """
    f = pd.read_parquet(
        PROCESSED_DIR / "features.parquet",
        columns=["user_id", "product_id", "ratio_usuario_producto",
                 "recencia_usuario_producto"],
    )
    f = f[f["user_id"].isin(usuarios)]
    f = f.sort_values(
        ["user_id", "ratio_usuario_producto", "recencia_usuario_producto",
         "product_id"],
        ascending=[True, False, True, True],
    )
    top = f.groupby("user_id").head(k)
    return top.groupby("user_id")["product_id"].agg(list).to_dict()


def main() -> None:
    args = parse_args()

    usuarios = usuarios_del_split(args.split)
    verdad = cargar_verdad(usuarios)
    if usuarios is None:
        usuarios = set(verdad)

    print(f"Baselines sobre el split '{args.split}'")
    print(f"  usuarios  {len(verdad):,}")
    print(f"  objetivos {sum(len(v) for v in verdad.values()):,}")

    modelos = {
        "popularidad": recomendaciones_popularidad(verdad, args.k),
        "recompra_personal": recomendaciones_recompra(usuarios, args.k),
    }

    segmentos = (
        pd.read_parquet(PROCESSED_DIR / "usuarios.parquet",
                        columns=["user_id", "segmento_usuario"])
        .set_index("user_id")["segmento_usuario"].loc[list(verdad)].to_dict()
    )

    tabla = pd.DataFrame(
        comparar(modelos, verdad, k=args.k, referencia="popularidad")
    )
    print("\nComparacion")
    print(tabla[["modelo", "precision", "recall", "recall_micro",
                 "hit_rate", "cobertura", "lift"]].round(4).to_string(index=False))

    det = evaluar(modelos["recompra_personal"], verdad, k=args.k,
                  segmentos=segmentos)
    por_seg = pd.DataFrame(det["segmentos"]).T
    print("\nRecompra personal por segmento")
    print(por_seg[["n_usuarios", "recall", "recall_micro", "hit_rate"]]
          .round(4).to_string())


if __name__ == "__main__":
    main()
