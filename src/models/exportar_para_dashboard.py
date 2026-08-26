# -*- coding: utf-8 -*-
"""
Exporta las recomendaciones ya calculadas, para que el dashboard no entrene.

El dashboard no puede depender de correr el modelo: entrenar tarda dos
minutos y necesita el parquet de features, que pesa 250 MB y no esta en el
repositorio. Este script corre el sistema una vez y deja archivos chicos y
autocontenidos que Streamlit lee directo.

Que deja
--------
recomendaciones_dashboard.parquet
    Formato largo, una fila por producto recomendado:
    user_id, segmento, bloque, posicion, product_id, product_name, aisle,
    acerto.

    `bloque` distingue "principal" de "sugerencia", que son las dos mitades
    del sistema y se muestran por separado en la interfaz.

    `acerto` dice si ese producto efectivamente aparecio en el proximo pedido
    del cliente. Sirve para armar ejemplos reales en el dashboard sin tener
    que recalcular nada.

metricas_dashboard.csv
    Los numeros de titular, uno por fila, con su nombre listo para mostrar.

Uso:
    python src/models/exportar_para_dashboard.py
    python src/models/exportar_para_dashboard.py --usuarios 3000
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
from src.models import boosting, recommender  # noqa: E402
from src.models.baselines import cargar_verdad  # noqa: E402
from src.models.regresion_logistica import VARIABLES, recomendar  # noqa: E402

K = 10


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--usuarios", type=int, default=None,
                   help="Exportar solo una muestra, para achicar el archivo. "
                        "La muestra respeta la proporcion de cada segmento.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    t0 = time.time()

    print("Leyendo features...")
    columnas = ["user_id", "product_id", "split", "segmento_usuario",
                "etiqueta"] + VARIABLES
    f = pd.read_parquet(PROCESSED_DIR / "features.parquet", columns=columnas)
    train, valid = f[f["split"] == "train"], f[f["split"] == "valid"]

    usuarios = set(valid["user_id"].unique())
    verdad = cargar_verdad(usuarios)
    segmentos = (valid.drop_duplicates("user_id")
                 .set_index("user_id")["segmento_usuario"].to_dict())

    inter = pd.read_parquet(PROCESSED_DIR / "interacciones.parquet",
                            columns=["user_id", "product_id"])
    inter = inter[inter["user_id"].isin(usuarios)]
    historial = {u: set(g) for u, g in inter.groupby("user_id")["product_id"]}
    reglas = recommender.cargar_reglas()

    print("Entrenando una vez...")
    p_lgb, _ = boosting.entrenar(train, valid)
    rankings = recomendar(valid, p_lgb, K)
    sistema = recommender.recomendar_lote(
        rankings, historial, segmentos, reglas, k=K)

    # Las metricas se calculan SIEMPRE sobre todos los usuarios, aunque
    # despues se exporte una muestra. Si salieran de la muestra, el dashboard
    # mostraria numeros distintos a los del informe y nadie sabria cual vale.
    m = evaluar(recommender.solo_principales(sistema), verdad, k=K,
                segmentos=segmentos, historial=historial)

    # La novedad se mide sobre el bloque de SUGERENCIAS, no sobre el
    # principal, y con el tamano de bloque de cada segmento. Medirla sobre el
    # principal da cero por definicion -ese bloque solo tiene historial- y
    # medirla con K=10 inventaria huecos que en la interfaz no existen: el
    # bloque de un heavy tiene un lugar, no diez.
    sugerencias = recommender.solo_sugerencias(sistema)
    novedad_seg, aciertos_nuevos, con_objetivo, con_acierto = {}, 0, 0, 0
    for seg, tamano in recommender.POLITICA_NOVEDAD.items():
        del_seg = [u for u in verdad if segmentos.get(u) == seg]
        if not del_seg or tamano <= 0:
            novedad_seg[seg] = {"hit_rate_novedad": 0.0, "aciertos": 0}
            continue
        n_seg = evaluar({u: sugerencias[u] for u in del_seg},
                        {u: verdad[u] for u in del_seg}, k=tamano,
                        historial={u: historial.get(u, set()) for u in del_seg},
                        estricto=False)["novedad"]
        novedad_seg[seg] = {
            "hit_rate_novedad": n_seg["hit_rate_novedad"],
            "aciertos": n_seg["aciertos_nuevos_totales"],
        }
        aciertos_nuevos += n_seg["aciertos_nuevos_totales"]
        con_objetivo += n_seg["n_con_objetivo_nuevo"]
        con_acierto += round(n_seg["hit_rate_novedad"]
                             * n_seg["n_con_objetivo_nuevo"])
    hit_rate_novedad = con_acierto / con_objetivo if con_objetivo else 0.0

    if args.usuarios:
        muestra = (pd.Series(segmentos).rename("segmento").reset_index()
                   .rename(columns={"index": "user_id"})
                   .groupby("segmento", group_keys=False)
                   .apply(lambda g: g.sample(
                       n=max(1, int(args.usuarios * len(g) / len(segmentos))),
                       random_state=42)))
        a_exportar = set(muestra["user_id"])
        print(f"  exportando una muestra de {len(a_exportar):,} usuarios")
    else:
        a_exportar = set(sistema)

    print("Armando el formato largo...")
    catalogo = pd.read_parquet(PROCESSED_DIR / "catalogo.parquet",
                               columns=["product_id", "product_name", "aisle"])

    filas = []
    for u in a_exportar:
        real = verdad[u]
        for bloque, productos in (("principal", sistema[u]["principales"]),
                                  ("sugerencia", sistema[u]["sugerencias"])):
            for i, pid in enumerate(productos, 1):
                filas.append({
                    "user_id": u,
                    "segmento": segmentos.get(u, ""),
                    "bloque": bloque,
                    "posicion": i,
                    "product_id": pid,
                    "acerto": pid in real,
                })

    largo = pd.DataFrame(filas).merge(catalogo, on="product_id", how="left")
    largo = largo.sort_values(["user_id", "bloque", "posicion"])

    REPORTS_DIR.mkdir(exist_ok=True)
    salida = REPORTS_DIR / "recomendaciones_dashboard.parquet"
    largo.to_parquet(salida, index=False)
    print(f"  {len(largo):,} filas, {salida.stat().st_size / 1e6:.1f} MB")

    # --- Los numeros de titular, con el nombre ya escrito ------------------
    metricas = [
        ("Hit Rate", m["hit_rate"], "porcentaje",
         "De cada 100 clientes, a cuantos les acertamos al menos un producto"),
        ("Recall", m["recall"], "decimal",
         "Que parte del proximo pedido logramos capturar, en promedio"),
        ("Precision", m["precision"], "decimal",
         "De los 10 que recomendamos, que proporcion acerto"),
        ("Cobertura", m["cobertura"], "porcentaje",
         "Que parte del catalogo llega a recomendarse"),
        ("Hit Rate en novedad", hit_rate_novedad, "porcentaje",
         "De los clientes que compraron algo nuevo, a cuantos les acertamos "
         "al menos un producto nuevo"),
        ("Aciertos de descubrimiento", aciertos_nuevos, "entero",
         "Productos nuevos acertados, sin resignar ninguno de recompra"),
        ("Usuarios evaluados", m["n_usuarios"], "entero",
         "Todos los de validacion, los mismos para los cinco sistemas"),
    ]
    df_m = pd.DataFrame(metricas, columns=["metrica", "valor", "formato",
                                           "descripcion"])
    df_m.to_csv(REPORTS_DIR / "metricas_dashboard.csv", index=False)

    por_seg = pd.DataFrame([
        {"segmento": s,
         "n_usuarios": d["n_usuarios"],
         "hit_rate": d["hit_rate"],
         "recall": d["recall"],
         "hit_rate_novedad": novedad_seg.get(s, {}).get("hit_rate_novedad", 0.0),
         "aciertos_novedad": novedad_seg.get(s, {}).get("aciertos", 0),
         "lugares_de_sugerencia": recommender.POLITICA_NOVEDAD.get(s, 0)}
        for s, d in m["segmentos"].items()
    ])
    por_seg.to_csv(REPORTS_DIR / "metricas_dashboard_segmento.csv", index=False)

    print("\n" + "=" * 60)
    print(df_m[["metrica", "valor"]].to_string(index=False))
    print("\n" + por_seg.round(4).to_string(index=False))
    print(f"\nGuardado en reports/recomendaciones_dashboard.parquet,")
    print("            reports/metricas_dashboard.csv y")
    print("            reports/metricas_dashboard_segmento.csv")
    print(f"Total {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
