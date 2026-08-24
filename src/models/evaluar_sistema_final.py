# -*- coding: utf-8 -*-
"""
Evalua el sistema final: dos bloques separados, con politica por segmento.

Es la configuracion que el equipo decidio el 24/08 y la que va a la Demo 2.
Produce los numeros de las dos mitades por separado, que es justamente el
punto: recompra y descubrimiento son problemas distintos y no se promedian.

Ademas verifica lo que la decision promete. Si el bloque de novedad va
aparte, el Top-10 tiene que quedar EXACTAMENTE igual que el del modelo solo.
El script lo comprueba y falla si no se cumple: es la clase de regresion que
ninguna metrica denunciaria por su cuenta, porque el hit rate seguiria dando
un numero verosimil.

Uso:
    python src/models/evaluar_sistema_final.py
"""
from __future__ import annotations

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


def main() -> None:
    t0 = time.time()

    print("Leyendo features...")
    columnas = ["user_id", "product_id", "split", "segmento_usuario",
                "etiqueta"] + VARIABLES
    f = pd.read_parquet(PROCESSED_DIR / "features.parquet", columns=columnas)
    train = f[f["split"] == "train"]
    valid = f[f["split"] == "valid"]

    usuarios = set(valid["user_id"].unique())
    verdad = cargar_verdad(usuarios)
    segmentos = (valid.drop_duplicates("user_id")
                 .set_index("user_id")["segmento_usuario"].to_dict())

    inter = pd.read_parquet(PROCESSED_DIR / "interacciones.parquet",
                            columns=["user_id", "product_id"])
    inter = inter[inter["user_id"].isin(usuarios)]
    historial = {u: set(g) for u, g in inter.groupby("user_id")["product_id"]}

    reglas = recommender.cargar_reglas()
    print(f"  {len(verdad):,} usuarios | {len(reglas):,} productos disparan reglas")

    print("Entrenando LightGBM...")
    t = time.time()
    p_lgb, modelo = boosting.entrenar(train, valid)
    rankings = recomendar(valid, p_lgb, K)
    print(f"  listo en {time.time() - t:.0f} s")

    sistema = recommender.recomendar_lote(
        rankings, historial, segmentos, reglas, k=K)
    principales = recommender.solo_principales(sistema)
    sugerencias = recommender.solo_sugerencias(sistema)

    # --- La verificacion que sostiene la decision -------------------------
    distintos = [u for u in rankings if principales[u] != rankings[u][:K]]
    if distintos:
        raise AssertionError(
            f"{len(distintos):,} usuarios tienen un Top-{K} distinto al del "
            "modelo. El bloque de novedad le esta sacando lugares al "
            "principal, que es exactamente lo que la decision del 24/08 "
            "buscaba evitar."
        )
    print(f"\n[OK] El Top-{K} es identico al del modelo solo, en los "
          f"{len(rankings):,} usuarios.")

    # --- Bloque principal: recompra --------------------------------------
    m = evaluar(principales, verdad, k=K, segmentos=segmentos,
                historial=historial)

    print("\n" + "=" * 66)
    print("BLOQUE PRINCIPAL - recompra")
    print("=" * 66)
    print(f"  Hit Rate    {m['hit_rate']:.4f}")
    print(f"  Recall      {m['recall']:.4f}")
    print(f"  Precision   {m['precision']:.4f}")
    print(f"  Cobertura   {m['cobertura']:.4f}")

    filas = [{"segmento": s,
              "n_usuarios": d["n_usuarios"],
              "hit_rate": round(d["hit_rate"], 4),
              "recall": round(d["recall"], 4)}
             for s, d in m["segmentos"].items()]
    print()
    print(pd.DataFrame(filas).to_string(index=False))

    # --- Bloque de novedad: descubrimiento --------------------------------
    # Se mide con su propio K, que es el tamano del bloque de cada segmento.
    # Usar K=10 aca inventaria huecos que en la interfaz no existen: el
    # bloque de un heavy tiene un solo lugar, no diez.
    print("\n" + "=" * 66)
    print("BLOQUE DE NOVEDAD - descubrimiento")
    print("=" * 66)

    filas_nov = []
    for segmento, tamano in recommender.POLITICA_NOVEDAD.items():
        del_seg = [u for u in verdad if segmentos.get(u) == segmento]
        if not del_seg or tamano <= 0:
            continue
        sug = {u: sugerencias[u] for u in del_seg}
        n = evaluar(sug, {u: verdad[u] for u in del_seg}, k=tamano,
                    historial={u: historial.get(u, set()) for u in del_seg},
                    estricto=False)["novedad"]
        filas_nov.append({
            "segmento": segmento,
            "lugares": tamano,
            "n_usuarios": len(del_seg),
            "precision_novedad": round(n["precision_novedad"], 4),
            "hit_rate_novedad": round(n["hit_rate_novedad"], 4),
            "aciertos_nuevos": n["aciertos_nuevos_totales"],
        })
    nov = pd.DataFrame(filas_nov)
    print(nov.to_string(index=False))

    total_nuevos = int(nov["aciertos_nuevos"].sum())
    print(f"\n  {total_nuevos:,} aciertos de descubrimiento, sin resignar "
          "ninguno de recompra.")
    print("  Los dos bloques no se promedian: son problemas distintos.")

    REPORTS_DIR.mkdir(exist_ok=True)
    nov.to_csv(REPORTS_DIR / "sistema_final_novedad.csv", index=False)
    pd.DataFrame(filas).to_csv(
        REPORTS_DIR / "sistema_final_recompra.csv", index=False)
    print(f"\nGuardado en reports/sistema_final_*.csv")
    print(f"Total {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
