# -*- coding: utf-8 -*-
"""
Mide la novedad del sistema actual y cuanto cuesta agregarla con reglas.

Responde con numeros la pregunta que quedo abierta despues de la Demo 1: si
las reglas de asociacion tienen que ocupar lugares del Top-10 o ir en un
bloque aparte.

El experimento es directo. Se arma un sistema hibrido que reserva M de los
10 lugares para productos que la persona NUNCA compro, elegidos por reglas
de asociacion, y se lo mide con el mismo protocolo de siempre. Variando M se
ve exactamente que se gana en novedad y que se paga en acierto.

Sobre el scoring de las reglas: a un candidato B se le asigna la confianza
MAXIMA entre todas las reglas A -> B donde A esta en el historial del
usuario. Se usa el maximo y no la suma porque sumar favorece a los productos
consecuentes populares, que es justo el sesgo que queremos evitar: si B
aparece en reglas con medio catalogo, acumula puntaje sin que eso signifique
que ESTA persona lo quiera.

Uso:
    python src/models/evaluar_novedad.py
    python src/models/evaluar_novedad.py --slots 0 1 2 3 5
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"

from src.evaluation.metrics import evaluar  # noqa: E402
from src.models import boosting  # noqa: E402
from src.models.baselines import cargar_verdad  # noqa: E402
from src.models.regresion_logistica import VARIABLES, recomendar  # noqa: E402

K = 10


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--slots", type=int, nargs="+", default=[0, 1, 2, 3],
                   help="Cuantos de los K lugares se reservan para novedad.")
    p.add_argument("--k", type=int, default=K)
    p.add_argument("--min-confianza", type=float, default=0.02,
                   help="Confianza minima de una regla para usarla.")
    p.add_argument("--reglas-por-producto", type=int, default=15,
                   help="Cuantos consecuentes se guardan por cada antecedente.")
    return p.parse_args()


def cargar_historial(usuarios: set[int]) -> dict[int, set[int]]:
    """
    Lo que cada usuario compro ANTES de la orden objetivo.

    Sale de interacciones, que es el historial: la orden objetivo no entra.
    Es el mismo universo de pares que alimenta al modelo, asi que por
    construccion todo lo que el recomendador puede rankear ya es conocido
    para la persona. De ahi que el sistema actual tenga novedad cero.
    """
    inter = pd.read_parquet(PROCESSED_DIR / "interacciones.parquet",
                            columns=["user_id", "product_id"])
    inter = inter[inter["user_id"].isin(usuarios)]
    return {u: set(g) for u, g in
            inter.groupby("user_id")["product_id"]}


def cargar_reglas(args) -> dict[int, list[tuple[int, float]]]:
    """
    {antecedente: [(consecuente, confianza), ...]} ya podado y ordenado.

    Solo complementarias: el archivo ya viene sin variantes ni pares del
    mismo pasillo, que son sustitutos y no agrandan la canasta.
    """
    ruta = REPORTS_DIR / "reglas_asociacion.csv"
    if not ruta.exists():
        raise FileNotFoundError(
            f"Falta {ruta}. Correr antes: python src/models/reglas_asociacion.py"
        )
    r = pd.read_csv(ruta, usecols=["producto_a_id", "producto_b_id", "confianza"])
    r = r[r["confianza"] >= args.min_confianza]
    r = r.sort_values("confianza", ascending=False)

    reglas: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for a, b, c in zip(r["producto_a_id"], r["producto_b_id"], r["confianza"]):
        if len(reglas[a]) < args.reglas_por_producto:
            reglas[a].append((int(b), float(c)))
    return dict(reglas)


def candidatos_nuevos(historial: set[int],
                      reglas: dict[int, list[tuple[int, float]]],
                      cuantos: int) -> list[int]:
    """
    Productos fuera del historial, disparados por lo que la persona SI compro.

    El desempate final por product_id no es decorativo: es la misma leccion
    que dejo la fuga del baseline. Si dos candidatos empatan en confianza,
    tiene que decidir un criterio explicito y no el orden del diccionario.
    """
    puntaje: dict[int, float] = {}
    for producto in historial:
        for consecuente, confianza in reglas.get(producto, ()):
            if consecuente in historial:
                continue
            if confianza > puntaje.get(consecuente, 0.0):
                puntaje[consecuente] = confianza

    ordenados = sorted(puntaje.items(), key=lambda x: (-x[1], x[0]))
    return [p for p, _ in ordenados[:cuantos]]


def construir_hibrido_por_segmento(recs_modelo: dict[int, list[int]],
                                   historial: dict[int, set[int]],
                                   reglas: dict[int, list[tuple[int, float]]],
                                   politica: dict[str, int],
                                   segmentos: dict[int, str],
                                   k: int) -> dict:
    """
    Igual que el hibrido, pero cada usuario recibe los lugares que le
    corresponden segun su segmento.

    Existe porque el precio de un lugar de novedad no es el mismo para todos:
    medido por segmento, al cliente nuevo le sale cuatro veces mas barato que
    al heavy. Una politica unica para todos paga el precio del heavy sin
    necesidad.
    """
    hibrido = {}
    for u, recs in recs_modelo.items():
        slots = politica.get(segmentos.get(u, ""), 0)
        if slots <= 0:
            hibrido[u] = recs[:k]
            continue
        nuevos = candidatos_nuevos(historial.get(u, set()), reglas, slots)
        if not nuevos:
            hibrido[u] = recs[:k]
            continue
        hibrido[u] = recs[:k - len(nuevos)] + nuevos
    return hibrido


def construir_hibrido(recs_modelo: dict[int, list[int]],
                      historial: dict[int, set[int]],
                      reglas: dict[int, list[tuple[int, float]]],
                      slots: int, k: int) -> tuple[dict, int]:
    """
    Reemplaza los ultimos `slots` lugares por productos nuevos de las reglas.

    Se sacan los ultimos y no los primeros a proposito: son los que el modelo
    puntuo mas bajo, o sea los que menos cuesta resignar.

    Si un usuario no tiene candidatos nuevos, se le deja el Top-K del modelo
    completo. No se deja el lugar vacio: un hueco es peor que una
    recomendacion mediocre, y ademas la precision lo penalizaria igual.
    """
    if slots <= 0:
        return recs_modelo, 0

    hibrido, con_novedad = {}, 0
    for u, recs in recs_modelo.items():
        nuevos = candidatos_nuevos(historial.get(u, set()), reglas, slots)
        if not nuevos:
            hibrido[u] = recs[:k]
            continue
        conservados = recs[:k - len(nuevos)]
        hibrido[u] = conservados + nuevos
        con_novedad += 1
    return hibrido, con_novedad


def main() -> None:
    args = parse_args()
    k = args.k
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
    print(f"  validacion: {len(verdad):,} usuarios")

    print("Leyendo historial e integrando reglas...")
    historial = cargar_historial(usuarios)
    reglas = cargar_reglas(args)
    print(f"  historial de {len(historial):,} usuarios")
    print(f"  {len(reglas):,} productos disparan al menos una regla")

    print("Entrenando LightGBM (una sola vez)...")
    t = time.time()
    p_lgb, modelo = boosting.entrenar(train, valid)
    recs_modelo = recomendar(valid, p_lgb, k)
    print(f"  listo en {time.time() - t:.0f} s")

    # Cuanto de lo que la gente compra es inalcanzable desde el historial.
    # Es el techo de la novedad, el equivalente al 55% de la recompra.
    obj_nuevos = sum(len(verdad[u] - historial.get(u, set())) for u in verdad)
    obj_totales = sum(len(verdad[u]) for u in verdad)
    con_obj_nuevo = sum(1 for u in verdad if verdad[u] - historial.get(u, set()))
    print(f"\n  {obj_nuevos:,} de {obj_totales:,} productos objetivo son nuevos "
          f"({obj_nuevos / obj_totales:.1%})")
    print(f"  {con_obj_nuevo:,} usuarios ({con_obj_nuevo / len(verdad):.1%}) "
          f"compraron al menos un producto nuevo")

    print("\nEvaluando cada configuracion...")
    filas, por_segmento = [], []
    for slots in sorted(set(args.slots)):
        recs, con_novedad = construir_hibrido(
            recs_modelo, historial, reglas, slots, k)
        m = evaluar(recs, verdad, k=k, segmentos=segmentos,
                    historial=historial)
        n = m["novedad"]

        for seg, ms in m["segmentos"].items():
            ns = ms["novedad"]
            por_segmento.append({
                "segmento": seg,
                "slots_novedad": slots,
                "n_usuarios": ms["n_usuarios"],
                "hit_rate": ms["hit_rate"],
                "recall": ms["recall"],
                "aciertos_totales": ms["aciertos_totales"],
                "pct_objetivo_nuevo": (ns["objetivos_nuevos_totales"]
                                       / ms["objetivos_totales"]),
                "novedad_ofrecida": ns["novedad_ofrecida"],
                "precision_novedad": ns["precision_novedad"],
                "hit_rate_novedad": ns["hit_rate_novedad"],
                "aciertos_nuevos": ns["aciertos_nuevos_totales"],
            })
        filas.append({
            "slots_novedad": slots,
            "hit_rate": m["hit_rate"],
            "recall": m["recall"],
            "precision": m["precision"],
            "cobertura": m["cobertura"],
            "novedad_ofrecida": n["novedad_ofrecida"],
            "precision_novedad": n["precision_novedad"],
            "recall_novedad": n["recall_novedad"],
            "hit_rate_novedad": n["hit_rate_novedad"],
            "aciertos_totales": m["aciertos_totales"],
            "aciertos_nuevos": n["aciertos_nuevos_totales"],
            "usuarios_con_novedad": con_novedad or len(verdad),
        })
        print(f"  {slots} lugares de novedad: listo")

    tabla = pd.DataFrame(filas)

    print("\n" + "=" * 72)
    print("RECOMPRA - lo que se paga")
    print("=" * 72)
    print(tabla[["slots_novedad", "hit_rate", "recall", "precision",
                 "cobertura"]].round(4).to_string(index=False))

    print("\n" + "=" * 72)
    print("NOVEDAD - lo que se gana")
    print("=" * 72)
    print(tabla[["slots_novedad", "novedad_ofrecida", "precision_novedad",
                 "recall_novedad", "hit_rate_novedad",
                 "aciertos_nuevos"]].round(4).to_string(index=False))

    base = tabla.iloc[0]
    if len(tabla) > 1:
        # El precio real, en aciertos y no en decimales de una metrica. Los
        # aciertos nuevos ya estan dentro de los totales, asi que la recompra
        # resignada es la caida del total MAS lo nuevo que entro a cubrirla.
        print("\nEl precio de cada lugar de novedad")
        print("  (recompra resignada por cada acierto nuevo ganado)")
        for _, fila in tabla.iloc[1:].iterrows():
            perdidos = base["aciertos_totales"] - fila["aciertos_totales"]
            ganados = fila["aciertos_nuevos"]
            recompra_resignada = perdidos + ganados
            ratio = recompra_resignada / ganados if ganados else float("nan")
            print(f"  {int(fila['slots_novedad'])} lugares: "
                  f"resigna {int(recompra_resignada):,} aciertos de recompra "
                  f"y gana {int(ganados):,} nuevos -> {ratio:.1f} a 1")

    # ----------------------------------------------------------------------
    # Por segmento. La pregunta es donde conviene arriesgar un lugar: no
    # tiene por que ser el mismo cliente al que mejor le predecimos la
    # recompra. Un cliente sin historial tiene poco que rankear, asi que el
    # lugar que resigna vale menos.
    # ----------------------------------------------------------------------
    seg = pd.DataFrame(por_segmento)
    orden = ["nuevo", "medio", "heavy"]
    seg["_o"] = seg["segmento"].map({s: i for i, s in enumerate(orden)})
    seg = seg.sort_values(["_o", "slots_novedad"]).drop(columns="_o")

    print("\n" + "=" * 72)
    print("CUANTO DE LO QUE COMPRA CADA SEGMENTO ES NUEVO PARA EL")
    print("=" * 72)
    base_seg = seg[seg["slots_novedad"] == 0]
    print(base_seg[["segmento", "n_usuarios", "pct_objetivo_nuevo"]]
          .round(4).to_string(index=False))

    print("\n" + "=" * 72)
    print("EL PRECIO DE UN LUGAR DE NOVEDAD, POR SEGMENTO")
    print("=" * 72)
    filas_seg = []
    for s in orden:
        b = seg[(seg["segmento"] == s) & (seg["slots_novedad"] == 0)]
        for slots in sorted(set(args.slots)):
            if slots == 0:
                continue
            h = seg[(seg["segmento"] == s) & (seg["slots_novedad"] == slots)]
            if b.empty or h.empty:
                continue
            perdidos = (b["aciertos_totales"].iloc[0]
                        - h["aciertos_totales"].iloc[0])
            ganados = h["aciertos_nuevos"].iloc[0]
            filas_seg.append({
                "segmento": s,
                "slots": slots,
                "precision_novedad": round(h["precision_novedad"].iloc[0], 4),
                "hit_rate_novedad": round(h["hit_rate_novedad"].iloc[0], 4),
                "aciertos_nuevos": int(ganados),
                "recompra_resignada": int(perdidos + ganados),
                "precio": (round((perdidos + ganados) / ganados, 1)
                           if ganados else None),
            })
    print(pd.DataFrame(filas_seg).to_string(index=False))
    print("\n'precio' = aciertos de recompra resignados por cada acierto")
    print("nuevo ganado. Mas bajo es mejor.")

    # ----------------------------------------------------------------------
    # La politica que sale de lo anterior: darle lugares de novedad al que
    # los aprovecha y ninguno al que los desperdicia. Se compara contra dar
    # un lugar a todos, que es la alternativa obvia.
    # ----------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("POLITICA POR SEGMENTO vs UN LUGAR PARA TODOS")
    print("=" * 72)
    politicas = {
        "sin novedad": {},
        "1 lugar a todos": {"nuevo": 1, "medio": 1, "heavy": 1},
        "2 nuevo / 1 medio / 0 heavy": {"nuevo": 2, "medio": 1, "heavy": 0},
        "3 nuevo / 1 medio / 0 heavy": {"nuevo": 3, "medio": 1, "heavy": 0},
    }
    comp = []
    for nombre, politica in politicas.items():
        recs = construir_hibrido_por_segmento(
            recs_modelo, historial, reglas, politica, segmentos, k)
        m = evaluar(recs, verdad, k=k, historial=historial)
        comp.append({
            "politica": nombre,
            "hit_rate": round(m["hit_rate"], 4),
            "recall": round(m["recall"], 4),
            "aciertos_totales": m["aciertos_totales"],
            "aciertos_nuevos": m["novedad"]["aciertos_nuevos_totales"],
            "hit_rate_novedad": round(m["novedad"]["hit_rate_novedad"], 4),
        })
    cdf = pd.DataFrame(comp)
    b = cdf.iloc[0]
    cdf["precio"] = [
        None if not f["aciertos_nuevos"] else
        round(((b["aciertos_totales"] - f["aciertos_totales"])
               + f["aciertos_nuevos"]) / f["aciertos_nuevos"], 1)
        for _, f in cdf.iterrows()
    ]
    print(cdf.to_string(index=False))

    cdf.to_csv(REPORTS_DIR / "novedad_politicas.csv", index=False)

    REPORTS_DIR.mkdir(exist_ok=True)
    salida = REPORTS_DIR / "novedad.csv"
    tabla.to_csv(salida, index=False)
    salida_seg = REPORTS_DIR / "novedad_por_segmento.csv"
    seg.to_csv(salida_seg, index=False)
    print(f"\nGuardado en {salida.relative_to(PROJECT_ROOT)} y "
          f"{salida_seg.relative_to(PROJECT_ROOT)}")
    print(f"Total {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
