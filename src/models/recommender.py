# -*- coding: utf-8 -*-
"""
El recomendador final: dos bloques separados, con politica por segmento.

El equipo decidio el 24/08 que las sugerencias de novedad van en un bloque
aparte y no adentro del Top-10, y que su tamano depende del segmento del
cliente. Este modulo implementa esa decision.

Devuelve dos listas, no una:

  principales   HASTA K productos de recompra. Sale del ranking de LightGBM
                sobre el historial del usuario. Es donde vive el KPI del
                proyecto y NO se toca: la novedad ya no le saca lugares.

  sugerencias   HASTA N productos de descubrimiento, con N segun el segmento.
                Productos que la persona nunca compro, disparados por reglas
                de asociacion sobre lo que si compra. Se muestra aparte, tipo
                "tambien podrias necesitar".

Los dos bloques devuelven HASTA esa cantidad, no exactamente esa cantidad.
El principal queda corto cuando el usuario no tiene suficiente historial que
rankear: le pasa al 6,2% de los usuarios, y al 14,3% entre los nuevos. El de
sugerencias queda corto cuando su historial no dispara suficientes reglas.
Quien muestre esto en una interfaz tiene que contemplar bloques incompletos,
y comunicarlo como "hasta 10" y "hasta 5".

Por que dos bloques y no uno mezclado
-------------------------------------
Medido: meter novedad adentro del Top-10 cuesta 6 aciertos de recompra por
cada acierto nuevo, porque cada lugar que se le da a lo desconocido se le
quita a una apuesta segura. Separados, el bloque de novedad no compite y no
resigna nada.

Por que el tamano depende del segmento
--------------------------------------
La calidad del bloque no es la misma para todos. La primera sugerencia
acierta el 4,5% de las veces en un cliente nuevo y el 1,7% en un heavy: mas
del doble de diferencia. El cliente nuevo todavia explora -mas de la mitad de
lo que compra nunca lo compro- mientras que el heavy tiene rutina.

Ojo con un matiz que es facil pasar por alto: los numeros que motivaron la
politica por segmento (2 al nuevo, 1 al medio, 0 al heavy) salieron del
escenario donde la novedad COMPETIA por lugares. En bloque aparte no compite,
asi que el criterio ya no es el costo sino cuanta irrelevancia se le muestra
al cliente. Ver POLITICA_NOVEDAD.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = PROJECT_ROOT / "reports"

K_PRINCIPALES = 10

# Cuantas sugerencias de novedad recibe cada segmento, COMO MAXIMO.
#
# Con el bloque aparte no hay un corte que los datos impongan: la precision
# baja de a poco y no se derrumba, asi que el criterio es de producto. La
# regla tiene dos partes, y las dos hacen falta:
#
#   1. Todos los segmentos reciben al menos UNA sugerencia, si hay candidata.
#   2. Se agregan lugares mientras el ULTIMO todavia acierte al menos el
#      2,5% de las veces.
#
#   nuevo   lugares 1 a 5 rinden 4,5 / 3,6 / 2,9 / 2,8 / 2,7 %  -> 5
#   medio   lugares 1 a 3 rinden 3,0 / 2,5 / 2,1 %              -> 2
#   heavy   el primer lugar rinde 1,7 %, debajo del piso        -> 1 por (1)
#
# La primera parte existe porque el piso gobierna cuantos AGREGAR, no si
# mostrar alguno. Al heavy le mostramos uno solo aunque rinda 1,7%: en un
# bloque aparte no le saca lugar a nada, un solo item no satura la interfaz,
# y aporta 144 aciertos que de otro modo no existirian. Sin esta regla
# explicita, el heavy -que es el 32% de los clientes- no veria nunca la
# seccion, y eso es una decision de producto demasiado grande para que quede
# implicita en un umbral.
#
# Son MAXIMOS, no cantidades fijas: un usuario cuyo historial no dispare
# suficientes reglas recibe menos. Hoy le pasa al 1,6% de los nuevos, al
# 0,2% de los medios y al 0,04% de los heavy. En la interfaz hay que
# comunicar "hasta 5" y no "5".
POLITICA_NOVEDAD = {
    "nuevo": 5,
    "medio": 2,
    "heavy": 1,
}

# Una regla se usa si al menos esta proporcion de quienes compran el
# antecedente compra tambien el consecuente.
MIN_CONFIANZA = 0.02

# Cuantos consecuentes se guardan por antecedente. Cortar aca evita cargar
# reglas de cola larga que nunca van a llegar al bloque.
REGLAS_POR_PRODUCTO = 15


def cargar_reglas(ruta: Path | None = None,
                  min_confianza: float = MIN_CONFIANZA,
                  por_producto: int = REGLAS_POR_PRODUCTO
                  ) -> dict[int, list[tuple[int, float]]]:
    """
    {antecedente: [(consecuente, confianza), ...]} ordenado y podado.

    El archivo trae solo reglas complementarias: las variantes del mismo
    producto y los pares del mismo pasillo ya se descartaron al generarlo,
    porque son sustitutos y no agrandan la canasta.
    """
    ruta = ruta or REPORTS_DIR / "reglas_asociacion.csv"
    if not ruta.exists():
        raise FileNotFoundError(
            f"Falta {ruta}. Correr antes: python src/models/reglas_asociacion.py"
        )
    r = pd.read_csv(ruta, usecols=["producto_a_id", "producto_b_id", "confianza"])
    r = r[r["confianza"] >= min_confianza].sort_values("confianza", ascending=False)

    reglas: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for a, b, c in zip(r["producto_a_id"], r["producto_b_id"], r["confianza"]):
        if len(reglas[a]) < por_producto:
            reglas[a].append((int(b), float(c)))
    return dict(reglas)


def sugerencias_novedad(historial: set[int],
                        reglas: dict[int, list[tuple[int, float]]],
                        cuantas: int) -> list[int]:
    """
    Productos fuera del historial, disparados por lo que la persona si compra.

    A cada candidato se le asigna la confianza MAXIMA entre las reglas que lo
    disparan, no la suma. Sumar favoreceria a los consecuentes populares, que
    es justo el sesgo a evitar: si un producto aparece en reglas con medio
    catalogo, acumula puntaje sin que eso signifique que ESTA persona lo
    quiera.

    El desempate final por product_id es explicito a proposito. Es la misma
    leccion que dejo la fuga del Sprint 1: si dos candidatos empatan, decide
    un criterio declarado y no el orden en que quedaron las filas.
    """
    if cuantas <= 0:
        return []

    puntaje: dict[int, float] = {}
    for producto in historial:
        for consecuente, confianza in reglas.get(producto, ()):
            if consecuente in historial:
                continue
            if confianza > puntaje.get(consecuente, 0.0):
                puntaje[consecuente] = confianza

    ordenados = sorted(puntaje.items(), key=lambda x: (-x[1], x[0]))
    return [p for p, _ in ordenados[:cuantas]]


def recomendar_usuario(ranking_modelo: list[int],
                       historial: set[int],
                       segmento: str,
                       reglas: dict[int, list[tuple[int, float]]],
                       k: int = K_PRINCIPALES,
                       politica: dict[str, int] | None = None) -> dict:
    """
    Los dos bloques para un usuario.

    `ranking_modelo` viene ya ordenado por el modelo; este modulo no lo
    reordena ni lo recorta mas alla de K. Que la novedad no pueda tocar el
    Top-10 es justamente el punto de la decision del equipo.
    """
    politica = politica if politica is not None else POLITICA_NOVEDAD
    cuantas = politica.get(segmento, 0)
    return {
        "principales": ranking_modelo[:k],
        "sugerencias": sugerencias_novedad(historial, reglas, cuantas),
    }


def recomendar_lote(rankings: dict[int, list[int]],
                    historial: dict[int, set[int]],
                    segmentos: dict[int, str],
                    reglas: dict[int, list[tuple[int, float]]],
                    k: int = K_PRINCIPALES,
                    politica: dict[str, int] | None = None
                    ) -> dict[int, dict]:
    """Lo mismo para todos los usuarios de una corrida."""
    return {
        u: recomendar_usuario(ranking, historial.get(u, set()),
                              segmentos.get(u, ""), reglas, k, politica)
        for u, ranking in rankings.items()
    }


def solo_principales(recomendaciones: dict[int, dict]) -> dict[int, list[int]]:
    """
    Extrae el Top-K para medirlo con `evaluar()`.

    Existe para dejar explicito que las metricas de recompra se calculan
    SOLO sobre el bloque principal. Mezclar el de novedad ahi adentro
    ensuciaria el KPI con productos que se muestran en otro lugar de la
    interfaz.
    """
    return {u: r["principales"] for u, r in recomendaciones.items()}


def solo_sugerencias(recomendaciones: dict[int, dict]) -> dict[int, list[int]]:
    """Extrae el bloque de novedad, que se mide con sus propias metricas."""
    return {u: r["sugerencias"] for u, r in recomendaciones.items()}
