# -*- coding: utf-8 -*-
"""
Tests de src/models/recommender.py

Lo que se verifica no es tanto el calculo como la DECISION: que el bloque de
novedad no pueda tocar el Top-10. Esa es la eleccion que tomo el equipo el
24/08 y la que un cambio distraido podria romper sin que ninguna metrica lo
note, porque el hit rate seguiria dando un numero verosimil.

Correr con:
    pytest tests/ -v
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.recommender import (  # noqa: E402
    sugerencias_novedad, recomendar_usuario, recomendar_lote,
    solo_principales, solo_sugerencias,
)


# --------------------------------------------------------------------------
# Caso de referencia
#
# El usuario compra 1 y 2. Las reglas dicen que quien compra 1 suele llevar
# 10 y 11, y quien compra 2 suele llevar 11 y 12.
# --------------------------------------------------------------------------
REGLAS = {
    1: [(10, 0.40), (11, 0.25)],
    2: [(11, 0.30), (12, 0.10)],
}
HISTORIAL = {1, 2}
RANKING = [1, 2, 3, 4, 5, 6, 7, 8, 9, 20, 21, 22]


def test_las_sugerencias_nunca_incluyen_lo_que_ya_compro():
    # El 1 y el 2 estan en el historial: por mas que disparen reglas, no
    # pueden volver como sugerencia.
    s = sugerencias_novedad(HISTORIAL, REGLAS, 5)
    assert 1 not in s and 2 not in s


def test_se_ordenan_por_confianza_descendente():
    # 10 tiene 0.40, 11 tiene max(0.25, 0.30) = 0.30, 12 tiene 0.10
    assert sugerencias_novedad(HISTORIAL, REGLAS, 3) == [10, 11, 12]


def test_un_candidato_toma_su_confianza_maxima_no_la_suma():
    # El 11 lo disparan las dos reglas: 0.25 desde el 1 y 0.30 desde el 2.
    # Si se sumaran daria 0.55 y quedaria primero, por delante del 10.
    # Sumar favoreceria a los consecuentes populares, que es el sesgo a
    # evitar: aparecer en muchas reglas no significa que ESTA persona lo
    # quiera.
    assert sugerencias_novedad(HISTORIAL, REGLAS, 1) == [10]


def test_empate_se_desempata_por_product_id():
    # Misma leccion que la fuga del Sprint 1: con empate decide un criterio
    # declarado, no el orden en que quedaron las filas.
    reglas = {1: [(30, 0.5), (20, 0.5)]}
    assert sugerencias_novedad({1}, reglas, 2) == [20, 30]


def test_pedir_cero_sugerencias_devuelve_lista_vacia():
    assert sugerencias_novedad(HISTORIAL, REGLAS, 0) == []


def test_usuario_sin_reglas_disparadas_no_rompe():
    assert sugerencias_novedad({999}, REGLAS, 3) == []


# --------------------------------------------------------------------------
# La decision del equipo: los dos bloques estan separados
# --------------------------------------------------------------------------
def test_el_top_10_no_cambia_por_la_politica_de_novedad():
    # El punto de la decision del 24/08: la novedad NO le saca lugares al
    # bloque principal. Si esto falla, volvimos al escenario que costaba 6
    # aciertos de recompra por cada acierto nuevo.
    sin = recomendar_usuario(RANKING, HISTORIAL, "heavy", REGLAS,
                             politica={"heavy": 0})
    con = recomendar_usuario(RANKING, HISTORIAL, "heavy", REGLAS,
                             politica={"heavy": 5})
    assert sin["principales"] == con["principales"]
    assert len(con["principales"]) == 10


def test_las_sugerencias_no_se_mezclan_con_las_principales():
    r = recomendar_usuario(RANKING, HISTORIAL, "nuevo", REGLAS,
                           politica={"nuevo": 3})
    assert set(r["principales"]) & set(r["sugerencias"]) == set()


def test_el_tamano_del_bloque_lo_define_el_segmento():
    politica = {"nuevo": 3, "medio": 2, "heavy": 1}
    for segmento, esperado in politica.items():
        r = recomendar_usuario(RANKING, HISTORIAL, segmento, REGLAS,
                               politica=politica)
        assert len(r["sugerencias"]) == esperado


def test_segmento_desconocido_no_recibe_sugerencias():
    # Mejor no mostrar nada que inventar un default silencioso.
    r = recomendar_usuario(RANKING, HISTORIAL, "?", REGLAS,
                           politica={"nuevo": 3})
    assert r["sugerencias"] == []
    assert len(r["principales"]) == 10


def test_el_ranking_del_modelo_se_respeta_tal_cual():
    r = recomendar_usuario(RANKING, HISTORIAL, "nuevo", REGLAS)
    assert r["principales"] == RANKING[:10]


# --------------------------------------------------------------------------
# Lote y extractores
# --------------------------------------------------------------------------
def test_recomendar_lote_arma_los_dos_bloques_por_usuario():
    rankings = {7: RANKING, 8: RANKING}
    historial = {7: HISTORIAL, 8: {1}}
    segmentos = {7: "nuevo", 8: "heavy"}
    politica = {"nuevo": 3, "heavy": 1}

    r = recomendar_lote(rankings, historial, segmentos, REGLAS,
                        politica=politica)
    assert len(r[7]["sugerencias"]) == 3
    assert len(r[8]["sugerencias"]) == 1
    assert solo_principales(r)[7] == RANKING[:10]
    assert solo_sugerencias(r)[8] == [10]


def test_usuario_sin_historial_no_recibe_sugerencias():
    # Sin historial no hay regla que disparar. Es el cold-start real: para
    # ese caso hace falta otra estrategia, no esta.
    r = recomendar_usuario(RANKING, set(), "nuevo", REGLAS,
                           politica={"nuevo": 5})
    assert r["sugerencias"] == []


# --------------------------------------------------------------------------
# Los tamanos son MAXIMOS, no cantidades fijas
#
# Lo senalo Dimas revisando el PR #8: la documentacion decia "10 productos" y
# "5 sugerencias", pero los dos bloques pueden venir cortos. El 6,2% de los
# usuarios recibe menos de 10 principales, y el 14,3% entre los nuevos. Si la
# interfaz asume que siempre vienen completos, se rompe con esos.
# --------------------------------------------------------------------------
def test_el_bloque_principal_puede_venir_corto():
    # Un usuario con solo 3 productos en su historial no puede recibir 10.
    r = recomendar_usuario([1, 2, 3], {1, 2, 3}, "nuevo", REGLAS,
                           politica={"nuevo": 2})
    assert len(r["principales"]) == 3


def test_el_bloque_de_sugerencias_puede_venir_corto():
    # Su historial dispara 3 candidatos, pero la politica pide 5.
    r = recomendar_usuario(RANKING, HISTORIAL, "nuevo", REGLAS,
                           politica={"nuevo": 5})
    assert len(r["sugerencias"]) == 3
    assert len(r["sugerencias"]) < 5


def test_todo_segmento_con_candidatas_recibe_al_menos_una():
    # Regla explicita de producto: el piso de precision gobierna cuantas
    # AGREGAR, no si mostrar alguna. El heavy recibe una aunque su primera
    # posicion rinda 1,7%, por debajo del piso del 2,5%.
    from src.models.recommender import POLITICA_NOVEDAD
    for segmento, cuantas in POLITICA_NOVEDAD.items():
        assert cuantas >= 1, (
            f"{segmento} quedo en 0. Si es intencional hay que cambiar este "
            "test y documentarlo: implica que ese segmento no ve nunca la "
            "seccion de sugerencias."
        )
