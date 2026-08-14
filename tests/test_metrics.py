# -*- coding: utf-8 -*-
"""
Tests de src/evaluation/metrics.py

Todos los casos usan usuarios inventados con las cuentas resueltas a mano. Es
deliberado: si un test no se puede verificar en papel, no sirve para detectar
un error de definicion, que es el tipo de error que mas caro sale aca. Un bug
de calculo se nota; una metrica bien programada pero mal definida se presenta
en la Demo sin que nadie la cuestione.

Correr con:
    pytest tests/ -v
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.metrics import (  # noqa: E402
    aciertos, precision_en_k, recall_en_k, acerto, evaluar, comparar,
)


# --------------------------------------------------------------------------
# Caso de referencia: 3 usuarios, cuentas hechas a mano
# --------------------------------------------------------------------------
RECOMENDACIONES = {
    1: [10, 20, 30, 40, 50],   # acierta 20 y 30
    2: [60, 70, 80],           # acierta 70; entrego solo 3 de 5
    3: [1, 2, 3, 4, 5],        # no acierta nada
}
VERDAD = {
    1: {20, 30, 99},
    2: {70},
    3: {100, 200},
}


# --------------------------------------------------------------------------
# Metricas por usuario
# --------------------------------------------------------------------------
def test_aciertos_cuenta_la_interseccion():
    assert aciertos([10, 20, 30, 40, 50], {20, 30, 99}, 5) == 2
    assert aciertos([1, 2, 3], {100}, 5) == 0


def test_solo_se_miran_los_primeros_k():
    # El acierto esta en la posicion 4; con k=3 no deberia contarse.
    assert aciertos([1, 2, 3, 20], {20}, 3) == 0
    assert aciertos([1, 2, 3, 20], {20}, 4) == 1


def test_precision_divide_por_k_no_por_lo_entregado():
    # El usuario 2 recibe 3 recomendaciones y acierta 1.
    # Con K=5 la precision es 1/5, no 1/3: los dos huecos se pagan.
    assert precision_en_k([60, 70, 80], {70}, 5) == pytest.approx(0.2)
    assert precision_en_k([60, 70, 80], {70}, 3) == pytest.approx(1 / 3)


def test_recall_divide_por_la_orden_real():
    assert recall_en_k([10, 20, 30, 40, 50], {20, 30, 99}, 5) == pytest.approx(2 / 3)
    assert recall_en_k([60, 70, 80], {70}, 5) == pytest.approx(1.0)


def test_recall_sin_orden_real_es_un_error_no_un_cero():
    with pytest.raises(ValueError):
        recall_en_k([1, 2, 3], set(), 5)


def test_acerto_es_binario():
    assert acerto([10, 20], {20, 30, 99}, 5) is True
    assert acerto([1, 2], {100}, 5) is False


def test_precision_y_recall_se_diferencian_solo_en_el_denominador():
    # El mismo acierto leido de dos formas: 3 de 10 mostrados, 3 de 12 comprados.
    recs = list(range(10))
    reales = set(range(3)) | set(range(100, 109))
    assert len(reales) == 12
    assert precision_en_k(recs, reales, 10) == pytest.approx(0.3)
    assert recall_en_k(recs, reales, 10) == pytest.approx(0.25)


# --------------------------------------------------------------------------
# Agregacion
# --------------------------------------------------------------------------
def test_evaluar_reproduce_las_cuentas_a_mano():
    r = evaluar(RECOMENDACIONES, VERDAD, k=5, tamano_catalogo=100)

    # precision = (2/5 + 1/5 + 0/5) / 3
    assert r["precision"] == pytest.approx(0.2)
    # recall = (2/3 + 1/1 + 0/2) / 3
    assert r["recall"] == pytest.approx(0.555555, abs=1e-5)
    # f1 = 2PR/(P+R) sobre los promedios
    assert r["f1"] == pytest.approx(0.294117, abs=1e-5)
    # 2 de 3 usuarios tienen al menos un acierto
    assert r["hit_rate"] == pytest.approx(2 / 3)
    # 13 productos distintos recomendados sobre 100
    assert r["cobertura"] == pytest.approx(0.13)
    assert r["n_usuarios"] == 3
    assert r["k"] == 5


def test_el_universo_lo_define_verdad_no_las_recomendaciones():
    # Un usuario recomendado que no esta en verdad no se evalua: no lo pedimos.
    recs = dict(RECOMENDACIONES)
    recs[999] = [1, 2, 3]
    r = evaluar(recs, VERDAD, k=5, tamano_catalogo=100)
    assert r["n_usuarios"] == 3


def test_usuario_sin_recomendaciones_falla_por_defecto():
    # Que falten usuarios suele ser sintoma de un merge que perdio filas.
    incompleto = {1: RECOMENDACIONES[1], 2: RECOMENDACIONES[2]}
    with pytest.raises(ValueError, match="no tienen"):
        evaluar(incompleto, VERDAD, k=5, tamano_catalogo=100)


def test_usuario_sin_recomendaciones_cuenta_como_cero_no_se_saltea():
    # Saltearlo premiaria al modelo por no contestar.
    incompleto = {1: RECOMENDACIONES[1], 2: RECOMENDACIONES[2]}
    r = evaluar(incompleto, VERDAD, k=5, tamano_catalogo=100, estricto=False)
    assert r["n_usuarios"] == 3
    assert r["sin_recomendaciones"] == 1
    assert r["precision"] == pytest.approx(0.2)  # identico: el 3 ya aportaba 0


def test_orden_real_vacia_es_un_error():
    with pytest.raises(ValueError, match="vacia"):
        evaluar({1: [1, 2]}, {1: set()}, k=5)


# --------------------------------------------------------------------------
# Micro y macro
#
# El caso de referencia esta elegido para que los dos den distinto: el usuario
# 2 compro un solo producto y lo acerto, asi que en el macro aporta un recall
# de 1.0 con peso 1/3, y en el micro aporta 1 acierto sobre 6 objetivos.
# --------------------------------------------------------------------------
def test_recall_micro_suma_aciertos_y_objetivos_sin_promediar():
    r = evaluar(RECOMENDACIONES, VERDAD, k=5, tamano_catalogo=100)
    # aciertos: 2 + 1 + 0 = 3 | objetivos: 3 + 1 + 2 = 6
    assert r["aciertos_totales"] == 3
    assert r["objetivos_totales"] == 6
    assert r["recall_micro"] == pytest.approx(0.5)


def test_micro_y_macro_no_son_el_mismo_numero():
    r = evaluar(RECOMENDACIONES, VERDAD, k=5, tamano_catalogo=100)
    assert r["recall"] == pytest.approx(0.555555, abs=1e-5)
    assert r["recall_micro"] == pytest.approx(0.5)
    assert r["recall"] != pytest.approx(r["recall_micro"])


def test_recall_macro_es_alias_de_recall():
    # `recall` se conserva por compatibilidad; el alias existe para que la
    # tabla de la Demo no dependa de recordar cual de los dos es.
    r = evaluar(RECOMENDACIONES, VERDAD, k=5, tamano_catalogo=100)
    assert r["recall_macro"] == r["recall"]


def test_la_precision_micro_y_la_macro_coinciden_por_construccion():
    # Todos los usuarios dividen por el mismo K, asi que promediar
    # aciertos/K da lo mismo que aciertos_totales/(N*K). Por eso el modulo
    # expone una sola precision: no hay dos numeros que reportar.
    r = evaluar(RECOMENDACIONES, VERDAD, k=5, tamano_catalogo=100)
    micro_a_mano = r["aciertos_totales"] / (r["n_usuarios"] * r["k"])
    assert r["precision"] == pytest.approx(micro_a_mano)


def test_micro_pondera_por_volumen_de_compra():
    # Dos usuarios: uno compra 1 producto y se lo acertamos; el otro compra 10
    # y no le acertamos ninguno. Macro dice 0.5, micro dice 1/11.
    recs = {1: [7], 2: [99]}
    verdad = {1: {7}, 2: set(range(100, 110))}
    r = evaluar(recs, verdad, k=5, tamano_catalogo=1000, estricto=False)
    assert r["recall"] == pytest.approx(0.5)
    assert r["recall_micro"] == pytest.approx(1 / 11)


def test_f1_micro_usa_el_recall_micro():
    r = evaluar(RECOMENDACIONES, VERDAD, k=5, tamano_catalogo=100)
    # 2 * 0.2 * 0.5 / (0.2 + 0.5)
    assert r["f1_micro"] == pytest.approx(0.285714, abs=1e-5)


def test_comparar_incluye_el_recall_micro_en_la_tabla():
    tabla = comparar({"a": RECOMENDACIONES}, VERDAD, k=5, tamano_catalogo=100)
    assert tabla[0]["recall_micro"] == pytest.approx(0.5)


def test_los_micro_tambien_se_calculan_por_segmento():
    segmentos = {1: "heavy", 2: "heavy", 3: "nuevo"}
    r = evaluar(RECOMENDACIONES, VERDAD, k=5,
                segmentos=segmentos, tamano_catalogo=100)
    # heavy: aciertos 2 + 1 = 3 sobre objetivos 3 + 1 = 4
    assert r["segmentos"]["heavy"]["recall_micro"] == pytest.approx(3 / 4)
    # y su macro es (2/3 + 1/1) / 2 = 5/6, distinto
    assert r["segmentos"]["heavy"]["recall"] == pytest.approx(5 / 6)


# --------------------------------------------------------------------------
# Segmentos
# --------------------------------------------------------------------------
def test_metricas_por_segmento():
    segmentos = {1: "heavy", 2: "heavy", 3: "nuevo"}
    r = evaluar(RECOMENDACIONES, VERDAD, k=5,
                segmentos=segmentos, tamano_catalogo=100)

    assert set(r["segmentos"]) == {"heavy", "nuevo"}
    assert r["segmentos"]["heavy"]["n_usuarios"] == 2
    assert r["segmentos"]["nuevo"]["n_usuarios"] == 1

    # heavy: recall = (2/3 + 1/1) / 2
    assert r["segmentos"]["heavy"]["recall"] == pytest.approx(5 / 6)
    # nuevo: el usuario 3 no acierta nada
    assert r["segmentos"]["nuevo"]["recall"] == pytest.approx(0.0)
    assert r["segmentos"]["nuevo"]["hit_rate"] == pytest.approx(0.0)


def test_el_promedio_global_puede_esconder_un_segmento_entero():
    # Es la razon por la que la rubrica y el marco de negocio piden desglose.
    segmentos = {1: "heavy", 2: "heavy", 3: "nuevo"}
    r = evaluar(RECOMENDACIONES, VERDAD, k=5,
                segmentos=segmentos, tamano_catalogo=100)
    assert r["hit_rate"] > 0.6                      # global se ve bien
    assert r["segmentos"]["nuevo"]["hit_rate"] == 0  # y sin embargo


def test_avisa_si_hay_usuarios_sin_segmento():
    r = evaluar(RECOMENDACIONES, VERDAD, k=5,
                segmentos={1: "heavy"}, tamano_catalogo=100)
    assert r["sin_segmento"] == 2


# --------------------------------------------------------------------------
# Cobertura
# --------------------------------------------------------------------------
def test_cobertura_castiga_al_modelo_que_repite_lo_mismo():
    # Todos reciben el mismo top-3: es el caso del baseline de popularidad.
    popularidad = {u: [7, 8, 9] for u in VERDAD}
    r = evaluar(popularidad, VERDAD, k=3, tamano_catalogo=100, estricto=False)
    assert r["cobertura"] == pytest.approx(0.03)


# --------------------------------------------------------------------------
# Usuarios completos
#
# "proporcion de usuarios con diez recomendaciones validas", del README.
# Detecta al modelo que solo puede llenar el Top-K para parte del padron.
# --------------------------------------------------------------------------
def test_usuarios_completos_detecta_al_que_no_llena_el_top_k():
    # El usuario 2 recibe 3 recomendaciones; con K=5 no esta completo.
    r = evaluar(RECOMENDACIONES, VERDAD, k=5, tamano_catalogo=100)
    assert r["usuarios_completos"] == pytest.approx(2 / 3)


def test_usuarios_completos_no_cuenta_repetidos():
    # Diez slots llenos con tres productos distintos no son diez validos.
    r = evaluar({1: [7, 7, 7, 8, 9]}, {1: {7}}, k=5, tamano_catalogo=100)
    assert r["usuarios_completos"] == pytest.approx(0.0)


def test_usuarios_completos_es_uno_cuando_todos_llenan():
    r = evaluar({u: [1, 2, 3] for u in VERDAD}, VERDAD, k=3,
                tamano_catalogo=100)
    assert r["usuarios_completos"] == pytest.approx(1.0)


def test_cobertura_solo_cuenta_lo_que_entra_en_el_top_k():
    r = evaluar({1: [10, 20, 30, 40, 50]}, {1: {20}},
                k=2, tamano_catalogo=100)
    assert r["cobertura"] == pytest.approx(0.02)  # 10 y 20, no los cinco


# --------------------------------------------------------------------------
# Comparacion entre modelos
# --------------------------------------------------------------------------
def test_comparar_calcula_lift_contra_la_referencia():
    peor = {u: [1, 2, 3, 4, 5] for u in VERDAD}   # no acierta nada
    tabla = comparar({"popularidad": peor, "modelo": RECOMENDACIONES},
                     VERDAD, k=5, referencia="popularidad",
                     tamano_catalogo=100, estricto=False)

    fila = {f["modelo"]: f for f in tabla}
    assert fila["popularidad"]["recall"] == pytest.approx(0.0)
    assert fila["modelo"]["recall"] == pytest.approx(0.555555, abs=1e-5)
    # Con referencia en cero el lift no esta definido
    assert fila["modelo"]["lift"] != fila["modelo"]["lift"]  # nan


def test_comparar_lift_con_referencia_distinta_de_cero():
    flojo = {1: [20, 0, 0, 0, 0], 2: [0, 0, 0, 0, 0], 3: [0, 0, 0, 0, 0]}
    tabla = comparar({"flojo": flojo, "bueno": RECOMENDACIONES},
                     VERDAD, k=5, referencia="flojo", tamano_catalogo=100)
    fila = {f["modelo"]: f for f in tabla}

    # flojo: recall = (1/3 + 0 + 0) / 3 = 0.11111
    assert fila["flojo"]["recall"] == pytest.approx(1 / 9, abs=1e-5)
    assert fila["flojo"]["lift"] == pytest.approx(0.0)
    # bueno: (0.55555 - 0.11111) / 0.11111 = 4.0
    assert fila["bueno"]["lift"] == pytest.approx(4.0, abs=1e-4)


def test_comparar_evalua_todos_sobre_los_mismos_usuarios():
    tabla = comparar({"a": RECOMENDACIONES, "b": RECOMENDACIONES},
                     VERDAD, k=5, tamano_catalogo=100)
    assert {f["n_usuarios"] for f in tabla} == {3}


def test_comparar_rechaza_una_referencia_inexistente():
    with pytest.raises(ValueError, match="referencia"):
        comparar({"a": RECOMENDACIONES}, VERDAD, k=5,
                 referencia="no_existe", tamano_catalogo=100)
