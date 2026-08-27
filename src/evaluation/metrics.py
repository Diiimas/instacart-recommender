# -*- coding: utf-8 -*-
"""
Metricas de evaluacion del recomendador.

Este modulo es deliberadamente independiente de los datos: recibe listas y
conjuntos de IDs y devuelve numeros. No sabe que existe Instacart, no lee
archivos y no importa pandas. Eso lo hace testeable sin tener el pipeline
listo, y garantiza que baseline y modelo se midan con exactamente el mismo
codigo.

Uso tipico:

    from src.evaluation.metrics import evaluar, comparar

    resultado = evaluar(recomendaciones, verdad, k=10, segmentos=segmentos)
    tabla = comparar({"popularidad": recs_pop, "recompra": recs_rec}, verdad)

Convenciones (propuestas, a confirmar con el equipo; ver docs/data_contract.md,
que deja la definicion del protocolo pendiente de ratificacion):

  - K = 10 por defecto.
  - Los promedios son MACRO: se calcula la metrica por usuario y despues se
    promedia entre usuarios. Cada persona pesa igual, sin importar cuanto
    compre. Es la lectura que corresponde a un KPI de negocio.
  - Ademas se devuelve el RECALL MICRO: aciertos totales sobre objetivos
    totales, sin pasar por el promedio por usuario. Ahi cada producto pesa
    igual, asi que quien compra 50 productos pesa mas que quien compra 3. Es
    la lectura de volumen agregado.
  - Precision divide por K, no por la cantidad de recomendaciones entregadas.
    Si el modelo devuelve menos de K, esa carencia se paga: en la interfaz
    real el espacio queda vacio igual.

Sobre micro y macro conviene tener presente una identidad que sorprende: bajo
la convencion de arriba, la precision MICRO y la MACRO son el mismo numero.
Como todos los usuarios tienen el mismo denominador K, promediar aciertos/K
entre N usuarios da exactamente aciertos_totales/(N*K). Por eso este modulo
expone una sola precision y dos recalls: el unico que cambia es el recall,
porque ahi el denominador es la cantidad de productos de cada persona y varia
de usuario en usuario.

Metricas que devuelve, y de donde sale cada una. Hoy el equipo tiene tres
documentos proponiendo listas distintas; esta las unifica y ninguna se
descarta:

  precision           README del repositorio
  recall              README del repositorio. Es el MACRO; se conserva con
                      este nombre por compatibilidad. Alias explicito:
                      recall_macro
  recall_micro        aciertos totales / objetivos totales. Es el numero con
                      el que Dimas evaluo los candidatos, incorporado aca para
                      que toda la tabla de la Demo salga del mismo codigo
  f1_micro            f1 calculado con el recall micro, para que la tabla sea
                      internamente consistente si se reporta esa columna
  f1                  handoff de Anastasia, como acompanamiento de las dos
                      anteriores
  hit_rate            KPI principal del marco de negocio de Leonardo, que lo
                      define como "% de pedidos en que el cliente suma al
                      menos un producto recomendado". Es una condicion binaria
                      por usuario, distinta de recall: si alguien compra 12
                      productos y le acertamos 3, recall es 3/12 y hit_rate
                      es 1
  cobertura           README del repositorio, y KPI de salud en el marco de
                      negocio
  usuarios_completos  "proporcion de usuarios con diez recomendaciones
                      validas", propuesta en el README

Bloque de NOVEDAD (opcional, requiere pasar `historial`). Todas las metricas
de arriba tratan igual a un producto que la persona compra siempre y a uno
que nunca compro, y para el negocio son cosas distintas: una reconstruye el
carrito y la otra lo agranda. Estas cuatro separan esa mitad del problema:

  novedad_ofrecida    que proporcion de los K lugares dedicamos a productos
                      fuera del historial del usuario. Es una decision de
                      diseno nuestra, no un resultado
  precision_novedad   de lo nuevo que ofrecimos, cuanto acerto
  recall_novedad      de lo nuevo que la persona compro, cuanto anticipamos
  hit_rate_novedad    a cuantos usuarios les acertamos al menos un producto
                      nuevo. Es el comparable con el 14% del Sprint 1

Nunca promediar novedad con recompra ni ponerlas en la misma escala: acertar
lo que alguien nunca compro es un problema mucho mas dificil y sus numeros
son de otro orden. Se reportan separadas a proposito.
"""
from __future__ import annotations

CATALOGO_INSTACART = 49_688


# --------------------------------------------------------------------------
# Metricas por usuario
#
# Las tres reciben lo mismo y solo cambian en el denominador. Estan separadas
# a proposito: es la forma mas facil de verificarlas a mano.
# --------------------------------------------------------------------------
def aciertos(recomendados: list[int], reales: set[int], k: int) -> int:
    """Cuantos de los primeros K recomendados estaban en la orden real."""
    return len(set(recomendados[:k]) & reales)


def precision_en_k(recomendados: list[int], reales: set[int], k: int) -> float:
    """Aciertos sobre K. Denominador: lo que mostramos."""
    return aciertos(recomendados, reales, k) / k


def recall_en_k(recomendados: list[int], reales: set[int], k: int) -> float:
    """Aciertos sobre el total de la orden real. Denominador: lo que compro."""
    if not reales:
        raise ValueError("El usuario no tiene productos en la orden real.")
    return aciertos(recomendados, reales, k) / len(reales)


def acerto(recomendados: list[int], reales: set[int], k: int) -> bool:
    """Si le acertamos al menos uno. Es la condicion del KPI de negocio."""
    return aciertos(recomendados, reales, k) > 0


# --------------------------------------------------------------------------
# Novedad
#
# Todo lo de arriba mide RECOMPRA: no distingue entre acertarle un producto
# que la persona compra siempre y uno que nunca compro. Para el negocio son
# cosas distintas: el primero reconstruye el carrito, el segundo lo agranda.
#
# Estas funciones necesitan un dato que las anteriores no usan: el historial
# de cada usuario. Un producto es NUEVO para alguien si no esta en su
# historial. Es una definicion por usuario, no por catalogo: la banana es
# nueva para quien nunca la compro, por mas que sea el producto mas vendido.
#
# Novedad NO es cobertura. La cobertura mira cuanto catalogo usamos sumando
# a todos los clientes; la novedad mira, para cada persona, cuanto de lo que
# le mostramos no conocia. Un sistema puede tener cobertura alta y novedad
# cero: alcanza con recomendarle a cada uno sus propios productos raros.
# --------------------------------------------------------------------------
def nuevos_recomendados(recomendados: list[int], historial: set[int],
                        k: int) -> set[int]:
    """De los primeros K, cuales no estaban en el historial del usuario."""
    return set(recomendados[:k]) - historial


def novedad_ofrecida(recomendados: list[int], historial: set[int],
                     k: int) -> float:
    """
    Que proporcion de los K lugares dedicamos a productos que no conoce.

    Divide por K y no por lo entregado, igual que precision: si dejamos
    lugares vacios, la novedad no sube por no haber ofrecido nada.

    Es una metrica de DISEnO, no de rendimiento: mide lo que decidimos
    mostrar, no si acertamos. Hoy da 0 porque el recomendador solo rankea
    productos del historial. Ese cero es informacion, no un error.
    """
    return len(nuevos_recomendados(recomendados, historial, k)) / k


def objetivos_nuevos(reales: set[int], historial: set[int]) -> set[int]:
    """Que compro esta vez que nunca habia comprado. El techo de la novedad."""
    return reales - historial


def aciertos_nuevos(recomendados: list[int], reales: set[int],
                    historial: set[int], k: int) -> int:
    """Cuantos productos nuevos le recomendamos que efectivamente compro."""
    return len(nuevos_recomendados(recomendados, historial, k) & reales)


# --------------------------------------------------------------------------
# Agregacion
# --------------------------------------------------------------------------
def _f1(precision: float, recall: float) -> float:
    """
    Media armonica de precision y recall.

    Se calcula sobre los promedios ya agregados, no promediando el F1 de cada
    usuario. Los dos criterios son validos, pero este deja la tabla de la demo
    verificable: quien quiera chequear que F1 = 2PR/(P+R) con las columnas a la
    vista, le va a dar.
    """
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _metricas(usuarios: list[int],
              recomendaciones: dict[int, list[int]],
              verdad: dict[int, set[int]],
              k: int,
              tamano_catalogo: int) -> dict:
    """Calcula el bloque de metricas para un conjunto dado de usuarios."""
    if not usuarios:
        return {"n_usuarios": 0, "precision": 0.0, "recall": 0.0,
                "recall_macro": 0.0, "recall_micro": 0.0,
                "f1": 0.0, "f1_micro": 0.0, "hit_rate": 0.0, "cobertura": 0.0,
                "usuarios_completos": 0.0,
                "aciertos_totales": 0, "objetivos_totales": 0}

    precisiones, recalls, hits, completos = [], [], 0, 0
    aciertos_totales, objetivos_totales = 0, 0
    recomendados_distintos: set[int] = set()

    for u in usuarios:
        recs = recomendaciones.get(u, [])
        reales = verdad[u]

        # Los acumuladores del micro no pasan por el promedio por usuario:
        # se suman aciertos y objetivos de todos y se dividen al final.
        aciertos_totales += aciertos(recs, reales, k)
        objetivos_totales += len(reales)

        precisiones.append(precision_en_k(recs, reales, k))
        recalls.append(recall_en_k(recs, reales, k))
        if acerto(recs, reales, k):
            hits += 1
        if len(set(recs[:k])) >= k:
            completos += 1
        recomendados_distintos.update(recs[:k])

    precision = sum(precisiones) / len(usuarios)
    recall = sum(recalls) / len(usuarios)
    recall_micro = aciertos_totales / objetivos_totales

    return {
        "n_usuarios": len(usuarios),
        "precision": precision,
        "recall": recall,
        "recall_macro": recall,       # alias explicito; `recall` es el macro
        "recall_micro": recall_micro,
        "f1": _f1(precision, recall),
        "f1_micro": _f1(precision, recall_micro),
        "hit_rate": hits / len(usuarios),
        "cobertura": len(recomendados_distintos) / tamano_catalogo,
        "usuarios_completos": completos / len(usuarios),
        "aciertos_totales": aciertos_totales,
        "objetivos_totales": objetivos_totales,
    }


def _metricas_novedad(usuarios: list[int],
                      recomendaciones: dict[int, list[int]],
                      verdad: dict[int, set[int]],
                      historial: dict[int, set[int]],
                      k: int) -> dict:
    """
    Bloque de novedad. Tres numeros con TRES DENOMINADORES DISTINTOS.

    Que sean distintos no es un descuido: cada uno responde una pregunta que
    solo tiene sentido sobre su propio subconjunto de usuarios.

      novedad_ofrecida   sobre TODOS los usuarios. Cuanto del carrito
                         dedicamos a lo desconocido. Es la perilla que
                         movemos nosotros.
      precision_novedad  sobre los usuarios A QUIENES LE OFRECIMOS algo
                         nuevo. De eso que arriesgamos, cuanto pego. No
                         esta definida para quien no recibio nada nuevo.
      recall_novedad     sobre los usuarios que COMPRARON algo nuevo. De lo
                         nuevo que efectivamente compro, cuanto anticipamos.
                         No esta definida para quien no compro nada nuevo.
      hit_rate_novedad   sobre los mismos que recall_novedad. A cuantos les
                         acertamos al menos un producto nuevo. Es el numero
                         comparable con el 14% que reporto el Sprint 1.

    Se devuelve el n de cada uno junto al valor. Sin eso, comparar dos
    corridas es adivinar: un promedio sobre 200 usuarios y otro sobre 26.000
    no se leen igual.
    """
    if not usuarios:
        return {"novedad_ofrecida": 0.0, "precision_novedad": 0.0,
                "recall_novedad": 0.0, "hit_rate_novedad": 0.0,
                "n_con_novedad_ofrecida": 0, "n_con_objetivo_nuevo": 0,
                "aciertos_nuevos_totales": 0, "objetivos_nuevos_totales": 0}

    ofrecidas, precisiones, recalls = [], [], []
    hits_nuevos = 0
    aciertos_nuevos_totales, objetivos_nuevos_totales = 0, 0

    for u in usuarios:
        recs = recomendaciones.get(u, [])
        reales = verdad[u]
        hist = historial.get(u, set())

        nuevos_recs = nuevos_recomendados(recs, hist, k)
        nuevos_obj = objetivos_nuevos(reales, hist)
        pegados = len(nuevos_recs & reales)

        aciertos_nuevos_totales += pegados
        objetivos_nuevos_totales += len(nuevos_obj)
        ofrecidas.append(len(nuevos_recs) / k)

        if nuevos_recs:
            precisiones.append(pegados / len(nuevos_recs))
        if nuevos_obj:
            recalls.append(pegados / len(nuevos_obj))
            if pegados > 0:
                hits_nuevos += 1

    n_obj_nuevo = len(recalls)
    return {
        "novedad_ofrecida": sum(ofrecidas) / len(usuarios),
        "precision_novedad": (sum(precisiones) / len(precisiones)
                              if precisiones else 0.0),
        "recall_novedad": (sum(recalls) / n_obj_nuevo if n_obj_nuevo else 0.0),
        "hit_rate_novedad": (hits_nuevos / n_obj_nuevo if n_obj_nuevo else 0.0),
        "n_con_novedad_ofrecida": len(precisiones),
        "n_con_objetivo_nuevo": n_obj_nuevo,
        "aciertos_nuevos_totales": aciertos_nuevos_totales,
        "objetivos_nuevos_totales": objetivos_nuevos_totales,
    }


def evaluar(recomendaciones: dict[int, list[int]],
            verdad: dict[int, set[int]],
            k: int = 10,
            segmentos: dict[int, str] | None = None,
            tamano_catalogo: int = CATALOGO_INSTACART,
            estricto: bool = True,
            historial: dict[int, set[int]] | None = None) -> dict:
    """
    Evalua un conjunto de recomendaciones contra la orden real de cada usuario.

    Parametros
    ----------
    recomendaciones : {user_id: [product_id, ...]} ordenado por relevancia.
    verdad          : {user_id: {product_id, ...}} la orden real a predecir.
                      Define el universo de usuarios evaluados.
    k               : cuantas recomendaciones se consideran.
    segmentos       : {user_id: nombre} opcional. Si se pasa, agrega el mismo
                      bloque de metricas desglosado por segmento.
    tamano_catalogo : denominador de la cobertura.
    estricto        : si True, falla cuando un usuario de `verdad` no tiene
                      recomendaciones. Ver nota abajo.
    historial       : {user_id: {product_id, ...}} lo que cada usuario compro
                      ANTES de la orden objetivo. Opcional. Si se pasa, agrega
                      el bloque "novedad". Un usuario que no aparece se trata
                      como sin historial, o sea que todo le es nuevo.

    Devuelve
    --------
    dict con las metricas globales y, si corresponde, las claves "segmentos"
    y "novedad".

    Sobre `estricto`
    ----------------
    Un usuario sin recomendaciones cuenta como cero aciertos, no se saltea.
    Saltearlo inflaria las metricas: el modelo quedaria premiado por no
    contestar. Pero como que falten usuarios suele ser un sintoma de otro
    problema (un merge que perdio filas, un filtro de mas), por defecto avisa
    en vez de seguir en silencio.
    """
    usuarios = sorted(verdad)

    sin_recomendaciones = [u for u in usuarios if not recomendaciones.get(u)]
    if sin_recomendaciones and estricto:
        raise ValueError(
            f"{len(sin_recomendaciones):,} usuarios de `verdad` no tienen "
            f"recomendaciones (ej.: {sin_recomendaciones[:5]}). "
            "Si es intencional, pasar estricto=False: van a contar como cero "
            "aciertos, que es lo correcto, pero conviene saberlo."
        )

    sin_verdad = [u for u in verdad if not verdad[u]]
    if sin_verdad:
        raise ValueError(
            f"{len(sin_verdad):,} usuarios tienen la orden real vacia. "
            "Recall no esta definido para ellos: hay que excluirlos de `verdad`."
        )

    resultado = _metricas(usuarios, recomendaciones, verdad, k, tamano_catalogo)
    resultado["k"] = k
    resultado["sin_recomendaciones"] = len(sin_recomendaciones)

    if historial is not None:
        resultado["novedad"] = _metricas_novedad(
            usuarios, recomendaciones, verdad, historial, k
        )

    if segmentos:
        por_segmento = {}
        for nombre in sorted({segmentos[u] for u in usuarios if u in segmentos}):
            del_segmento = [u for u in usuarios if segmentos.get(u) == nombre]
            por_segmento[nombre] = _metricas(
                del_segmento, recomendaciones, verdad, k, tamano_catalogo
            )
            if historial is not None:
                por_segmento[nombre]["novedad"] = _metricas_novedad(
                    del_segmento, recomendaciones, verdad, historial, k
                )
        resultado["segmentos"] = por_segmento

        sin_segmento = [u for u in usuarios if u not in segmentos]
        if sin_segmento:
            resultado["sin_segmento"] = len(sin_segmento)

    return resultado


# --------------------------------------------------------------------------
# Comparacion entre modelos
# --------------------------------------------------------------------------
def comparar(modelos: dict[str, dict[int, list[int]]],
             verdad: dict[int, set[int]],
             k: int = 10,
             referencia: str | None = None,
             metrica_lift: str = "recall",
             **kwargs) -> list[dict]:
    """
    Evalua varios modelos sobre EXACTAMENTE los mismos usuarios y devuelve la
    tabla de comparacion que va a la Demo.

    `referencia` es el modelo contra el que se calcula el lift, que segun el
    marco de negocio es el baseline de popularidad. Si no se pasa, se usa el
    primero del diccionario.

    Lift = (metrica del modelo - metrica de la referencia) / metrica de la
    referencia. Aisla cuanto aporta la personalizacion sobre una regla simple.

    Devuelve una lista de dicts, lista para `pd.DataFrame(...)`. No importa
    pandas a proposito, para que este modulo se pueda testear sin dependencias.
    """
    if not modelos:
        raise ValueError("No hay modelos para comparar.")

    referencia = referencia or next(iter(modelos))
    if referencia not in modelos:
        raise ValueError(f"La referencia '{referencia}' no esta entre los modelos.")

    filas = []
    for nombre, recomendaciones in modelos.items():
        m = evaluar(recomendaciones, verdad, k=k, **kwargs)
        filas.append({
            "modelo": nombre,
            "precision": m["precision"],
            "recall": m["recall"],
            "recall_micro": m["recall_micro"],
            "f1": m["f1"],
            "hit_rate": m["hit_rate"],
            "cobertura": m["cobertura"],
            "n_usuarios": m["n_usuarios"],
        })

    base = next(f for f in filas if f["modelo"] == referencia)[metrica_lift]
    for f in filas:
        f["lift"] = (f[metrica_lift] - base) / base if base else float("nan")

    # Todos se evaluaron sobre `verdad`, asi que el conjunto de usuarios es el
    # mismo por construccion. Este control existe para que se vea que existe:
    # si alguna vez alguien cambia la firma, que rompa aca y no en la demo.
    n = {f["n_usuarios"] for f in filas}
    if len(n) != 1:
        raise AssertionError(
            f"Los modelos se evaluaron sobre distinta cantidad de usuarios: {n}. "
            "La comparacion no es valida."
        )

    return filas


# --------------------------------------------------------------------------
# Verificacion rapida
#
# Tres usuarios inventados, con las cuentas hechas a mano. Correr con:
#     python src/evaluation/metrics.py
# Los tests formales estan en tests/test_metrics.py
# --------------------------------------------------------------------------
def _autocomprobacion() -> None:
    recomendaciones = {
        1: [10, 20, 30, 40, 50],   # acierta 20 y 30 -> 2 de 5
        2: [60, 70, 80],           # acierta 70 -> 1, pero entrego solo 3
        3: [1, 2, 3, 4, 5],        # no acierta nada
    }
    verdad = {
        1: {20, 30, 99},           # compro 3 productos
        2: {70},                   # compro 1
        3: {100, 200},             # compro 2
    }

    r = evaluar(recomendaciones, verdad, k=5, tamano_catalogo=100)

    print("Verificacion con 3 usuarios inventados (K=5, catalogo=100)")
    print(f"  precision     {r['precision']:.4f}   esperado 0.2000")
    print(f"  recall macro  {r['recall']:.4f}   esperado 0.5556")
    print(f"  recall micro  {r['recall_micro']:.4f}   esperado 0.5000")
    print(f"  f1            {r['f1']:.4f}   esperado 0.2941")
    print(f"  f1 micro      {r['f1_micro']:.4f}   esperado 0.2857")
    print(f"  hit_rate      {r['hit_rate']:.4f}   esperado 0.6667")
    print(f"  cobertura     {r['cobertura']:.4f}   esperado 0.1300")
    print()
    print("Las cuentas, para poder rehacerlas en papel:")
    print("  precision    = (2/5 + 1/5 + 0/5) / 3 = 0.6/3")
    print("  recall macro = (2/3 + 1/1 + 0/2) / 3 = 1.6667/3")
    print("  recall micro = (2 + 1 + 0) aciertos / (3 + 1 + 2) objetivos = 3/6")
    print("  f1           = 2*0.2*0.5556 / (0.2+0.5556)")
    print("  f1 micro     = 2*0.2*0.5    / (0.2+0.5)")
    print("  hit_rate     = 2 usuarios con al menos un acierto / 3")
    print("  cobertura    = 13 productos distintos recomendados / 100")
    print()
    print("Micro y macro difieren porque el usuario 2 pesa un tercio en el")
    print("macro y un sexto en el micro: compro un solo producto.")

    # ----------------------------------------------------------------------
    historial = {
        1: {20, 40, 60},   # ya conocia 20, 40 y 60
        2: {60, 70},
        3: {1, 2},
    }
    n = evaluar(recomendaciones, verdad, k=5, tamano_catalogo=100,
                historial=historial)["novedad"]

    print()
    print("Bloque de novedad, con el mismo ejemplo")
    print(f"  novedad_ofrecida   {n['novedad_ofrecida']:.4f}   esperado 0.4667")
    print(f"  precision_novedad  {n['precision_novedad']:.4f}   esperado 0.1111")
    print(f"  recall_novedad     {n['recall_novedad']:.4f}   esperado 0.2500")
    print(f"  hit_rate_novedad   {n['hit_rate_novedad']:.4f}   esperado 0.5000")
    print()
    print("Las cuentas:")
    print("  nuevos recomendados  u1 {10,30,50}  u2 {80}  u3 {3,4,5}")
    print("  objetivos nuevos     u1 {30,99}     u2 {}    u3 {100,200}")
    print("  aciertos nuevos      u1 1           u2 0     u3 0")
    print("  ofrecida  = (3/5 + 1/5 + 3/5) / 3 usuarios")
    print("  precision = (1/3 + 0/1 + 0/3) / 3 con novedad ofrecida")
    print("  recall    = (1/2 + 0/2) / 2 con objetivo nuevo  <- u2 no cuenta")
    print("  hit_rate  = 1 usuario con acierto nuevo / 2 con objetivo nuevo")
    print()
    print("El usuario 2 queda fuera de recall y hit_rate porque no compro")
    print("nada nuevo: no hay contra que medirlo. Por eso cada metrica")
    print("reporta su propio n.")


if __name__ == "__main__":
    _autocomprobacion()
