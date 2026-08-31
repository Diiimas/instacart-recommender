"""
Dashboard del sistema de recomendacion - Basket Analytics (Instacart).

Correr:  streamlit run dashboard.py
Requisitos:  pip install streamlit duckdb pandas altair

Los numeros salen de los CSV reales de reports/ y de recomendaciones_dashboard.parquet
(las recomendaciones ya calculadas por el modelo final, sistema de dos bloques).
El explorador en vivo usa ese parquet: no hace falta entrenar ni tener data/processed.
"""
import os
from pathlib import Path

import duckdb
import pandas as pd
import altair as alt
import streamlit as st

# ---------------------------------------------------------------- Config + estilo
st.set_page_config(page_title="Basket Analytics · Recomendador Instacart",
                   layout="wide", initial_sidebar_state="collapsed")

AZUL, NAVY, ROJO, TINTA2, AZULCL = "#12408f", "#0a2a5e", "#e4002b", "#39568c", "#a9c7ea"

st.markdown(f"""<style>
    .stApp {{ background:#f2f7fd; }}
    h1,h2,h3,h4,h5 {{ color:{NAVY} !important; }}
    .stApp, [data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li {{ color:#12233f; }}
    .card {{ background:#fff;border:1px solid #d3e0f2;border-radius:14px;padding:16px 18px;height:100%; }}
    .card .big {{ font-size:28px;font-weight:800;color:{AZUL};line-height:1.05; }}
    .card .lbl {{ color:{TINTA2};font-size:12.5px;margin-top:6px; }}
    .seg .name {{ font-weight:700;color:{NAVY};font-size:18px; }}
    .seg .pct  {{ float:right;font-weight:800;color:{AZUL};font-size:20px; }}
    .b-prin {{ background:#eaf1fb;border:1px solid #a9c7ea;border-radius:14px;padding:16px 18px; }}
    .b-sug  {{ background:#fdecef;border:1px solid #f4b8c2;border-radius:14px;padding:16px 18px; }}
    .m-rep {{ font-size:32px;font-weight:800;color:{AZUL}; }}
    .m-desc{{ font-size:32px;font-weight:800;color:{ROJO}; }}
    .callout {{ background:#fff;border-left:4px solid {ROJO};border-radius:0 12px 12px 0;padding:14px 16px;margin-top:12px; }}
    .good {{ background:#fff;border-left:4px solid {AZUL};border-radius:0 12px 12px 0;padding:14px 16px;margin-top:12px; }}
    .brand {{ color:{TINTA2};font-size:13px;font-weight:700;letter-spacing:.08em; }}
    .stTabs [data-baseweb="tab-list"] {{ gap:4px; flex-wrap:wrap; }}
</style>""", unsafe_allow_html=True)

# ---------------------------------------------------------------- Rutas y datos
try:
    HERE = Path(__file__).parent
except NameError:
    HERE = Path.cwd()

def _find_root(p: Path) -> Path:
    for c in [p, *p.parents]:
        if (c / "data" / "processed").exists() or (c / ".git").exists():
            return c
    return p
ROOT = _find_root(Path.cwd().resolve())

def _reports_dir():
    for c in [HERE / "reports", ROOT / "reports", HERE, ROOT]:
        if (c / "comparacion_final.csv").exists() or (c / "recomendaciones_dashboard.parquet").exists():
            return c
    return HERE / "reports"
REP = _reports_dir()
PARQ = REP / "recomendaciones_dashboard.parquet"
LOGO = HERE / "logo_ba.png"

@st.cache_data
def load_csv(name):
    p = REP / name
    return pd.read_csv(p) if p.exists() else None

@st.cache_data
def rec_usuario(uid: int):
    if not PARQ.exists():
        return None
    con = duckdb.connect()
    return con.execute(f"""SELECT segmento, bloque, posicion, product_name, aisle, acerto
        FROM '{PARQ.as_posix()}' WHERE user_id={int(uid)} ORDER BY bloque DESC, posicion""").fetchdf()

@st.cache_data
def clientes_ejemplo():
    """(default, lista) de clientes de ejemplo balanceados por segmento (nuevos, medios, heavy) y con
    acierto variado: casos donde acierta mucho, con novedad, y alguno flojo. Mezclados, sin etiquetar."""
    if not PARQ.exists():
        return None, []
    con = duckdb.connect()
    df = con.execute(f"""SELECT user_id, ANY_VALUE(segmento) seg,
          SUM(CASE WHEN bloque='principal'  AND acerto THEN 1 ELSE 0 END) ap,
          SUM(CASE WHEN bloque='sugerencia' AND acerto THEN 1 ELSE 0 END) asg
        FROM '{PARQ.as_posix()}' GROUP BY user_id""").fetchdf()
    sel = []
    for s in ["nuevo", "medio", "heavy"]:
        d = df[df.seg == s]
        wow    = d[d.asg > 0].nlargest(2, "ap")          # aciertan recompra y novedad
        fuerte = d.nlargest(2, "ap")                      # recompra fuerte
        medias = d[(d.ap >= 4) & (d.ap <= 7)].head(2)     # a medias
        flojo  = d[d.ap <= 2].head(1)                     # flojos (honestidad)
        sel.append(pd.concat([wow, fuerte, medias, flojo]).drop_duplicates("user_id").head(7))
    out = pd.concat(sel).drop_duplicates("user_id")
    ids = sorted(int(u) for u in out["user_id"])
    wowall = out[out.asg > 0].nlargest(1, "ap")
    default = int(wowall["user_id"].iloc[0]) if len(wowall) else (ids[0] if ids else 13)
    return default, ids

# ---------------------------------------------------------------- Narrativa fija
PITCH = ("Le arma a cada cliente su proximo carrito a partir del historial: un bloque de recompra con lo "
         "habitual y, aparte, un bloque de sugerencias con productos nuevos. Medido con honestidad.")
DATOS = [("206.209", "clientes con historial"), ("49.688", "productos en el catalogo"),
         ("32,4 M", "compras analizadas"), ("3,4 M", "pedidos")]
FEATMAP = {
    "recencia_usuario_producto": "Recencia (hace cuanto lo compro)", "ratio_usuario_producto": "Ratio (en que % de pedidos aparece)",
    "cadencia_par": "Cadencia del par (cada cuanto lo recompra)", "dias_registrados_desde_ultima_compra": "Dias desde su ultima compra",
    "reorder_rate_producto": "Recomprabilidad del producto", "vencimiento": "Vencimiento (toca reponer)",
    "ciclos_desde_ultima_compra": "Ciclos desde la ultima compra", "freq_usuario_producto": "Frecuencia (veces que lo compro)",
    "reorder_rate_usuario": "Recompra del usuario", "ciclos_totales": "Ciclos totales del usuario",
    "dias_hasta_orden_objetivo": "Dias hasta la orden objetivo", "recencia_relativa": "Recencia relativa",
    "popularidad_producto": "Popularidad del producto", "posicion_media_carrito": "Posicion en el carrito",
    "posicion_relativa": "Posicion relativa en el carrito", "productos_distintos": "Variedad de productos del usuario",
}
REGLAS_PASILLO = [
    ("Vinos tintos", "Vinos blancos", 38.9, 7738), ("Cervezas", "Vinos tintos", 22.7, 5317),
    ("Reposteria (decoracion)", "Masas y mezclas para hornear", 7.4, 4286), ("Higiene oral", "Jabones y lociones", 7.1, 4367),
    ("Lavanderia", "Productos de limpieza", 6.3, 13928), ("Tofu y sustitutos de carne", "Congelados veganos", 6.2, 16122),
    ("Detergente de vajilla", "Lavanderia", 5.5, 12096),
]
MODELOS_NOMBRE = {"popularidad": "Popularidad (baseline)", "recompra_personal": "Recompra personal",
                  "heuristica": "Heuristica", "regresion_logistica": "Regresion logistica", "lightgbm": "LightGBM (elegido)"}
SEG_ORDEN = ["nuevo", "medio", "heavy"]
SEG_INFO = {  # nombre, % clientes, n clientes (validacion), descripcion
    "nuevo": ("Nuevos", 29, "7.616 de validacion", "5 pedidos o menos. Poco historial: es donde mas pesa el descubrimiento."),
    "medio": ("Medios", 39, "10.304 de validacion", "6 a 15 pedidos. Comportamiento mixto entre habito y exploracion."),
    "heavy": ("Heavy", 32, "8.323 de validacion", "16 pedidos o mas. Muy predecibles: la recompra rinde al maximo."),
}

# ---------------------------------------------------------------- Encabezado
head = st.columns([1, 8])
if LOGO.exists():
    head[0].image(str(LOGO), width=90)
with head[1]:
    st.markdown("<div class='brand'>BASKET ANALYTICS</div>", unsafe_allow_html=True)
    st.title("Sistema de recomendacion personalizado · Instacart")
    st.markdown(PITCH)
st.caption("Sprint 2 · numeros reales de reports/ y de recomendaciones_dashboard.parquet (modelo final, dos bloques). "
           "Comparacion de modelos sobre 26.243 usuarios de validacion.")

tabs = st.tabs(["1· Problema", "2· Clientes", "3· Predecimos", "4· Que mira el modelo",
                "5· Dos bloques", "6· Complementos", "7· Ejemplos", "8· Decision", "Explorador en vivo"])
t1, t2, t3, t4, t5, t6, t7, t8, t9 = tabs

# ---------------------------------------------------------------- 1. Problema
with t1:
    st.subheader("El problema y la oportunidad")
    st.write("Predecir el carrito habitual de cada cliente para que comprar sea mas rapido y el carrito, mas grande.")
    st.write("Instacart es un supermercado online. Su ingreso depende del tamano del carrito y la frecuencia de "
             "recompra. Como el costo de preparar y entregar una orden es practicamente fijo (cuesta casi lo mismo un "
             "carrito de 8 productos que uno de 12), cada producto adicional es ingreso con casi ningun costo extra: "
             "margen casi puro.")
    cols = st.columns(4)
    for col, (big, lbl) in zip(cols, DATOS):
        col.markdown(f"<div class='card'><div class='big'>{big}</div><div class='lbl'>{lbl}</div></div>", unsafe_allow_html=True)
    st.markdown("<div class='good'><b>La oportunidad:</b> hoy el cliente rearma su carrito a mano en cada compra. "
                "Ayudarlo a recomprar mas rapido y sugerirle lo que le falta significa carritos mas grandes y mas margen.</div>",
                unsafe_allow_html=True)

# ---------------------------------------------------------------- 2. Clientes
with t2:
    st.subheader("A quien le recomendamos")
    seg = load_csv("metricas_dashboard_segmento.csv")
    st.write("Separamos a los clientes por profundidad de historial. Pasa algo interesante: la recompra sube con la "
             "antiguedad (al heavy le acertamos casi siempre), pero el descubrimiento va al reves, rinde mas en los "
             "nuevos. Por eso el bloque de sugerencias es mas grande justo donde mas se necesita.")
    if seg is not None:
        s = seg.set_index("segmento")
        cols = st.columns(3)
        for col, k in zip(cols, SEG_ORDEN):
            nom, pct, n, desc = SEG_INFO[k]
            hr = s.loc[k, "hit_rate"] * 100
            hn = s.loc[k, "hit_rate_novedad"] * 100
            lug = int(s.loc[k, "lugares_de_sugerencia"])
            with col:
                st.markdown(f"<div class='card seg'><span class='name'>{nom}</span><span class='pct'>{pct}%</span>"
                            f"<div class='lbl'>{n}</div></div>", unsafe_allow_html=True)
                st.markdown(f"Recompra (acierto): **:blue[{hr:.0f}%]**")
                st.markdown(f"Sugerencias: **:red[hasta {lug}]** · acierto de novedad **{hn:.1f}%**")
                st.caption(desc)
        st.markdown("<div class='good'>La recompra rinde al maximo en <b>heavy</b> (91% de acierto); el descubrimiento "
                    "rinde al maximo en <b>nuevos</b> (16,5% de acierto de novedad, contra 2,3% en heavy). Un promedio "
                    "global escondia estas dos direcciones opuestas.</div>", unsafe_allow_html=True)
    else:
        st.warning("No encontre reports/metricas_dashboard_segmento.csv.")

# ---------------------------------------------------------------- 3. Predecimos (+ el limite es el dato)
with t3:
    st.subheader("Que tan bien predecimos la recompra")
    cf = load_csv("comparacion_final.csv")
    if cf is not None:
        cf = cf.copy()
        cf["Modelo"] = cf["modelo"].map(MODELOS_NOMBRE).fillna(cf["modelo"])
        cf["Hit Rate"] = (cf["hit_rate"] * 100).round(1)
        cf["Recall@10"] = cf["recall"].round(3)
        cf["Cobertura"] = (cf["cobertura"] * 100).round(1)
        cf["Lift"] = cf["lift"].round(2)
        st.write("Cinco sistemas, el mismo protocolo y los mismos 26.243 usuarios de validacion.")
        met = st.radio("Metrica a comparar:", ["Hit Rate", "Recall@10", "Lift"], horizontal=True)
        ch = alt.Chart(cf).mark_bar(cornerRadius=5).encode(
            x=alt.X("Modelo", sort=list(cf["Modelo"]), axis=alt.Axis(labelAngle=-15, title=None)),
            y=alt.Y(f"{met}:Q", title=met),
            color=alt.condition(alt.datum.Modelo == "LightGBM (elegido)", alt.value(ROJO), alt.value(AZUL)),
            tooltip=["Modelo", "Hit Rate", "Recall@10", "Cobertura", "Lift"]).properties(height=300)
        st.altair_chart(ch, width='stretch')
        st.dataframe(cf[["Modelo", "Hit Rate", "Recall@10", "Cobertura", "Lift"]], hide_index=True, width='stretch')
        st.markdown("<div class='good'><b>El Hit Rate pasa de 46 % (popularidad) a 87 % (LightGBM), casi el doble.</b> "
                    "LightGBM gana en todas las metricas (Recall@10 0,356 y lift 4,07x): capta la recompra cuatro veces "
                    "mejor que recomendar lo popular. Es el modelo elegido.</div>", unsafe_allow_html=True)
    st.markdown("#### Y despues probamos exprimirlo mas")
    st.write("Cinco caminos para mejorar el modelo: seis variables nuevas, objetivo de ranking (lambdarank), mas datos "
             "de entrenamiento, hiperparametros con Optuna y el tamano del carrito. Ninguno movio la aguja.")
    apn = load_csv("aporte_variables_nuevas.csv")
    cur = load_csv("curva_aprendizaje.csv")
    c1, c2 = st.columns(2)
    if apn is not None:
        a0 = apn.iloc[0]["hit_rate"] * 100
        a1 = apn.iloc[1]["hit_rate"] * 100
        c1.markdown(f"<div class='card'><div class='lbl'>Sumar 6 variables nuevas</div>"
                    f"<div class='big'>{a0:.2f}% &rarr; {a1:.2f}%</div><div class='lbl'>Hit Rate: casi sin cambio</div></div>",
                    unsafe_allow_html=True)
    if cur is not None:
        r0 = cur.iloc[0]["recall"]; r1 = cur.iloc[-1]["recall"]
        c2.markdown(f"<div class='card'><div class='lbl'>Cuadruplicar los datos (25% &rarr; 100%)</div>"
                    f"<div class='big'>Recall {r0:.3f} &rarr; {r1:.3f}</div><div class='lbl'>plano: no faltan datos</div></div>",
                    unsafe_allow_html=True)
    st.markdown("<div class='callout'><b>El limite no es el modelo, es el dato.</b> Es lo que muestran estos "
                "experimentos: falta informacion que el historial de compras no tiene, como precio o promociones, y el "
                "dataset de Instacart no la trae. Saberlo evita perder tiempo afinando un modelo que ya toco su techo.</div>",
                unsafe_allow_html=True)

# ---------------------------------------------------------------- 4. Que mira el modelo
with t4:
    st.subheader("Que mira el modelo para decidir")
    imp = load_csv("importancia_lightgbm.csv")
    if imp is not None:
        imp = imp.copy()
        imp["Variable"] = imp["variable"].map(FEATMAP).fillna(imp["variable"])
        imp["Peso (%)"] = (imp["pct_gain"] * 100).round(1)
        imp = imp.sort_values("Peso (%)", ascending=False)
        st.write("El modelo no pesa todas las variables igual. Estas son las que mas aportan a su decision (por "
                 "ganancia). Confirma lo que ya mostraba el EDA.")
        topn = st.slider("Cuantas variables mostrar:", 5, min(18, len(imp)), 10)
        top = imp.head(topn).iloc[::-1]
        ch = alt.Chart(top).mark_bar(cornerRadius=4, color=AZUL).encode(
            x=alt.X("Peso (%):Q", title="Peso en la decision (%)"),
            y=alt.Y("Variable:N", sort=list(top["Variable"]), title=None),
            tooltip=["Variable", "Peso (%)"]).properties(height=34 * topn)
        st.altair_chart(ch, width='stretch')
        r, ra = imp.iloc[0], imp.iloc[1]
        st.markdown(f"<div class='good'><b>{r['Variable'].split(' (')[0]} y {ra['Variable'].split(' (')[0]} "
                    f"explican el {(r['Peso (%)']+ra['Peso (%)']):.0f}% de la decision.</b> El habito de cada cliente "
                    "manda. La <b>posicion en el carrito</b> y la hora casi no aportan (menos de 0,4% cada una), tal como "
                    "anticipaba el EDA.</div>", unsafe_allow_html=True)

# ---------------------------------------------------------------- 5. Dos bloques
with t5:
    st.subheader("El sistema final: dos bloques")
    seg = load_csv("metricas_dashboard_segmento.csv")
    st.write("El sistema devuelve dos cosas separadas, en lugares distintos de la pantalla. No las mezclamos, para no "
             "vender de mas ni tapar el techo real.")
    c1, c2 = st.columns(2)
    c1.markdown("<div class='b-prin'><h3>Carrito habitual (recompra)</h3><div class='m-rep'>hasta 10</div>"
                "<p>productos. Es lo que el cliente vuelve a comprar. Le acertamos al menos uno a <b>87 de cada 100</b> "
                "clientes (79.401 aciertos). Puede traer menos de 10 si el cliente tiene poco historial.</p></div>",
                unsafe_allow_html=True)
    c2.markdown("<div class='b-sug'><h3>Tambien podrias necesitar (novedad)</h3><div class='m-desc'>hasta 5 / 2 / 1</div>"
                "<p>sugerencias de productos nuevos, segun el segmento (nuevo / medio / heavy). Suman <b>1.964 aciertos "
                "de descubrimiento sin resignar ninguno de recompra</b>, porque van en un bloque aparte.</p></div>",
                unsafe_allow_html=True)
    if seg is not None:
        s = seg.copy()
        s["Segmento"] = s["segmento"].map({"nuevo": "Nuevos", "medio": "Medios", "heavy": "Heavy"})
        s["Recompra (Hit Rate)"] = (s["hit_rate"] * 100).round(1).astype(str) + " %"
        s["Sugerencias (hasta)"] = s["lugares_de_sugerencia"].astype(int)
        s["Acierto de novedad"] = (s["hit_rate_novedad"] * 100).round(1).astype(str) + " %"
        s["Aciertos nuevos"] = s["aciertos_novedad"].astype(int)
        s = s.set_index("segmento").loc[SEG_ORDEN].reset_index(drop=True)
        st.dataframe(s[["Segmento", "Recompra (Hit Rate)", "Sugerencias (hasta)", "Acierto de novedad", "Aciertos nuevos"]],
                     hide_index=True, width='stretch')
    st.markdown("<div class='callout'><b>Los tamanos son maximos, no fijos.</b> El carrito puede traer menos de 10 "
                "productos (le pasa al 6 % de los clientes, y al 14 % de los nuevos), y las sugerencias tambien pueden "
                "venir incompletas. En pantalla se comunica como <b>hasta 10</b> y <b>hasta 5 / 2 / 1</b>.</div>",
                unsafe_allow_html=True)
    st.markdown("<div class='good'>Y ojo con la <b>cobertura</b> (39 % del catalogo): mide amplitud, no capacidad de "
                "descubrir. Por eso el descubrimiento se mide aparte, con su propio acierto de novedad.</div>",
                unsafe_allow_html=True)

# ---------------------------------------------------------------- 6. Complementos
with t6:
    st.subheader("Lo que se compra junto: complementos")
    st.write("El bloque de sugerencias se apoya en reglas de asociacion: productos que suelen ir juntos. Lo medimos con "
             "el **lift**: cuantas veces mas aparecen dos cosas en el mismo carrito comparado con el azar. Un lift de 6 "
             "significa que van juntos seis veces mas de lo que su popularidad explicaria.")
    dfr = pd.DataFrame(REGLAS_PASILLO, columns=["Si el carrito tiene", "Suele aparecer", "Lift", "Carritos juntos"])
    dfr["Carritos juntos"] = dfr["Carritos juntos"].map(lambda x: f"{x:,}".replace(",", "."))
    ch = alt.Chart(dfr.iloc[::-1]).mark_bar(cornerRadius=4, color=AZUL).encode(
        x=alt.X("Lift:Q", title="Lift (1 = azar)"),
        y=alt.Y("Si el carrito tiene:N", sort=list(dfr["Si el carrito tiene"].iloc[::-1]), title=None),
        tooltip=["Si el carrito tiene", "Suele aparecer", "Lift"]).properties(height=44 * len(dfr))
    st.altair_chart(ch, width='stretch')
    st.dataframe(dfr, hide_index=True, width='stretch')
    st.markdown("<div class='good'><b>Ejemplos claros:</b> vinos tintos con blancos, lavanderia con limpieza, tofu con "
                "congelados veganos. Detalle completo en el notebook 05.</div>", unsafe_allow_html=True)

# ---------------------------------------------------------------- 7. Ejemplos (parquet)
with t7:
    st.subheader("Ejemplos reales")
    st.write("Tres clientes reales, uno por segmento, con sus dos bloques y lo que acerto de verdad (columna de la base "
             "de recomendaciones).")
    demo = {"Nuevo (70368)": 70368, "Medio (164710)": 164710, "Heavy (2544)": 2544}
    for tb, (lbl, uid) in zip(st.tabs(list(demo.keys())), demo.items()):
        with tb:
            df = rec_usuario(uid)
            if df is None or df.empty:
                st.info("Falta el parquet de recomendaciones.")
                continue
            for bloque, titulo in [("principal", "Carrito habitual (recompra)"), ("sugerencia", "Tambien podrias necesitar")]:
                d = df[df["bloque"] == bloque]
                if d.empty:
                    continue
                ok = int(d["acerto"].sum())
                st.markdown(f"**{titulo}** · {len(d)} productos · :blue[{ok} acertaron]")
                tabla = pd.DataFrame({"#": d["posicion"].astype(int), "Producto": d["product_name"].values,
                                      "Pasillo": d["aisle"].values,
                                      "Acerto": ["Si" if a else "-" for a in d["acerto"]]})
                st.dataframe(tabla, hide_index=True, width='stretch')

# ---------------------------------------------------------------- 8. Decision
with t8:
    st.subheader("La decision y los proximos pasos")
    c1, c2 = st.columns(2)
    c1.markdown("<div class='good'><h3>Que activar ahora</h3><p><b>El sistema de dos bloques.</b> La recompra le acierta "
                "el carrito habitual a casi 9 de cada 10 clientes, y las sugerencias suman descubrimiento sin quitarle "
                "lugar a lo que si necesita.</p></div>", unsafe_allow_html=True)
    c2.markdown("<div class='callout'><h3>El techo, con honestidad</h3><p><b>El limite es el dato.</b> Cinco experimentos "
                "lo confirman: para crecer hace falta informacion nueva (precio, promociones), no un modelo mas complejo.</p></div>",
                unsafe_allow_html=True)
    st.markdown("**Backlog Sprint 2**")
    st.markdown(
        "- Desplegar el sistema de dos bloques y dejar este dashboard como demo final.\n"
        "- Enriquecer el dato: sumar precio y promociones para destrabar el descubrimiento.\n"
        "- Ampliar el motor de complementos (reglas de asociacion, notebook 05).\n"
        "- Manual de usuario con los cinco KPIs (a cargo de Leo).")
    st.caption("Una sola idea para el stakeholder: activar el sistema ahora, porque funciona y su impacto es medible; el "
               "descubrimiento crece cuando enriquezcamos el dato.")

# ---------------------------------------------------------------- Explorador en vivo (parquet)
with t9:
    st.subheader("Explora las recomendaciones de un cliente")
    if PARQ.exists():
        default_uid, hint_ids = clientes_ejemplo()
        if "uid_input" not in st.session_state:
            st.session_state["uid_input"] = int(default_uid or 13)
        def _set_uid(c):
            st.session_state["uid_input"] = int(c)
        if hint_ids:
            st.write("Toca un cliente de ejemplo (o escribi un user_id abajo):")
            percol = 7
            for i in range(0, len(hint_ids), percol):
                fila = hint_ids[i:i + percol]
                cols = st.columns(percol)
                for col, cid in zip(cols, fila):
                    col.button(str(cid), key=f"cli_{cid}", on_click=_set_uid, args=(cid,), width='stretch')
        uid = st.number_input("user_id (de los 26.243 de validacion)", min_value=1, max_value=210000, step=1, key="uid_input")
        df = rec_usuario(int(uid))
        if df is None or df.empty:
            st.warning("Ese user_id no esta en la base de validacion. Proba con uno de los sugeridos.")
        else:
            seg = str(df["segmento"].iloc[0]).capitalize() if "segmento" in df.columns and len(df) else "?"
            st.markdown(f"Cliente **{int(uid)}** · segmento **:blue[{seg}]**")
            c1, c2 = st.columns(2)
            for col, bloque, titulo, cls in [(c1, "principal", "Carrito habitual (recompra)", "blue"),
                                             (c2, "sugerencia", "Tambien podrias necesitar (novedad)", "red")]:
                d = df[df["bloque"] == bloque]
                with col:
                    if d.empty:
                        st.markdown(f"**{titulo}**"); st.caption("Sin productos para este cliente.")
                        continue
                    ok = int(d["acerto"].sum())
                    badge = f":{cls}[{ok} acertaron]"
                    st.markdown(f"**{titulo}** · {len(d)} · {badge}")
                    tabla = pd.DataFrame({"#": d["posicion"].astype(int), "Producto": d["product_name"].values,
                                          "Acerto": ["Si" if a else "-" for a in d["acerto"]]})
                    st.dataframe(tabla, hide_index=True, width='stretch')
        st.caption("Datos de recomendaciones_dashboard.parquet (modelo final, ya calculado). No necesita entrenar ni data/processed.")
    else:
        st.info("Falta reports/recomendaciones_dashboard.parquet. Copialo del repo (rama del modelado) a la carpeta reports/.")

st.divider()
st.caption("Basket Analytics · Proyecto Final Data Science · Instacart · sistema de dos bloques (Sprint 2).")
