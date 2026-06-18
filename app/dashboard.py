"""
Dashboard comercial de Mercado Libre — Streamlit.

Tabs:
  Resumen           — KPIs con deltas vs periodo anterior
  Volumen & Facturacion — GMV+unidades, marca, medio de entrega
  Ritmo & Tendencia — serie diaria + MA7 + campanas, MoM/YoY
  Conversion        — embudo, scatter visitas vs CVR
  Concentracion     — Pareto ABC, productos sin conversion, velocidad SKU
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src import metricas  # noqa: E402

st.set_page_config(page_title="Dashboard Mercado Libre", page_icon="📊", layout="wide")


# --------------------------------------------------------------------------- #
# Configuración de clientes ML                                                 #
# --------------------------------------------------------------------------- #
def _ml_client(nombre: str):
    """
    Construye un MLClient desde Streamlit secrets si están configurados.
    Devuelve None si no hay secrets ML para ese cliente (usa datos sintéticos).
    """
    try:
        # La sección puede llamarse [nombre] o [ml_nombre]
        sec = st.secrets.get(nombre) or st.secrets.get(f"ml_{nombre}")
        if sec:
            from src.ml_client import from_secrets
            key = f"ml_client_{nombre}"
            # Reusar el cliente de session_state para no re-refreshar el token
            # en cada rerun de Streamlit (cada interacción recorre el script entero).
            if key not in st.session_state:
                st.session_state[key] = from_secrets(nombre, sec)
            return st.session_state[key]
    except Exception:
        pass
    return None


# --------------------------------------------------------------------------- #
# Cache en disco — /tmp persiste entre reruns dentro de la misma instancia    #
# --------------------------------------------------------------------------- #
from datetime import date as _date, timedelta as _td

# v2: el filtro de fechas de /orders/search estaba mal (faltaba prefijo
# "order."), así que el cache v1 quedó con datos de todo el año. Bump → ignora v1.
_DISK_CACHE = Path("/tmp/ml_dash_cache_v2")


def _pk(nombre: str, tag: str, kind: str) -> Path:
    return _DISK_CACHE / f"{nombre}_{tag}_{kind}.parquet"


def _disk_load(nombre: str, tag: str):
    """Carga (ventas, visitas, preguntas) desde disco. None si no existe."""
    try:
        paths = [_pk(nombre, tag, k) for k in ("v", "vis", "preg")]
        if all(p.exists() for p in paths):
            return tuple(pd.read_parquet(p) for p in paths)
    except Exception:
        pass
    return None


def _disk_save(v, vis, preg, nombre: str, tag: str) -> None:
    try:
        _DISK_CACHE.mkdir(parents=True, exist_ok=True)
        for df, kind in [(v, "v"), (vis, "vis"), (preg, "preg")]:
            df.to_parquet(_pk(nombre, tag, kind), index=False)
    except Exception:
        pass


def _concat3(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    if a.empty:
        return b
    if b.empty:
        return a
    return pd.concat([a, b], ignore_index=True)


@st.cache_data(ttl=None, show_spinner="⏳ Cargando historial de ML…")
def _datos_historico(nombre: str, desde: str, hasta: str, usar_ml: bool):
    """
    Meses cerrados (antes del mes actual) — datos inmutables.
    TTL infinito en sesión + parquet en /tmp para sobrevivir reinicios.
    """
    if usar_ml:
        tag = desde.replace("-", "") + "_" + hasta.replace("-", "")
        cached = _disk_load(nombre, tag)
        if cached:
            return cached
    d, h = pd.Timestamp(desde), pd.Timestamp(hasta)
    client = _ml_client(nombre) if usar_ml else None
    result = metricas.cargar_datos_cliente(nombre, d, h, client)
    if usar_ml:
        tag = desde.replace("-", "") + "_" + hasta.replace("-", "")
        _disk_save(*result, nombre, tag)
    return result


@st.cache_data(ttl=3600, show_spinner="🔄 Actualizando datos del mes…")
def _datos_reciente(nombre: str, desde: str, hasta: str, usar_ml: bool):
    """Mes en curso — puede cambiar. TTL 1 hora."""
    d, h = pd.Timestamp(desde), pd.Timestamp(hasta)
    client = _ml_client(nombre) if usar_ml else None
    return metricas.cargar_datos_cliente(nombre, d, h, client)


def _cargar_rango(
    nombre: str,
    desde: pd.Timestamp,
    hasta: pd.Timestamp,
    usar_ml: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Divide en histórico (meses cerrados, cacheados en disco) + reciente (mes actual, TTL 1h).
    Los meses cerrados se persisten en /tmp/ml_dash_cache como parquet.
    """
    hoy = _date.today()
    mes_actual = pd.Timestamp(_date(hoy.year, hoy.month, 1))

    if usar_ml and desde < mes_actual <= hasta:
        hasta_hist = mes_actual - pd.Timedelta(days=1)
        v_h, vis_h, preg_h = _datos_historico(nombre, str(desde.date()), str(hasta_hist.date()), True)
        v_r, vis_r, preg_r = _datos_reciente(nombre, str(mes_actual.date()), str(hasta.date()), True)
        return _concat3(v_h, v_r), _concat3(vis_h, vis_r), _concat3(preg_h, preg_r)

    if usar_ml and hasta < mes_actual:
        return _datos_historico(nombre, str(desde.date()), str(hasta.date()), True)

    return _datos_reciente(nombre, str(desde.date()), str(hasta.date()), usar_ml)


@st.cache_data(ttl=None, show_spinner="⏳ Cargando histórico 24 meses…")
def _historico_cliente(nombre: str, usar_ml: bool):
    """Ventas de los últimos 24 meses para MoM/YoY. Reutiliza el cache histórico."""
    hoy = _date.today()
    mes_actual = pd.Timestamp(_date(hoy.year, hoy.month, 1))
    hasta_hist = mes_actual - pd.Timedelta(days=1)
    desde_24m = pd.Timestamp(hoy - _td(days=730))
    if usar_ml:
        v_h, _, _ = _datos_historico(nombre, str(desde_24m.date()), str(hasta_hist.date()), True)
        v_r, _, _ = _datos_reciente(nombre, str(mes_actual.date()), str(pd.Timestamp(hoy).date()), True)
        return _concat3(v_h, v_r)
    ventas, _, _ = metricas.cargar_datos()
    return ventas[ventas["cliente_ml"] == nombre]


# Datos sintéticos para saber qué clientes hay cuando no hay ML configurado
@st.cache_data
def _sinteticos():
    return metricas.cargar_datos()

# --------------------------------------------------------------------------- #
# Autenticacion por token en la URL                                            #
# --------------------------------------------------------------------------- #
# Sin secrets configurados (dev local) → modo admin sin restriccion.
# Con secrets: token admin ve todo; token seller ve solo su cliente.
params = st.query_params
token_url = params.get("token", "")
cliente_url = params.get("cliente", "")

locked_cliente: str | None = None  # None = admin (selector libre)

try:
    tokens: dict = dict(st.secrets.get("tokens", {}))
    if tokens:
        admin_token = tokens.get("admin", "")
        if token_url and token_url == admin_token:
            locked_cliente = None  # admin
        elif token_url and cliente_url:
            esperado = tokens.get(cliente_url, "")
            if esperado and token_url == esperado:
                locked_cliente = cliente_url
            else:
                st.error("🔒 Token inválido o expirado. Pedí tu link actualizado.")
                st.stop()
        else:
            # Sin token → acceso admin (útil mientras no hay secrets configurados)
            locked_cliente = None
except Exception:
    locked_cliente = None  # dev local sin secrets.toml

# --------------------------------------------------------------------------- #
# Sidebar                                                                      #
# --------------------------------------------------------------------------- #
st.sidebar.title("📊 Mercado Libre")

# Determinar lista de clientes disponibles
_v_sint, _, _ = _sinteticos()
_clientes_sint = sorted(_v_sint["cliente_ml"].unique().tolist())
try:
    _clientes_ml = metricas.clientes_configurados(dict(st.secrets))
except Exception:
    _clientes_ml = []
clientes = _clientes_ml if _clientes_ml else _clientes_sint

if locked_cliente:
    cliente = locked_cliente
    st.sidebar.markdown(f"**Cuenta:** {cliente}")
else:
    cliente = st.sidebar.selectbox("Cliente", clientes)

PRESETS = {"7 días": 7, "30 días": 30, "90 días": 90, "1 año": 365, "Personalizado": None}
preset = st.sidebar.radio("Período", list(PRESETS.keys()), index=1)

hasta_max = pd.Timestamp(_date.today())
desde_min = pd.Timestamp(_date.today() - _td(days=364))

if PRESETS[preset] is not None:
    dias = PRESETS[preset]
    hasta = hasta_max
    desde = hasta - pd.Timedelta(days=dias - 1)
else:
    desde_sel = st.sidebar.date_input(
        "Desde", value=(hasta_max - pd.Timedelta(days=29)).date(),
        min_value=desde_min.date(), max_value=hasta_max.date(),
    )
    hasta_sel = st.sidebar.date_input(
        "Hasta", value=hasta_max.date(),
        min_value=desde_min.date(), max_value=hasta_max.date(),
    )
    desde = pd.Timestamp(desde_sel)
    hasta = pd.Timestamp(hasta_sel)
    dias = max(1, (hasta - desde).days + 1)

desde_prev = desde - pd.Timedelta(days=dias)
hasta_prev = desde - pd.Timedelta(days=1)

# Datos filtrados — ML real si hay secrets [ml_{cliente}], sintético si no.
_usar_ml = bool(_ml_client(cliente))
v_act, vis_act, preg_act = _cargar_rango(cliente, desde, hasta, _usar_ml)
v_prev, vis_prev, _ = _cargar_rango(cliente, desde_prev, hasta_prev, _usar_ml)

if _usar_ml:
    st.sidebar.success("✅ Datos reales de ML")

kpi = metricas.kpis_periodo(v_act, vis_act)
kpi_prev = metricas.kpis_periodo(v_prev, vis_prev)


def _delta(campo: str, sufijo: str = "%") -> str | None:
    var = metricas.variacion_pct(kpi[campo], kpi_prev[campo])
    return None if var is None else f"{var:+.1f}{sufijo}"


def _money(n: float) -> str:
    """Formato monetario compacto y legible: $814.3M, $417K, $950."""
    n = float(n)
    if abs(n) >= 1_000_000:
        return f"${n / 1_000_000:,.1f}M"
    if abs(n) >= 1_000:
        return f"${n / 1_000:,.0f}K"
    return f"${n:,.0f}"


# --------------------------------------------------------------------------- #
# Tabs                                                                         #
# --------------------------------------------------------------------------- #
st.title(f"📈 {cliente}")
st.caption(f"{preset} · {desde.date()} → {hasta.date()} · vs período anterior")

tab_res, tab_vol, tab_ritmo, tab_conv, tab_conc = st.tabs([
    "Resumen", "Volumen & Facturación", "Ritmo & Tendencia", "Conversión", "Concentración"
])


# =================================================================== RESUMEN #
with tab_res:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("GMV", _money(kpi['ingreso']), _delta("ingreso"), help=f"${kpi['ingreso']:,.0f}")
    c2.metric(
        "Órdenes pagadas", f"{kpi['ordenes']:,}", _delta("ordenes"),
        help=(
            f"Solo ventas concretadas (pagadas). No incluye canceladas.\n\n"
            f"Total en ML: {kpi['ordenes_total']:,} = "
            f"{kpi['ordenes']:,} pagadas + {kpi['canceladas']:,} canceladas."
        ),
    )
    c3.metric("Ticket promedio", _money(kpi['ticket_promedio']), _delta("ticket_promedio"),
              help=f"${kpi['ticket_promedio']:,.0f}")
    c4.metric("Conversión", f"{kpi['conversion']:.2f}%", _delta("conversion", " pts"))

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Unidades", f"{kpi['unidades']:,}", _delta("unidades"))
    c6.metric("Visitas", f"{kpi['visitas']:,}", _delta("visitas"))
    c7.metric("Comisiones ML", _money(kpi['comisiones']), _delta("comisiones"),
              help=f"${kpi['comisiones']:,.0f}")
    c8.metric(
        "Cancelaciones", f"{kpi['tasa_cancelacion']:.1f}%",
        help=f"{kpi['canceladas']:,} de {kpi['ordenes_total']:,} órdenes totales.",
    )

    st.divider()
    col1, col2 = st.columns([2, 1])
    with col1:
        serie = metricas.serie_diaria(v_act)
        fig = px.area(serie, x="fecha", y="ingreso", labels={"ingreso": "GMV $", "fecha": ""})
        fig.update_traces(line_color="#2563eb", fillcolor="rgba(37,99,235,0.12)")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        cat = metricas.por_categoria(v_act)
        fig_cat = px.pie(cat, names="categoria", values="ingreso", hole=0.5)
        fig_cat.update_traces(textposition="outside", textinfo="label+percent")
        st.plotly_chart(fig_cat, use_container_width=True)


# ================================================= VOLUMEN & FACTURACION #
with tab_vol:
    st.subheader("GMV por marca + Ticket promedio")
    st.caption(
        "Las barras apiladas muestran cuánto aporta cada marca al GMV total. "
        "La línea punteada es el ticket promedio: si sube mientras el GMV baja, "
        "vendiste menos pero más caro."
    )

    gmv_marca = metricas.gmv_por_marca_tiempo(v_act)
    ticket_d = metricas.ticket_diario(v_act)

    if not gmv_marca.empty:
        pivot = gmv_marca.pivot(index="fecha", columns="marca", values="ingreso").fillna(0).reset_index()
        marcas = [c for c in pivot.columns if c != "fecha"]
        colores = px.colors.qualitative.Set2

        fig_stack = make_subplots(specs=[[{"secondary_y": True}]])
        for i, m in enumerate(marcas):
            fig_stack.add_trace(
                go.Scatter(
                    x=pivot["fecha"], y=pivot[m], name=m,
                    stackgroup="one", fill="tonexty",
                    line=dict(color=colores[i % len(colores)], width=0.5),
                ),
                secondary_y=False,
            )
        fig_stack.add_trace(
            go.Scatter(
                x=ticket_d["fecha"], y=ticket_d["ticket"],
                name="Ticket promedio", mode="lines",
                line=dict(color="#1e293b", width=2, dash="dot"),
            ),
            secondary_y=True,
        )
        fig_stack.update_yaxes(title_text="GMV $", secondary_y=False)
        fig_stack.update_yaxes(title_text="Ticket $ (promedio)", secondary_y=True)
        fig_stack.update_layout(legend=dict(orientation="h", y=-0.15), margin=dict(t=10))
        st.plotly_chart(fig_stack, use_container_width=True)

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Por medio de entrega")
        me = metricas.por_medio_entrega(v_act)
        fig_me = px.bar(
            me, x="medio_entrega", y="ingreso", color="medio_entrega",
            color_discrete_sequence=px.colors.qualitative.Pastel,
            labels={"ingreso": "GMV $", "medio_entrega": ""},
            text_auto=".2s",
        )
        fig_me.update_layout(showlegend=False)
        st.plotly_chart(fig_me, use_container_width=True)

        total_me = me["ingreso"].sum()
        me["pct"] = (me["ingreso"] / total_me * 100).round(1)
        st.dataframe(
            me[["medio_entrega", "ordenes", "unidades", "ingreso", "pct"]]
            .rename(columns={"medio_entrega": "Medio", "ordenes": "Órdenes",
                             "unidades": "Unidades", "ingreso": "GMV $", "pct": "%"}),
            use_container_width=True, hide_index=True,
        )

    with col2:
        st.subheader("Top productos — GMV vs Unidades")
        st.caption("Las barras son GMV; las etiquetas muestran las unidades vendidas.")
        top = metricas.top_productos(v_act)
        fig_top = px.bar(
            top.sort_values("ingreso"),
            x="ingreso", y="titulo", orientation="h",
            color="ingreso", color_continuous_scale="Blues",
            text="unidades",
            labels={"ingreso": "GMV $", "titulo": ""},
        )
        fig_top.update_traces(texttemplate="%{text} u.", textposition="inside")
        fig_top.update_coloraxes(showscale=False)
        st.plotly_chart(fig_top, use_container_width=True)


# ================================================= RITMO & TENDENCIA #
with tab_ritmo:
    st.subheader("Ventas diarias con media móvil 7 días")
    st.caption(
        "La línea fina son las ventas reales (ruidosas). La línea gruesa es la media "
        "móvil de 7 días: suaviza el ruido del día a día y muestra la tendencia real. "
        "Los sombreados son campañas de ML."
    )

    serie_ma = metricas.serie_con_ma(v_act)

    fig_ma = go.Figure()
    fig_ma.add_trace(go.Scatter(
        x=serie_ma["fecha"], y=serie_ma["ingreso"],
        name="GMV diario", line=dict(color="#93c5fd", width=1), opacity=0.7,
    ))
    fig_ma.add_trace(go.Scatter(
        x=serie_ma["fecha"], y=serie_ma["ma7"],
        name="Media móvil 7d", line=dict(color="#2563eb", width=2.5),
    ))

    # Sombreado de campañas dentro del periodo visible
    for nombre, ini, fin in metricas.CAMPANAS:
        if ini <= hasta and fin >= desde:
            fig_ma.add_vrect(
                x0=max(ini, desde), x1=min(fin, hasta),
                fillcolor="#fbbf24", opacity=0.25, line_width=0,
                annotation_text=nombre, annotation_position="top left",
            )

    fig_ma.update_layout(
        yaxis_title="GMV $", xaxis_title="",
        legend=dict(orientation="h", y=-0.15), margin=dict(t=10),
    )
    st.plotly_chart(fig_ma, use_container_width=True)

    st.divider()
    st.subheader("Comparativa mensual — MoM y YoY")
    st.caption(
        "MoM (month-over-month): cuánto creció vs el mes anterior. "
        "YoY (year-over-year): vs el mismo mes del año pasado — elimina la estacionalidad. "
        "Si noviembre siempre pega fuerte, el YoY te dice si creciste ADEMÁS del efecto seasonal."
    )

    mom = metricas.mom_yoy(_historico_cliente(cliente, _usar_ml))
    mom_show = mom.copy()
    mom_show["GMV $"] = mom_show["ingreso"].map("${:,.0f}".format)
    mom_show["MoM %"] = mom_show["mom_pct"].map(lambda x: f"{x:+.1f}%" if pd.notna(x) else "—")
    mom_show["YoY %"] = mom_show["yoy_pct"].map(lambda x: f"{x:+.1f}%" if pd.notna(x) else "—")

    st.dataframe(
        mom_show[["mes_label", "GMV $", "ordenes", "MoM %", "YoY %"]]
        .rename(columns={"mes_label": "Mes", "ordenes": "Órdenes"}),
        use_container_width=True, hide_index=True,
    )

    fig_mom = go.Figure()
    m_clean = mom.dropna(subset=["mom_pct"])
    colors_mom = ["#16a34a" if v >= 0 else "#dc2626" for v in m_clean["mom_pct"]]
    fig_mom.add_trace(go.Bar(
        x=m_clean["mes_label"], y=m_clean["mom_pct"],
        marker_color=colors_mom, name="MoM %",
    ))
    fig_mom.update_layout(yaxis_title="Variación %", xaxis_title="", margin=dict(t=10))
    fig_mom.add_hline(y=0, line_dash="solid", line_color="#94a3b8")
    st.plotly_chart(fig_mom, use_container_width=True)


# ======================================================== CONVERSION #
with tab_conv:
    st.subheader("Embudo: visitas → preguntas → ventas")
    st.caption(
        "Donde se rompe el embudo te dice qué optimizar. "
        "Muchas preguntas + pocas ventas = fricción de info (precio, garantía, cuotas). "
        "Muchas visitas + pocas preguntas = el título o la foto no enganchan."
    )

    f = metricas.funnel(v_act, vis_act, preg_act)
    c1, c2, c3 = st.columns(3)
    c1.metric("Visitas", f"{f['visitas']:,}")
    c2.metric("Preguntas", f"{f['preguntas']:,}", f"{f['preguntas']/f['visitas']*100:.1f}% de visitas")
    c3.metric("Ventas", f"{f['ventas']:,}", f"{f['ventas']/f['visitas']*100:.2f}% de visitas")

    fig_funnel = go.Figure(go.Funnel(
        y=["Visitas", "Preguntas", "Ventas"],
        x=[f["visitas"], f["preguntas"], f["ventas"]],
        textposition="inside", textinfo="value+percent initial",
        marker=dict(color=["#3b82f6", "#f59e0b", "#16a34a"]),
    ))
    fig_funnel.update_layout(margin=dict(t=10, l=0, r=0))
    st.plotly_chart(fig_funnel, use_container_width=True)

    st.divider()
    st.subheader("Scatter: visitas vs conversión por publicación")
    st.caption(
        "**Burbuja grande** = mucho GMV. **Derecha** = mucho tráfico. **Arriba** = buena conversión. "
        "Ideal: arriba a la derecha. "
        "Peligro: abajo a la derecha (mucho tráfico que no convierte = problema de precio/foto). "
        "Oportunidad: arriba a la izquierda (convierte bien pero le falta tráfico = invertir en visitas)."
    )

    scatter_df = metricas.conversion_por_publicacion(v_act, vis_act)
    if not scatter_df.empty:
        fig_sc = px.scatter(
            scatter_df,
            x="visitas", y="conversion",
            size="ingreso", color="marca",
            hover_name="titulo",
            hover_data={"visitas": True, "conversion": ":.2f", "ingreso": ":,.0f", "ordenes": True},
            size_max=50,
            labels={"visitas": "Visitas", "conversion": "Conversión %", "marca": "Marca"},
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig_sc.update_layout(margin=dict(t=10))
        st.plotly_chart(fig_sc, use_container_width=True)


# ======================================================= CONCENTRACION #
with tab_conc:
    st.subheader("Curva de Pareto — concentración de GMV por SKU")
    st.caption(
        "Las barras son el GMV de cada SKU (orden desc). La línea es el % acumulado. "
        "A = primeros SKUs hasta el 70% del GMV (críticos — si caen, duele). "
        "B = hasta el 90%. C = cola larga. "
        "Si el 80% del GMV está en 1-2 SKUs, tenés riesgo de concentración."
    )

    pareto = metricas.pareto_sku(v_act)
    if not pareto.empty:
        abc_colors = {"A": "#dc2626", "B": "#f59e0b", "C": "#6b7280"}
        fig_par = make_subplots(specs=[[{"secondary_y": True}]])
        fig_par.add_trace(
            go.Bar(
                x=pareto["titulo"], y=pareto["ingreso"],
                name="GMV $",
                marker_color=[abc_colors.get(str(a), "#6b7280") for a in pareto["abc"]],
            ),
            secondary_y=False,
        )
        fig_par.add_trace(
            go.Scatter(
                x=pareto["titulo"], y=pareto["pct_acum"],
                name="% acumulado", mode="lines+markers",
                line=dict(color="#1e293b", width=2),
            ),
            secondary_y=True,
        )
        fig_par.add_hline(y=80, line_dash="dash", line_color="#f59e0b", secondary_y=True,
                          annotation_text="80%", annotation_position="right")
        fig_par.update_yaxes(title_text="GMV $", secondary_y=False)
        fig_par.update_yaxes(title_text="% Acumulado", range=[0, 105], secondary_y=True)
        fig_par.update_layout(
            xaxis=dict(tickangle=-30), margin=dict(t=10),
            legend=dict(orientation="h", y=-0.25),
        )
        st.plotly_chart(fig_par, use_container_width=True)

        with st.expander("Ver tabla ABC completa"):
            st.dataframe(
                pareto[["titulo", "marca", "ingreso", "pct_acum", "abc"]]
                .rename(columns={"titulo": "Producto", "marca": "Marca",
                                 "ingreso": "GMV $", "pct_acum": "% Acum", "abc": "Clase"}),
                use_container_width=True, hide_index=True,
            )

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Velocidad de venta por SKU")
        st.caption("Unidades vendidas por día en el período. Detecta estrellas en ascenso y productos frenados.")
        vel = metricas.velocidad_sku(v_act)
        fig_vel = px.bar(
            vel.head(10).sort_values("unidades_dia"),
            x="unidades_dia", y="titulo", orientation="h",
            color="unidades_dia", color_continuous_scale="Greens",
            labels={"unidades_dia": "Unidades/día", "titulo": ""},
        )
        fig_vel.update_coloraxes(showscale=False)
        st.plotly_chart(fig_vel, use_container_width=True)

    with col2:
        st.subheader("Publicaciones sin conversión")
        st.caption("Items con ≥30 visitas y cero ventas en el período. Candidatos a optimizar o dar de baja.")
        sin_conv = metricas.productos_sin_conversion(v_act, vis_act)
        if sin_conv.empty:
            st.info("Todos los items con tráfico suficiente vendieron algo en este período.")
        else:
            st.dataframe(
                sin_conv[["titulo", "visitas", "ordenes"]]
                .rename(columns={"titulo": "Producto", "visitas": "Visitas", "ordenes": "Ventas"}),
                use_container_width=True, hide_index=True,
            )
