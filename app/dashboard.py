"""
Dashboard interactivo de Mercado Libre (Streamlit).

Streamlit convierte un script de Python en una web: cada widget (selectbox,
slider) que el usuario toca vuelve a correr el script de arriba a abajo con el
nuevo valor. No escribis HTML ni JS: solo Python.

Correr local:
    streamlit run app/dashboard.py

Este archivo SOLO arma la interfaz y llama a src/metricas.py para los calculos.
La separacion calculo / vista es a proposito (ver comentario en metricas.py).
"""

from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

# Permite importar src/metricas.py al correr desde la raiz del repo.
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src import metricas  # noqa: E402

st.set_page_config(page_title="Dashboard Mercado Libre", page_icon="📊", layout="wide")


# @st.cache_data: Streamlit guarda el resultado en memoria y no relee los CSV
# en cada interaccion. Clave para que el dashboard sea rapido.
@st.cache_data
def _datos():
    return metricas.cargar_datos()


ventas, visitas = _datos()

# --------------------------------------------------------------------------- #
# Sidebar: filtros (cliente + periodo)                                         #
# --------------------------------------------------------------------------- #
st.sidebar.title("📊 Mercado Libre")
st.sidebar.caption("Panel de metricas por cliente")

clientes = sorted(ventas["cliente_ml"].unique())
cliente = st.sidebar.selectbox("Cliente (cuenta ML)", clientes)

PRESETS = {
    "Ultimos 7 dias": 7,
    "Ultimos 30 dias": 30,
    "Ultimos 90 dias": 90,
    "Ultimo anio": 365,
}
preset = st.sidebar.radio("Periodo", list(PRESETS.keys()), index=1)
dias = PRESETS[preset]

hasta = pd.Timestamp(ventas["fecha"].max())
desde = hasta - pd.Timedelta(days=dias - 1)
# Periodo anterior de igual largo, para calcular las variaciones (deltas).
desde_prev = desde - pd.Timedelta(days=dias)
hasta_prev = desde - pd.Timedelta(days=1)

# --------------------------------------------------------------------------- #
# Filtrado de datos                                                            #
# --------------------------------------------------------------------------- #
v_act = metricas.filtrar(ventas, cliente, desde, hasta)
vis_act = metricas.filtrar(visitas, cliente, desde, hasta)
v_prev = metricas.filtrar(ventas, cliente, desde_prev, hasta_prev)
vis_prev = metricas.filtrar(visitas, cliente, desde_prev, hasta_prev)

kpi = metricas.kpis_periodo(v_act, vis_act)
kpi_prev = metricas.kpis_periodo(v_prev, vis_prev)

# --------------------------------------------------------------------------- #
# Encabezado + tarjetas de KPIs                                                #
# --------------------------------------------------------------------------- #
st.title(f"Resumen de {cliente}")
st.caption(f"{preset} · {desde.date()} a {hasta.date()} · comparado con el periodo anterior")


def _delta(campo: str, sufijo: str = "%") -> str | None:
    var = metricas.variacion_pct(kpi[campo], kpi_prev[campo])
    return None if var is None else f"{var:+.1f}{sufijo}"


c1, c2, c3, c4 = st.columns(4)
c1.metric("Ingresos", f"${kpi['ingreso']:,.0f}", _delta("ingreso"))
c2.metric("Ordenes", f"{kpi['ordenes']:,}", _delta("ordenes"))
c3.metric("Ticket promedio", f"${kpi['ticket_promedio']:,.0f}", _delta("ticket_promedio"))
c4.metric("Conversion", f"{kpi['conversion']:.2f}%", _delta("conversion", " pts"))

c5, c6, c7, c8 = st.columns(4)
c5.metric("Unidades", f"{kpi['unidades']:,}", _delta("unidades"))
c6.metric("Visitas", f"{kpi['visitas']:,}", _delta("visitas"))
c7.metric("Comisiones ML", f"${kpi['comisiones']:,.0f}", _delta("comisiones"))
c8.metric("Tasa cancelacion", f"{kpi['tasa_cancelacion']:.1f}%")

st.divider()

# --------------------------------------------------------------------------- #
# Graficos                                                                     #
# --------------------------------------------------------------------------- #
col_izq, col_der = st.columns([2, 1])

with col_izq:
    st.subheader("Ingresos por dia")
    serie = metricas.serie_diaria(v_act)
    fig = px.area(serie, x="fecha", y="ingreso", labels={"ingreso": "Ingreso $", "fecha": ""})
    fig.update_traces(line_color="#2563eb", fillcolor="rgba(37,99,235,0.15)")
    st.plotly_chart(fig, use_container_width=True)

with col_der:
    st.subheader("Por categoria")
    cat = metricas.por_categoria(v_act)
    fig_cat = px.pie(cat, names="categoria", values="ingreso", hole=0.5)
    st.plotly_chart(fig_cat, use_container_width=True)

st.subheader("Top productos")
top = metricas.top_productos(v_act, n=10)
fig_top = px.bar(
    top.sort_values("ingreso"),
    x="ingreso", y="titulo", orientation="h",
    labels={"ingreso": "Ingreso $", "titulo": ""},
)
fig_top.update_traces(marker_color="#16a34a")
st.plotly_chart(fig_top, use_container_width=True)

# Vista mensual: comparar varios meses (estacionalidad). Usa TODO el historico
# del cliente, no solo el periodo filtrado, para que se vea la tendencia anual.
st.subheader("Evolucion mensual (historico completo)")
mensual = metricas.serie_mensual(metricas.filtrar(ventas, cliente))
fig_mes = px.bar(mensual, x="mes", y="ingreso", labels={"ingreso": "Ingreso $", "mes": ""})
fig_mes.update_traces(marker_color="#7c3aed")
st.plotly_chart(fig_mes, use_container_width=True)

with st.expander("Ver datos crudos del periodo"):
    st.dataframe(v_act, use_container_width=True)
