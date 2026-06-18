"""
Metricas del negocio.

Regla: este modulo SOLO calcula — devuelve numeros, DataFrames, dicts.
El dashboard (app/dashboard.py) dibuja los resultados.
Separar calculo de visualizacion permite testear los numeros sin levantar la web.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Campanas para shading en graficos. Importadas desde dashboard.py.
CAMPANAS: list[tuple[str, pd.Timestamp, pd.Timestamp]] = [
    ("CyberMonday", pd.Timestamp("2025-11-03"), pd.Timestamp("2025-11-05")),
    ("Navidad", pd.Timestamp("2025-12-24"), pd.Timestamp("2025-12-26")),
    ("Enamorados", pd.Timestamp("2026-02-14"), pd.Timestamp("2026-02-14")),
    ("Hot Sale", pd.Timestamp("2026-05-11"), pd.Timestamp("2026-05-13")),
]


# --------------------------------------------------------------------------- #
# Carga                                                                        #
# --------------------------------------------------------------------------- #
def cargar_datos() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Lee los CSV sintéticos. Si no existen los genera al vuelo."""
    if not (DATA_DIR / "ventas.csv").exists():
        from data import generar_datos
        generar_datos.main()
    ventas = pd.read_csv(DATA_DIR / "ventas.csv", parse_dates=["fecha"])
    visitas = pd.read_csv(DATA_DIR / "visitas.csv", parse_dates=["fecha"])
    preguntas = pd.read_csv(DATA_DIR / "preguntas.csv", parse_dates=["fecha"])
    return ventas, visitas, preguntas


_ML_CRED_KEYS = {"client_id", "client_secret", "access_token", "refresh_token"}


def _es_seccion_ml(valor) -> bool:
    """True si una sección de secrets parece un bloque de credenciales ML."""
    try:
        claves = set(dict(valor).keys())
    except Exception:
        return False
    # Requiere al menos client_id + access_token para considerarse seller ML
    return {"client_id", "access_token"}.issubset(claves)


def clientes_configurados(secrets: dict | None = None) -> list[str]:
    """
    Devuelve la lista de sellers con credenciales ML configuradas en secrets.

    Detecta CUALQUIER sección que contenga credenciales ML (client_id +
    access_token), sin importar cómo se llame. Acepta tanto [ml_seller] como
    [SELLER] directamente. Excluye [tokens] y otras secciones de config.
    """
    if secrets is None:
        return sorted(
            pd.read_csv(DATA_DIR / "ventas.csv")["cliente_ml"].unique().tolist()
            if (DATA_DIR / "ventas.csv").exists()
            else ["tienda_norte", "deco_hogar", "tech_outlet"]
        )
    nombres = []
    for k, v in secrets.items():
        if k == "tokens":
            continue
        if _es_seccion_ml(v):
            # Quitar prefijo ml_ si lo tiene, para mostrar nombre limpio
            nombres.append(k[3:] if k.startswith("ml_") else k)
    return sorted(nombres)


def cargar_datos_cliente(
    nombre: str,
    desde: pd.Timestamp,
    hasta: pd.Timestamp,
    ml_client=None,  # MLClient | None
    solo_ventas: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Carga datos de UN cliente para el rango desde..hasta.

    Si ml_client es None → usa los CSV sintéticos filtrados.
    Si ml_client es un MLClient → llama a la API real de ML.

    solo_ventas=True: modo liviano (solo órdenes), para series largas MoM/YoY.

    Esta separación permite usar el mismo dashboard con datos reales o
    sintéticos sin cambiar ninguna otra función de metricas.py.
    """
    if ml_client is not None:
        from src.ml_fetch import fetch_all
        return fetch_all(ml_client, desde.date(), hasta.date(), full=not solo_ventas)

    # Fallback sintético
    ventas, visitas, preguntas = cargar_datos()
    v = filtrar(ventas, nombre, desde, hasta)
    vis = filtrar(visitas, nombre, desde, hasta)
    preg = filtrar(preguntas, nombre, desde, hasta)
    return v, vis, preg


# --------------------------------------------------------------------------- #
# Filtros                                                                      #
# --------------------------------------------------------------------------- #
def filtrar(
    df: pd.DataFrame,
    cliente: str | None = None,
    desde: pd.Timestamp | None = None,
    hasta: pd.Timestamp | None = None,
) -> pd.DataFrame:
    m = pd.Series(True, index=df.index)
    if cliente:
        m &= df["cliente_ml"] == cliente
    if desde is not None:
        m &= df["fecha"] >= desde
    if hasta is not None:
        m &= df["fecha"] <= hasta
    return df[m]


def solo_pagadas(ventas: pd.DataFrame) -> pd.DataFrame:
    return ventas[ventas["estado"] == "pagado"]


# --------------------------------------------------------------------------- #
# KPIs del periodo                                                             #
# --------------------------------------------------------------------------- #
def kpis_periodo(ventas: pd.DataFrame, visitas: pd.DataFrame) -> dict:
    pagadas = solo_pagadas(ventas)
    ingreso = pagadas["ingreso"].sum()
    unidades = pagadas["unidades"].sum()
    ordenes = len(pagadas)
    canceladas = int((ventas["estado"] == "cancelado").sum())
    total_visitas = visitas["visitas"].sum()
    return {
        "ingreso": float(ingreso),
        "unidades": int(unidades),
        "ordenes": ordenes,
        "canceladas": canceladas,
        "ordenes_total": ordenes + canceladas,
        "ticket_promedio": float(ingreso / ordenes) if ordenes else 0.0,
        "visitas": int(total_visitas),
        "conversion": float(ordenes / total_visitas * 100) if total_visitas else 0.0,
        "comisiones": float(pagadas["comision_ml"].sum()),
        "tasa_cancelacion": float(canceladas / (ordenes + canceladas) * 100) if (ordenes + canceladas) else 0.0,
    }


def variacion_pct(actual: float, anterior: float) -> float | None:
    if anterior == 0:
        return None
    return round((actual - anterior) / anterior * 100, 1)


# --------------------------------------------------------------------------- #
# Series temporales                                                            #
# --------------------------------------------------------------------------- #
def serie_diaria(ventas: pd.DataFrame) -> pd.DataFrame:
    pagadas = solo_pagadas(ventas)
    return (
        pagadas.resample("D", on="fecha")
        .agg(ingreso=("ingreso", "sum"), unidades=("unidades", "sum"), ordenes=("order_id", "count"))
        .reset_index()
    )


def serie_con_ma(ventas: pd.DataFrame, window: int = 7) -> pd.DataFrame:
    """Serie diaria de GMV + media movil para suavizar el ruido."""
    s = serie_diaria(ventas)
    s["ma7"] = s["ingreso"].rolling(window, min_periods=1).mean()
    return s


def serie_mensual(ventas: pd.DataFrame) -> pd.DataFrame:
    pagadas = solo_pagadas(ventas)
    s = (
        pagadas.resample("ME", on="fecha")
        .agg(ingreso=("ingreso", "sum"), ordenes=("order_id", "count"))
        .reset_index()
    )
    s["mes"] = s["fecha"].dt.strftime("%Y-%m")
    return s


def gmv_por_marca_tiempo(ventas: pd.DataFrame) -> pd.DataFrame:
    """GMV diario por marca, para el area chart apilado."""
    pagadas = solo_pagadas(ventas)
    return (
        pagadas.groupby(["fecha", "marca"])["ingreso"]
        .sum()
        .reset_index()
    )


def ticket_diario(ventas: pd.DataFrame) -> pd.DataFrame:
    """Ticket promedio por dia, para la linea superpuesta en el area chart."""
    pagadas = solo_pagadas(ventas)
    s = (
        pagadas.resample("D", on="fecha")
        .agg(ingreso=("ingreso", "sum"), ordenes=("order_id", "count"))
        .reset_index()
    )
    s["ticket"] = (s["ingreso"] / s["ordenes"].replace(0, float("nan"))).round(0)
    return s


# --------------------------------------------------------------------------- #
# MoM y YoY                                                                   #
# --------------------------------------------------------------------------- #
def mom_yoy(ventas: pd.DataFrame) -> pd.DataFrame:
    """
    Tabla mensual con variacion MoM y YoY.

    MoM (month-over-month): compara cada mes con el mes anterior.
    YoY (year-over-year): compara cada mes con el mismo mes del anio anterior.
    YoY es el que elimina la estacionalidad: si noviembre siempre pega,
    el YoY te dice si creciste ADEMAS del efecto estacional.
    """
    mensual = serie_mensual(ventas).copy()
    mensual["mom_pct"] = mensual["ingreso"].pct_change() * 100

    # Para YoY: unir con la misma tabla desplazada un anio
    base = mensual[["fecha", "ingreso"]].copy()
    base["fecha_prev"] = base["fecha"] + pd.DateOffset(years=1)
    yoy_map = base.set_index("fecha_prev")["ingreso"].rename("ingreso_prev_year")
    mensual = mensual.join(yoy_map, on="fecha")
    mensual["yoy_pct"] = (mensual["ingreso"] / mensual["ingreso_prev_year"] - 1) * 100

    mensual["mes_label"] = mensual["fecha"].dt.strftime("%b %Y")
    return mensual[["mes_label", "fecha", "ingreso", "ordenes", "mom_pct", "yoy_pct"]].sort_values("fecha")


# --------------------------------------------------------------------------- #
# Ritmo por SKU                                                                #
# --------------------------------------------------------------------------- #
def velocidad_sku(ventas: pd.DataFrame) -> pd.DataFrame:
    """
    Unidades vendidas por dia por SKU.
    Util para detectar 'estrellas en ascenso' y 'productos que se detienen'.
    """
    pagadas = solo_pagadas(ventas)
    periodo_dias = max(1, (pagadas["fecha"].max() - pagadas["fecha"].min()).days + 1)
    t = (
        pagadas.groupby(["titulo", "marca", "categoria"])
        .agg(unidades=("unidades", "sum"), ingreso=("ingreso", "sum"))
        .reset_index()
    )
    t["unidades_dia"] = (t["unidades"] / periodo_dias).round(2)
    return t.sort_values("unidades_dia", ascending=False)


# --------------------------------------------------------------------------- #
# Rankings                                                                     #
# --------------------------------------------------------------------------- #
def top_productos(ventas: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    pagadas = solo_pagadas(ventas)
    return (
        pagadas.groupby("titulo")
        .agg(ingreso=("ingreso", "sum"), unidades=("unidades", "sum"))
        .sort_values("ingreso", ascending=False)
        .head(n)
        .reset_index()
    )


def por_categoria(ventas: pd.DataFrame) -> pd.DataFrame:
    return (
        solo_pagadas(ventas).groupby("categoria")["ingreso"]
        .sum().sort_values(ascending=False).reset_index()
    )


def por_provincia(ventas: pd.DataFrame) -> pd.DataFrame:
    return (
        solo_pagadas(ventas).groupby("provincia")["ingreso"]
        .sum().sort_values(ascending=False).reset_index()
    )


def por_medio_entrega(ventas: pd.DataFrame) -> pd.DataFrame:
    pagadas = solo_pagadas(ventas)
    return (
        pagadas.groupby("medio_entrega")
        .agg(ingreso=("ingreso", "sum"), ordenes=("order_id", "count"), unidades=("unidades", "sum"))
        .reset_index()
        .sort_values("ingreso", ascending=False)
    )


# --------------------------------------------------------------------------- #
# Conversion y embudo                                                          #
# --------------------------------------------------------------------------- #
def funnel(ventas: pd.DataFrame, visitas: pd.DataFrame, preguntas: pd.DataFrame) -> dict:
    """
    Embudo: visitas -> preguntas -> ventas.

    Donde se rompe el embudo te dice que optimizar:
    - Pocas preguntas / muchas visitas: el titulo/foto no engancha (problema de trafico frio).
    - Muchas preguntas / pocas ventas: hay friccion informativa (precio, garantia, cuotas).
      Aca es donde pega el agente de IA.
    - Pocas visitas / buena conversion: el producto convierte bien pero le falta trafico.
    """
    return {
        "visitas": int(visitas["visitas"].sum()),
        "preguntas": int(preguntas["preguntas"].sum()),
        "ventas": int(len(solo_pagadas(ventas))),
    }


def conversion_por_publicacion(ventas: pd.DataFrame, visitas: pd.DataFrame) -> pd.DataFrame:
    """
    Tasa de conversion (CVR) por item = ordenes / visitas.
    Fuente principal de la scatter chart.
    """
    pagadas = solo_pagadas(ventas)
    vis_agg = visitas.groupby(["item_id", "titulo", "marca"])["visitas"].sum().reset_index()
    ven_agg = pagadas.groupby("item_id").agg(
        ordenes=("order_id", "count"),
        ingreso=("ingreso", "sum"),
    ).reset_index()
    m = vis_agg.merge(ven_agg, on="item_id", how="left").fillna(0)
    m["conversion"] = (m["ordenes"] / m["visitas"] * 100).where(m["visitas"] > 0, 0.0)
    return m.sort_values("conversion", ascending=False)


# --------------------------------------------------------------------------- #
# Concentracion / Pareto                                                       #
# --------------------------------------------------------------------------- #
def pareto_sku(ventas: pd.DataFrame) -> pd.DataFrame:
    """
    Curva de Pareto por SKU.

    La regla 80/20: en la mayoria de los negocios el 20% de los SKUs genera
    el 80% de la venta. El Pareto te muestra exactamente cuantos SKUs son
    criticos y cuales son 'cola larga'.
    Clasificacion ABC:
      A = primeros SKUs hasta el 70% del GMV (criticos)
      B = hasta el 90%
      C = el resto (cola)
    """
    pagadas = solo_pagadas(ventas)
    t = (
        pagadas.groupby(["titulo", "marca"])["ingreso"]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )
    total = t["ingreso"].sum()
    t["pct_acum"] = t["ingreso"].cumsum() / total * 100
    t["abc"] = pd.cut(t["pct_acum"], bins=[0, 70, 90, 100], labels=["A", "B", "C"], right=True)
    return t


def productos_sin_conversion(ventas: pd.DataFrame, visitas: pd.DataFrame, min_visitas: int = 30) -> pd.DataFrame:
    """
    Items con visitas suficientes pero cero ventas pagadas en el periodo.
    Candidatos a optimizar (precio/foto/titulo) o dar de baja.
    """
    pagadas = solo_pagadas(ventas)
    vis_agg = visitas.groupby(["item_id", "titulo"])["visitas"].sum().reset_index()
    ven_agg = pagadas.groupby("item_id")["order_id"].count().reset_index(name="ordenes")
    m = vis_agg.merge(ven_agg, on="item_id", how="left").fillna(0)
    return (
        m[(m["visitas"] >= min_visitas) & (m["ordenes"] == 0)]
        .sort_values("visitas", ascending=False)
        .reset_index(drop=True)
    )
