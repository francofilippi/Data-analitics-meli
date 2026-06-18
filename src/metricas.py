"""
Metricas del negocio: aca vive el "data science".

Cada funcion toma un DataFrame de pandas y devuelve un numero, una serie o una
tabla agregada. El dashboard (app/dashboard.py) solo llama estas funciones y
dibuja el resultado: separar el CALCULO de la VISUALIZACION es la regla de oro
para que el codigo se entienda y se pueda testear.

Conceptos de pandas que vas a ver repetidos aca:
  - filtrado booleano:   df[df["col"] == valor]
  - agregacion:          df.groupby("col")["otra"].sum()
  - series temporales:   df.resample("D" / "ME", on="fecha")
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


# --------------------------------------------------------------------------- #
# 1. Carga                                                                     #
# --------------------------------------------------------------------------- #
def cargar_datos() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Lee los CSV y convierte 'fecha' a tipo fecha real (no texto)."""
    ventas = pd.read_csv(DATA_DIR / "ventas.csv", parse_dates=["fecha"])
    visitas = pd.read_csv(DATA_DIR / "visitas.csv", parse_dates=["fecha"])
    return ventas, visitas


# --------------------------------------------------------------------------- #
# 2. Filtros (por cliente y por rango de fechas)                               #
# --------------------------------------------------------------------------- #
def filtrar(
    df: pd.DataFrame,
    cliente: str | None = None,
    desde: pd.Timestamp | None = None,
    hasta: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Devuelve un subconjunto del df. None en un parametro = no filtra por eso."""
    m = pd.Series(True, index=df.index)  # mascara que arranca toda en True
    if cliente:
        m &= df["cliente_ml"] == cliente
    if desde is not None:
        m &= df["fecha"] >= desde
    if hasta is not None:
        m &= df["fecha"] <= hasta
    return df[m]


def solo_pagadas(ventas: pd.DataFrame) -> pd.DataFrame:
    """Para ingresos reales se cuentan solo ordenes 'pagado' (no canceladas)."""
    return ventas[ventas["estado"] == "pagado"]


# --------------------------------------------------------------------------- #
# 3. KPIs del periodo (los numeros grandes de las tarjetas)                    #
# --------------------------------------------------------------------------- #
def kpis_periodo(ventas: pd.DataFrame, visitas: pd.DataFrame) -> dict:
    """
    Calcula los indicadores clave de un periodo ya filtrado.

    Tasa de conversion = ordenes / visitas. Es la metrica reina del e-commerce:
    de cada 100 personas que entran a ver, cuantas compran.
    """
    pagadas = solo_pagadas(ventas)
    ingreso = pagadas["ingreso"].sum()
    unidades = pagadas["unidades"].sum()
    ordenes = len(pagadas)
    total_visitas = visitas["visitas"].sum()

    return {
        "ingreso": float(ingreso),
        "unidades": int(unidades),
        "ordenes": ordenes,
        # ticket promedio = cuanto deja en promedio cada orden
        "ticket_promedio": float(ingreso / ordenes) if ordenes else 0.0,
        "visitas": int(total_visitas),
        "conversion": float(ordenes / total_visitas * 100) if total_visitas else 0.0,
        "comisiones": float(pagadas["comision_ml"].sum()),
        # cancelaciones: salud de la operacion
        "tasa_cancelacion": (
            float((ventas["estado"] == "cancelado").mean() * 100) if len(ventas) else 0.0
        ),
    }


def variacion_pct(actual: float, anterior: float) -> float | None:
    """
    Cuanto crecio/cayo un KPI vs el periodo anterior, en %.
    None cuando no hay base de comparacion (periodo anterior en cero).
    """
    if anterior == 0:
        return None
    return round((actual - anterior) / anterior * 100, 1)


# --------------------------------------------------------------------------- #
# 4. Series temporales (para los graficos de tendencia)                        #
# --------------------------------------------------------------------------- #
def serie_diaria(ventas: pd.DataFrame) -> pd.DataFrame:
    """Ingreso y ordenes por dia. resample('D') agrupa por dia calendario."""
    pagadas = solo_pagadas(ventas)
    s = (
        pagadas.resample("D", on="fecha")
        .agg(ingreso=("ingreso", "sum"), ordenes=("order_id", "count"))
        .reset_index()
    )
    return s


def serie_mensual(ventas: pd.DataFrame) -> pd.DataFrame:
    """
    Lo mismo pero por mes ('ME' = month end). Sirve para 'comparar varios
    periodos mensuales' y ver estacionalidad (ej: el pico de noviembre).
    """
    pagadas = solo_pagadas(ventas)
    s = (
        pagadas.resample("ME", on="fecha")
        .agg(ingreso=("ingreso", "sum"), ordenes=("order_id", "count"))
        .reset_index()
    )
    s["mes"] = s["fecha"].dt.strftime("%Y-%m")
    return s


# --------------------------------------------------------------------------- #
# 5. Rankings y cortes (group by)                                              #
# --------------------------------------------------------------------------- #
def top_productos(ventas: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Top N productos por ingreso. El clasico groupby + sort + head."""
    pagadas = solo_pagadas(ventas)
    t = (
        pagadas.groupby("titulo")
        .agg(ingreso=("ingreso", "sum"), unidades=("unidades", "sum"))
        .sort_values("ingreso", ascending=False)
        .head(n)
        .reset_index()
    )
    return t


def por_categoria(ventas: pd.DataFrame) -> pd.DataFrame:
    pagadas = solo_pagadas(ventas)
    return (
        pagadas.groupby("categoria")["ingreso"]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )


def por_provincia(ventas: pd.DataFrame) -> pd.DataFrame:
    pagadas = solo_pagadas(ventas)
    return (
        pagadas.groupby("provincia")["ingreso"]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )
