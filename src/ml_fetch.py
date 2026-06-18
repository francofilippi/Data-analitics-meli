"""
Fetchers de la API de Mercado Libre → DataFrames.

Cada función devuelve un DataFrame con exactamente las mismas columnas que
generar_datos.py produce para datos sintéticos. Esto permite que metricas.py
funcione igual con datos reales o sintéticos — solo cambia la fuente.

Endpoints usados:
  /orders/search               — ventas (paginado)
  /items?ids=...               — detalles de items (batch de 20)
  /items/{id}/visits/time_window — visitas diarias por publicación
  /questions/search            — preguntas (paginado)
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from src.ml_client import MLClient

# --------------------------------------------------------------------------- #
# Helpers de mapeo                                                             #
# --------------------------------------------------------------------------- #
_LISTING_TYPE_MAP = {
    "gold_special": "clasica",
    "gold_pro": "clasica",
    "free": "clasica",
    "gold_premium": "premium",
    "gold": "premium",
}

_LOGISTIC_MAP = {
    "fulfillment": "Full",
    "self_service": "Flex",
    "me2": "MercadoEnvios2",
    "not_specified": "Retiro en local",
    "drop_off": "MercadoEnvios2",
    "xd_drop_off": "MercadoEnvios2",
}

COMISION_PCT = {"clasica": 0.13, "premium": 0.18}


def _tipo_pub(listing_type_id: str) -> str:
    return _LISTING_TYPE_MAP.get(listing_type_id, "clasica")


def _medio_entrega(shipping: dict) -> str:
    if not shipping:
        return "Retiro en local"
    logistic = shipping.get("logistic_type", "")
    mode = shipping.get("shipping_mode", "")
    return _LOGISTIC_MAP.get(logistic) or _LOGISTIC_MAP.get(mode, "MercadoEnvios2")


def _iso(d: date, eod: bool = False) -> str:
    """Formato ISO con timezone -03:00 que espera la API de ML."""
    t = "23:59:59.999" if eod else "00:00:00.000"
    return f"{d}T{t}-03:00"


# --------------------------------------------------------------------------- #
# 1. Órdenes                                                                   #
# --------------------------------------------------------------------------- #
def fetch_orders(client: MLClient, desde: date, hasta: date) -> pd.DataFrame:
    """
    Descarga órdenes pagadas Y canceladas en el rango de fechas.

    Por qué incluir canceladas: la tasa de cancelación es un KPI.
    Filtramos 'pagado' en metricas.solo_pagadas() igual que con datos sintéticos.
    """
    filas: list[dict] = []

    for status in ("paid", "cancelled"):
        orders = client.paginate(
            "/orders/search",
            params={
                "seller": client.seller_id,
                "order.status": status,
                "date_created.from": _iso(desde),
                "date_created.to": _iso(hasta, eod=True),
                "sort": "date_asc",
            },
        )
        for order in orders:
            for oi in order.get("order_items", []):
                tipo = _tipo_pub(oi.get("listing_type_id", ""))
                unidades = oi.get("quantity", 1)
                precio = float(oi.get("unit_price", 0))
                ingreso = unidades * precio
                filas.append({
                    "order_id": order["id"],
                    "fecha": pd.Timestamp(order["date_created"]).date(),
                    "cliente_ml": client.nombre,
                    "item_id": oi["item"]["id"],
                    "titulo": oi["item"].get("title", ""),
                    "categoria": "",     # enriquecido en fetch_all
                    "marca": "",
                    "tipo_publicacion": tipo,
                    "medio_entrega": _medio_entrega(order.get("shipping") or {}),
                    "unidades": unidades,
                    "precio_unitario": precio,
                    "ingreso": ingreso,
                    "comision_ml": round(ingreso * COMISION_PCT.get(tipo, 0.13), 2),
                    "costo_envio": float(order.get("shipping_cost") or 0),
                    "provincia": "",     # requiere /shipments/{id}, se agrega luego
                    "estado": "pagado" if status == "paid" else "cancelado",
                })

    if not filas:
        return pd.DataFrame(columns=[
            "order_id", "fecha", "cliente_ml", "item_id", "titulo",
            "categoria", "marca", "tipo_publicacion", "medio_entrega",
            "unidades", "precio_unitario", "ingreso", "comision_ml",
            "costo_envio", "provincia", "estado",
        ])
    return pd.DataFrame(filas)


# --------------------------------------------------------------------------- #
# 2. Detalles de items (categoría, atributos, marca)                          #
# --------------------------------------------------------------------------- #
def fetch_item_details(client: MLClient, item_ids: list[str]) -> dict[str, dict]:
    """
    Trae categoria y marca de cada item.
    La API acepta hasta 20 IDs por llamada (batch).

    Lección de DS: el "enrichment" — joinear tu tabla de hechos (órdenes)
    con una tabla de dimensiones (atributos de items). Exactamente como
    haría un join en SQL o un merge en pandas.
    """
    details: dict[str, dict] = {}
    # Batch de 20 para no superar el límite de la API
    for i in range(0, len(item_ids), 20):
        batch = item_ids[i : i + 20]
        resp = client.get("/items", params={"ids": ",".join(batch)})
        # La respuesta es una lista [{code: 200, body: {...}}, ...]
        if isinstance(resp, list):
            for entry in resp:
                if entry.get("code") == 200:
                    body = entry["body"]
                    item_id = body["id"]
                    # Buscar marca en los atributos del item
                    marca = ""
                    for attr in body.get("attributes", []):
                        if attr.get("id") in ("BRAND", "brand"):
                            marca = attr.get("value_name", "")
                            break
                    details[item_id] = {
                        "categoria": body.get("category_id", ""),
                        "marca": marca,
                    }
    return details


# --------------------------------------------------------------------------- #
# 3. Visitas diarias                                                           #
# --------------------------------------------------------------------------- #
def fetch_visits_daily(
    client: MLClient,
    item_ids: list[str],
    desde: date,
    hasta: date,
) -> pd.DataFrame:
    """
    Visitas por día por item usando el endpoint time_window.

    El endpoint devuelve los últimos N días en unidades 'day'.
    Para períodos > 90 días usamos múltiples llamadas solapadas.

    Lección de DS: trabajar con limitaciones de APIs reales — a veces el dato
    que querés no está en un solo endpoint y hay que construirlo combinando
    varias llamadas o interpolando.
    """
    dias_total = (hasta - desde).days + 1
    filas: list[dict] = []

    for item_id in item_ids:
        try:
            # ML acepta hasta 90 días por llamada
            dias_req = min(dias_total, 90)
            resp = client.get(
                f"/items/{item_id}/visits/time_window",
                params={
                    "last": dias_req,
                    "unit": "day",
                    "ending": hasta.isoformat(),
                },
            )
            for entry in resp.get("results", []):
                dia = pd.Timestamp(entry["date"]).date()
                if desde <= dia <= hasta:
                    filas.append({
                        "fecha": dia,
                        "cliente_ml": client.nombre,
                        "item_id": item_id,
                        "visitas": entry.get("total", 0),
                    })
        except Exception:
            # Si falla para un item, seguimos con los demás
            continue

    return pd.DataFrame(filas) if filas else pd.DataFrame(
        columns=["fecha", "cliente_ml", "item_id", "visitas"]
    )


# --------------------------------------------------------------------------- #
# 4. Preguntas                                                                 #
# --------------------------------------------------------------------------- #
def fetch_questions(client: MLClient, desde: date, hasta: date) -> pd.DataFrame:
    """
    Preguntas recibidas en el período.

    Importante para el embudo: Visitas → Preguntas → Ventas.
    Si hay muchas preguntas y pocas ventas = fricción de info.
    """
    questions = client.paginate(
        "/questions/search",
        params={
            "seller_id": client.seller_id,
            "date_created_from": _iso(desde),
            "date_created_to": _iso(hasta, eod=True),
            "sort_fields": "date_created",
            "sort_types": "ASC",
        },
    )
    filas: list[dict] = []
    for q in questions:
        filas.append({
            "fecha": pd.Timestamp(q["date_created"]).date(),
            "cliente_ml": client.nombre,
            "item_id": q.get("item_id", ""),
            "preguntas": 1,
        })

    if not filas:
        return pd.DataFrame(columns=["fecha", "cliente_ml", "item_id", "preguntas"])

    # Agrupar por fecha + item_id (igual que el formato sintético)
    df = pd.DataFrame(filas)
    return (
        df.groupby(["fecha", "cliente_ml", "item_id"])["preguntas"]
        .sum()
        .reset_index()
    )


# --------------------------------------------------------------------------- #
# 5. fetch_all — punto de entrada principal                                   #
# --------------------------------------------------------------------------- #
def fetch_all(
    client: MLClient,
    desde: date,
    hasta: date,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Descarga y transforma todos los datos de un seller en el rango de fechas.
    Devuelve (ventas, visitas, preguntas) con el mismo schema que generar_datos.py.
    """
    # 1. Órdenes
    ventas = fetch_orders(client, desde, hasta)

    # 2. Enriquecer con detalles de items (categoría, marca)
    if not ventas.empty:
        item_ids = ventas["item_id"].unique().tolist()
        details = fetch_item_details(client, item_ids)
        ventas["categoria"] = ventas["item_id"].map(
            lambda iid: details.get(iid, {}).get("categoria", "Sin categoría")
        )
        ventas["marca"] = ventas["item_id"].map(
            lambda iid: details.get(iid, {}).get("marca", "Sin marca")
        )
        # Enriquecer títulos si están vacíos
        titulo_map = ventas.groupby("item_id")["titulo"].first().to_dict()
    else:
        item_ids = []
        titulo_map = {}

    # 3. Visitas
    visitas = fetch_visits_daily(client, item_ids, desde, hasta)
    if not visitas.empty and titulo_map:
        visitas["titulo"] = visitas["item_id"].map(titulo_map).fillna("")
        visitas["categoria"] = visitas["item_id"].map(
            lambda iid: details.get(iid, {}).get("categoria", "")
            if "details" in dir()
            else ""
        )
        visitas["marca"] = visitas["item_id"].map(
            lambda iid: details.get(iid, {}).get("marca", "")
            if "details" in dir()
            else ""
        )

    # 4. Preguntas
    preguntas = fetch_questions(client, desde, hasta)
    if not preguntas.empty and titulo_map:
        preguntas["titulo"] = preguntas["item_id"].map(titulo_map).fillna("")

    # Asegurar tipos de fecha consistentes
    for df in [ventas, visitas, preguntas]:
        if not df.empty and "fecha" in df.columns:
            df["fecha"] = pd.to_datetime(df["fecha"])

    return ventas, visitas, preguntas
