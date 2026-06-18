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

from concurrent.futures import ThreadPoolExecutor, as_completed
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
    "not_specified": "Retiro / Acordar",
    "custom": "Retiro / Acordar",
    "drop_off": "MercadoEnvios2",
    "xd_drop_off": "MercadoEnvios2",
    "cross_docking": "MercadoEnvios2",
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
                "order.date_created.from": _iso(desde),
                "order.date_created.to": _iso(hasta, eod=True),
                "sort": "date_asc",
            },
        )
        for order in orders:
            shipping_id = (order.get("shipping") or {}).get("id", "")
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
                    "shipping_id": str(shipping_id) if shipping_id else "",
                    "medio_entrega": "MercadoEnvios2",  # enriquecido vía /shipments en fetch_all
                    "unidades": unidades,
                    "precio_unitario": precio,
                    "ingreso": ingreso,
                    "comision_ml": round(ingreso * COMISION_PCT.get(tipo, 0.13), 2),
                    "costo_envio": float(order.get("shipping_cost") or 0),
                    "provincia": "",     # enriquecido vía /shipments en fetch_all
                    "estado": "pagado" if status == "paid" else "cancelado",
                })

    if not filas:
        return pd.DataFrame(columns=[
            "order_id", "fecha", "cliente_ml", "item_id", "titulo",
            "categoria", "marca", "tipo_publicacion", "shipping_id", "medio_entrega",
            "unidades", "precio_unitario", "ingreso", "comision_ml",
            "costo_envio", "provincia", "estado",
        ])
    return pd.DataFrame(filas)


# --------------------------------------------------------------------------- #
# 1b. Envíos — logistic_type (medio de entrega real) + provincia              #
# --------------------------------------------------------------------------- #
def fetch_shipments(client: MLClient, shipping_ids: list[str]) -> dict[str, dict]:
    """
    Para cada envío trae el logistic_type real (Flex/Full/ME2/Retiro) y la
    provincia del comprador. /orders/search NO trae el logistic_type, solo el
    id del envío — por eso hay que pegarle a /shipments/{id}.

    Se consulta en paralelo (4 workers) y se cachea en el parquet histórico,
    así solo se hace una vez por envío.
    """
    info: dict[str, dict] = {}

    def _one(sid: str) -> tuple[str, dict]:
        try:
            s = client.get(f"/shipments/{sid}")
            logistic = s.get("logistic_type") or (s.get("logistic") or {}).get("type", "")
            addr = s.get("receiver_address") or {}
            provincia = (addr.get("state") or {}).get("name", "") or ""
            return sid, {
                "medio_entrega": _LOGISTIC_MAP.get(logistic, "MercadoEnvios2"),
                "provincia": provincia or "Sin dato",
            }
        except Exception:
            return sid, {}

    ids = [s for s in shipping_ids if s]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for sid, data in pool.map(_one, ids):
            if data:
                info[sid] = data
    return info


# --------------------------------------------------------------------------- #
# 2. Detalles de items (categoría, atributos, marca)                          #
# --------------------------------------------------------------------------- #
# Cache de nombres de categoría: el id (ej MLA22195) nunca cambia de nombre.
_CAT_NOMBRE_CACHE: dict[str, str] = {}


def _nombre_categoria(client: MLClient, cat_id: str) -> str:
    """Resuelve el nombre legible de una categoría desde su id (cacheado)."""
    if not cat_id:
        return "Sin categoría"
    if cat_id in _CAT_NOMBRE_CACHE:
        return _CAT_NOMBRE_CACHE[cat_id]
    try:
        data = client.get(f"/categories/{cat_id}")
        nombre = data.get("name", cat_id)
    except Exception:
        nombre = cat_id
    _CAT_NOMBRE_CACHE[cat_id] = nombre
    return nombre


def fetch_item_details(client: MLClient, item_ids: list[str]) -> dict[str, dict]:
    """
    Trae categoria (nombre legible) y marca de cada item.
    La API acepta hasta 20 IDs por llamada (batch).

    Lección de DS: el "enrichment" — joinear tu tabla de hechos (órdenes)
    con una tabla de dimensiones (atributos de items). Exactamente como
    haría un join en SQL o un merge en pandas.
    """
    details: dict[str, dict] = {}
    # Batch de 20 para no superar el límite de la API
    for i in range(0, len(item_ids), 20):
        batch = item_ids[i : i + 20]
        resp = client.get(
            "/items",
            params={"ids": ",".join(batch), "attributes": "id,title,category_id,attributes"},
        )
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
                        "titulo": body.get("title", ""),
                        "categoria": _nombre_categoria(client, body.get("category_id", "")),
                        "marca": marca or "Sin marca",
                    }
    return details


def fetch_active_item_ids(client: MLClient) -> list[str]:
    """
    IDs de TODAS las publicaciones activas del seller.

    Necesario para conversión fidedigna: si solo miramos visitas de items que
    vendieron, la tasa de conversión queda inflada (faltan los items con
    tráfico y cero ventas). Con el catálogo completo el embudo es real.
    """
    try:
        return client.paginate(
            f"/users/{client.seller_id}/items/search",
            params={"status": "active"},
        )
    except Exception:
        return []


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
    dias_req = min(dias_total, 90)
    filas: list[dict] = []

    def _fetch_one(item_id: str) -> list[dict]:
        try:
            resp = client.get(
                f"/items/{item_id}/visits/time_window",
                params={"last": dias_req, "unit": "day", "ending": hasta.isoformat()},
            )
            return [
                {
                    "fecha": pd.Timestamp(e["date"]).date(),
                    "cliente_ml": client.nombre,
                    "item_id": item_id,
                    "visitas": e.get("total", 0),
                }
                for e in resp.get("results", [])
                if desde <= pd.Timestamp(e["date"]).date() <= hasta
            ]
        except Exception:
            return []

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(_fetch_one, iid): iid for iid in item_ids}
        for fut in as_completed(futures):
            filas.extend(fut.result())

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
    # Filtro de fecha del lado cliente: el endpoint a veces ignora el rango,
    # así garantizamos que solo contamos preguntas dentro del período.
    filas: list[dict] = []
    for q in questions:
        dia = pd.Timestamp(q["date_created"]).date()
        if not (desde <= dia <= hasta):
            continue
        filas.append({
            "fecha": dia,
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
    full: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Descarga y transforma todos los datos de un seller en el rango de fechas.
    Devuelve (ventas, visitas, preguntas) con el mismo schema que generar_datos.py.

    full=False: modo liviano para series largas (MoM/YoY). Solo trae órdenes —
    omite envíos, visitas, catálogo y preguntas, que para esos cálculos no se
    usan y dispararían miles de llamadas (una /shipments por orden).
    """
    # Obtener seller_id automáticamente si no fue configurado manualmente
    client.ensure_seller_id()

    # 1. Órdenes
    ventas = fetch_orders(client, desde, hasta)

    if not full:
        vacio_vis = pd.DataFrame(columns=["fecha", "cliente_ml", "item_id", "visitas"])
        vacio_preg = pd.DataFrame(columns=["fecha", "cliente_ml", "item_id", "preguntas"])
        if not ventas.empty:
            ventas["fecha"] = pd.to_datetime(ventas["fecha"])
        return ventas, vacio_vis, vacio_preg

    # Catálogo de items para visitas/conversión: TODOS los activos + los que
    # vendieron en el período (pueden estar pausados hoy). La unión evita
    # inflar la conversión por mirar solo items que vendieron.
    sold_ids = ventas["item_id"].unique().tolist() if not ventas.empty else []
    active_ids = fetch_active_item_ids(client)
    all_ids = sorted(set(sold_ids) | set(active_ids))

    # 2. Detalles de items (título, categoría, marca) para todo el catálogo
    details = fetch_item_details(client, all_ids) if all_ids else {}
    titulo_map = {iid: d.get("titulo", "") for iid, d in details.items()}

    if not ventas.empty:
        ventas["categoria"] = ventas["item_id"].map(
            lambda iid: details.get(iid, {}).get("categoria", "Sin categoría")
        )
        ventas["marca"] = ventas["item_id"].map(
            lambda iid: details.get(iid, {}).get("marca", "Sin marca")
        )
        # Completar títulos faltantes con los del catálogo
        ventas["titulo"] = ventas.apply(
            lambda r: r["titulo"] or titulo_map.get(r["item_id"], ""), axis=1
        )

        # 2b. Envíos: medio de entrega real (Flex/Full/ME2) + provincia
        ship_ids = [s for s in ventas["shipping_id"].unique().tolist() if s]
        envios = fetch_shipments(client, ship_ids)
        ventas["medio_entrega"] = ventas["shipping_id"].map(
            lambda sid: envios.get(sid, {}).get("medio_entrega", "Retiro / Acordar")
        )
        ventas["provincia"] = ventas["shipping_id"].map(
            lambda sid: envios.get(sid, {}).get("provincia", "Sin dato")
        )

    # 3. Visitas — de TODO el catálogo, no solo lo que vendió
    visitas = fetch_visits_daily(client, all_ids, desde, hasta)
    if not visitas.empty:
        visitas["titulo"] = visitas["item_id"].map(titulo_map).fillna("")
        visitas["categoria"] = visitas["item_id"].map(
            lambda iid: details.get(iid, {}).get("categoria", "")
        )
        visitas["marca"] = visitas["item_id"].map(
            lambda iid: details.get(iid, {}).get("marca", "")
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
