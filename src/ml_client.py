"""
Cliente HTTP para la API de Mercado Libre.

Responsabilidades:
- Adjuntar el Bearer token en cada request.
- Auto-refresh cuando el access_token expira (HTTP 401).
- Retry básico en rate limit (HTTP 429).
- Paginación de resultados con offset/limit.

El access_token de ML dura 6 horas.
El refresh_token dura 6 meses; cuando se usa, ML devuelve uno nuevo.
Como Streamlit secrets son estáticos, guardamos el token refrescado en
session_state para no perderlo entre reruns dentro de la misma sesión.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

import requests

ML_BASE = "https://api.mercadolibre.com"
_SESSION = requests.Session()


@dataclass
class MLClient:
    nombre: str          # nombre del cliente en el dashboard (ej: "tienda_norte")
    client_id: str
    client_secret: str
    access_token: str
    refresh_token: str
    seller_id: str = ""  # se obtiene automaticamente de /users/me si no se configura
    refreshed: bool = field(default=False, repr=False)

    def __post_init__(self):
        self._lock = threading.Lock()

    def ensure_seller_id(self) -> None:
        """Obtiene el seller_id desde /users/me si no está configurado."""
        if not self.seller_id:
            data = self.get("/users/me")
            self.seller_id = str(data["id"])

    # ----------------------------------------------------------------------- #
    # Autenticacion                                                             #
    # ----------------------------------------------------------------------- #
    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.access_token}"}

    def refresh(self) -> None:
        """Refresca el access_token con el refresh_token actual (thread-safe)."""
        with self._lock:
            resp = _SESSION.post(
                f"{ML_BASE}/oauth/token",
                data={
                    "grant_type": "refresh_token",
                    "client_id": self.client_id,
                    "client_secret": self.client_secret,
                    "refresh_token": self.refresh_token,
                },
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
            self.access_token = data["access_token"]
            self.refresh_token = data["refresh_token"]
            self.refreshed = True

    # ----------------------------------------------------------------------- #
    # HTTP                                                                     #
    # ----------------------------------------------------------------------- #
    def get(self, path: str, params: dict | None = None, _retry: bool = True) -> dict:
        url = f"{ML_BASE}{path}"
        resp = _SESSION.get(url, headers=self._headers(), params=params, timeout=15)

        if resp.status_code == 401 and _retry:
            self.refresh()
            return self.get(path, params, _retry=False)

        if resp.status_code == 429:
            time.sleep(2)
            return self.get(path, params, _retry=False)

        resp.raise_for_status()
        return resp.json()

    def paginate(self, path: str, params: dict | None = None, limit: int = 50) -> list[dict]:
        """Recorre todas las páginas de un endpoint paginado con offset/limit."""
        params = dict(params or {})
        params["limit"] = limit
        params["offset"] = 0
        results: list[dict] = []

        while True:
            data = self.get(path, params)
            page = data.get("results", [])
            results.extend(page)
            paging = data.get("paging", {})
            params["offset"] += limit
            if params["offset"] >= paging.get("total", len(results)):
                break

        return results


def from_secrets(nombre: str, sec: dict) -> MLClient:
    """Construye un MLClient desde un bloque de secrets de Streamlit.
    seller_id es opcional — si no está, se obtiene automáticamente de /users/me.
    """
    return MLClient(
        nombre=nombre,
        client_id=str(sec["client_id"]),
        client_secret=str(sec["client_secret"]),
        access_token=str(sec["access_token"]),
        refresh_token=str(sec["refresh_token"]),
        seller_id=str(sec.get("seller_id", "")),
    )
