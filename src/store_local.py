"""
Respaldo del histórico en archivos parquet versionados en el repo.

Por qué (además de Supabase): la API de ML solo devuelve órdenes de los
últimos ~12 meses, así que el YoY depende de tener "fotos" de los meses viejos
guardadas en algún lado durable. Supabase cumple ese rol, pero un proyecto
free-tier se pausa por inactividad y puede perder datos. Estos archivos son un
respaldo independiente: viven en git, se despliegan con la app (solo-lectura en
Streamlit Cloud) y sobreviven aunque Supabase desaparezca.

Layout en disco (commiteado):
    data/historico/{cliente}/{YYYY-MM}__{kind}.parquet

  kind = 'v' | 'vis' | 'preg'   → datos completos (envíos/visitas/preguntas)
  kind = 'lv'                   → ventas livianas (solo órdenes, MoM/YoY)

Mismo interfaz que src/store.py (enabled / load_months_range / save_month /
stats) para que el dashboard pueda usar ambos de forma simétrica.

Poblá estos archivos con scripts/snapshot_historico.py (vuelca Supabase → repo)
y commiteá data/historico/.
"""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path

import pandas as pd

# Raíz del repo → data/historico (un nivel arriba de src/)
BASE = Path(__file__).resolve().parent.parent / "data" / "historico"

_FNAME_RE = re.compile(r"^(\d{4})-(\d{2})__([a-z]+)\.parquet$")


def _safe(cliente: str) -> str:
    """Nombre de carpeta seguro para el filesystem."""
    return re.sub(r"[^A-Za-z0-9_.-]", "_", str(cliente))


def enabled() -> bool:
    """Hay respaldo local si existe el directorio (aunque esté vacío lo creamos al guardar)."""
    return BASE.exists()


def _path(cliente: str, mes, kind: str) -> Path:
    m = mes if isinstance(mes, date) else pd.Timestamp(mes).date()
    return BASE / _safe(cliente) / f"{m:%Y-%m}__{kind}.parquet"


def save_month(cliente: str, mes, kind: str, df: pd.DataFrame) -> None:
    """
    Guarda el DataFrame de (cliente, mes, kind) como parquet en el repo.
    Omite los meses vacíos: un archivo vacío bloquearía el fallback a ML/Supabase
    en la lectura (un df no-None se usa tal cual), y no aporta nada al histórico.
    """
    try:
        if df is None or df.empty:
            return
        p = _path(cliente, mes, kind)
        p.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(p, index=False)
    except Exception:
        pass


def load_months_range(
    cliente: str, desde, hasta, kinds: tuple[str, ...]
) -> dict[tuple, pd.DataFrame]:
    """
    Lee todos los meses guardados del rango [desde, hasta] para los kinds pedidos.
    Devuelve dict keyed por (mes_date, kind) → DataFrame, igual que store.load_months_range.
    mes_date es el primer día del mes (date), para coincidir con las claves de Supabase.
    """
    result: dict[tuple, pd.DataFrame] = {}
    cdir = BASE / _safe(cliente)
    if not cdir.is_dir():
        return result
    d0 = desde if isinstance(desde, date) else pd.Timestamp(desde).date()
    d1 = hasta if isinstance(hasta, date) else pd.Timestamp(hasta).date()
    kinds_set = set(kinds)
    for p in cdir.glob("*.parquet"):
        m = _FNAME_RE.match(p.name)
        if not m:
            continue
        year, month, kind = int(m.group(1)), int(m.group(2)), m.group(3)
        if kind not in kinds_set:
            continue
        mes_date = date(year, month, 1)
        if not (d0 <= mes_date <= d1):
            continue
        try:
            result[(mes_date, kind)] = pd.read_parquet(p)
        except Exception:
            continue
    return result


def stats() -> pd.DataFrame:
    """Resumen de lo guardado en disco: por cliente y kind, meses / filas / rango."""
    filas = []
    if BASE.is_dir():
        for cdir in sorted(BASE.iterdir()):
            if not cdir.is_dir():
                continue
            agg: dict = {}
            for p in cdir.glob("*.parquet"):
                m = _FNAME_RE.match(p.name)
                if not m:
                    continue
                kind = m.group(3)
                mes_date = date(int(m.group(1)), int(m.group(2)), 1)
                try:
                    n = len(pd.read_parquet(p))
                except Exception:
                    n = 0
                a = agg.setdefault(kind, {"meses": 0, "filas": 0, "desde": mes_date, "hasta": mes_date})
                a["meses"] += 1
                a["filas"] += n
                a["desde"] = min(a["desde"], mes_date)
                a["hasta"] = max(a["hasta"], mes_date)
            for kind, a in sorted(agg.items()):
                filas.append({"cliente": cdir.name, "kind": kind, **a})
    return pd.DataFrame(filas)
