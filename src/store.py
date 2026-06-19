"""
Almacenamiento persistente del histórico en Postgres (Supabase).

Por qué: el caché de Streamlit vive en /tmp y se borra al redeployar o dormir
la app. Para que un seller que entra esporádicamente (hoy, en 15 días o en 2
meses) cargue rápido, los meses cerrados (inmutables) se guardan una sola vez
acá y se leen al instante. A ML solo se le piden los meses que faltan + el mes
en curso.

Cada (cliente, mes, kind) guarda un DataFrame serializado como parquet:
  kind = 'v' | 'vis' | 'preg'        → datos completos (con envíos/visitas)
  kind = 'lv'                        → ventas livianas (solo órdenes, MoM/YoY)

Configuración: poné el connection string de Supabase en los Secrets de
Streamlit como:

    [supabase]
    dsn = "postgresql://postgres.<ref>:<password>@<host>.pooler.supabase.com:6543/postgres"

(o como variable de entorno DATABASE_URL para correr local).
"""

from __future__ import annotations

import io
import os

import pandas as pd

_TABLE = "ml_cache"
_conn = None


def _dsn() -> str | None:
    try:
        import streamlit as st
        sec = st.secrets.get("supabase", {})
        dsn = (dict(sec).get("dsn") if sec else None) or st.secrets.get("DATABASE_URL")
        if dsn:
            return str(dsn)
    except Exception:
        pass
    return os.environ.get("DATABASE_URL")


def enabled() -> bool:
    return bool(_dsn())


def _connect():
    import psycopg2
    dsn = _dsn()
    if "sslmode=" not in dsn:
        dsn += ("&" if "?" in dsn else "?") + "sslmode=require"
    conn = psycopg2.connect(dsn, connect_timeout=10)
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {_TABLE} (
                cliente    text NOT NULL,
                mes        date NOT NULL,
                kind       text NOT NULL,
                data       bytea NOT NULL,
                rows       integer NOT NULL DEFAULT 0,
                updated_at timestamptz NOT NULL DEFAULT now(),
                PRIMARY KEY (cliente, mes, kind)
            )
            """
        )
    return conn


def _get_conn():
    """Conexión cacheada; reconecta si se cayó."""
    global _conn
    try:
        if _conn is None or _conn.closed:
            _conn = _connect()
        else:
            # ping rápido
            with _conn.cursor() as cur:
                cur.execute("SELECT 1")
    except Exception:
        _conn = _connect()
    return _conn


def load_month(cliente: str, mes, kind: str) -> pd.DataFrame | None:
    """Devuelve el DataFrame guardado para (cliente, mes, kind) o None."""
    try:
        conn = _get_conn()
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT data FROM {_TABLE} WHERE cliente=%s AND mes=%s AND kind=%s",
                (cliente, mes, kind),
            )
            row = cur.fetchone()
        if not row:
            return None
        return pd.read_parquet(io.BytesIO(bytes(row[0])))
    except Exception:
        return None


def load_months_range(
    cliente: str, desde, hasta, kinds: tuple[str, ...]
) -> dict[tuple, pd.DataFrame]:
    """
    Trae todos los meses del rango [desde, hasta] para los kinds pedidos en
    UNA sola query. Devuelve dict keyed por (mes_date, kind) → DataFrame.
    Mucho más rápido que N llamadas a load_month cuando el rango tiene varios meses.
    """
    result: dict[tuple, pd.DataFrame] = {}
    try:
        conn = _get_conn()
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT mes, kind, data FROM {_TABLE}
                WHERE cliente = %s
                  AND mes >= %s AND mes <= %s
                  AND kind = ANY(%s)
                """,
                (cliente, desde, hasta, list(kinds)),
            )
            for mes, kind, data in cur.fetchall():
                result[(mes, kind)] = pd.read_parquet(io.BytesIO(bytes(data)))
    except Exception:
        pass
    return result


def save_month(cliente: str, mes, kind: str, df: pd.DataFrame) -> None:
    """Guarda (upsert) el DataFrame para (cliente, mes, kind)."""
    try:
        import psycopg2
        buf = io.BytesIO()
        df.to_parquet(buf, index=False)
        conn = _get_conn()
        with conn.cursor() as cur:
            cur.execute(
                f"""
                INSERT INTO {_TABLE} (cliente, mes, kind, data, rows, updated_at)
                VALUES (%s, %s, %s, %s, %s, now())
                ON CONFLICT (cliente, mes, kind)
                DO UPDATE SET data = EXCLUDED.data, rows = EXCLUDED.rows, updated_at = now()
                """,
                (cliente, mes, kind, psycopg2.Binary(buf.getvalue()), len(df)),
            )
    except Exception:
        pass
