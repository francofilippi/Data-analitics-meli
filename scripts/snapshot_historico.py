"""
Vuelca el histórico guardado en Supabase a archivos parquet versionados en el
repo (data/historico/), como respaldo independiente y durable.

Por qué: la API de ML solo devuelve órdenes de los últimos ~12 meses, así que el
YoY depende de tener "fotos" de los meses viejos. Supabase cumple ese rol, pero
un proyecto free-tier se pausa por inactividad y puede perder datos. Estos
archivos viven en git y sobreviven aunque Supabase desaparezca. El dashboard los
lee con prioridad sobre Supabase (ver src/store_local.py).

Uso (desde la raíz del repo):
    # con el DSN en el entorno
    DATABASE_URL="postgresql://...supabase.../postgres" python scripts/snapshot_historico.py
    # o tomándolo de .streamlit/secrets.toml ([supabase].dsn)
    python scripts/snapshot_historico.py

Después:
    git add data/historico && git commit -m "snapshot histórico" && git push

Cuándo correrlo: una vez al mes (o cuando quieras congelar el último mes cerrado
antes de que ML lo descarte). Idealmente automatizado por GitHub Action.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _dsn_desde_secrets() -> str | None:
    """Lee [supabase].dsn o DATABASE_URL de .streamlit/secrets.toml si existe."""
    p = ROOT / ".streamlit" / "secrets.toml"
    if not p.exists():
        return None
    try:
        import tomllib  # Python 3.11+
    except ImportError:
        try:
            import tomli as tomllib
        except ImportError:
            return None
    with open(p, "rb") as f:
        data = tomllib.load(f)
    sup = data.get("supabase", {})
    return sup.get("dsn") or data.get("DATABASE_URL")


def main() -> None:
    # store._dsn() lee os.environ["DATABASE_URL"] cuando no hay runtime de Streamlit.
    if not os.environ.get("DATABASE_URL"):
        dsn = _dsn_desde_secrets()
        if dsn:
            os.environ["DATABASE_URL"] = str(dsn)

    from src import store, store_local

    if not store.enabled():
        print("ERROR: no hay DSN de Supabase. Pasá DATABASE_URL o configurá "
              ".streamlit/secrets.toml con [supabase].dsn.")
        sys.exit(1)

    diag = store.diagnose()
    if not diag.get("ok"):
        print(f"ERROR conectando a Supabase: {diag.get('error')}")
        sys.exit(1)
    print(f"Conectado a Supabase ({diag.get('filas', 0)} filas en ml_cache).")

    st = store.stats()
    if st.empty or "cliente" not in st.columns:
        print("Supabase está vacío — no hay nada para snapshotear.")
        return

    kinds = ("v", "vis", "preg", "lv")
    total_archivos = 0
    for cliente in sorted(st["cliente"].unique()):
        sub = st[st["cliente"] == cliente]
        desde = min(sub["desde"])
        hasta = max(sub["hasta"])
        meses = store.load_months_range(cliente, desde, hasta, kinds)
        n_cliente = 0
        for (mes, kind), df in meses.items():
            if df is None or df.empty:
                continue
            store_local.save_month(cliente, mes, kind, df)
            n_cliente += 1
        total_archivos += n_cliente
        print(f"  {cliente}: {n_cliente} archivos ({desde} → {hasta})")

    print(f"\nListo: {total_archivos} archivos en {store_local.BASE}")
    print("Ahora: git add data/historico && git commit -m 'snapshot histórico' && git push")


if __name__ == "__main__":
    main()
