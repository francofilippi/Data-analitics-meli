"""
Refresca los access_token y refresh_token de todos los sellers configurados
en un archivo .streamlit/secrets.toml local, e imprime el bloque actualizado
para pegar en Streamlit Cloud.

Uso (desde la raiz del repo, con el .venv activado):
    python scripts/refresh_tokens.py

Cuándo correrlo:
  - Si el dashboard muestra "Error 401" al cargar datos reales.
  - Como rutina cada 5 meses para renovar el refresh_token antes de que expire.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Necesita toml (Python 3.11+ lo trae built-in como tomllib para lectura)
try:
    import tomllib  # Python 3.11+
except ImportError:
    try:
        import tomli as tomllib  # pip install tomli
    except ImportError:
        print("ERROR: instalá tomli:  pip install tomli")
        sys.exit(1)

import requests

ML_BASE = "https://api.mercadolibre.com"
SECRETS_PATH = Path(__file__).resolve().parent.parent / ".streamlit" / "secrets.toml"


def refresh_token(client_id: str, client_secret: str, refresh_token: str) -> dict:
    resp = requests.post(
        f"{ML_BASE}/oauth/token",
        data={
            "grant_type": "refresh_token",
            "client_id": client_id,
            "client_secret": client_secret,
            "refresh_token": refresh_token,
        },
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()


def main() -> None:
    if not SECRETS_PATH.exists():
        print(f"No encontré {SECRETS_PATH}")
        print("Copiá .streamlit/secrets.toml.example a .streamlit/secrets.toml y completalo.")
        sys.exit(1)

    with open(SECRETS_PATH, "rb") as f:
        secrets = tomllib.load(f)

    ml_sections = {k: v for k, v in secrets.items() if k.startswith("ml_")}
    if not ml_sections:
        print("No hay secciones [ml_*] en secrets.toml.")
        sys.exit(1)

    print("Refrescando tokens...\n")
    updated: dict[str, dict] = {}

    for section, cfg in ml_sections.items():
        nombre = section[3:]
        try:
            data = refresh_token(cfg["client_id"], cfg["client_secret"], cfg["refresh_token"])
            updated[section] = {
                **cfg,
                "access_token": data["access_token"],
                "refresh_token": data["refresh_token"],
            }
            print(f"  ✅ {nombre} — OK")
        except Exception as e:
            print(f"  ❌ {nombre} — ERROR: {e}")
            updated[section] = cfg

    print()
    print("=" * 60)
    print("PEGÁ ESTO EN Streamlit Cloud → App settings → Secrets")
    print("(solo las secciones [ml_*] — el resto de secrets no cambia)")
    print("=" * 60)
    for section, cfg in updated.items():
        print(f"\n[{section}]")
        for k, v in cfg.items():
            print(f'{k} = "{v}"')


if __name__ == "__main__":
    main()
