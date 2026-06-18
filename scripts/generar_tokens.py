"""
Genera tokens aleatorios por cliente y muestra:
  1. El bloque [tokens] para pegar en Streamlit Cloud → App settings → Secrets
  2. Las URLs privadas para compartir con cada seller

Uso:
    python scripts/generar_tokens.py --url https://dashboardpy-xxx.streamlit.app
"""

from __future__ import annotations

import argparse
import secrets

CLIENTES = ["tienda_norte", "deco_hogar", "tech_outlet"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="https://TU_APP.streamlit.app", help="URL base de la app")
    args = parser.parse_args()
    base = args.url.rstrip("/")

    tokens: dict[str, str] = {"admin": secrets.token_urlsafe(20)}
    for c in CLIENTES:
        tokens[c] = secrets.token_urlsafe(20)

    print("=" * 60)
    print("1. PEGÁ ESTO EN Streamlit Cloud → App settings → Secrets")
    print("=" * 60)
    print("[tokens]")
    for k, v in tokens.items():
        print(f'{k} = "{v}"')

    print()
    print("=" * 60)
    print("2. URLs PARA COMPARTIR")
    print("=" * 60)
    print(f"Admin (vos):  {base}/?token={tokens['admin']}")
    print()
    for c in CLIENTES:
        print(f"{c}:  {base}/?cliente={c}&token={tokens[c]}")


if __name__ == "__main__":
    main()
