"""
Generador de datos sinteticos de Mercado Libre.

Genera tres tablas:
  ventas.csv   — una fila por linea de orden
  visitas.csv  — visitas diarias por publicacion
  preguntas.csv — preguntas diarias por publicacion (parte del embudo)
"""

from __future__ import annotations

import random
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from faker import Faker

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
Faker.seed(SEED)

DATA_DIR = Path(__file__).parent
DIAS_DE_HISTORIA = 365

CLIENTES = ["tienda_norte", "deco_hogar", "tech_outlet"]

# (titulo, categoria, precio_base, popularidad, marca)
CATALOGO = {
    "tienda_norte": [
        ("Zapatillas Running Pro", "Calzado", 85000, 5, "Nike"),
        ("Campera Inflable", "Indumentaria", 62000, 3, "The North Face"),
        ("Mochila Urbana 25L", "Accesorios", 38000, 4, "Quechua"),
        ("Remera Algodon Premium", "Indumentaria", 18000, 6, "Adidas"),
        ("Medias Pack x3", "Indumentaria", 9000, 8, "Adidas"),
        ("Gorra Trucker", "Accesorios", 12000, 2, "Nike"),
    ],
    "deco_hogar": [
        ("Lampara de Pie Nordica", "Iluminacion", 95000, 3, "Loft"),
        ("Juego de Sabanas Queen", "Textil", 54000, 5, "Cannon"),
        ("Cuadro Decorativo 60x90", "Decoracion", 28000, 4, "Maisons"),
        ("Organizador Modular", "Organizacion", 33000, 6, "Loft"),
        ("Vela Aromatica Set", "Decoracion", 14000, 7, "Maisons"),
        ("Espejo Redondo 50cm", "Decoracion", 41000, 2, "Cannon"),
    ],
    "tech_outlet": [
        ("Auriculares Bluetooth", "Audio", 72000, 6, "Sony"),
        ("Cargador USB-C 65W", "Accesorios", 25000, 7, "Samsung"),
        ("Mouse Gamer RGB", "Perifericos", 38000, 5, "Logitech"),
        ("Teclado Mecanico", "Perifericos", 89000, 3, "Logitech"),
        ("Webcam Full HD", "Perifericos", 56000, 2, "Sony"),
        ("Hub USB 7 Puertos", "Accesorios", 31000, 4, "Kingston"),
    ],
}

PROVINCIAS = [
    ("Buenos Aires", 0.38), ("CABA", 0.20), ("Cordoba", 0.12),
    ("Santa Fe", 0.10), ("Mendoza", 0.07), ("Tucuman", 0.05),
    ("Otras", 0.08),
]

MEDIOS_ENTREGA = [
    ("MercadoEnvios2", 0.50),
    ("Full", 0.25),
    ("Flex", 0.20),
    ("Retiro en local", 0.05),
]

TIPO_PUBLICACION = [("clasica", 0.55), ("premium", 0.45)]
COMISION_PCT = {"clasica": 0.13, "premium": 0.18}

# Picos de campanas: (inicio, fin, multiplicador de demanda)
CAMPANAS_BOOST: list[tuple[date, date, float]] = [
    (date(2025, 11, 3), date(2025, 11, 5), 2.5),   # CyberMonday
    (date(2025, 12, 24), date(2025, 12, 26), 1.8),  # Navidad
    (date(2026, 2, 14), date(2026, 2, 14), 1.5),    # Dia Enamorados
    (date(2026, 5, 11), date(2026, 5, 13), 2.2),    # Hot Sale
]


def _elegir_ponderado(opciones: list[tuple]) -> str:
    return random.choices([o[0] for o in opciones], weights=[o[1] for o in opciones], k=1)[0]


def _factor_estacional(dia: date, inicio: date) -> float:
    factor = 1.0
    if dia.weekday() >= 5:
        factor *= 1.35
    if dia.month == 11:
        factor *= 1.8
    if dia.month in (1, 2):
        factor *= 0.8
    dias_transcurridos = (dia - inicio).days
    factor *= 1 + (dias_transcurridos / DIAS_DE_HISTORIA) * 0.4
    for c_ini, c_fin, boost in CAMPANAS_BOOST:
        if c_ini <= dia <= c_fin:
            factor *= boost
            break
    return factor


def generar() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    filas_ventas: list[dict] = []
    filas_visitas: list[dict] = []
    filas_preguntas: list[dict] = []
    order_id = 1000
    inicio = date.today() - timedelta(days=DIAS_DE_HISTORIA)

    for offset in range(DIAS_DE_HISTORIA):
        dia = inicio + timedelta(days=offset)
        estacional = _factor_estacional(dia, inicio)

        for cliente in CLIENTES:
            for titulo, categoria, precio_base, popularidad, marca in CATALOGO[cliente]:
                item_id = f"MLA{abs(hash((cliente, titulo))) % 900000 + 100000}"

                visitas_base = popularidad * 18 * estacional
                visitas = int(np.random.poisson(visitas_base))
                filas_visitas.append({
                    "fecha": dia, "cliente_ml": cliente,
                    "item_id": item_id, "titulo": titulo,
                    "categoria": categoria, "marca": marca,
                    "visitas": visitas,
                })

                # Preguntas: ~12% de las visitas generan una pregunta.
                # Si hay muchas preguntas y pocas ventas = friccion de info.
                preguntas = int(np.random.binomial(visitas, 0.12))
                filas_preguntas.append({
                    "fecha": dia, "cliente_ml": cliente,
                    "item_id": item_id, "titulo": titulo,
                    "preguntas": preguntas,
                })

                tasa_conv = 0.02 + (popularidad / 200)
                ordenes_hoy = np.random.binomial(visitas, tasa_conv)

                for _ in range(ordenes_hoy):
                    order_id += 1
                    tipo = _elegir_ponderado(TIPO_PUBLICACION)
                    unidades = random.choices([1, 2, 3], weights=[0.8, 0.15, 0.05])[0]
                    precio = round(precio_base * random.uniform(0.95, 1.05), -2)
                    ingreso = precio * unidades
                    estado = random.choices(["pagado", "cancelado"], weights=[0.96, 0.04])[0]
                    filas_ventas.append({
                        "order_id": order_id,
                        "fecha": dia,
                        "cliente_ml": cliente,
                        "item_id": item_id,
                        "titulo": titulo,
                        "categoria": categoria,
                        "marca": marca,
                        "tipo_publicacion": tipo,
                        "medio_entrega": _elegir_ponderado(MEDIOS_ENTREGA),
                        "unidades": unidades,
                        "precio_unitario": precio,
                        "ingreso": ingreso,
                        "comision_ml": round(ingreso * COMISION_PCT[tipo], 2),
                        "costo_envio": random.choice([0, 0, 0, 4500, 6000]),
                        "provincia": _elegir_ponderado(PROVINCIAS),
                        "estado": estado,
                    })

    return pd.DataFrame(filas_ventas), pd.DataFrame(filas_visitas), pd.DataFrame(filas_preguntas)


def main() -> None:
    ventas, visitas, preguntas = generar()
    DATA_DIR.mkdir(exist_ok=True)
    ventas.to_csv(DATA_DIR / "ventas.csv", index=False)
    visitas.to_csv(DATA_DIR / "visitas.csv", index=False)
    preguntas.to_csv(DATA_DIR / "preguntas.csv", index=False)
    print(f"OK -> {len(ventas):,} ventas | {len(visitas):,} visitas | {len(preguntas):,} preguntas")
    print(f"   periodo: {ventas['fecha'].min()} a {ventas['fecha'].max()}")
    print(f"   GMV total: ${ventas[ventas['estado']=='pagado']['ingreso'].sum():,.0f}")


if __name__ == "__main__":
    main()
