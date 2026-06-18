"""
Generador de datos sinteticos de Mercado Libre.

Por que sinteticos: para aprender data science necesitamos datos YA, sin pelear
con credenciales de la API. Estos datos imitan la forma de lo que despues va a
devolver la API real de ML (ordenes + visitas), asi el resto del codigo
(metricas, dashboard) no cambia cuando enchufemos datos reales.

Genera dos tablas (los dos "datasets" base de cualquier analisis de e-commerce):

  1. ventas.csv  -> una fila por linea de orden (que se vendio, cuanto, a quien)
  2. visitas.csv -> visitas diarias por publicacion (el "trafico" del negocio)

Con esas dos tablas se calcula casi todo: ingresos, ticket promedio, unidades,
y la metrica reina del e-commerce -> tasa de conversion = ventas / visitas.

Uso:
    python data/generar_datos.py
"""

from __future__ import annotations

import random
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from faker import Faker

# Semilla fija = resultados reproducibles. En DS esto importa: que el analisis
# de errores el mismo dataset cada vez que corres el script.
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
fake = Faker("es_AR")
Faker.seed(SEED)

DATA_DIR = Path(__file__).parent
DIAS_DE_HISTORIA = 365  # un anio para poder comparar meses y ver estacionalidad

# Varios "clientes" (cuentas de ML). El dashboard sera multi-cliente: cada uno
# entra a su link y ve SOLO sus datos. Por eso todo lleva una columna cliente_ml.
CLIENTES = ["tienda_norte", "deco_hogar", "tech_outlet"]

# Catalogo de productos por cliente. categoria + precio_base + popularidad.
# popularidad pondera cuanto se vende cada item (productos "estrella" vs cola larga).
CATALOGO = {
    "tienda_norte": [
        ("Zapatillas Running Pro", "Calzado", 85000, 5),
        ("Campera Inflable", "Indumentaria", 62000, 3),
        ("Mochila Urbana 25L", "Accesorios", 38000, 4),
        ("Remera Algodon Premium", "Indumentaria", 18000, 6),
        ("Medias Pack x3", "Indumentaria", 9000, 8),
        ("Gorra Trucker", "Accesorios", 12000, 2),
    ],
    "deco_hogar": [
        ("Lampara de Pie Nordica", "Iluminacion", 95000, 3),
        ("Juego de Sabanas Queen", "Textil", 54000, 5),
        ("Cuadro Decorativo 60x90", "Decoracion", 28000, 4),
        ("Organizador Modular", "Organizacion", 33000, 6),
        ("Vela Aromatica Set", "Decoracion", 14000, 7),
        ("Espejo Redondo 50cm", "Decoracion", 41000, 2),
    ],
    "tech_outlet": [
        ("Auriculares Bluetooth", "Audio", 72000, 6),
        ("Cargador USB-C 65W", "Accesorios", 25000, 7),
        ("Mouse Gamer RGB", "Perifericos", 38000, 5),
        ("Teclado Mecanico", "Perifericos", 89000, 3),
        ("Webcam Full HD", "Perifericos", 56000, 2),
        ("Hub USB 7 Puertos", "Accesorios", 31000, 4),
    ],
}

PROVINCIAS = [
    ("Buenos Aires", 0.38), ("CABA", 0.20), ("Cordoba", 0.12),
    ("Santa Fe", 0.10), ("Mendoza", 0.07), ("Tucuman", 0.05),
    ("Otras", 0.08),
]

TIPO_PUBLICACION = [("clasica", 0.55), ("premium", 0.45)]
# La comision de ML depende del tipo: premium cobra mas pero da cuotas sin interes.
COMISION_PCT = {"clasica": 0.13, "premium": 0.18}


def _elegir_ponderado(opciones: list[tuple]) -> str:
    """Elige una opcion segun su peso. opciones = [(valor, peso), ...]."""
    valores = [o[0] for o in opciones]
    pesos = [o[1] for o in opciones]
    return random.choices(valores, weights=pesos, k=1)[0]


def _factor_estacional(dia: date) -> float:
    """
    Multiplicador de demanda segun el dia. Imita patrones reales:
    - Fin de semana vende mas que dias de semana.
    - Hay un pico en noviembre (Black Friday / CyberMonday).
    - Leve crecimiento a lo largo del anio (el negocio crece).
    """
    factor = 1.0
    if dia.weekday() >= 5:           # sabado / domingo
        factor *= 1.35
    if dia.month == 11:              # temporada alta
        factor *= 1.8
    if dia.month in (1, 2):          # verano, baja el consumo
        factor *= 0.8
    # tendencia: cada dia transcurrido suma un poquito (negocio en crecimiento)
    dias_desde_inicio = (dia - (date.today() - timedelta(days=DIAS_DE_HISTORIA))).days
    factor *= 1 + (dias_desde_inicio / DIAS_DE_HISTORIA) * 0.4
    return factor


def generar() -> tuple[pd.DataFrame, pd.DataFrame]:
    filas_ventas: list[dict] = []
    filas_visitas: list[dict] = []
    order_id = 1000

    inicio = date.today() - timedelta(days=DIAS_DE_HISTORIA)

    for offset in range(DIAS_DE_HISTORIA):
        dia = inicio + timedelta(days=offset)
        estacional = _factor_estacional(dia)

        for cliente in CLIENTES:
            for titulo, categoria, precio_base, popularidad in CATALOGO[cliente]:
                # --- VISITAS del dia para esta publicacion ---
                # base proporcional a la popularidad, con ruido aleatorio (Poisson
                # modela bien conteos de eventos como visitas).
                visitas_base = popularidad * 18 * estacional
                visitas = int(np.random.poisson(visitas_base))
                filas_visitas.append({
                    "fecha": dia,
                    "cliente_ml": cliente,
                    "item_id": f"MLA{abs(hash((cliente, titulo))) % 900000 + 100000}",
                    "titulo": titulo,
                    "categoria": categoria,
                    "visitas": visitas,
                })

                # --- VENTAS: cada visita convierte con cierta probabilidad ---
                # tasa de conversion realista de ML: ~2% a 6% segun el item.
                tasa_conv = 0.02 + (popularidad / 200)
                ordenes_hoy = np.random.binomial(visitas, tasa_conv)

                for _ in range(ordenes_hoy):
                    order_id += 1
                    tipo = _elegir_ponderado(TIPO_PUBLICACION)
                    unidades = random.choices([1, 2, 3], weights=[0.8, 0.15, 0.05])[0]
                    # variacion de precio +/-5% (promos, descuentos)
                    precio = round(precio_base * random.uniform(0.95, 1.05), -2)
                    ingreso = precio * unidades
                    # 4% de las ordenes terminan canceladas/devueltas
                    estado = random.choices(
                        ["pagado", "cancelado"], weights=[0.96, 0.04]
                    )[0]
                    filas_ventas.append({
                        "order_id": order_id,
                        "fecha": dia,
                        "cliente_ml": cliente,
                        "item_id": f"MLA{abs(hash((cliente, titulo))) % 900000 + 100000}",
                        "titulo": titulo,
                        "categoria": categoria,
                        "tipo_publicacion": tipo,
                        "unidades": unidades,
                        "precio_unitario": precio,
                        "ingreso": ingreso,
                        "comision_ml": round(ingreso * COMISION_PCT[tipo], 2),
                        "costo_envio": random.choice([0, 0, 0, 4500, 6000]),
                        "provincia": _elegir_ponderado(PROVINCIAS),
                        "estado": estado,
                    })

    ventas = pd.DataFrame(filas_ventas)
    visitas = pd.DataFrame(filas_visitas)
    return ventas, visitas


def main() -> None:
    ventas, visitas = generar()
    DATA_DIR.mkdir(exist_ok=True)
    ventas.to_csv(DATA_DIR / "ventas.csv", index=False)
    visitas.to_csv(DATA_DIR / "visitas.csv", index=False)
    print(f"OK -> {len(ventas):,} ventas y {len(visitas):,} filas de visitas")
    print(f"   clientes: {', '.join(CLIENTES)}")
    print(f"   periodo: {ventas['fecha'].min()} a {ventas['fecha'].max()}")
    print(f"   ingreso total simulado: ${ventas['ingreso'].sum():,.0f}")


if __name__ == "__main__":
    main()
