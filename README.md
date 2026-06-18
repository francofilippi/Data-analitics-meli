# Dashboard de Analitica — Mercado Libre

Proyecto para **aprender data science / analisis de datos construyendo dashboards reales**: cada cliente de Mercado Libre podra entrar a un link y ver las metricas de su cuenta (ingresos, ordenes, conversion, productos top, evolucion mensual).

Stack: **Python + pandas + Plotly + Streamlit**. Empezamos con datos sinteticos realistas y mas adelante conectamos la API real de Mercado Libre.

---

## Estructura

```
Data-analitics-meli/
├── data/
│   └── generar_datos.py     # genera datos sinteticos (ventas.csv + visitas.csv)
├── src/
│   └── metricas.py          # el "cerebro": calculos con pandas (KPIs, series, rankings)
├── app/
│   └── dashboard.py         # el dashboard interactivo (Streamlit)
├── notebooks/
│   └── 01_exploracion.ipynb # notebook para aprender pandas explorando los datos
└── requirements.txt
```

Regla de diseno: **el calculo vive en `src/metricas.py`, la vista en `app/dashboard.py`**. Asi cada parte se entiende y se testea por separado.

---

## Como correrlo

```bash
# 1. Crear entorno virtual e instalar dependencias
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Generar los datos de ejemplo
python data/generar_datos.py

# 3. Levantar el dashboard
streamlit run app/dashboard.py     # abre http://localhost:8501

# (opcional) Explorar los datos en notebook
jupyter notebook notebooks/01_exploracion.ipynb
```

---

## Datos reales + persistencia (Streamlit Cloud)

Las credenciales ML de cada seller y los tokens de acceso van en **Secrets** de
Streamlit (ver `.streamlit/secrets.toml.example`), nunca en el repo.

El histórico se guarda en **Supabase (Postgres)** para que sea durable: cada mes
cerrado se baja de ML una sola vez y se guarda; cuando un seller entra, lee los
meses guardados al instante y solo consulta el mes en curso. Sin Supabase el
caché es efímero (`/tmp`) y se pierde al redeployar o dormir la app.

Setup: creá un proyecto en supabase.com y pegá el connection string del
*Connection Pooler* en Secrets como `[supabase].dsn`. La tabla `ml_cache` se
crea sola. Desde el dashboard (admin) → **⚙️ Mantenimiento → Pre-cargar 1 año**
para llenar el histórico de todos los sellers de una.

---

## Conceptos de data science que vas a aprender

| Tema | Donde |
|---|---|
| Cargar y limpiar datos | `src/metricas.py` → `cargar_datos` |
| Filtrado booleano | `filtrar`, notebook §2 |
| Agregaciones (`groupby`) | `top_productos`, `por_categoria`, notebook §3 |
| Series temporales (`resample`) | `serie_diaria`, `serie_mensual`, notebook §4 |
| KPIs de e-commerce | `kpis_periodo` (conversion, ticket promedio, etc.) |
| Comparacion entre periodos | `variacion_pct` (los deltas de las tarjetas) |
| Visualizacion interactiva | `app/dashboard.py` (Plotly + Streamlit) |

---

## Roadmap

- [x] **Fase 1 — Fundamentos:** datos sinteticos + metricas con pandas + dashboard base multi-cliente.
- [ ] **Fase 2 — Datos reales:** conectar la API de Mercado Libre (OAuth) y reemplazar los CSV.
- [ ] **Fase 3 — Analisis avanzado:** tendencias, cohortes, deteccion de productos en caida, pronostico de ventas.
- [ ] **Fase 4 — Deploy:** subir el dashboard con un link por cliente y autenticacion.

---

## Metricas del dashboard (que significan)

- **Ingresos:** suma de ventas pagadas en el periodo.
- **Ordenes:** cantidad de ventas pagadas.
- **Ticket promedio:** ingreso / ordenes — cuanto deja cada venta en promedio.
- **Conversion:** ordenes / visitas × 100 — de cada 100 que miran, cuantos compran. La metrica reina del e-commerce.
- **Tasa de cancelacion:** % de ordenes canceladas/devueltas — salud de la operacion.

Los deltas (`+12.3%`) comparan cada KPI contra el periodo anterior de igual duracion.
