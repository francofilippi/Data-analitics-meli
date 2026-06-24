# Respaldo durable del histórico (backup del YoY)

Archivos parquet por mes, **versionados en git**, como respaldo independiente de
Supabase. La API de Mercado Libre solo devuelve órdenes de los últimos ~12 meses,
así que el YoY depende de tener "fotos" de los meses viejos guardadas en algún
lado que no se borre. Supabase cumple ese rol, pero un proyecto free-tier se
pausa por inactividad y puede perder datos; estos archivos sobreviven a eso.

## Layout

```
data/historico/{cliente}/{YYYY-MM}__{kind}.parquet
```

- `kind = v | vis | preg` → datos completos (envíos / visitas / preguntas)
- `kind = lv` → ventas livianas (solo órdenes; es lo que usa el MoM/YoY)

El dashboard los lee con **prioridad sobre Supabase** (ver `src/store_local.py`).

## Cómo poblarlos / actualizarlos

```bash
# desde la raíz del repo, con el DSN de Supabase en el entorno
DATABASE_URL="postgresql://...supabase.../postgres" python scripts/snapshot_historico.py
git add data/historico && git commit -m "snapshot histórico" && git push
```

O dejá que el GitHub Action `snapshot-historico.yml` lo haga solo cada mes
(necesita el secret `DATABASE_URL` en el repo).
