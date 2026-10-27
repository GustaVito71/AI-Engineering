# Bloque 5 — Reporte de estrés (LIDR_6)

Generado por `evals/stress/run.py` a partir de `results.csv`. Ver el docstring del módulo para qué mide cada barrido y las limitaciones de `--http`.

## 1. Resumen

| Métrica | Valor |
| --- | --- |
| Latencia P50 (ms) | 7027.3 |
| Latencia P95 (ms) | 13697.9 |
| Coste acumulado por escenario (USD, promedio entre repeticiones) | s/d |
| Hit rate caché exacta | 0.0% |
| Hit rate caché semántica | 0.0% |
| Hit rate de cualquier caché (fallback en --http) | 66.7% |
| Recall medio del fact-tracker (todas las filas con hechos) | 100.0% |

## 2. Curvas

### 2a. Latencia vs. tokens

_Sin datos de tokens (ningún turno en proceso con `turn_observed` completo)._

### 2b. Coste acumulado vs. turno

_Sin datos del barrido de memoria._

### 2c. Recall vs. N

| Perfil | N (turnos) | Recall al último turno |
| --- | --- | --- |
| growing | 1 | 100% |

## 3. Dónde empieza a romperse mi CAG, y por qué

**Memoria conversacional.** En los N medidos en esta corrida ([1]) el recall al último turno se mantuvo en 100%: no alcanzó a verse una fuga con este largo de conversación. La ventana deslizante es la sospechosa natural (los hechos viejos dependen de haber sido promovidos a ancla o resumen antes de salir de la ventana; ver `app/sessions/compression.py`), pero esta corrida mide el síntoma (qué se recuerda), no la causa exacta turno por turno.

**Caché y adjuntos.** Ninguna de las dos cachés acertó en esta corrida (revisar que Redis esté arriba y la clave del proveedor de embeddings esté configurada antes de sacar conclusiones). Más de fondo: memoria conversacional, adjuntos y caché son tres caminos que en este servicio NO se solapan (`estimate_conversation` nunca consulta ninguna caché, y `estimate_structured`, el único con caché, no acepta adjuntos ni sesión). Es ahí donde mi CAG empieza a romperse: cualquier escenario real que combine las tres cosas a la vez (una conversación larga, con adjuntos, que además debería cachear lo repetido) hoy no tiene un camino único que lo cubra.
