# lidr_6

Entrega de la práctica LIDR_6: reporte de estrés, construida sobre el servicio de `lidr_5`. El servicio en sí, para levantarlo y correrlo, vive en `lidr_5`: `docker-compose.yml`, `.env.example`, `pyproject.toml` y `app/` no se duplican aquí.

## Qué contiene

| Carpeta | Contenido |
| --- | --- |
| `evals/` | Métricas y runner heredados de bloques anteriores (`metrics.py`, `run.py`, `golden_dataset.json`) |
| `evals/stress/` | El Bloque 5 propiamente: `scenarios.py` (escenarios de estrés), `attachments.py` (fixtures de adjuntos), `metrics.py`/`run.py` (barrido y métricas), `REPORT.md`/`REPORT.pdf` (reporte final), `results.csv` (datos crudos de la corrida), y los CSV/reportes auxiliares de verificación (`cache_check.*`, `http_check.*`, `resultados_adjuntos.csv`) |
| `test/` | Tests unitarios del Bloque 5 y de los bloques de evals previos |

## Entrega

Esta carpeta (rama [`entrega/lidr_6`](https://github.com/GustaVito71/AI-Engineering/tree/entrega/lidr_6)) es la entrega oficial de la práctica LIDR_6. El servicio sobre el que corre vive en el repo [`lidr_5`](https://github.com/GustaVito71/AI-Engineering/tree/entrega/lidr_5) (rama separada del mismo repo): ahí están `docker-compose.yml`, `.env.example`, `pyproject.toml` y el resto de `app/` necesarios para levantar el servicio y correr `evals/stress/run.py`.

Para levantar el servicio y correr el barrido de estrés, ver el README de `lidr_5`.
