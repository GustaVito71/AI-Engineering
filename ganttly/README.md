# Ganttly

Mini aplicación de **AI Engineering services**: expone capacidades de IA a través de una **API FastAPI** con una **interfaz Streamlit** como frontend. Empaquetada con **uv** y dockerizada en dos servicios (`api` y `ui`) con **Docker Compose**.

## Objetivo de corto plazo

- Mostrar un servicio backend con **FastAPI** (endpoints, CORS) listo para integrar capacidades de IA.
- Mostrar un frontend liviano con **Streamlit** conectado a ese backend.
- Aplicar **uv** como gestor de dependencias y empaquetado (reproducible vía `uv.lock`).
- Mostrar **multi-stage Dockerfile** y orquestación con **Docker Compose** para correr backend y frontend juntos.

## Objetivo de largo plazo

Producto SaaS de inteligencia artificial que, conectado a un proveedor de gestión de proyectos, genera un **diagrama de Gantt** a partir de una épica y sus tareas y subtareas.

- **Conector a Jira:** dado `épica → tareas → subtareas`, construir un cronograma con fechas, dependencias y camino crítico.
- **Agnóstico del proveedor de PM:** modelo canónico propio (definición unificada de épica/tarea/subtarea) + capa de adaptadores (patrón puerto/adaptador). Futuro: Asana, Linear, Azure DevOps, etc.
- **Agnóstico del LLM:** AI Gateway con puerto propio (`complete`, `complete_json`, `embed`). Sustituible entre OpenAI, Anthropic o modelos locales sin tocar el núcleo.
- **Lógica híbrida:** la IA solo resuelve lo ambiguo (estimación de duración, dependencias implícitas, riesgos); fechas y camino crítico los calcula un motor determinista (**CPM/PERT**).
- **Salida estructurada:** la IA retorna JSON validado por schema; si falla, fallback a heurísticas.
- **SaaS multi-tenant:** PostgreSQL con Row-Level Security + cola de trabajos asíncronos (Redis + workers) para sync, pipeline de IA y exportación a PNG/PDF.
- **CI/CD y testing:** GitHub Actions (CI, evals de LLM, CD) con pirámide de tests: unit del núcleo, adaptadores con mocks, evals con golden-set y E2E con Playwright.

## Estructura

```
ganttly/
├── main.py                 # Ejemplo mínimo FastAPI (raíz e items)
├── app/main.py             # API Ganttly (título, versión, CORS, health check)
├── frontend/app.py         # UI Streamlit (solapa Chat + solapa Dashboard)
├── pyproject.toml           # Metadatos y dependencias (uv)
├── uv.lock                  # Lockfile reproducible
├── Dockerfile               # Multi-stage: builder, api y ui
├── compose.yaml             # Servicios api y ui con puertos expuestos
├── .python-version          # Versión de Python del proyecto
├── .gitignore
└── .dockerignore
```

## Requisitos

- Docker (BuildKit) o en su lugar Python 3.11+ con `uv`.

## Ejecución con Docker Compose

```bash
cd ganttly
docker compose up --build
```

| Servicio | URL | Puerto |
|----------|-----|--------|
| API (FastAPI) | http://localhost:8000 | 8000 |
| UI (Streamlit) | http://localhost:8501 | 8501 |
| Documentación API | http://localhost:8000/docs | 8000 |

## Ejecución local con uv

```bash
cd ganttly
uv sync
uv run api      # o: uvicorn app.main:app --reload
uv run ui       # o: streamlit run frontend/app.py
```

## Endpoints de la API

| Método | Ruta | Descripción |
|--------|------|-------------|
| `GET` | `/` | Estado del servicio (`{"service": "ganttly", "status": "ok"}`) |
| `GET` | `/items/{item_id}` | Ejemplo con path y query param (en `main.py` raíz) |

## Estado

- [x] API con endpoints base
- [x] UI con pestañas Chat y Dashboard
- [x] Contenedores `api` y `ui` vía Compose
- [ ] Chat con IA
- [ ] Conector Jira
- [ ] Motor CPM/Gantt
