# Ganttly

Mini aplicación de **AI Engineering services**: expone capacidades de IA a través de una **API FastAPI** con una **interfaz Streamlit** como frontend. Empaquetada con **uv** y dockerizada en dos servicios (`api` y `ui`) con **Docker Compose**.

## Objetivo educativo

- Mostrar un servicio backend con **FastAPI** (endpoints, CORS) listo para integrar capacidades de IA.
- Mostrar un frontend liviano con **Streamlit** conectado a ese backend.
- Aplicar **uv** como gestor de dependencias y empaquetado (reproducible vía `uv.lock`).
- Mostrar **multi-stage Dockerfile** y orquestación con **Docker Compose** para correr backend y frontend juntos.

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

- [ ] API con endpoints base
- [ ] UI con pestañas Chat y Dashboard
- [ ] Integración de IA pendiente (Chat)
- [ ] Contenedores `api` y `ui` vía Compose