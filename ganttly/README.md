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

## Arquitectura

**Hexagonal (puertos y adaptadores) sobre un monolito modular con bordes asíncronos y multi-tenencia aislada.** Todo lo externo (proveedores de PM, LLM, Webhooks) entra por puertos; todo lo lento fluye por una cola de trabajos; el dominio solo conoce modelos canónicos y schemas validados.

### Diagrama

```
┌────────────────────────────────────────────────────────────┐
│  UI (Streamlit / Frontend)  ← agnóstico de proveedores y LLM│
└───────────────────────────┬────────────────────────────────┘
                            │ REST / WebSocket
┌───────────────────────────▼────────────────────────────────┐
│  API FastAPI — Monolito modular                             │
│  core · providers · ai · scheduling · gantt                 │
│  auth multi-tenant · orquesta workers                       │
└───────┬──────────────────────────────┬─────────────────────┘
        │ encola jobs                  │ recibe eventos
┌───────▼──────────┐          ┌────────▼──────────────────────┐
│ Queue (Redis)    │          │ Webhooks Jira →               │
│ Workers:         │          │ ChangeFeed canónico           │
│  sync · IA · CPM │          │ (item.created, updated, ...)  │
│  render Gantt    │          └───────────────────────────────┘
└───────┬──────────┘
        │
┌───────▼─────────────────────────────────────────────────────┐
│ Postgres (RLS multi-tenant) + pgvector (embeddings)          │
│ S3 (export PNG/PDF) · KMS (tokens OAuth cifrados)            │
└──────────────────────────────────────────────────────────────┘

        Puertos (contratos, agnósticos del vendor):
   ProjectProvider  ←  Jira │ Asana │ Linear │ Azure DevOps
   LLMProvider      ←  OpenAI │ Anthropic │ local (LiteLLM)
```

### Capas y patrones

| Capa | Patrón | Responsabilidad |
|------|--------|-----------------|
| Núcleo (`core/`) | Modelo canónico + DDD liviano | `WorkItem`, `Epic`, `Dependency`, `Timeline`; sin conocimientos de vendors |
| Proveedores PM | Puerto/adaptador (`ProjectProvider`) | `JiraAdapter`, futuro `AsanaAdapter`, `LinearAdapter`; mapping a modelo canónico |
| IA (`ai/`) | Gateway (`LLMProvider`) + salida estructurada | Estimación de duración, dependencias implícitas, riesgos; JSON validado por schema (Pydantic), fallback heurístico |
| Scheduling | CPM/PERT (determinista) | Fechas, dependencias y camino crítico a partir del output de IA |
| Gantt | Render agnóstico | Frappe-Gantt (UI) / Playwright → PNG/PDF |
| Datos | Multi-tenant RLS | Un Postgres, `tenant_id` + Row-Level Security |
| Sincronización | Event-driven + ChangeFeed | Webhooks normalizados a eventos canónicos; cache de resultados por versión |
| CI/CD | GitHub Actions | `ci.yml`, `eval-llm.yml`, `cd.yml`; deploy a PaaS (fase 1) |

### CAG (Cache-Augmented Generation)

No se usa RAG: el insumo por épica es **estructurado y acotado** (ya proviene del sync), cabe completo en contexto. En su lugar, dos caches:

1. **Cache de contexto (adaptador LLM):** el prefijo estable (system prompt + épica canónica) se cachea vía prompt/KV caching del vendor. Las llamadas siguientes (estimación, dependencias, riesgos) solo pagan los tokens del query. Vive dentro del adaptador (Anthropic `cache_control`, OpenAI automático) — el núcleo solo ve `cache_strategy`.
2. **Cache de resultados (pipeline):** cada snapshot de sync tiene un `version_hash`. Si la épica no cambió desde la última generación, se devuelve el Gantt guardado **sin gastar tokens**. Es la mayor ganancia de costo.

### Integración (flujo end-to-end)

1. Usuario conecta su cuenta de Jira (OAuth 2.0) → tokens cifrados en KMS.
2. `GET /epics/{id}/gantt` → se consulta el cache por `version_hash` → si existe, devolver.
3. Si no: worker hace sync de la jerarquía (`épica → tareas → subtareas`) a modelo canónico.
4. Pipeline IA: estimaciones, dependencias implícitas y riesgos (LLM con cache de contexto).
5. Motor CPM/PERT: fechas y camino crítico (determinista, sin LLM).
6. El Gantt se guarda versionado y se devuelve al UI; exportación opcional a PNG/PDF (S3).
7. Un webhook de Jira emite `ChangeFeed` → invalida el cache y re-genera si aplica.

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
