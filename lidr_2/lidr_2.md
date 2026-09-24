# lidr_2 — Estimador CAG (Entregable AI Engineering 2026-09)

> **Rol del documento:** guía ejecutable. Fue escrito para entregarse a un agente de IA
> (rol: ingeniero de IA y desarrollador experto) y permitirle reproducir, mantener o
> continuar esta misma tarea de forma autónoma, de principio a fin. No sustituye al
> código: lo ordena y lo enlaza.
>
> **Repo:** `https://github.com/GustaVito71/AI-Engineering` (monorepo) · subcarpeta `lidr_2/`
> **Fecha de referencia:** 2026-09-24 · commit de entrega `8405174`, merge `b70d5bb`
> **Estado:** entregado. 21/21 tests verdes, ruff limpio, CI verde en GitHub Actions.

---

## 1. Objetivo buscado

Construir un **servicio web HTTP (API)** que:

1. Recibe una **transcripción de reunión** (texto libre) por `POST`.
2. La enriquece con **contexto de ejemplos de estimaciones** previas (técnica **CAG**,
   Cache-Augmented Generation: el "cache" son ejemplos inyectados en el *system prompt*,
   no recuperación de embedding como RAG).
3. Llama a un **LLM** (OpenAI o Anthropic, intercambiables) para que genere una
   **estimación de duración de proyecto en horas**, con desglose por tareas, equipo
   recomendado y duración en semanas.
4. Devuelve la estimación como JSON, **con transparencia de coste** (tokens reales +
   precio) y **sin esconder fallas** (truncado, errores de proveedor, coste desconocido).

Es el ejercicio 02 del curso AI Engineering 2026-09: un "Estimador CAG".
La validación del curso exige (ver §9): arranque sin errores, keys solo en `.env`,
`/health` 200 sin key, `/estimate` 200 con transcripción, salida inspirada en los
ejemplos del contexto, Swagger en `/docs`, y `.env` en `.gitignore`.

---

## 2. Flujo end-to-end

```
cliente ──POST /api/v1/estimate {transcription}──▶ router (validación 50..50000 chars)
                                                      │ 422 si la entrada es inválida (0 tokens gastados)
                                                      ▼
                                              service (dominio)
                                                      │
                          build_system_prompt (instrucciones + 4 ejemplos CAG +
                          transcripción envuelta en delimitar impredecible)
                                                      │
                                              provider (OpenAI | Anthropic)
                                                      │  timeout=30s, max_retries=2
                                                      ▼
                         respuesta LLM: texto + usage (input/output tokens) + finish_reason
                                                      │
                                              resultado: estimation + truncated + cost_usd
                                              (coste con snapshot LLMPrice; None si no se puede calcular)
                                                      ▼
                             200 JSON  |  502 error proveedor  |  503 sin key/config
```

Puntos no negociables de diseño:

- **La entrada es la factura:** se valida antes de tocar el LLM (límites configurables,
  por defecto 50..50.000 caracteres). Un `422` no gasta un token.
- **El truncado se declara,** no se disimula: `truncated: true|false`.
- **El coste se calcula de tokens reales** de la respuesta; si el modelo no está en el
  snapshot de precios → `cost_usd: null` + `cost_note` explicativo. Nunca `0.0` inventado.
- **Un error del proveedor no se filtra al cliente como texto crudo:** se traduce a
  error de dominio (`LLMServiceError`) y a HTTP (`502`/`503`).

---

## 3. Decisiones tomadas (y por qué)

| Decisión | Alternativa descartada | Razón |
|---|---|---|
| **CAG** (ejemplos en system prompt) | RAG (embeddings + retrieval) | El ejercicio pide CAG; para ~4 ejemplos fijos el retrieval agrega latencia, tokens e infraestructura sin ganancia. Fácil de probar de forma determinista. |
| **Cache de ejemplos como dato, no como prompt armado** (`context/examples.py`: `ESTIMATION_EXAMPLES` frozen dataclasses + `build_system_prompt()`) | Prompt monolítico hardcodeado en el router | Separa el conocimiento de dominio del ensamblado; permite `test_los_ejemplos_llegan_al_system_prompt` y validar cada ejemplo individualmente. |
| **Delimitador impredecible** alrededor de la transcripción | Separador fijo tipo `---` | Un separador fijo puede aparecer en el texto del usuario y romper el prompt (prompt injection / confusión de roles). Se genera una etiqueta aleatoria por request. |
| **Dos proveedores bajo interfaz común** (`BaseProvider.chat` → `Message`/`LLMResponse` con `usage` y `finish_reason`) | Un solo proveedor | El ejercicio exige OpenAI+Anthropic; la interfaz común deja la lógica de dominio (validación, coste, errores) en un solo lugar. |
| **Lógica de negocio en capas** (router → service → context/config/providers) | Todo en el router | Inyectable y testeable con fake; el router solo valida entrada y traduce HTTP. El service no sabe nada de HTTP. |
| **Manejo de errores de dominio** (`LLMConfigurationError`, `LLMServiceError`) | Dejar propagar excepciones del SDK | Una excepción del SDK (p. ej. `AuthenticationError`) filtrada al cliente es un leak de detalle interno. Se traduce: config → `503`, proveedor → `502`. |
| **Precios por snapshot** (`llmprice-kit`, paquete `llmprice`, clase `LLMPrice`) | Catálogo hardcodeado | Los precios cambian; un snapshot de una librería mantenida evita inventar precios. Si el modelo no existe → `None` + nota. |
| **`package = false` en `pyproject.toml`** | Layout `src/` / empaquetado | Es una aplicación, no una librería; `uv sync` no debe intentar construir un paquete inexistente. |
| **`APP_PORT=8001` por defecto** | 8000 | El 8000 suele estar tomado por Docker (en este entorno, por el proyecto Ganttly). |
| **`app/__main__.py`** para `uv run python -m app` | Solo `uvicorn app.main:app` | Ambos funcionan; `-m app` lee `APP_PORT` de configuración y es más idiomático para scripts. |
| **`/health` independiente de la key** | Health que falla sin key | El orquestador (y el curso) necesitan saber que el servicio está vivo aunque el LLM no esté configurado: responde 200 con `llm_configured: false`. |
| **Una sola carpeta `test/`** (no `tests/`) | — | Requisito explícito del ejercicio. |
| **Validación de límites con `@model_validator`** en el modelo de request | `min_length`/`max_length` de Pydantic | Mismo resultado (422), pero los límites salen de `Settings` (configurables por `.env`) en lugar de quedar fijos en el schema. |
| **Transcripción canónica como archivo con marcadores** (`datos/transcripcion_reunion.md`, entre `<!-- transcripcion -->` y `<!-- /transcripcion -->`) | Texto pegado en docs/comandos | Es el parámetro del ejercicio; extraerlo por marcadores permite al pipeline y a los tests consumir exactamente el mismo dato. |
| **CI en la raíz del monorepo** (`.github/workflows/ci.yml`) | `.github/workflows` dentro de `lidr_2/` | **GitHub Actions solo descubre workflows en el `.github` de la raíz del repo.** El CI anidado nunca se ejecutó (verificado empíricamente: la API de workflows estaba vacía y solo existía el check-suite de Sentry). |

Decisiones de robustez aprendidas de la revisión del instructor (`revision_ejercicio_02.md`):

- `timeout=30.0` y `max_retries=2` en los clientes OpenAI/Anthropic (el default del SDK,
  600 s con reintentos, puede retener un worker una hora).
- Techo de entrada (`ESTIMATION_MAX_CHARS`) para no facturar entradas gigantes.
- Lectura de `usage` y `finish_reason` reales de la respuesta.
- `def create_estimation` **síncrono** (no `async`): el SDK es bloqueante; un handler
  `async` que bloquea es peor que uno síncrono. Queda un comentario en el código
  explicándolo para que nadie lo "arregle" a futuro.
- Comentarios verificados/útiles, no decorativos.

---

## 4. Estructura del proyecto (mapa)

```
lidr_2/
├── pyproject.toml                 # deps + [tool.uv] package=false + pytest config
├── uv.lock                        # lockfile (CI usa uv sync --locked: el CI falla si está desincronizado)
├── README.md                      # alcance honesto, uso, sección "Validación automática", "Alcance (y deuda declarada)"
├── .env.example                   # plantilla (NUNCA valores reales); .env real en .gitignore
├── .gitignore                     # línea 2: .env
├── app/
│   ├── __init__.py
│   ├── __main__.py                # uv run python -m app  → uvicorn con APP_PORT
│   ├── main.py                    # create_app(): /, /health, router; health sin key
│   ├── config.py                  # Settings (pydantic-settings): proveedor, modelos, timeouts, límites, APP_PORT; is_configured; normalización de vacíos
│   ├── context/
│   │   ├── __init__.py
│   │   └── examples.py            # EstimationExample (frozen); ESTIMATION_EXAMPLES (4); build_system_prompt()
│   ├── services/
│   │   ├── __init__.py
│   │   ├── errors.py              # LLMConfigurationError, LLMServiceError
│   │   ├── llm_service.py         # generate_estimation(transcription, provider, cfg) -> EstimationResult; mapea errores de dominio
│   │   └── pricing.py             # estimate_cost(model, usage) -> (cost_usd | None, cost_note) usando LLMPrice
│   ├── providers/
│   │   ├── __init__.py
│   │   ├── base.py                # Message, LLMResponse, BaseProvider(ABC).chat()
│   │   ├── errors.py
│   │   ├── factory.py             # create_provider(provider_name) -> BaseProvider
│   │   ├── openai_provider.py     # SDK OpenAI, timeout/max_retries, usage + finish_reason
│   │   └── anthropic_provider.py  # SDK Anthropic (sin `temperature` en SDK 1.8+, ver tests)
│   └── routers/
│       ├── __init__.py
│       └── estimations.py         # EstimationRequest (validación 50..50000); EstimationResponse; POST /estimate
├── test/
│   ├── conftest.py                # fixtures: fake provider, client, ejemplos
│   ├── test_estructura.py         # estructura de carpetas; secrets; transcripción canónica; pipeline en raíz
│   └── (resto de la suite: 21 tests, ver §8)
├── datos/
│   └── transcripcion_reunion.md   # transcripción canónica entre marcadores <!-- transcripcion -->
│                                  # y <!-- /transcripcion --> (~2651 chars)
└── ../.github/workflows/ci.yml    # pipeline: uv sync --locked → pytest -q → ruff check .
                                   # filtra paths lidr_2/** y working-directory: lidr_2
```

Capas: **router → service → context/config/providers**. Regla de oro: *ningún router
conoce el SDK, ningún servicio conoce HTTP.*

---

## 5. Contratos técnicos

### Configuración (`.env`, ver `.env.example`)

| Variable | Default | Notas |
|---|---|---|
| `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` | vacío | al menos una; si ninguna → `/estimate` → `503` |
| `LLM_PROVIDER` | `openai` | `openai` \| `anthropic` |
| `LLM_MODEL` | vacío | vacío = se resuelve el default del proveedor |
| `OPENAI_MODEL` | `gpt-4o-mini` | |
| `ANTHROPIC_MODEL` | `claude-haiku-4-5` | |
| `LLM_TIMEOUT` | `30.0` | segundos, en ambos SDK |
| `LLM_MAX_RETRIES` | `2` | |
| `LLM_MAX_TOKENS` | `2000` | techo de salida |
| `ESTIMATION_MIN_CHARS` | `50` | 422 si la entrada es menor |
| `ESTIMATION_MAX_CHARS` | `50000` | 422 si la entrada es mayor (coste) |
| `APP_PORT` | `8001` | default para no chocar con Docker (8000) |

### Endpoints

`GET /health` → 200 siempre (independiente de la key):
```json
{"service": "estimador-cag", "status": "ok", "env": "…", "llm_configured": true|false,
 "provider": "openai", "model": "gpt-4o-mini"}
```

`POST /api/v1/estimate` — request:
```json
{"transcription": "texto de 50 a 50000 caracteres"}
```
Response 200 (`EstimationResponse`):
```json
{"estimation": "## Alcance\n…", "truncated": false, "model": "gpt-4o-mini-2024-07-18",
 "provider": "openai",
 "usage": {"input_tokens": 2007, "output_tokens": 257},
 "cost_usd": 0.000455, "cost_note": "LLMPrice (snapshot 2026.4.3 (de abril))"}
```

Errores:

| HTTP | Caso |
|---|---|
| 422 | entrada fuera de límites (no se llama al LLM) |
| 503 | LLM no configurado (`LLMConfigurationError`) |
| 502 | fallo del proveedor (`LLMServiceError`, sin filtrar el texto del SDK) |

`GET /docs` → Swagger (OpenAPI generado por FastAPI).

### Formato de salida esperado (CAG)

La estimación usa `##` para secciones `Alcance`, `Desglose`, `Total`, `Supuestos`,
`Riesgos`, y `**Equipo recomendado**` / `**Duración estimada**` — el mismo patrón que
los 4 ejemplos del context. Verificación: `test_los_ejemplos_son_aritmeticamente_coherentes`
y `test_cada_ejemplo_incluye_equipo_y_duracion_estimada`.

---

## 6. Acciones realizadas (estado a 2026-09-24)

1. Proyecto FastAPI creado con `uv` (`pyproject.toml`, `uv.lock`, `package=false`).
2. `app/config.py` con defaults coherentes por proveedor y `/health` que sobrevive sin key.
3. `app/context/examples.py`: cache CAG (4 ejemplos) + `build_system_prompt()` + delimitador.
4. `app/providers/`: `openai_provider.py` y `anthropic_provider.py` con timeout/max_retries,
   lectura de `usage`/`finish_reason`, factory e interfaz común.
5. `app/services/llm_service.py`: traducción de errores de dominio, `EstimationResult`.
6. `app/services/pricing.py`: `estimate_cost()` con snapshot LLMPrice y política de `None`.
7. `app/routers/estimations.py` + `app/main.py`: endpoints, validación de límites.
8. Suite de tests: **21 tests**, 0 fallos, ruff limpio.
9. `datos/transcripcion_reunion.md` (canónica, con marcadores).
10. README con alcance honesto y sección de validación automática.
11. CI llevado a la **raíz del monorepo** (ver §3) y **verde en GitHub** (3 runs: push main,
    push branch, pull_request — todos success, 21 passed).
12. Verificación en vivo: server en `:8001`, `/health` 200, `/docs` 200, `/estimate` 200 con
    la transcripción canónica (gpt-4o-mini, 80 horas, coste ~$0.00045).

---

## 7. Guía de reproducción autónoma (runbook para el agente)

Reglas de operación del entorno:

- **Git lo maneja el humano.** No commitees, no pushees, no crees ramas salvo que se te pida
  explícitamente. Dejá el repo listo para que el humano decida.
- **Prohibido escribir secrets en código.** Las keys viven en `.env` (gitignored) y en `.env.example`
  como plantilla vacía.
- El puerto 8000 está reservado (Docker/Ganttly). Usá 8001.
- `rg` no está instalado en este entorno; usá grep o la herramienta de búsqueda del agente.
- No combines `kill`/`pkill` con comandos siguientes en la misma invocación de bash: el wrapper
  aborta ("Unknown: ChildProcess.kill"). Hacé el kill en su propio paso.

### Paso 0 — Contexto

Leé este documento, `README.md`, el código de `app/` y `test/`. Mirá
`/home/gustavo/Descargas/revision_ejercicio_02.md` (revisión del instructor): es la lista de
errores y aciertos que este entregable ya incorpora — no los reintroduzcas.

### Paso 1 — Bootstrap

```bash
cd <monorepo>/lidr_2
uv sync --all-groups          # instala deps + grupo dev
uv run pytest -q              # debe pasar ya (21 tests)
```
Si el entorno es nuevo: crear proyecto con `uv init`, declarar deps del §4/§5
(fastapi, uvicorn[standard], pydantic, pydantic-settings, python-dotenv, structlog,
openai, anthropic, llmprice-kit; dev: pytest, pytest-asyncio, httpx, ruff) y `package=false`.

### Paso 2 — Configuración

Crear `app/config.py` (pydantic-settings) con todos los campos de la tabla §5:
- `is_configured` → True si hay al menos una key.
- Campos opcionales vacíos se normalizan al default (ver `test_campos_opcionales_vacios_se_normalizan_al_default`).
- `app_port: int = 8001`.
Crear `.env.example` copiando la tabla §5 y `.gitignore` con `.env`.

### Paso 3 — Contexto CAG

Crear `app/context/examples.py`:
- `EstimationExample` (frozen): transcripción resumida, alcance, desglose, total_horas,
  equipo recomendado, duracion_semanas.
- `ESTIMATION_EXAMPLES`: 4 ejemplos realistas (rango 56–88 h).
- `build_system_prompt(examples=None)`: instrucciones + ejemplos serializados + regla de
  envolver la transcripción en un delimitador **impredecible** (etiqueta aleatoria por request).

### Paso 4 — Proveedores

- `BaseProvider.chat(messages, ...) -> LLMResponse` con `Message` (role/content),
  `usage` (input/output tokens) y `finish_reason` como datos tipados.
- `openai_provider.py`: `OpenAI(api_key=…, timeout=30.0, max_retries=2)`, enviar
  `max_tokens`, leer `usage` + `finish_reason` y traducir a `LLMResponse`.
- `anthropic_provider.py`: mismo contrato. **Atención SDK:** desde 1.8 quitó el parámetro
  `temperature` (ver test de regresión). Nunca asumas que el SDK acepta kwargs que usabas.
- `factory.py`: `create_provider(name)` resuelve por nombre.

### Paso 5 — Servicios

- `errors.py`: `LLMConfigurationError` (sin key/modelo) y `LLMServiceError` (fallo del proveedor).
- `pricing.py`: `estimate_cost(model, usage) -> tuple[float | None, str | None]` con
  `LLMPrice()` snapshot; modelo desconocido → `(None, "LLMPrice (snapshot …) — precio no disponible")`.
- `llm_service.py`: `generate_estimation(transcription, provider, cfg) -> EstimationResult`
  (estimation, truncated, model, provider, usage, cost_usd, cost_note). Captura excepciones del
  SDK y las traduce a dominio **antes** de llegar al router.

### Paso 6 — API

- `routers/estimations.py`: `EstimationRequest` (transcription + `@model_validator` con límites
  de Settings); `POST /estimate` **síncrono** (comentario explicando por qué); traduce errores de
  dominio a 502/503 (ver §5); `EstimationResponse` con todos los campos.
- `main.py`: `create_app()` con `/`, `/health` (200 sin key, `llm_configured`), router.
- `__main__.py`: `uvicorn.run("app.main:app", port=settings.app_port)`.

### Paso 7 — Pruebas (TDD hacia atrás: escribí tests contra los contratos §5)

Los 21 tests cubren (nombres reales de la suite):
- CAG: `test_los_ejemplos_llegan_al_system_prompt`, `test_cada_ejemplo_incluye_equipo_y_duracion_estimada`,
  `test_los_ejemplos_son_aritmeticamente_coherentes`, `test_los_roles_van_en_orden_y_la_transcripcion_es_dato`,
  `test_la_transcripcion_va_envuelta_en_delimitador_impredecible`.
- Robusteza: `test_entrada_corta_no_llama_al_llm`, `test_entrada_demasiado_larga_no_llama_al_llm`,
  `test_truncado_se_expone_como_flag_no_como_200_completo`, `test_error_del_proveedor_no_se_filtra_al_cliente`.
- Config: `test_health_responde_sin_key`, `test_el_modelo_se_resuelve_desde_el_default_del_proveedor`,
  `test_campos_opcionales_vacios_se_normalizan_al_default`, `test_el_puerto_default_es_8001_para_no_chocar_con_docker`,
  `test_estimate_sin_key_devuelve_503`, `test_estimate_completo_con_fake`.
- Coste: `test_coste_se_calcula_de_tokens_reales`, `test_coste_es_none_si_no_se_puede_calcular_con_datos_reales`.
- Estructura (`test_estructura.py`): `test_la_estructura_de_carpetas_es_la_del_ejercicio`,
  `test_los_secretos_no_viven_en_el_repo`, `test_la_transcripcion_canonica_sirve_al_endpoint`,
  `test_el_pipeline_vive_en_la_raiz_del_monorepo`.

Proof-of-concept en vivo (opcional, cuesta fracciones de centavo): un par de llamadas reales a
OpenAI/Anthropic para confirmar ambos adaptadores.

### Paso 8 — Transcripción canónica

Crear `datos/transcripcion_reunion.md` con el texto de la reunión entre
`<!-- transcripcion -->` y `<!-- /transcripcion -->`. Regla crítica: **esos marcadores no deben
aparecer en ninguna otra parte del archivo** (ni siquiera dentro de backticks), porque la
extracción usa `split()` por marcador.

### Paso 9 — README

`README.md` con: qué es, alcance (y deuda declarada), cómo correr, ejemplo de uso con la
transcripción canónica, validación automática (estructura + pipeline), y el "por qué" del CI en la raíz.

### Paso 10 — CI

Crear `.github/workflows/ci.yml` **en la raíz del monorepo** (no dentro de `lidr_2/`):
- `on.push` y `on.pull_request` con `paths: ["lidr_2/**", ".github/workflows/ci.yml"]`.
- `defaults.run.working-directory: lidr_2`.
- Steps: checkout → setup-uv → `uv sync --locked --all-groups` (falla si el lock está
  desincronizado) → `pytest -q` → `ruff check .`.

### Paso 11 — Verificación final (siempre ejecutar antes de entregar)

```bash
uv run pytest -q                 # 21 passed
uv run ruff check .              # 0 errores
uv run python -m app             # o uvicorn app.main:app --reload --port 8001
curl -s http://127.0.0.1:8001/health
curl -s -X POST http://127.0.0.1:8001/api/v1/estimate \
  -H 'Content-Type: application/json' \
  -d "$(python3 -c "
import json
md = open('datos/transcripcion_reunion.md').read()
t = md.split('<!-- transcripcion -->')[1].split('<!-- /transcripcion -->')[0].strip()
print(json.dumps({'transcription': t}, ensure_ascii=False))
")" | jq '{model, truncated, cost_usd, cost_note, usage}'
```
Y en GitHub (después del push del humano): `gh run list` → verde.

---

## 8. Criterios de aceptación (checklist que debe quedar ✅)

1. `uv run uvicorn app.main:app --reload` arranca sin errores.
2. Las API keys se cargan desde `.env`; nunca en el código.
3. `GET /health` responde 200 (con o sin key).
4. `POST /api/v1/estimate` recibe una transcripción y devuelve una estimación.
5. La estimación está inspirada en los ejemplos del contexto inyectado (mismo formato de secciones).
6. Swagger accesible en `/docs`.
7. `.env` está en `.gitignore`.
8. Solicitudes fuera del rango configurable → 422 sin llamar al LLM (0 tokens).
9. `truncated` y el coste real se exponen honestamente (`cost_usd: null` + `cost_note` si no
   se puede calcular; nunca 0.0 inventado).
10. Los errores del proveedor llegan al cliente como 502/503 tipados, sin filtrar el texto del SDK.
11. Suite completa verde + ruff limpio.
12. CI corre en GitHub (workflow en la raíz del monorepo).

---

## 9. Pendientes y mejoras opcionales

- **Test de secreto más estricto** (contra `git`): el check actual no usa
  `git ls-files --error-unmatch .env`; se puede reforzar para fallar si `.env` estuviera
  trackeado, no solo presente en disco.
- **Reintento/cola para AFIP**: es tema de la transcripción canónica, no del servicio; si el
  alcance crece, ver los pendientes anotados en `datos/transcripcion_reunion.md` (notas de
  crédito compartidas, políticas de reintento).
- **Alertas/observabilidad**: el proyecto usa `structlog`; no hay Sentry. `ganttly` lo usa
  como referencia de configuración.

---

## 10. Referencias en el repo

- `README.md` — uso y alcance honesto (leelo antes de tocar nada).
- `app/` — código (contratos en §5).
- `test/` — 21 tests (nombres en §7 paso 7).
- `datos/transcripcion_reunion.md` — parámetro del ejercicio.
- `../.github/workflows/ci.yml` — pipeline (2 niveles arriba).
- `/home/gustavo/Descargas/revision_ejercicio_02.md` — revisión del instructor (externo al repo).