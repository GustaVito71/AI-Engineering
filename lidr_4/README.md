# Estimator — Servicio IA de estimación de software

Servicio IA en FastAPI que estima proyectos de software a partir de un formulario tipado. Es la pieza Python del programa **Master en AI Engineering**: un endpoint pensado para ser consumido por un backend de negocio (Rails, Streamlit u otro), no por un usuario final.

A partir de la **Sesión 04** el contrato es deliberadamente estrecho:
- entrada tipada (`description` + tres enums),
- salida en texto libre,
- prompt fuera del código, versionado en `app/prompts/<use_case>/<version>/`: plantillas Jinja2 más un `examples.yaml` con los datos de esa versión.

La inteligencia adicional (output estructurado, guardrails, cache semántico) se construye encima de esta base en directo.

## Estado

| Pieza | Estado |
|---|---|
| Contrato de entrada (`EstimationRequest` / `EstimationResponse`) | Hecho (WU2) |
| Prompts versionados (`system.j2`, `user.j2`, `examples.yaml`, `loader.py`) | Hecho (WU4) |
| Wrapper LLM (`app/services/llm_wrapper.py`), caché, router `POST /api/v1/estimate`, errores y trazabilidad | Hecho (WU5) |
| Cliente Streamlit | Hecho (WU6) |
| Salida estructurada, guardrails, caché semántico | Pendiente (WU7–WU10) |

El detalle de cada unidad está en [`PLAN.md`](PLAN.md) §6.

**Convenciones:** los mensajes al usuario (errores HTTP, textos de la UI y de Swagger), los comentarios y los docstrings van en español; los identificadores (variables, funciones, métodos, clases) van en inglés.

## Cómo levantar

```bash
cd lidr_4
cp .env.example .env  # completá las API keys del modelo primario y del de respaldo
uv sync
uv run python -m app            # puerto APP_PORT (default 8001)
uv run python -m app --reload   # con recarga
```

`python -m app` usa `APP_PORT` de Settings. `uvicorn app.main:app` a secas usa el 8000, que suele estar tomado. El servicio queda en `http://localhost:8001` (Swagger en `/docs`, health en `/health`).

El servicio arranca aunque falten las API keys. Con la configuración por defecto:

- **Sin `OPENAI_API_KEY`** (primario) no hay estimaciones: `/health` informa `llm_configured: false` y `POST /api/v1/estimate` responde 503 nombrando la variable.
- **Sin `ANTHROPIC_API_KEY`** (respaldo) las estimaciones funcionan solo con el primario. Se avisa en tres lugares: el log (evento `respaldo_no_disponible`), `/health` (`fallback_configured: false` y `avisos`) y cada respuesta del endpoint (`avisos`), que Streamlit muestra encima de la estimación.

Redis es opcional: con `REDIS_URL` vacío la caché queda desactivada, y si Redis no responde el servicio sigue sin caché (fail soft).

### Probar el endpoint

```bash
curl -X POST http://localhost:8001/api/v1/estimate \
  -H "Content-Type: application/json" \
  -d '{
    "description": "A small B2B SaaS to manage employee equipment loans across teams. Role-based access, audit trail, weekly digest.",
    "project_type": "web_saas",
    "detail_level": "medium",
    "output_format": "phases_table"
  }'
```

Respuesta:

```json
{
  "text": "| phase | duration_weeks | cost_eur | confidence_pct | …",
  "prompt_version": "v1",
  "avisos": []
}
```

Errores posibles:

| Código | Cuándo | `detail` |
|---|---|---|
| 422 | La entrada no cumple el contrato (longitud de `description`, valores de los enums) | Lista de errores de validación de Pydantic |
| 503 | Falta la API key del modelo primario | Nombra la variable, por ejemplo *"Falta OPENAI_API_KEY para el modelo primario openai/gpt-4o-mini."* |
| 504 | El proveedor no respondió a tiempo (agotados los reintentos y el respaldo) | Mensaje genérico en español |
| 502 | Cualquier otro fallo del proveedor | Mensaje genérico en español |

En 502 y 504 el detalle real del proveedor no llega al cliente: queda en el log (ver **Trazabilidad**).

### Cliente Streamlit

El cliente Streamlit es un formulario que construye el JSON y muestra el `text` recibido. Consume la API por HTTP:

```bash
cd lidr_4
uv run streamlit run streamlit_app.py
# Abrir http://localhost:8501
```

La URL del servicio se lee de `ESTIMATOR_API_BASE_URL` (default `http://localhost:8001`).

## Cómo testar

```bash
cd lidr_4
uv run pytest
uv run ruff check . && uv run ruff format --check .
```

La batería corre en unos segundos, sin red y sin Redis (las llamadas al LLM se simulan con `mock_response` de LiteLLM y Redis con fakeredis):

- `test/test_schemas.py` — validaciones del `EstimationRequest` (longitudes, enums, campos obligatorios) y de `Settings` (techo del operador sobre `description`, `LOG_LEVEL`).
- `test/test_prompts.py` — render de la versión `v1`:
  - **Plantilla:** `description` dentro de `<project_description>`, bloques condicionales por `output_format` y `detail_level`, `StrictUndefined` falla temprano, una versión inexistente lanza `TemplateNotFound`, el prompt empieza sin líneas en blanco.
  - **Ejemplos:** horas y costes calculados desde las tarifas y redondeados (5 h / 50 €), totales que cuadran con las fases, una fila por línea en los tres formatos, numeración de `line_items` desde 1 en cada ejemplo, resumen de equipo armado con los `label`/`plural` del YAML.
  - **Validación del YAML** (con una versión temporal en `tmp_path`): rol no declarado en `rates` → error; rol sin `label` → error; un rol nuevo aparece en `<scope>`, partidas y resumen sin tocar Python.
  - **Caché:** el YAML se lee y se calcula una sola vez por versión, y cada llamada recibe su propia copia.
- `test/test_frontend.py` — cliente HTTP del formulario con transporte mockeado: el endpoint es `/api/v1/estimate` (no `/stream`), el payload son las cuatro claves del contrato, un 422 de FastAPI llega como lista de `msg`, y los `avisos` de la respuesta llegan al cliente (vacíos si la API no los envía).

- `test/test_llm_wrapper.py` — el wrapper LLM:
  - **Claves:** cada deployment recibe la clave de su proveedor; si falta la del primario, `LLMConfigurationError` nombrando la variable; si falta la del respaldo, arranca solo con el primario, con aviso en el log y en `avisos`.
  - **Router y respaldo:** el Router tiene los dos modelos, y si el primario falla responde el de respaldo.
  - **Resultado:** `model` es el nombre del modelo (no el id interno del deployment), `provider` coincide, se respeta `prompt_version` y el coste se calcula.
  - **Configuración:** reintentos, timeout y `max_tokens` salen de `Settings`.
  - **Coste:** si `completion_cost` falla, el coste es 0 y la respuesta sale igual.
  - **Trazabilidad:** evento `estimacion_completada` en llamada normal, con respaldo y con acierto de caché; el prompt nunca va al log.
- `test/test_cache.py` — la caché: ida y vuelta, TTL, clave que cambia con el prompt, entrada corrupta, Redis caído (lectura, escritura y una estimación completa), caché desactivada, y que la app abra **un solo** cliente de Redis y el wrapper use ese.
- `test/test_estimate_endpoint.py` — el endpoint con la app real: 200 normal, 503 si falta la clave del primario (sin filtrar la clave configurada), estimación con aviso si falta la del respaldo, `/health` con y sin claves, 502/504 con mensaje limpio ante fallos del proveedor y el detalle en el log.
- `test/test_logging.py` — los loggers de LiteLLM: con `LOG_LEVEL=DEBUG` quedan en `WARNING` y una estimación completa no deja la descripción del cliente en la salida; sus advertencias salen una sola vez, con el formato de structlog, y sin claves de API.

Los tests no dependen del `.env` ni de las variables de entorno de quien los corre: `test/conftest.py` los aísla y cada test fija lo que usa. Los que dependen del proveedor corren dos veces, con OpenAI y con Anthropic como primario.

## Estructura del proyecto

```
lidr_4/
├── app/
│   ├── __main__.py                    # Lanzador: python -m app [--port N] [--reload]
│   ├── main.py                        # FastAPI app, lifespan, /health, parche OpenAPI
│   ├── config.py                      # Settings (Pydantic Settings, .env)
│   ├── cache.py                       # Cliente Redis único (lifespan), lectura/escritura fail soft
│   ├── tracing.py                     # emitir(): eventos de trazabilidad con structlog
│   ├── dependencies.py                # LLMWrapper perezoso, con la caché del lifespan
│   ├── routers/
│   │   └── estimations.py             # POST /api/v1/estimate, errores 502/504
│   ├── schemas/
│   │   └── estimation.py              # EstimationRequest, EstimationResponse, enums
│   ├── prompts/
│   │   ├── loader.py                  # Carga examples.yaml, aritmética genérica, render
│   │   └── estimation/
│   │       └── v1/
│   │           ├── system.j2          # rol + reglas + bloques condicionales + macros de presentación
│   │           ├── user.j2            # bloque <project_description>
│   │           └── examples.yaml      # roles (label, plural, tarifa), redondeo, few-shot
│   └── services/
│       ├── cache.py                   # Caché exact-match de estimaciones (sobre app/cache.py)
│       └── llm_wrapper.py             # LiteLLM Router con respaldo, coste y trazabilidad
├── test/
│   ├── conftest.py
│   ├── test_schemas.py
│   ├── test_prompts.py
│   ├── test_llm_wrapper.py
│   ├── test_cache.py
│   ├── test_estimate_endpoint.py
│   └── test_frontend.py
├── streamlit_app.py                   # Formulario que consume /api/v1/estimate
├── PLAN.md                            # Plan de construcción y decisiones
└── pyproject.toml
```

### Versionado de prompts

La estructura `app/prompts/<use_case>/<version>/` no es opcional: `v1/` ya existe desde el primer día porque versionar un prompt es la forma más barata de habilitar A/B testing y rollback en producción. Cuando una iteración del prompt se cocina, se crea `v2/` al lado y `render_estimation_prompt(request, version="v2")` lo recoge sin tocar router ni schemas.

Cada versión tiene tres archivos, con responsabilidades separadas:

| Archivo | Qué contiene | Qué decide |
|---|---|---|
| `examples.yaml` | Roles (`label`, `plural`, `eur_per_hour`), redondeo (`hours_base`, `cost_base`), `productive_hours_per_week` y los ejemplos few-shot (fases con equipo, semanas y confianza) | Los **datos** |
| `system.j2` | Rol del modelo, reglas, formatos de salida, niveles de detalle y las macros que muestran los ejemplos | Toda la **presentación**: etiquetas, plurales, resumen de equipo, maquetación por `output_format` |
| `user.j2` | El bloque `<project_description>` | La entrada del usuario |

`loader.py` es común a todas las versiones y solo hace **aritmética genérica**: horas por rol (semanas × horas productivas × personas, redondeado), coste (horas × tarifa, redondeado), totales y personas máximas por rol. No decide cómo se muestra nada. Valida el YAML al cargarlo (todo rol usado en un ejemplo tiene que estar en `rates`, y cada rol tiene que declarar `label`, `plural` y `eur_per_hour`) y cachea el resultado por versión.

Consecuencias:

- Agregar un rol es solo editar `examples.yaml`: aparece en `<scope>`, en las partidas y en el resumen del equipo. El orden de `rates` es el orden en que se muestra.
- Una `v2/` necesita su propio `examples.yaml` con esa misma forma.
- Las versiones publicadas son inmutables: corregir `v1` es sacar `v2`, no editar `v1` (ver `PROMPT_VERSION` en `.env.example`).

Lo que vive **fuera** de la versión (en código): el contrato (`EstimationRequest`), el switch de versión, el wrapper y la aritmética de los ejemplos (`loader.py`). Todo lo demás (rol del modelo, reglas, ejemplos, tarifas, formatos de salida, niveles de detalle) vive en `v1/`. Si para cambiar el comportamiento del modelo hay que tocar Python, la separación está rota.

## Llamada al LLM, caché y trazabilidad

**Router con respaldo.** `llm_wrapper.py` arma un Router de LiteLLM con dos deployments, `PRIMARY_MODEL` y `FALLBACK_MODEL`, cada uno con la clave de su proveedor. Las llamadas van al primario; si falla tras `LLM_MAX_RETRIES` reintentos, responde el de respaldo. El Router es el único dueño de reintentos y respaldo (PLAN.md §2). El wrapper se construye en la primera request, no al arrancar, para que el servicio levante aunque falte una clave. Si falta la clave del respaldo, el Router se arma solo con el primario y cada respuesta lleva el aviso en `avisos`.

**Caché exact-match.** Antes de llamar al proveedor se busca la respuesta en Redis. La clave es un hash del system prompt y el user prompt completos, así que cambiar la plantilla invalida la caché sola. Hay un único cliente de Redis, el que crea el lifespan (`app/cache.py`), y es el que se cierra al apagar. Si Redis está caído, lento o tiene una entrada corrupta, la estimación sale igual sin caché (fail soft, con timeouts de 1 s).

**Trazabilidad.** Cada estimación deja un evento en el log, emitido con `app/tracing.py`:

| Evento | Nivel | Campos |
|---|---|---|
| `estimacion_completada` | `info` | `modelo`, `proveedor`, `uso_respaldo`, `desde_cache`, `tokens_prompt`, `tokens_completion`, `coste_usd`, `coste_evitado_usd`, `latencia_ms`, `prompt_version` |
| `estimacion_fallida` | `error` | `codigo_http` (502/504), `tipo_error`, `detalle` (el mensaje real del proveedor) |
| `respaldo_no_disponible` | `warning` | `modelo_respaldo`, `detalle` (el mismo aviso que recibe el usuario). Se emite una vez, al construir el wrapper |

`coste_usd` es lo que costó esa request: con acierto de caché vale 0 y lo que costó la respuesta original va a `coste_evitado_usd`. `uso_respaldo` marca las respuestas del modelo de respaldo, que pueden costar bastante más que el primario (PLAN.md §7). Los eventos llevan métricas, nunca el texto del prompt ni la descripción del cliente.

## Variables de entorno

Referencia completa y comentada en `.env.example`. Las principales:

| Variable | Default | Notas |
|---|---|---|
| `OPENAI_API_KEY` | — | Clave del proveedor `openai`. Se exige al usarse, no al arrancar |
| `ANTHROPIC_API_KEY` | — | Clave del proveedor `anthropic`. Hacen falta las claves de los proveedores de `PRIMARY_MODEL` y `FALLBACK_MODEL` |
| `PRIMARY_MODEL` | `openai/gpt-4o-mini` | Deployment principal del Router. El prefijo decide el provider |
| `FALLBACK_MODEL` | `anthropic/claude-haiku-4-5` | Se usa si el primario falla. Tiene que ser distinto del primario |
| `LLM_TIMEOUT` | `30.0` | Segundos por llamada al proveedor |
| `LLM_MAX_RETRIES` | `2` | Reintentos del Router antes de pasar al respaldo |
| `LLM_MAX_TOKENS` | `4000` | Tope de tokens de la respuesta |
| `PROMPT_VERSION` | `v1` | Versión de la plantilla de prompt. Hoy no invalida la caché (su clave ya incluye el prompt completo); con el caché semántico de WU10 será su mecanismo de invalidación |
| `REDIS_URL` | `redis://localhost:6379/0` | Vacío = caché desactivada |
| `CACHE_TTL` | `86400` | Segundos |
| `DESCRIPCION_MIN_CHARS` / `DESCRIPCION_MAX_CHARS` | `20` / `2000` | Techo del operador; solo puede estrechar el contrato |
| `APP_ENV` | `local` | Se muestra en `/health` |
| `LOG_LEVEL` | `INFO` | Nivel de los logs propios. Las librerías HTTP, el SDK y LiteLLM quedan fijas en `WARNING`: ningún nivel escribe el prompt |
| `APP_PORT` | `8001` | Puerto de `python -m app` |
| `ESTIMATOR_API_BASE_URL` | `http://localhost:8001` | Lo lee el cliente Streamlit |

`get_settings()` es un singleton cacheado con `lru_cache`: cualquier cambio en `.env` requiere reiniciar el servicio (no basta con `--reload`).

---

> Este proyecto forma parte del **Master en AI Engineering** y es la base sobre la que se construye en directo el resto de la Sesión 04 (output estructurado, guardrails, cache semántico).
