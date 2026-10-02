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
| Cliente Streamlit | Hecho (WU6) |
| Wrapper LLM (`app/services/llm_wrapper.py`) y router `POST /api/v1/estimate` | **Pendiente (WU5)** |

Hasta que WU5 exista, `app.main` no arranca: importa `app.routers.estimations`, que todavía no está escrito. El detalle de cada unidad está en [`PLAN.md`](PLAN.md) §6.

## Cómo levantar

> Requiere WU5 (ver **Estado**).

```bash
cd lidr_4
cp .env.example .env  # añade al menos OPENAI_API_KEY o ANTHROPIC_API_KEY
uv sync
uv run python -m app            # puerto APP_PORT (default 8001)
uv run python -m app --reload   # con recarga
```

`python -m app` usa `APP_PORT` de Settings. `uvicorn app.main:app` a secas usa el 8000, que suele estar tomado. El servicio queda en `http://localhost:8001` (Swagger en `/docs`, health en `/health`).

Redis es opcional: con `REDIS_URL` vacío el cache queda desactivado, y si Redis no responde el servicio sigue sin cache (fail soft).

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
  "prompt_version": "v1"
}
```

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

La batería corre en milisegundos sin tocar APIs externas:

- `test/test_schemas.py` — validaciones del `EstimationRequest` (longitudes, enums, campos obligatorios) y de `Settings` (techo del operador sobre `description`, `LOG_LEVEL`).
- `test/test_prompts.py` — render de la versión `v1`:
  - **Plantilla:** `description` dentro de `<project_description>`, bloques condicionales por `output_format` y `detail_level`, `StrictUndefined` falla temprano, una versión inexistente lanza `TemplateNotFound`, el prompt empieza sin líneas en blanco.
  - **Ejemplos:** horas y costes calculados desde las tarifas y redondeados (5 h / 50 €), totales que cuadran con las fases, una fila por línea en los tres formatos, numeración de `line_items` desde 1 en cada ejemplo, resumen de equipo armado con los `label`/`plural` del YAML.
  - **Validación del YAML** (con una versión temporal en `tmp_path`): rol no declarado en `rates` → error; rol sin `label` → error; un rol nuevo aparece en `<scope>`, partidas y resumen sin tocar Python.
  - **Caché:** el YAML se lee y se calcula una sola vez por versión, y cada llamada recibe su propia copia.
- `test/test_frontend.py` — cliente HTTP del formulario con transporte mockeado: el endpoint es `/api/v1/estimate` (no `/stream`), el payload son las cuatro claves del contrato, y un 422 de FastAPI llega como lista de `msg`.

Los tests del endpoint y del wrapper LLM llegan con WU5.

## Estructura del proyecto

```
lidr_4/
├── app/
│   ├── __main__.py                    # Lanzador: python -m app [--port N] [--reload]
│   ├── main.py                        # FastAPI app, lifespan, /health, parche OpenAPI
│   ├── config.py                      # Settings (Pydantic Settings, .env)
│   ├── cache.py                       # Cliente Redis fail-soft (keying semántico en WU10)
│   ├── tracing.py                     # emitir(): eventos de trazabilidad con structlog
│   ├── dependencies.py                # Singletons de cache + LLMWrapper (se completa en WU5)
│   ├── routers/
│   │   └── estimations.py             # POST /api/v1/estimate            (WU5, pendiente)
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
│       └── llm_wrapper.py             # LiteLLM Router con fallback y coste (WU5, pendiente)
├── test/
│   ├── conftest.py
│   ├── test_schemas.py
│   ├── test_prompts.py
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

## Variables de entorno

Referencia completa y comentada en `.env.example`. Las principales:

| Variable | Default | Notas |
|---|---|---|
| `OPENAI_API_KEY` | — | Requerido al menos uno de los dos; se exige al usarse, no al arrancar |
| `ANTHROPIC_API_KEY` | — | Requerido al menos uno de los dos |
| `PRIMARY_MODEL` | `openai/gpt-4o-mini` | Deployment principal del Router. El prefijo decide el provider |
| `FALLBACK_MODEL` | `anthropic/claude-haiku-4-5` | Se usa si el primario falla. Tiene que ser distinto del primario |
| `PROMPT_VERSION` | `v1` | Versión de prompt. Es el mecanismo de invalidación del caché semántico |
| `REDIS_URL` | `redis://localhost:6379/0` | Vacío = cache desactivado |
| `CACHE_TTL` | `86400` | Segundos |
| `DESCRIPCION_MIN_CHARS` / `DESCRIPCION_MAX_CHARS` | `20` / `2000` | Techo del operador; solo puede estrechar el contrato |
| `APP_ENV` | `local` | Se muestra en `/health` |
| `LOG_LEVEL` | `INFO` | `DEBUG` vuelca prompt y respuesta completos al log |
| `APP_PORT` | `8001` | Puerto de `python -m app` |
| `ESTIMATOR_API_BASE_URL` | `http://localhost:8001` | Lo lee el cliente Streamlit |

`get_settings()` es un singleton cacheado con `lru_cache`: cualquier cambio en `.env` requiere reiniciar el servicio (no basta con `--reload`).

---

> Este proyecto forma parte del **Master en AI Engineering** y es la base sobre la que se construye en directo el resto de la Sesión 04 (output estructurado, guardrails, cache semántico).
