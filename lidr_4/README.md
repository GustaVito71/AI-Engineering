# Estimator — Servicio IA de estimación de software

Servicio IA en FastAPI que estima proyectos de software a partir de un formulario tipado. Es la pieza Python del programa **Master en AI Engineering**: un endpoint pensado para ser consumido por un backend de negocio (Rails, Streamlit u otro), no por un usuario final.

A partir de la **Sesión 04** el contrato es deliberadamente estrecho:
- entrada tipada (`description` + tres enums),
- salida en texto libre (`/api/v1/estimate`) o estructurada y validada (`/api/v1/estimate/structured`),
- prompt fuera del código, versionado en `app/prompts/<use_case>/<version>/`: plantillas Jinja2 (`system.j2`, `user.j2` y `examples.j2`, que `system.j2` incluye con `{% include %}`) más un `examples.yaml` con los datos de esa versión.

La inteligencia adicional (guardrails, cache semántico) se construye encima de esta base en directo.

## Estado

| Pieza | Estado |
|---|---|
| Contrato de entrada (`EstimationRequest` / `EstimationResponse`) | Hecho (WU2) |
| Prompts versionados (`system.j2`, `user.j2`, `examples.j2`, `examples.yaml`, `loader.py`) | Hecho (WU4) |
| Wrapper LLM (`app/services/llm_wrapper.py`), caché, router `POST /api/v1/estimate`, errores y trazabilidad | Hecho (WU5) |
| Cliente Streamlit | Hecho (WU6) |
| Slice de punta a punta: formulario → API → proveedor → caché → formulario | Hecho (WU7) |
| Salida estructurada: `StructuredResult` con Instructor sobre el Router, prompt `v3`, `POST /api/v1/estimate/structured`, interruptor en el formulario | Hecho, sin streaming (WU8) |
| Streaming de la salida estructurada (`ijson`, eventos por fase) | Pendiente (WU8) |
| Guardrails, caché semántico | Pendiente (WU9–WU10) |

El detalle de cada unidad está en [`PLAN.md`](PLAN.md) §6.

**Convenciones:** los mensajes al usuario (errores HTTP, textos de la UI y de Swagger), los comentarios y los docstrings van en español; los identificadores (variables, funciones, métodos, clases) van en inglés. Quedan en español, porque son datos y no identificadores, los nombres de los eventos y de los campos de log (`estimacion_completada`, `codigo_http`). Las plantillas `v1` y `v2` conservan sus macros (`miles`, `decimales`): son versiones publicadas e inmutables.

## Cómo levantar

Todo corre en Docker: Redis, la API y el frontend Streamlit (`docker-compose.yml`).

```bash
cd lidr_4
cp .env.example .env            # completá las API keys del modelo primario y del de respaldo
docker compose up --build -d    # construir y levantar los tres servicios
docker compose logs -f api      # logs de la API (trazabilidad, avisos)
docker compose down             # parar todo
```

| Servicio | Dirección | Notas |
|---|---|---|
| API | `http://localhost:8001` | Swagger en `/docs`, health en `/health` |
| Frontend | `http://localhost:8501` | Formulario Streamlit |
| Redis | solo dentro de Docker | Caché de estimaciones; no se publica hacia afuera |

Los puertos se publican solo en `127.0.0.1`: nada queda accesible desde la red local. Las claves salen del `.env` en tiempo de ejecución y nunca entran a la imagen (`.dockerignore`). Dentro de Docker los servicios se alcanzan por su nombre, así que el compose pisa dos valores del `.env`: `REDIS_URL=redis://redis:6379/0` y `ESTIMATOR_API_BASE_URL=http://api:8001`.

Un cambio de código requiere reconstruir: `docker compose up --build -d`. Un cambio en `.env` basta con reiniciar: `docker compose up -d` (recrea los contenedores cuya configuración cambió).

El servicio arranca aunque falten las API keys. Con la configuración por defecto:

- **Sin `OPENAI_API_KEY`** (primario) no hay estimaciones: `/health` informa `llm_configured: false` y `POST /api/v1/estimate` responde 503 nombrando la variable.
- **Sin `ANTHROPIC_API_KEY`** (respaldo) las estimaciones funcionan solo con el primario. Se avisa en tres lugares: el log (evento `respaldo_no_disponible`), `/health` (`fallback_configured: false` y `warnings`) y cada respuesta del endpoint (`warnings`), que Streamlit muestra encima de la estimación.

Redis es opcional para la API: si no responde, el servicio sigue sin caché (fail soft), y con `REDIS_URL` vacío la caché queda desactivada.

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
  "text": "| Fase | Semanas | Coste (EUR) | Confianza (%) | …",
  "prompt_version": "v2",
  "warnings": []
}
```

`text` llega sin las etiquetas `<estimation>` que envuelven los ejemplos del prompt: el modelo a veces las copia en su respuesta, y el endpoint las quita antes de devolverla.

Para usar otra versión publicada del prompt sin reiniciar el servicio, se agrega `?prompt_version=` a la URL (por ejemplo, `http://localhost:8001/api/v1/estimate?prompt_version=v1`). Sin el parámetro se usa `PROMPT_VERSION`. `prompt_version` en la respuesta indica siempre la versión que se usó.

Errores posibles:

| Código | Cuándo | `detail` |
|---|---|---|
| 422 | La entrada no cumple el contrato (longitud de `description`, valores de los enums) | Lista de errores de validación de Pydantic |
| 422 | `?prompt_version=` pide una versión que no existe | *"La versión de prompt 'v9' no existe. Versiones disponibles: v1, v2."* |
| 503 | Falta la API key del modelo primario | Nombra la variable, por ejemplo *"Falta OPENAI_API_KEY para el modelo primario openai/gpt-4o-mini."* |
| 504 | El proveedor no respondió a tiempo (agotados los reintentos y el respaldo) | Mensaje genérico en español |
| 502 | Cualquier otro fallo del proveedor | Mensaje genérico en español |

En 502 y 504 el detalle real del proveedor no llega al cliente: queda en el log (ver **Trazabilidad**).

### Salida estructurada

`POST /api/v1/estimate/structured` recibe el mismo cuerpo y devuelve la estimación como datos validados contra `StructuredResult` (`app/schemas/structured_estimation.py`):

```bash
curl -X POST http://localhost:8001/api/v1/estimate/structured \
  -H "Content-Type: application/json" \
  -d '{
    "description": "A small B2B SaaS to manage employee equipment loans across teams. Role-based access, audit trail, weekly digest.",
    "project_type": "web_saas",
    "detail_level": "detailed",
    "output_format": "phases_table"
  }'
```

```json
{
  "estimation": {
    "phases": [
      {"name": "Descubrimiento", "summary": "Entrevistas con RR. HH. e IT…",
       "duration_weeks": 1.0, "hours": 70, "cost_eur": 3950.0, "confidence_pct": 85,
       "assumptions": ["…"], "risks": [{"risk": "…", "mitigation": "…"}]}
    ],
    "team": [{"role": "Desarrollador", "headcount": 2}],
    "totals": {"hours": 565, "cost_eur": 34050.0, "duration_weeks": 9.0},
    "summary": "SaaS B2B de préstamo de equipos… El grueso del esfuerzo está en la implementación.",
    "confidence_pct": 75
  },
  "prompt_version": "v3",
  "cached": false,
  "warnings": []
}
```

Cómo funciona: `StructuredResult` (Pydantic) → Instructor → Router de LiteLLM → proveedor. Instructor envía el schema como `response_format` y valida la respuesta; si no lo cumple, le devuelve el error al modelo y vuelve a preguntar, hasta `STRUCTURED_MAX_RETRIES` veces. Lo hace sobre `Router.acompletion`, así que el respaldo, los reintentos ante fallos del proveedor, la caché y la trazabilidad son los mismos que en `/estimate`.

- **Qué es un error y qué un aviso.** Los rangos de cada campo (mínimos y topes: entre 1 y 8 fases, hasta 52 semanas y 1.000.000 EUR por fase, hasta 104 semanas en total, entre otros), los nombres de fase únicos y la coherencia del rechazo son condiciones del schema: si no se cumplen, se vuelve a preguntar. Que los totales no coincidan con la suma de las fases **no** es un error: los totales los declara el modelo (PLAN.md §3.4) y la estimación se devuelve igual, con el aviso en `warnings` y el evento `totales_no_cuadran` en el log.
- **Descripciones que no alcanzan.** Si el modelo no puede estimar con al menos 30 % de confianza, lo dice: `summary` empieza con «Fuera de alcance:» y explica qué falta, y la estimación es una sola fase «Sin estimar» con todo en cero. Es una respuesta 200, no un error; queda en el log como `estimacion_fuera_de_alcance`, y el formulario muestra solo la explicación. Una confianza baja sin el prefijo, o el prefijo con cifras, se vuelve a pedir.
- **Resumen y confianza.** `summary` y `confidence_pct` resumen la estimación completa, y cada fase trae su propio `summary` con lo que se hace en ella. El prompt pide calcular los totales a partir de las fases, sumar y comprobar antes de responder (bloque `<totals>`).
- **`cached`** dice si la estimación salió de la caché, sin llamada al proveedor.
- **`output_format` no cambia la respuesta.** El modelo devuelve siempre la misma estructura; el formulario decide si mostrarla como tabla, partidas o narrativa. `detail_level` sí cambia: `summary` deja `assumptions` y `risks` vacíos, `medium` agrega supuestos y `detailed` agrega supuestos y riesgos con su mitigación.
- **Cada intento se paga.** El coste y los tokens de la respuesta suman todos los intentos, no solo el aceptado.
- **Origen.** El resumen general y por fase, la confianza global con «Fuera de alcance:», los topes, el bloque `<totals>` y `cached` se tomaron de la solución de referencia `session_4_live/estimator`. De ella no se tomó la llamada a Instructor sobre `litellm.completion` (perdería el respaldo del Router), la tabla de precios escrita a mano ni re-preguntar cuando los totales no cuadran.
- **Versiones.** Solo acepta versiones de salida estructurada (hoy, `v3`); sin `?prompt_version=` usa `STRUCTURED_PROMPT_VERSION`. Pedirle `v2`, o pedirle `v3` a `/estimate`, da un 422 que dice en qué endpoint pedirla.

Errores propios, además de los de `/estimate`:

| Código | Cuándo | `detail` |
|---|---|---|
| 422 | `?prompt_version=` pide una versión de texto libre | *"La versión de prompt 'v2' es de texto libre: pedila en POST /api/v1/estimate. Versiones disponibles en este endpoint: v3."* |
| 502 | Ningún intento del modelo cumplió el schema | Mensaje genérico en español; intentos, coste y último error van al log |
| 503 | `STRUCTURED_PROMPT_VERSION` no es una versión de salida estructurada publicada | Nombra la variable y las versiones disponibles |

### Cliente Streamlit

El cliente Streamlit es un formulario que construye el JSON y muestra la estimación recibida. Consume la API por HTTP y queda en `http://localhost:8501` al levantar el compose.

La URL de la API se lee de `ESTIMATOR_API_BASE_URL`; en Docker es `http://api:8001`.

En la barra lateral, **Versión del prompt** elige con qué versión se estima: "Predeterminada (v2)" deja que decida `PROMPT_VERSION` en el servicio, y cada versión publicada se pide con `?prompt_version=`. La lista sale de `prompt_versions` en `/health`, así que una versión nueva aparece sola. Si la API no respondía al abrir la página, queda solo "Predeterminada"; **Probar conexión** la vuelve a leer.

Con el interruptor **Salida estructurada** el formulario llama a `/api/v1/estimate/structured` y el selector ofrece las versiones estructuradas (`structured_prompt_versions` en `/health`). La estimación se muestra con el resumen y los totales arriba (horas, coste, duración y confianza), las fases en el formato elegido (tabla, partidas o narrativa) con la descripción de cada una, el equipo y, si el nivel de detalle los pide, los supuestos y riesgos de cada fase. Si el modelo rechazó estimar, se muestra solo la explicación de qué falta. Si la estimación salió de la caché, lo indica debajo.

## Cómo testar

Los tests corren fuera de Docker, con las dependencias de desarrollo:

```bash
cd lidr_4
uv sync
uv run pytest
uv run ruff check . && uv run ruff format --check .
```

La batería corre en unos segundos, sin red y sin Redis (las llamadas al LLM se simulan con `mock_response` de LiteLLM y Redis con fakeredis):

- `test/test_schemas.py` — validaciones del `EstimationRequest` (longitudes, enums, campos obligatorios) y de `Settings` (techo del operador sobre `description`, `LOG_LEVEL`).
- `test/test_prompts.py` — render de la versión `v1`:
  - **Plantilla:** `description` dentro de `<project_description>`, bloques condicionales por `output_format` y `detail_level`, `StrictUndefined` falla temprano, una versión inexistente lanza `TemplateNotFound`, el prompt empieza sin líneas en blanco, y `system.j2` incluye el `examples.j2` de su propia versión (también en una versión copiada).
  - **Ejemplos:** horas y costes calculados desde las tarifas y redondeados (5 h / 50 €), totales que cuadran con las fases, una fila por línea en los tres formatos, numeración de `line_items` desde 1 en cada ejemplo, resumen de equipo armado con los `label`/`plural` del YAML.
  - **Validación del YAML** (con una versión temporal en `tmp_path`): rol no declarado en `rates` → error; rol sin `label` → error; un rol nuevo aparece en `<scope>`, partidas y resumen sin tocar Python.
  - **Caché:** el YAML se lee y se calcula una sola vez por versión, y cada llamada recibe su propia copia.
- `test/test_prompts_v2.py` — render de la versión `v2`: las 36 combinaciones sin texto en inglés, regla de idioma en castellano, tarifas con coma decimal, tabla, elementos de línea y narrativo con números en formato castellano, concordancia de singular y plural, y la misma aritmética que `v1`.
- `test/test_frontend.py` — cliente HTTP del formulario con transporte mockeado: el endpoint es `/api/v1/estimate` (no `/stream`), el payload son las cuatro claves del contrato, un 422 de FastAPI llega como lista de `msg`, y los `warnings` de la respuesta llegan al cliente (vacíos si la API no los envía). Versión del prompt: la elegida viaja en `?prompt_version=` y la predeterminada no manda el parámetro; las opciones salen de `/health` y, si la API no responde o es anterior a `prompt_versions`, queda solo la predeterminada.
- `test/test_e2e.py` — el slice de punta a punta (WU7) con todas las piezas reales: el formulario de Streamlit (`_build_payload` + `_estimate`), la API con su lifespan, el prompt `v2`, el wrapper con el Router de LiteLLM, la caché y la limpieza de etiquetas. Solo se simulan el proveedor (`mock_response`) y Redis (fakeredis). Cubre el camino normal (prompt que recibe el proveedor, texto limpio, trazabilidad), la caché (acierto, otra descripción, texto cacheado que se limpia al responder), la configuración (sin clave de respaldo, sin clave del primario, sin Redis) y los fallos (504 y 502 en castellano con el detalle solo en el log, un fallo que no queda en caché, un 422 que no llega al proveedor). Incluye el selector de versión: las opciones salen del `/health` real y la versión elegida decide el prompt que recibe el proveedor.

- `test/test_llm_wrapper.py` — el wrapper LLM:
  - **Claves:** cada deployment recibe la clave de su proveedor; si falta la del primario, `LLMConfigurationError` nombrando la variable; si falta la del respaldo, arranca solo con el primario, con aviso en el log y en `warnings`.
  - **Router y respaldo:** el Router tiene los dos modelos, y si el primario falla responde el de respaldo.
  - **Resultado:** `model` es el nombre del modelo (no el id interno del deployment), `provider` coincide, se respeta `prompt_version` y el coste se calcula.
  - **Configuración:** reintentos, timeout y `max_tokens` salen de `Settings`.
  - **Coste:** si `completion_cost` falla, el coste es 0 y la respuesta sale igual.
  - **Trazabilidad:** evento `estimacion_completada` en llamada normal, con respaldo y con acierto de caché; el prompt nunca va al log.
- `test/test_cache.py` — la caché: ida y vuelta, TTL, clave que cambia con el prompt, entrada corrupta, Redis caído (lectura, escritura y una estimación completa), caché desactivada, y que la app abra **un solo** cliente de Redis y el wrapper use ese.
- `test/test_estimate_endpoint.py` — el endpoint con la app real: 200 normal, 503 si falta la clave del primario (sin filtrar la clave configurada), estimación con aviso si falta la del respaldo, `/health` con y sin claves, 502/504 con mensaje limpio ante fallos del proveedor y el detalle en el log; `?prompt_version=` (sin query se usa `PROMPT_VERSION`, la query la pisa, una versión inexistente o con `..` da 422 sin llamar al modelo); `/health` lista las versiones disponibles (`prompt_versions`).
- **Salida estructurada (WU8):**
  - `test/test_structured_schema.py` — qué rechaza el schema (y se vuelve a preguntar: mínimos, topes, textos, rechazos mal formados, fases estimadas en cero) y qué es solo un aviso (totales que no cuadran, con los números en formato castellano); un rechazo bien formado se acepta; `phases` va primero y el resumen al final.
  - `test/test_prompts_v3.py` — `v3` es la única versión estructurada y `v1`/`v2` no se tocaron; los ejemplos del prompt son JSON válido contra el schema y con totales que cuadran, con resumen y confianza, y el último es un rechazo en cero; el bloque `<totals>` y las reglas de rechazo salen de las constantes del schema; `detail_level` decide supuestos y riesgos y `output_format` no cambia el prompt; las 36 combinaciones en castellano; regla anti-instrucciones.
  - `test/test_llm_wrapper_structured.py` — Instructor sobre el Router: respuesta válida, respuesta inválida o JSON roto que se vuelve a pedir con el error, intentos agotados, coste y tokens que suman todos los intentos (también en el error), fallos del proveedor que llegan como excepciones de LiteLLM, respaldo, caché y trazabilidad.
  - `test/test_estimate_structured_endpoint.py` — 200, avisos de totales, 422 por versión del otro tipo o inexistente (en los dos endpoints), 503 por `STRUCTURED_PROMPT_VERSION` inválida o sin clave, 502 por salida inválida o fallo del proveedor, 504, `/health` y OpenAPI.
  - `test/test_frontend_structured.py` — cliente HTTP del endpoint estructurado, selector de versión y lo que se muestra (tabla, partidas, narrativa, equipo, números en castellano).
  - `test/test_e2e_structured.py` — el slice completo con Instructor y el Router reales: del formulario a la tabla, reintento, intentos agotados (502), caché, texto y estructurada sin compartir caché, aviso de totales y selector desde `/health`.
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
│   ├── tracing.py                     # emit(): eventos de trazabilidad con structlog
│   ├── dependencies.py                # LLMWrapper perezoso, con la caché del lifespan
│   ├── routers/
│   │   └── estimations.py             # POST /api/v1/estimate y /estimate/structured, errores 502/504
│   ├── schemas/
│   │   ├── estimation.py              # EstimationRequest, EstimationResponse, enums
│   │   └── structured_estimation.py   # StructuredResult (contrato del LLM), StructuredEstimationResponse
│   ├── prompts/
│   │   ├── loader.py                  # Carga examples.yaml, aritmética genérica, render
│   │   └── estimation/
│   │       ├── v1/                    # estimación en inglés
│   │       │   ├── system.j2          # rol + reglas + bloques condicionales; incluye examples.j2
│   │       │   ├── user.j2            # bloque <project_description>
│   │       │   ├── examples.j2        # presentación de los few-shot según output_format y detail_level
│   │       │   └── examples.yaml      # roles (label, plural, tarifa), redondeo, datos de los few-shot
│   │       ├── v2/                    # la misma estimación en castellano (versión por defecto)
│   │       └── v3/                    # salida estructurada (JSON), en castellano
│   └── services/
│       ├── cache.py                   # Caché exact-match de estimaciones (sobre app/cache.py)
│       └── llm_wrapper.py             # LiteLLM Router con respaldo, coste y trazabilidad; Instructor para la salida estructurada
├── test/
│   ├── conftest.py
│   ├── test_schemas.py
│   ├── test_prompts.py
│   ├── test_prompts_v2.py
│   ├── test_prompts_v3.py
│   ├── test_structured_schema.py
│   ├── test_llm_wrapper_structured.py
│   ├── test_estimate_structured_endpoint.py
│   ├── test_frontend_structured.py
│   ├── test_e2e_structured.py
│   ├── test_llm_wrapper.py
│   ├── test_cache.py
│   ├── test_estimate_endpoint.py
│   ├── test_logging.py
│   ├── test_frontend.py
│   └── test_e2e.py
├── Dockerfile                         # Imagen única para la API y el frontend (dependencias de uv.lock)
├── docker-compose.yml                 # Redis + API + frontend
├── .dockerignore                      # Deja afuera de la imagen el .env, los tests y los caches
├── streamlit_app.py                   # Formulario que consume /api/v1/estimate
├── PLAN.md                            # Plan de construcción y decisiones
└── pyproject.toml
```

### Versionado de prompts

La estructura `app/prompts/<use_case>/<version>/` no es opcional: `v1/` ya existe desde el primer día porque versionar un prompt es la forma más barata de habilitar A/B testing y rollback en producción. Cuando una iteración del prompt se cocina, se crea `v2/` al lado y `render_estimation_prompt(request, version="v2")` lo recoge sin tocar router ni schemas.

Cada versión tiene cuatro archivos, con responsabilidades separadas:

| Archivo | Qué contiene | Qué decide |
|---|---|---|
| `examples.yaml` | Roles (`label`, `plural`, `eur_per_hour`), redondeo (`hours_base`, `cost_base`), `productive_hours_per_week`, los ejemplos few-shot (fases con equipo, semanas y confianza) y, opcionalmente, el tipo de salida (`output`) | Los **datos** |
| `system.j2` | Rol del modelo, reglas, formatos de salida y niveles de detalle. Al final incluye `examples.j2` con `{% include "estimation/" ~ version ~ "/examples.j2" %}`: la versión es una variable, así que una versión copiada incluye sus propios ejemplos y no los de la original | Las **instrucciones** al modelo |
| `examples.j2` | Las macros que muestran los ejemplos few-shot | La **presentación** de los ejemplos: etiquetas, plurales, resumen de equipo, maquetación por `output_format` y `detail_level` |
| `user.j2` | El bloque `<project_description>` | La entrada del usuario |

`loader.py` es común a todas las versiones y solo hace **aritmética genérica**: horas por rol (semanas × horas productivas × personas, redondeado), coste (horas × tarifa, redondeado), totales y personas máximas por rol. No decide cómo se muestra nada. Valida el YAML al cargarlo (todo rol usado en un ejemplo tiene que estar en `rates`, y cada rol tiene que declarar `label`, `plural` y `eur_per_hour`) y cachea el resultado por versión.

Consecuencias:

- Agregar un rol es solo editar `examples.yaml`: aparece en `<scope>`, en las partidas y en el resumen del equipo. El orden de `rates` es el orden en que se muestra.
- Una versión nueva necesita sus propios `examples.yaml` y `examples.j2`: se copian de la anterior y se cambia lo que haga falta.
- Las versiones publicadas son inmutables: corregir `v1` es sacar `v2`, no editar `v1` (ver `PROMPT_VERSION` en `.env.example`).

Versiones publicadas:

| Versión | Idioma de la estimación | Notas |
|---|---|---|
| `v1` | Inglés | Instrucciones, ejemplos y etiquetas en inglés; números como `29,850` |
| `v2` (por defecto) | Castellano | Mismos datos y la misma aritmética que `v1`; instrucciones, ejemplos y etiquetas en castellano; números como `29.850` y `62,50`; registro impersonal |
| `v3` (salida estructurada) | Castellano | Mismos números que `v2`, más un resumen y una confianza por ejemplo, un resumen por fase y un quinto ejemplo de rechazo («Fuera de alcance:»). Pide un objeto JSON que cumpla `StructuredResult`; los ejemplos son ese JSON (filtro `json` del loader, con tildes legibles). `output_format` no cambia el prompt |

**Tipo de salida.** Cada versión declara su tipo con la clave `output` de su `examples.yaml`: `text` (el valor por defecto, así que `v1` y `v2` no se tocaron) o `structured`. `available_versions(output)` filtra por tipo y cada endpoint acepta solo las suyas: `/estimate` las de texto, `/estimate/structured` las estructuradas. Una versión nueva de cualquier tipo entra declarando su `output`, sin tocar código.

La versión se elige por petición con `?prompt_version=` o, si no se indica, con `PROMPT_VERSION` (o `STRUCTURED_PROMPT_VERSION` en `/estimate/structured`). El endpoint solo acepta versiones publicadas (un directorio `vN/` con `system.j2`); el valor nunca llega a armar una ruta de archivo sin haberse comparado antes con esa lista.

Lo que vive **fuera** de la versión (en código): el contrato (`EstimationRequest`), el switch de versión, el wrapper y la aritmética de los ejemplos (`loader.py`). Todo lo demás (rol del modelo, reglas, ejemplos, tarifas, formatos de salida, niveles de detalle) vive en `v1/`. Si para cambiar el comportamiento del modelo hay que tocar Python, la separación está rota.

## Llamada al LLM, caché y trazabilidad

**Router con respaldo.** `llm_wrapper.py` arma un Router de LiteLLM con dos deployments, `PRIMARY_MODEL` y `FALLBACK_MODEL`, cada uno con la clave de su proveedor. Las llamadas van al primario; si falla tras `LLM_MAX_RETRIES` reintentos, responde el de respaldo. El Router es el único dueño de reintentos y respaldo (PLAN.md §2). El wrapper se construye en la primera request, no al arrancar, para que el servicio levante aunque falte una clave. Si falta la clave del respaldo, el Router se arma solo con el primario y cada respuesta lleva el aviso en `warnings`.

**Caché exact-match.** Antes de llamar al proveedor se busca la respuesta en Redis. La clave es un hash del system prompt y el user prompt completos, así que cambiar la plantilla invalida la caché sola. Hay un único cliente de Redis, el que crea el lifespan (`app/cache.py`), y es el que se cierra al apagar. Si Redis está caído, lento o tiene una entrada corrupta, la estimación sale igual sin caché (fail soft, con timeouts de 1 s).

**Trazabilidad.** Cada estimación deja un evento en el log, emitido con `app/tracing.py`:

| Evento | Nivel | Campos |
|---|---|---|
| `estimacion_completada` | `info` | `modelo`, `proveedor`, `uso_respaldo`, `desde_cache`, `tokens_prompt`, `tokens_completion`, `coste_usd`, `coste_evitado_usd`, `latencia_ms`, `prompt_version`. En la salida estructurada, además `salida` (`estructurada`) e `intentos` (0 si salió de la caché); tokens y coste suman todos los intentos |
| `estimacion_fallida` | `error` | `codigo_http` (502/504), `tipo_error`, `detalle` (el mensaje real del proveedor). En la salida estructurada, además `salida` y, si ningún intento cumplió el schema, `intentos` y `coste_usd` |
| `totales_no_cuadran` | `warning` | `prompt_version`, `modelo`, `detalle` (los mismos avisos que recibe el usuario) |
| `estimacion_fuera_de_alcance` | `info` | `prompt_version`, `modelo`, `confianza_pct`. El modelo rechazó estimar porque la descripción no alcanza |
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
| `PROMPT_VERSION` | `v2` | Versión de la plantilla de prompt. Hoy no invalida la caché (su clave ya incluye el prompt completo); con el caché semántico de WU10 será su mecanismo de invalidación |
| `STRUCTURED_PROMPT_VERSION` | `v3` | Versión por defecto de `/estimate/structured`. Tiene que ser de salida estructurada; si no, ese endpoint responde 503 |
| `STRUCTURED_MAX_RETRIES` | `2` | Veces que se vuelve a preguntar al modelo si la respuesta no cumple el schema. Cada intento se paga |
| `REDIS_URL` | `redis://localhost:6379/0` | Vacío = caché desactivada. En Docker lo fija el compose: `redis://redis:6379/0` |
| `CACHE_TTL` | `86400` | Segundos |
| `DESCRIPTION_MIN_CHARS` / `DESCRIPTION_MAX_CHARS` | `20` / `2000` | Techo del operador; solo puede estrechar el contrato. Los nombres anteriores, `DESCRIPCION_MIN_CHARS` / `DESCRIPCION_MAX_CHARS`, se siguen leyendo; si están los dos, manda el nuevo |
| `APP_ENV` | `local` | Se muestra en `/health` |
| `LOG_LEVEL` | `INFO` | Nivel de los logs propios. Las librerías HTTP, el SDK, LiteLLM e Instructor quedan fijas en `WARNING`: ningún nivel escribe el prompt |
| `APP_PORT` | `8001` | Puerto de `python -m app` (fuera de Docker; el compose usa 8001) |
| `ESTIMATOR_API_BASE_URL` | `http://localhost:8001` | Lo lee el cliente Streamlit. En Docker lo fija el compose: `http://api:8001` |

`get_settings()` es un singleton cacheado con `lru_cache`: cualquier cambio en `.env` requiere reiniciar el servicio (`docker compose up -d`).

---

> Este proyecto forma parte del **Master en AI Engineering** y es la base sobre la que se construye en directo el resto de la Sesión 04 (guardrails, cache semántico).
