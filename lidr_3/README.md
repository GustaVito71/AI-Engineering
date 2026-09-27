# Estimador CAG

Servicio **FastAPI** que genera una estimación de duración (en horas) de un proyecto de software a partir de la transcripción de una reunión, usando **CAG (Cache-Augmented Generation)**.

## Qué es CAG y qué hace acá

A diferencia de RAG, no hay base vectorial ni búsqueda por similaridad: el insumo por petición (una transcripción de reunión) es acotado y cabe completo en contexto. El "cache" son **ejemplos de referencia que viven en el system prompt**, de forma estable, en cada llamada:

```
system: instrucciones + 4 ejemplos (transcripción → estimación modelo)   ← el cache
user:   <transcripcion-{marca aleatoria}> …transcripción… </transcripcion-{marca}>
```

Los ejemplos pesan más que las instrucciones. Por eso:

- **Son el sistema.** Cada total de `app/context/examples.py` está verificado a mano contra la suma de su desglose. Un ejemplo incoherente contamina todas las estimaciones.
- **Están separados de su serialización** (`get_examples_as_text`). Cuando la fuente migre a una base vectorial (RAG), solo cambia ese punto y nadie más se entera.
- **El prompt previene el anclaje**: dice al modelo que los ejemplos son referencia de granularidad/orden de magnitud, no una plantilla de cifras.
- Hay un test que verifica el invariante del ejercicio: `estimation` de cada ejemplo aparece en el system prompt. Si ese test se rompe, no hay CAG.

## Requisitos

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) como gestor de paquetes
- Cuenta activa en OpenAI Platform y/o Anthropic con créditos disponibles
- API key disponible como variable de entorno o en `.env`

## Puesta en marcha

```bash
cp .env.example .env      # y pegá al menos una API key
uv sync --all-groups
uv run python -m app      # arranca en 127.0.0.1:8001 (APP_PORT)
```

> El default del servicio es el **8001** porque el 8000 lo suele tener tomado otro
> proceso (p. ej. Ganttly en Docker). Cambiá `APP_PORT` en `.env`, o
> `uv run python -m app --port <n>`. Si usás `uvicorn app.main:app` a secas,
> uvicorn usa su default (8000) e ignora `APP_PORT`.

El frontend (Streamlit) va en **otra terminal**, y espera la APIlevantada:

```bash
uv run streamlit run streamlit_app.py     # UI en 127.0.0.1:8501
```

| Recurso | URL |
|---|---|
| API | http://localhost:8001 |
| `/health` | http://localhost:8001/health |
| `/docs` | http://localhost:8001/docs |
| UI (Streamlit) | http://localhost:8501 |

> La UI lee `ESTIMATOR_API_BASE_URL` (default `http://localhost:8001`) del
> entorno, y además se puede cambiar en vivo desde el sidebar.

## Uso

El **parámetro del ejercicio es `datos/transcripcion_reunion.md`** (la
transcripción de reunión canónica entre los marcadores
`<!-- transcripcion -->` / `<!-- /transcripcion -->`). Con un comando:

```bash
curl -X POST http://localhost:8001/api/v1/estimate \
  -H 'Content-Type: application/json' \
  -d "$(python3 - <<'PY'
import json
md = open('datos/transcripcion_reunion.md').read()
texto = md.split('<!-- transcripcion -->')[1].split('<!-- /transcripcion -->')[0].strip()
print(json.dumps({'transcription': texto}))
PY
)"
```

O con una transcripción a mano:

```bash
curl -X POST http://localhost:8001/api/v1/estimate \
  -H 'Content-Type: application/json' \
  -d '{"transcription": "Reunión: el cliente pide una landing con formulario de contacto…"}'
```

La respuesta incluye `truncated` (si es `true`, la respuesta se cortó por `max_tokens` y **no está completa**) y el coste de la llamada: `cost_usd` con `cost_note` explicando la fuente (base de precios de LLMPrice) o, si es `null`, por qué no se pudo estimar. El servicio no oculta ninguna de las dos señales.

Configurar el proveedor: `LLM_PROVIDER=openai|anthropic` en `.env`. El modelo es opcional (vacío = default del proveedor → `gpt-4o-mini` / `claude-haiku-4-5`).

### Streaming (SSE)

`POST /api/v1/estimate/stream` es la misma estimación, pero la respuesta se
consume como `text/event-stream` para que el texto aparezca a medida que el
modelo lo genera:

```bash
curl -N -X POST http://localhost:8001/api/v1/estimate/stream \
  -H 'Content-Type: application/json' \
  -d '{"transcription": "Reunión: el cliente pide una landing con formulario de contacto…"}'
```

`-N` desactiva el buffer de curl: se ven los eventos a medida que llegan.

| Evento | Payload | Cuándo |
|---|---|---|
| `meta` | `{"proveedor", "camino"}` | primero, siempre. `camino` es `primario`, `secundario` (fallback), `cache` o `degradado` |
| `delta` | `{"texto"}` | fragmentos del Markdown, en orden |
| `estimation` | estimación completa (misma forma que `/estimate`) | último, una vez |

Reglas del contrato (cubiertas por `test/test_streaming.py`):

- **Fallo antes del primer byte** → error HTTP real (503 sin key, 502 si ambos proveedores fallan, 422 si la transcripción está fuera de límites). No hay stream que cortar: el cliente recibe un status y un `detail`.
- **Fallo después del primer fragmento** → el stream ya empezó, no se puede cambiar el status: viaja como evento `error` y el cliente debe tratarlo como estimado incompleto.
- **Cache hit** → un único evento `estimation`, sin `delta`. No se llama al proveedor.
- **Fallback** → solo puede ocurrir antes del primer fragmento (de ahí que solo haya un salto configurable, `llm_fallback`). Un `RateLimitError` o `5xx` del primario habilita el secundario; el resto de `4xx` no.

## Frontend (Streamlit)

`streamlit_app.py` es un cliente delgado de **turno único**: pegás la
transcripción y la estimación aparece en el chat mientras se genera. Cada
transcripción nueva reemplaza el turno anterior (el modelo es stateless por
petición, así que un historial multi-turno sería inventado).

Lo que **no** hace el frontend, a propósito: pedir API keys (viven en el `.env`
del backend) y mantener historial entre turnos.

Para probarlo de punta a punta:

```bash
# 1. API arriba, en una terminal
uv run python -m app

# 2. UI arriba, en otra
uv run streamlit run streamlit_app.py

# 3. En http://localhost:8501: "Probar conexión" (sidebar) y después estimar
```

Desde la terminal, sin abrir la UI, los mismos chequeos:

```bash
# La API está sana y con key
curl -s http://localhost:8001/health

# Todas las rutas publicadas (si falta /estimate/stream, hay un proceso viejo
# ocupando el puerto)
curl -s http://localhost:8001/openapi.json | uv run python -c \
  "import json,sys; [print(m.upper(), p) for p, ops in json.load(sys.stdin)['paths'].items() for m in ops]"

# El stream evento por evento
curl -N -X POST http://localhost:8001/api/v1/estimate/stream \
  -H 'Content-Type: application/json' -d '{"transcription": "Reunión: …"}'
```

Casos borde que vale la pena ver a mano:

| Qué probar | Qué tenés que ver |
|---|---|
| Transcripción de < 50 caracteres | 422 con el detalle de validación; **no** se gasta un token |
| API apagada (Ctrl+C) y estimar desde la UI | "No se pudo conectar…", no un traceback |
| API apagada y estimar por `curl -N` | 502 con el detalle de los dos proveedores |
| Repetir la misma transcripción | `camino: cache` y un único evento `estimation` |
| `Probar conexión` sin key en `.env` | "Conectado: … FALTA API KEY" + aviso de qué variable poner |

Si el puerto está tomado (el clásico `address already in use`):

```bash
ss -tlnp | grep 8001     # qué PID lo tiene
kill <PID>               # matarlo y relanzar
```

> Este escenario aparece seguido en la práctica: un proceso viejo de otra
> sesión queda sirviendo una versión anterior de la app (por ejemplo, sin
> `/estimate/stream`) y los 404 del backend se confunden con un bug del
> frontend. `curl /openapi.json` resuelve el diagnóstico en un segundo.

`test/test_frontend.py` cubre el contrato del parser contra el output real de
`_evento_sse` del router, más el cliente SSE con `httpx.MockTransport` y un
`IteratorByteStream` (body sin leer, como en el cable) para reproducir los
errores pre-primer-byte.

## Logs y trazabilidad

El servicio loguea con **structlog** a la salida estándar del proceso que corre
la API. No hay archivo de log: si la API está en tu terminal, los eventos
aparecen ahí; si la=background, redirigilos vos.

```bash
uv run python -m app                                  # eventos en vivo
nohup uv run python -m app > /tmp/lidr_api.log 2>&1 &  # en segundo plano
tail -f /tmp/lidr_api.log                             # seguirlos después
```

`LOG_LEVEL` (INFO por defecto) gobierna **sólo los eventos del proyecto**. Las
librerías HTTP de terceros quedan pineadas a WARNING, y eso es una decisión de
seguridad, no de comfort: `httpx` y la familia `httpcore` loguean los *headers*
completos de request y response en DEBUG, y así se terminaba escribiendo en el
log el `openai-organization` de tu cuenta y las cookies `set-cookie` de
Cloudflare. WARNING y arriba siguen pasando, así que los warnings y errores
reales de terceros no se pierden.

> Si agregás una librería HTTP nueva, pineá la **raíz** de su namespace a
> WARNING. Fijar los hijos no sirve: un logger hijo no listado de un padre
> `NOTSET` cae al nivel del root y vuelve a volcar todo. (Pinear `httpcore`
> cuando httpx 0.28 ya usa `httpcore2` fue exactamente ese error, y el test lo
> detectó.) `test/test_tracing.py` recorre el árbol real de loggers y falla si
> alguno queda por debajo de WARNING, así que un paquete renombrado te avisa en
> el pipeline en vez de filtrar en silencio.

### Las tres dimensiones

| Dimensión | Dónde | Qué responde |
| --- | --- | --- |
| Contenido | `contenido_intento` (DEBUG) | Prompt completo y respuesta literal |
| Camino | `intento_proveedor` + `estimacion_completada` | ¿Cache, primario o fallback? ¿Cuántos intentos? |
| Costo | `estimacion_completada` | Tokens reales × LLMPrice, con `None` + nota si no hay precio |

```text
[info] estimacion_completada  camino=primario cache_hit=False uso_fallback=False
       intentos_proveedor=1 modelo=gpt-4o-mini-2024-07-18
       costo_usd=0.0003589 costo_nota='LLMPrice (snapshot 2026.4.3 (de abril))'
       input_tokens=1431 output_tokens=244
       latencia_cache_ms=None latencia_llm_ms=3183.6 latencia_total_ms=3206.4
```

- `latencia_llm_ms` y `ttft_ms` son métricas **exclusivas de `/estimate/stream`**;
  el endpoint JSON las reporta como `None`. `ttft_ms` mide desde el pedido hasta
  el primer fragmento de contenido, así que incluye la red y el procesamiento
  del modelo (≈900 ms con `gpt-4o-mini`, no los 0.5 ms de un cronómetro mal
  colocado). Con fallback, ambas incluyen el intento fallido del primario: es la
  latencia que realmente sufrió el usuario, y `intento_proveedor` conserva el
  desglose por intento.
- `contenido_intento` es DEBUG a propósito: con el nivel INFO de producción
  el contenido sensible no llega al log. Subilo solo cuando estés depurando.
- Los fallos del proveedor se registran sanitizados (`provider_error` con tipo
  y status), nunca con el cuerpo crudo de la respuesta del proveedor.

## Tests y lint

```bash
uv run pytest -q      # suite completa
uv run ruff check .   # lint
uv run ruff format .  # formato
```

La suite cubre, entre otros: el invariante CAG (ejemplos en el system prompt), que la transcripción viaja como dato delimitado, que una entrada fuera de límites NO llama al LLM (no se gasta un token), que un error del proveedor no se filtra al cliente, que `/health` responde aunque falte la API key, el coste (tokens reales × precios de LLMPrice, con `None` + nota cuando no se puede calcular), y **la estructura de carpetas del ejercicio** (`test/test_estructura.py`). Los tests mockean el LLM: corren sin keys.

## Validación automática (pipeline)

"Que la estructura sea la correcta y el servicio funcione" se valida con dos
mecanismos:

1. **Tests locales** (basta `uv run pytest -q`): `test/test_estructura.py`
   falla si falta/queda renombrada una carpeta o archivo que el ejercicio
   exige, o si la transcripción canónica deja de ser un parámetro válido.
   El resto de la suite prueba el flujo completo (recibir → inyectar contexto
   → "LLM" → respuesta) con proveedores mockeados, así el pipeline corre sin
   API keys.
2. **CI automático** (`.github/workflows/ci.yml` en la **raíz del monorepo**):
   se ejecuta en cada `push` y `pull_request` (filtrado a `lidr_2/**` en paths)
   con la misma suite + lint. `uv sync --locked` falla si `uv.lock` está
   desincronizado del `pyproject.toml` — así "se me olvidó regenerar el lock"
   es un error de CI hoy y no una sorpresa después.

> Por qué en la raíz: GitHub Actions solo descubre workflows en el
> `.github/workflows` de la raíz del repo (aunque sí recorre sus subcarpetas).
> Un `lidr_2/.github/workflows/ci.yml` nunca se ejecuta — de hecho el pipeline
> no corrió hasta que se movió. `defaults.run.working-directory: lidr_2`
> ejecuta la suite dentro de este proyecto.

```text
push / pull_request
  └─ CI (ubuntu) ──→ uv sync --locked --all-groups
                   └─ pytest -q   (incl. test_estructura.py)
                   └─ ruff check .
```

## Estructura

```
app/
├── main.py            # FastAPI, /health independiente de la configuración
├── config.py          # Settings: defaults coherentes por proveedor
├── context/           # el cache CAG: ejemplos + build_system_prompt
├── services/          # dominio: LLMServiceError, delimitador, truncado, streaming
├── providers/         # adaptadores OpenAI/Anthropic (interfaz común)
└── routers/           # validación de entrada + traducción a HTTP (JSON y SSE)
streamlit_app.py       # cliente de turno único; el parser SSE vive acá
test/                  # suite + test/test_estructura.py (valida este árbol)
datos/transcripcion_reunion.md   # la transcripción canónica (parámetro)
../.github/workflows/ci.yml      # pipeline (raíz del monorepo, scope lidr_2/**)
```

`test/test_estructura.py` verifica esta estructura de forma automática: si
alguien renombra `test/`, mueve `datos/` o borra un archivo que el ejercicio
pide, el pipeline queda en rojo.

Capas: router → service → context/config/providers. Ningún router conoce el SDK, ningún servicio conoce HTTP.

## Alcance (y deuda declarada)

- **No hay CORS, y ahora hay una razón real**: el cliente de la UI es `httpx`
  dentro del proceso de Streamlit, no el navegador del usuario (que habla con
  Streamlit por su websocket). El navegador nunca cruza el origen, así que CORS
  no aplica. Si algún día la UI se sirve como HTML plano desde otro origen,
  ahí sí va CORS con `allow_origins` explícito desde Settings.
- **No hay base de datos**: decisión del ejercicio, no algo pendiente.
- **No hay autenticación**: la API es local/de aprendizaje.
- El coste está acotado: la entrada tiene techo (`ESTIMATION_MAX_CHARS`, validado antes de llamar al LLM), el cliente tiene `timeout`/`max_retries` explícitos y la salida tiene `max_tokens`.
- **El coste se reporta por llamada** (`cost_usd` + `cost_note`) usando la base de precios de LLMPrice (snapshot local, sin red). No hay `GET /usage`: no es parte del alcance del ejercicio.