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

| Recurso | URL |
|---|---|
| API | http://localhost:8001 |
| `/health` | http://localhost:8001/health |
| `/docs` | http://localhost:8001/docs |

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

## Tests y lint

```bash
uv run pytest -q      # suite completa (depende de cuántos tests corran)
uv run ruff check .   # lint
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
├── services/          # dominio: LLMServiceError, delimitador, truncado
├── providers/         # adaptadores OpenAI/Anthropic (interfaz común)
└── routers/           # validación de entrada + traducción a HTTP
test/                  # suite + test/test_estructura.py (valida este árbol)
datos/transcripcion_reunion.md   # la transcripción canónica (parámetro)
../.github/workflows/ci.yml      # pipeline (raíz del monorepo, scope lidr_2/**)
```

`test/test_estructura.py` verifica esta estructura de forma automática: si
alguien renombra `test/`, mueve `datos/` o borra un archivo que el ejercicio
pide, el pipeline queda en rojo.

Capas: router → service → context/config/providers. Ningún router conoce el SDK, ningún servicio conoce HTTP.

## Alcance (y deuda declarada)

- **No hay frontend ni CORS** a propósito: no existe navegador llamando a esta API. Si aparece, se agrega CORS con `allow_origins` explícito desde Settings.
- **No hay base de datos**: decisión del ejercicio, no algo pendiente.
- **No hay autenticación**: la API es local/de aprendizaje.
- El coste está acotado: la entrada tiene techo (`ESTIMATION_MAX_CHARS`, validado antes de llamar al LLM), el cliente tiene `timeout`/`max_retries` explícitos y la salida tiene `max_tokens`.
- **El coste se reporta por llamada** (`cost_usd` + `cost_note`) usando la base de precios de LLMPrice (snapshot local, sin red). No hay `GET /usage`: no es parte del alcance del ejercicio.