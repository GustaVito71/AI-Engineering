# Estimator — Servicio IA de estimación de software

Servicio IA en FastAPI que estima proyectos de software a partir de un formulario tipado. Es la pieza Python del programa **Master en AI Engineering**: un endpoint pensado para ser consumido por un backend de negocio (Rails, Streamlit u otro), no por un usuario final.

A partir de la **Sesión 04** el contrato es deliberadamente estrecho:
- entrada tipada (`description` + tres enums),
- salida en texto libre (`/api/v1/estimate`), estructurada y validada (`/api/v1/estimate/structured`) o estructurada con su presentación, también validada (`/api/v1/estimate/rendered`),
- prompt fuera del código, versionado en `app/prompts/<use_case>/<version>/`: plantillas Jinja2 (`system.j2`, `user.j2` y `examples.j2`, que `system.j2` incluye con `{% include %}`) más un `examples.yaml` con los datos de esa versión.

La inteligencia adicional (guardrails, cache semántico) se construye encima de esta base en directo.

## Estado

| Pieza | Estado |
|---|---|
| Contrato de entrada (`EstimationRequest` / `EstimationResponse`) | Hecho (WU2) |
| Prompts versionados (`system.j2`, `user.j2`, `examples.j2`, `examples.yaml`, `loader.py`) | Hecho (WU4) |
| Wrapper LLM (`app/services/llm_wrapper.py`), caché, router `POST /api/v1/estimate`, errores y trazabilidad | Hecho (WU5) |
| Servicio de estimación (`app/services/estimation.py`): el pipeline entre el router y el wrapper, donde entran los guardrails | Hecho |
| Guardrails de entrada (`app/guardrails/input.py`): inyección de prompt, datos personales y moderación de OpenAI, en los tres endpoints | Hecho (WU9 A) |
| Cliente Streamlit | Hecho (WU6) |
| Slice de punta a punta: formulario → API → proveedor → caché → formulario | Hecho (WU7) |
| Salida estructurada: `StructuredResult` con Instructor sobre el Router, prompt `v3`, `POST /api/v1/estimate/structured`, interruptor en el formulario | Hecho, sin streaming (WU8) |
| Salida renderizada: `RenderedResult` (la presentación de `output_format` en `rendered`, validada contra las cifras), prompt `v4`, `POST /api/v1/estimate/rendered`, segundo interruptor en el formulario | Hecho |
| Streaming de la salida estructurada (`ijson`, eventos por fase) | Pendiente (WU8) |
| Guardrail de salida (horas contra el equipo, aviso de confianza baja), `IncompleteJSONError` y política de reintento | Hecho (WU9 B) |
| Caché semántica (`app/services/semantic_cache.py`): embeddings en Redis plano, umbral, mismas cifras, solo estimaciones sanas, modo `log_only`; opt-in | Hecho (WU10) |

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
| 422 | La entrada no cumple el contrato (longitud de `description` medida sin los espacios de los extremos, valores de los enums, campos que no están en el contrato) | Lista de errores de validación de Pydantic |
| 422 | `?prompt_version=` pide una versión que no existe | *"La versión de prompt 'v9' no existe. Versiones disponibles: v1, v2."* |
| 422 | Un guardrail de entrada rechazó la descripción (instrucciones dirigidas al modelo, o contenido que marca la moderación) | Mensaje fijo por regla, sin repetir lo que escribió el usuario |
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

- **Qué es un error y qué un aviso.** Los rangos de cada campo (mínimos y topes: entre 1 y 8 fases, hasta 52 semanas y 1.000.000 EUR por fase, hasta 104 semanas en total, entre otros), los nombres de fase únicos y la coherencia del rechazo son condiciones del schema: si no se cumplen, se vuelve a preguntar. Los totales (horas y coste) que se alejan **más de 5 %** de la suma de las fases también son un error del schema (`TOTALS_MAX_DEVIATION_PCT`): se vuelve a preguntar, y el mensaje nombra qué total y con qué cifras. Una diferencia de **hasta 5 %** no es un error: los totales los declara el modelo (PLAN.md §3.4) y la estimación se devuelve igual, con el aviso en `warnings` y el evento `totales_no_cuadran` en el log. La duración total no se compara: las fases pueden solaparse.
- **Descripciones que no alcanzan.** Si el modelo no puede estimar con al menos 30 % de confianza, lo dice: `summary` empieza con «Fuera de alcance:» y explica qué falta, y la estimación es una sola fase «Sin estimar» con todo en cero. Es una respuesta 200, no un error; queda en el log como `estimacion_fuera_de_alcance`, y el formulario muestra solo la explicación. Una confianza baja sin el prefijo, o el prefijo con cifras, se vuelve a pedir.
- **Resumen y confianza.** `summary` y `confidence_pct` resumen la estimación completa, y cada fase trae su propio `summary` con lo que se hace en ella. El prompt pide calcular los totales a partir de las fases, sumar y comprobar antes de responder (bloque `<totals>`).
- **`cached`** dice si la estimación salió de la caché, sin llamada al proveedor.
- **`output_format` no cambia la respuesta.** El modelo devuelve siempre la misma estructura; el formulario decide si mostrarla como tabla, partidas o narrativa (en `/estimate/rendered` sí la cambia: ver **Salida renderizada**). `detail_level` sí cambia: `summary` deja `assumptions` y `risks` vacíos, `medium` agrega supuestos y `detailed` agrega supuestos y riesgos con su mitigación.
- **Cada intento se paga.** El coste y los tokens de la respuesta suman todos los intentos, no solo el aceptado.
- **Origen.** El resumen general y por fase, la confianza global con «Fuera de alcance:», los topes, el bloque `<totals>` y `cached` se tomaron de la solución de referencia `session_4_live/estimator`. De ella no se tomó la llamada a Instructor sobre `litellm.completion` (perdería el respaldo del Router), la tabla de precios escrita a mano ni re-preguntar cuando los totales no cuadran.
- **Versiones.** Solo acepta versiones de salida estructurada (hoy, `v3`); sin `?prompt_version=` usa `STRUCTURED_PROMPT_VERSION`. Pedirle `v2` o `v4`, o pedirle `v3` a `/estimate` o a `/estimate/rendered`, da un 422 que dice en qué endpoint pedirla.

Errores propios, además de los de `/estimate`:

| Código | Cuándo | `detail` |
|---|---|---|
| 422 | `?prompt_version=` pide una versión de texto libre | *"La versión de prompt 'v2' es de texto libre: pedila en POST /api/v1/estimate. Versiones disponibles en este endpoint: v3."* |
| 502 | Ningún intento del modelo cumplió el schema (incluida la regla de horas contra el equipo) | Mensaje genérico en español; intentos, coste y último error van al log |
| 502 | La respuesta llegó cortada por `max_tokens` (`IncompleteJSONError`) | Mensaje propio, sin reintentar; intentos, coste y `max_tokens` van al log |
| 503 | `STRUCTURED_PROMPT_VERSION` no es una versión de salida estructurada publicada | Nombra la variable y las versiones disponibles |

### Salida renderizada

`POST /api/v1/estimate/rendered` recibe el mismo cuerpo y devuelve lo mismo que `/estimate/structured`, validado contra `RenderedResult`: la estimación trae además `rendered`, la presentación en Markdown que pidió `output_format`, escrita por el modelo.

```json
{
  "estimation": {
    "phases": ["…"], "team": ["…"], "totals": {"hours": 565, "cost_eur": 34050.0, "duration_weeks": 9.0},
    "summary": "…", "confidence_pct": 75,
    "rendered": "| Fase | Semanas | Horas | Coste (EUR) | Confianza (%) |\n|---|---|---|---|---|\n| Descubrimiento | 1 | 70 | 3.950 | 85 |\n…\n| **Total** | 9 | 565 | 34.050 | 75 |"
  },
  "prompt_version": "v4",
  "cached": false,
  "warnings": []
}
```

- **`output_format` entra al prompt.** El prompt es parte del contrato: estructura fija (las plantillas), variables de la request (la descripción) y parámetros del usuario (tipo, nivel de detalle y formato). En `v4`, el bloque `<output_format>` pide en `rendered` una tabla de fases, una lista numerada con una partida por fase o un párrafo por fase, cada una con su línea de totales. El texto va en un campo propio del objeto, así que no compite con el schema.
- **Guardrail de salida.** El validador de `RenderedResult` recibe `output_format` por el contexto de validación de Instructor (`context={"output_format": …}`) y comprueba la forma pedida y que las horas y el coste de cada fase y de los totales aparezcan en `rendered` con las mismas cifras que los campos (en formato castellano, `34.050`, o sin separador, `34050`). Si no, Instructor vuelve a preguntar con el error; agotados `STRUCTURED_MAX_RETRIES` reintentos, 502. Un rechazo («Fuera de alcance:») lleva en `rendered` el mismo texto que `summary`, sin tabla ni lista.
- **Lo demás es igual que en `/estimate/structured`.** Los totales que no cuadran hasta 5 % siguen siendo un aviso, y por encima se re-pregunta (el validador compara `rendered` con los campos, no con la suma de las fases), y la caché, el respaldo y los errores del proveedor son los mismos.
- **Versiones.** Solo acepta versiones de salida renderizada (hoy, `v4`); sin `?prompt_version=` usa `RENDERED_PROMPT_VERSION`. Los tres endpoints rechazan con 422 las versiones de los otros dos y dicen dónde pedirlas; `RENDERED_PROMPT_VERSION` inválida es un 503 que nombra la variable.

### Cliente Streamlit

El cliente Streamlit es un formulario que construye el JSON y muestra la estimación recibida. Consume la API por HTTP y queda en `http://localhost:8501` al levantar el compose.

La URL de la API se lee de `ESTIMATOR_API_BASE_URL`; en Docker es `http://api:8001`.

En la barra lateral, **Versión del prompt** elige con qué versión se estima: "Predeterminada (v2)" deja que decida `PROMPT_VERSION` en el servicio, y cada versión publicada se pide con `?prompt_version=`. La lista sale de `prompt_versions` en `/health`, así que una versión nueva aparece sola. Si la API no respondía al abrir la página, queda solo "Predeterminada"; **Probar conexión** la vuelve a leer.

Con el interruptor **Salida estructurada** el formulario llama a `/api/v1/estimate/structured` y el selector ofrece las versiones estructuradas (`structured_prompt_versions` en `/health`). La estimación se muestra con el resumen y los totales arriba (horas, coste, duración y confianza), las fases en el formato elegido (tabla, partidas o narrativa) con la descripción de cada una, el equipo y, si el nivel de detalle los pide, los supuestos y riesgos de cada fase. Si el modelo rechazó estimar, se muestra solo la explicación de qué falta. Si la estimación salió de la caché, lo indica debajo.

Debajo está el interruptor **Salida renderizada**: el formulario llama a `/api/v1/estimate/rendered` y el selector ofrece las versiones renderizadas (`rendered_prompt_versions` en `/health`). Se muestran el resumen y los totales, el `rendered` del modelo tal cual, el equipo y los supuestos y riesgos. Los dos interruptores son excluyentes: encender uno apaga el otro, y con los dos apagados la salida es texto libre.

## Cómo testar

Los tests corren fuera de Docker, con las dependencias de desarrollo:

```bash
cd lidr_4
uv sync
uv run pytest
uv run ruff check . && uv run ruff format --check .
```

La batería corre en unos segundos, sin red y sin Redis (las llamadas al LLM se simulan con `mock_response` de LiteLLM y Redis con fakeredis):

- `test/test_schemas.py` — validaciones del `EstimationRequest` (longitudes, enums, campos obligatorios, descripción en blanco o de relleno, recorte de los extremos, campos fuera del contrato) y de `Settings` (techo del operador sobre `description`, `LOG_LEVEL`).
- `test/test_prompts.py` — render de la versión `v1`:
  - **Plantilla:** `description` dentro de `<project_description>`, bloques condicionales por `output_format` y `detail_level`, `StrictUndefined` falla temprano, una versión inexistente lanza `TemplateNotFound`, el prompt empieza sin líneas en blanco, y `system.j2` incluye el `examples.j2` de su propia versión (también en una versión copiada).
  - **Ejemplos:** horas y costes calculados desde las tarifas y redondeados (5 h / 50 €), totales que cuadran con las fases, una fila por línea en los tres formatos, numeración de `line_items` desde 1 en cada ejemplo, resumen de equipo armado con los `label`/`plural` del YAML.
  - **Validación del YAML** (con una versión temporal en `tmp_path`): rol no declarado en `rates` → error; rol sin `label` → error; un rol nuevo aparece en `<scope>`, partidas y resumen sin tocar Python.
  - **Caché:** el YAML se lee y se calcula una sola vez por versión, y cada llamada recibe su propia copia.
- `test/test_prompts_v2.py` — render de la versión `v2`: las 36 combinaciones sin texto en inglés, regla de idioma en castellano, tarifas con coma decimal, tabla, elementos de línea y narrativo con números en formato castellano, concordancia de singular y plural, y la misma aritmética que `v1`.
- `test/test_frontend.py` — cliente HTTP del formulario con transporte mockeado: el endpoint es `/api/v1/estimate` (no `/stream`), el payload son las cuatro claves del contrato, un 422 de FastAPI llega como lista de `msg`, y los `warnings` de la respuesta llegan al cliente (vacíos si la API no los envía). Versión del prompt: la elegida viaja en `?prompt_version=` y la predeterminada no manda el parámetro; las opciones salen de `/health` y, si la API no responde o es anterior a `prompt_versions`, queda solo la predeterminada.
- `test/test_e2e.py` — el slice de punta a punta (WU7) con todas las piezas reales: el formulario de Streamlit (`_build_payload` + `_estimate`), la API con su lifespan, el prompt `v2`, el wrapper con el Router de LiteLLM, la caché y la limpieza de etiquetas. Solo se simulan el proveedor (`mock_response`) y Redis (fakeredis). Cubre el camino normal (prompt que recibe el proveedor, texto limpio, trazabilidad), la caché (acierto, otra descripción, texto cacheado que se limpia al responder), la configuración (sin clave de respaldo, sin clave del primario, sin Redis) y los fallos (504 y 502 en castellano con el detalle solo en el log, un fallo que no queda en caché, un 422 que no llega al proveedor). Incluye el selector de versión: las opciones salen del `/health` real y la versión elegida decide el prompt que recibe el proveedor.

- `test/test_estimation_service.py` — el servicio de estimación con un wrapper falso: la caché exacta (un acierto no renderiza ni llama al wrapper, emite el evento con `desde_cache`, la clave lleva modelo, versión, parámetros y la descripción ya oculta, entradas rotas cuentan como fallo, un rechazo o un error no tocan la caché), arma el prompt de la versión pedida, quita las etiquetas que el modelo copia, junta los avisos de los guardrails, los del wrapper y los totales que no cuadran, y deja los eventos de negocio; deja pasar tal cual los errores del proveedor, y un prompt que no se puede armar sale como error interno (500) y no como un 502 del proveedor. Con los guardrails: una descripción rechazada no llega al wrapper (422 en los tres endpoints), el prompt lleva la descripción con los datos personales ocultos, y la app arma los guardrails de la configuración una sola vez. El aviso de confianza baja (con el evento, en el umbral, no en un rechazo, también desde la caché) y una entrada de caché que ya no cumple la regla de horas se descarta.
- `test/test_output_guardrails.py` — el descarte por datos personales o prompt volcado en los tres endpoints y los avisos de salida (confianza baja con umbral configurable, totales que no cuadran, fuera de alcance), la configuración y el cableado por endpoint.
- `test/test_detectors.py` — los detectores compartidos (normalización, datos personales con dígito de control, prompt volcado y que sus frases siguen en las plantillas).
- `test/test_input_guardrails.py` — los guardrails de entrada, sin red ni claves: cada regla de inyección rechaza lo suyo y no una descripción normal (incluye variantes con tildes, mayúsculas, letras de ancho completo y caracteres invisibles), las etiquetas prohibidas cubren las que usan las plantillas, los datos personales se ocultan y las cifras del proyecto no (IBAN, tarjeta y DNI/NIE con su dígito de control), la moderación recibe el texto ya oculto, rechaza lo que marca y falla abierta, ningún evento lleva el texto de la descripción, y el adaptador de OpenAI se prueba contra un transporte HTTP falso. También la configuración: valores por defecto, vacío = no configurado, y qué queda apagado en cada caso.
- `test/test_llm_wrapper.py` — el wrapper LLM:
  - **Claves:** cada deployment recibe la clave de su proveedor; si falta la del primario, `LLMConfigurationError` nombrando la variable; si falta la del respaldo, arranca solo con el primario, con aviso en el log y en `warnings`.
  - **Router y respaldo:** el Router tiene los dos modelos, y si el primario falla responde el de respaldo.
  - **Resultado:** `model` es el nombre del modelo (no el id interno del deployment), `provider` coincide, se respeta `prompt_version` y el coste se calcula.
  - **Configuración:** reintentos, timeout y `max_tokens` salen de `Settings`.
  - **Coste:** si `completion_cost` falla, el coste es 0 y la respuesta sale igual.
  - **Trazabilidad:** evento `estimacion_completada` en llamada normal, con respaldo y el prompt nunca va al log; el wrapper no guarda caché y siempre llama al proveedor (el evento de los aciertos lo emite el servicio).
- `test/test_semantic_cache.py` — la caché semántica con fakeredis y embeddings a mano: acierta con lo parecido, no con otro espacio, otro modelo de embeddings, otras cifras ni por debajo del umbral; solo guarda estimaciones sanas; TTL, tope de entradas, `log_only`; fail soft (embedding o Redis caídos, entradas corruptas o de otra dimensión); la configuración; el adaptador de embeddings; y en el servicio, que busca solo si la exacta falló, copia el acierto a la exacta y avisa.
- `test/test_cache.py` — la caché: ida y vuelta, TTL, clave que cambia con cada parte (modelo, versión, parámetros, descripción), entrada corrupta, Redis caído (lectura, escritura y una estimación completa), caché desactivada, y que la app abra **un solo** cliente de Redis y el wrapper use ese.
- `test/test_estimate_endpoint.py` — el endpoint con la app real: 200 normal, 503 si falta la clave del primario (sin filtrar la clave configurada), estimación con aviso si falta la del respaldo, `/health` con y sin claves, 502/504 con mensaje limpio ante fallos del proveedor y el detalle en el log; `?prompt_version=` (sin query se usa `PROMPT_VERSION`, la query la pisa, una versión inexistente o con `..` da 422 sin llamar al modelo); `/health` lista las versiones disponibles (`prompt_versions`); una descripción en blanco es 422 sin llamar al modelo, el prompt recibe la descripción recortada y un campo fuera del contrato es 422 en los tres endpoints.
- **Salida estructurada (WU8):**
  - `test/test_structured_schema.py` — qué rechaza el schema (y se vuelve a preguntar: mínimos, topes, textos, rechazos mal formados, fases estimadas en cero) y qué es solo un aviso (totales que no cuadran hasta 5 %, con los números en formato castellano; por encima de 5 % se rechaza); un rechazo bien formado se acepta; `phases` va primero y el resumen al final. La regla de horas contra el equipo: fase y total, el borde (40 h por persona y semana), que el tope crece con el equipo y suma los roles. Las tres de alucinación: coste por hora (con su margen y que las tarifas siguen siendo las de las plantillas), duración total entre la fase más larga y la suma, y las cifras con unidad de los resúmenes.
  - `test/test_prompts_v3.py` — `v3` es la única versión estructurada y `v1`/`v2` no se tocaron; los ejemplos del prompt son JSON válido contra el schema y con totales que cuadran, con resumen y confianza, y el último es un rechazo en cero; el bloque `<totals>` y las reglas de rechazo salen de las constantes del schema; `detail_level` decide supuestos y riesgos y `output_format` no cambia el prompt; las 36 combinaciones en castellano; regla anti-instrucciones.
  - `test/test_llm_wrapper_structured.py` — Instructor sobre el Router: respuesta válida, respuesta inválida o JSON roto que se vuelve a pedir con el error, intentos agotados, coste y tokens que suman todos los intentos (también en el error), fallos del proveedor que llegan como excepciones de LiteLLM, respaldo, caché y trazabilidad. Horas imposibles que se re-preguntan o agotan los intentos; respuesta cortada por `max_tokens`: no se reintenta, trae el coste y el presupuesto, también tras un reintento.
  - `test/test_estimate_structured_endpoint.py` — 200, avisos de totales, 422 por versión del otro tipo o inexistente (en los dos endpoints), 503 por `STRUCTURED_PROMPT_VERSION` inválida o sin clave, 502 por salida inválida o fallo del proveedor, 504, `/health` y OpenAPI. 502 con mensaje propio y campos en el log para una respuesta cortada.
  - `test/test_frontend_structured.py` — cliente HTTP del endpoint estructurado, selector de versión y lo que se muestra (tabla, partidas, narrativa, equipo, números en castellano).
  - `test/test_e2e_structured.py` — el slice completo con Instructor y el Router reales: del formulario a la tabla, reintento, intentos agotados (502), caché, texto y estructurada sin compartir caché, aviso de totales (hasta 5 %), reintento por un total muy desviado y selector desde `/health`.
- **Salida renderizada:**
  - `test/test_rendered_schema.py` — el validador de `rendered`: cada formato acepta su forma y rechaza la de los otros dos, una cifra distinta de los campos o una fase que falta se nombran, los totales tienen que aparecer, un total mal sumado sigue siendo un aviso, el rechazo es solo texto con el prefijo, sin contexto solo se validan los campos y `rendered` va al final del schema.
  - `test/test_prompts_v4.py` — `v4` es la única versión renderizada y `v3` sigue siendo la única estructurada; fuera de `<output_format>` y de los ejemplos el prompt es el de `v3`; cada `output_format` produce un prompt distinto; los ejemplos pasan el validador en los 9 casos y, sin `rendered`, son los de `v3`; las 36 combinaciones en castellano.
  - `test/test_estimate_rendered_endpoint.py` — 200, el endpoint pide `RenderedResult` con el `output_format` de la request (y `/structured` sigue sin contexto), 422 entre los tres endpoints, 503 por `RENDERED_PROMPT_VERSION`, 502, 504, `/health` y OpenAPI.
  - `test/test_frontend_rendered.py` — cliente HTTP del endpoint, modo de salida, y los interruptores excluyentes ejecutando la app con `AppTest`.
  - `test/test_e2e_rendered.py` — el slice completo con Instructor y el Router reales: el contexto llega al validador, una cifra distinta se corrige con un reintento que lleva el error, una tabla cuando se pidió narrativa agota los intentos (502), caché y formatos sin compartir caché.
- `test/test_logging.py` — los loggers de LiteLLM: con `LOG_LEVEL=DEBUG` quedan en `WARNING` y una estimación completa no deja la descripción del cliente en la salida; sus advertencias salen una sola vez, con el formato de structlog, y sin claves de API.

Los tests no dependen del `.env` ni de las variables de entorno de quien los corre: `test/conftest.py` los aísla y cada test fija lo que usa. Los que dependen del proveedor corren dos veces, con OpenAI y con Anthropic como primario.

**Medición con el modelo real (`test/medir_v4.py`).** No es un test: llama al proveedor, cuesta dinero y no tiene aprobado ni fallo, por eso no empieza con `test_` y pytest no lo recoge. Corre los 9 casos (detalle × formato) `-n` veces (3 por defecto) con el modelo y las claves del `.env`, sin caché, y anota por ejecución los intentos y el motivo de cada reintento (si el validador de `rendered` rechaza respuestas correctas), el coste, los avisos de totales, las fases con coste por hora fuera de las tarifas del prompt, las cifras sin el redondeo del prompt y, entre repeticiones, la variación de horas y coste. Desde la raíz del proyecto: `uv run python test/medir_v4.py` (escribe `medir_v4.csv`, una fila por ejecución, y `medir_v4_fases.csv`, una fila por fase, para ver cuál queda fuera de tarifa o sin redondear). No corre dentro de Docker: la imagen no incluye `test/`.

## Estructura del proyecto

```
lidr_4/
├── app/
│   ├── __main__.py                    # Lanzador: python -m app [--port N] [--reload]
│   ├── main.py                        # FastAPI app, lifespan, /health, parche OpenAPI
│   ├── config.py                      # Settings (Pydantic Settings, .env)
│   ├── cache.py                       # Cliente Redis único (lifespan), lectura/escritura fail soft
│   ├── tracing.py                     # emit(): eventos de trazabilidad con structlog
│   ├── dependencies.py                # LLMWrapper perezoso; la caché del lifespan va al servicio
│   ├── routers/
│   │   └── estimations.py             # POST /api/v1/estimate, /estimate/structured y /estimate/rendered, errores 502/504
│   ├── schemas/
│   │   ├── estimation.py              # EstimationRequest, EstimationResponse, enums
│   │   └── structured_estimation.py   # StructuredResult y RenderedResult (contratos del LLM) y sus respuestas
│   ├── prompts/
│   │   ├── loader.py                  # Carga examples.yaml, aritmética genérica, render
│   │   └── estimation/
│   │       ├── v1/                    # estimación en inglés
│   │       │   ├── system.j2          # rol + reglas + bloques condicionales; incluye examples.j2
│   │       │   ├── user.j2            # bloque <project_description>
│   │       │   ├── examples.j2        # presentación de los few-shot según output_format y detail_level
│   │       │   └── examples.yaml      # roles (label, plural, tarifa), redondeo, datos de los few-shot
│   │       ├── v2/                    # la misma estimación en castellano (versión por defecto)
│   │       ├── v3/                    # salida estructurada (JSON), en castellano
│   │       └── v4/                    # salida estructurada con la presentación de output_format en rendered
│   ├── guardrails/
│   │   ├── input.py                   # Entrada: inyección, datos personales, moderación
│   │   ├── output.py                  # Salida: descarta PII y prompt volcado; avisos de confianza baja y totales
│   │   └── detectors.py               # Detectores puros compartidos (normalización, datos personales, prompt volcado)
│   └── services/
│       ├── cache.py                   # Caché exact-match (la usa EstimationService; clave sin render)
│       ├── semantic_cache.py          # Caché semántica: embeddings, Redis plano, coseno en la app (WU10)
│       ├── estimation.py              # EstimationService: guardrails de entrada, render del prompt, llamada, eventos y avisos
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
│   ├── test_prompts_v4.py
│   ├── test_rendered_schema.py
│   ├── test_estimate_rendered_endpoint.py
│   ├── test_frontend_rendered.py
│   ├── test_e2e_rendered.py
│   ├── test_estimation_service.py
│   ├── test_input_guardrails.py
│   ├── test_output_guardrails.py
│   ├── test_detectors.py
│   ├── test_llm_wrapper.py
│   ├── test_cache.py
│   ├── test_semantic_cache.py
│   ├── test_estimate_endpoint.py
│   ├── test_logging.py
│   ├── test_frontend.py
│   ├── test_e2e.py
│   └── medir_v4.py                    # Medición con el modelo real (no es un test)
├── Dockerfile                         # Imagen única para la API y el frontend (dependencias de uv.lock)
├── docker-compose.yml                 # Redis + API + frontend
├── .dockerignore                      # Deja afuera de la imagen el .env, los tests y los caches
├── streamlit_app.py                   # Formulario que consume /api/v1/estimate (y /structured o /rendered)
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
| `v4` (salida renderizada) | Castellano | El prompt de `v3` con un bloque `<output_format>` que pide la presentación en el campo `rendered` (tabla, partidas o narrativa, con las mismas cifras en formato castellano). Los ejemplos son los de `v3` más su `rendered` en el formato pedido, armado con el filtro `number` del loader (la misma función que usa el validador) |

**Tipo de salida.** Cada versión declara su tipo con la clave `output` de su `examples.yaml`: `text` (el valor por defecto, así que `v1` y `v2` no se tocaron), `structured` o `rendered`. `available_versions(output)` filtra por tipo y cada endpoint acepta solo las suyas: `/estimate` las de texto, `/estimate/structured` las estructuradas y `/estimate/rendered` las renderizadas. Una versión nueva de cualquier tipo entra declarando su `output`, sin tocar código.

La versión se elige por petición con `?prompt_version=` o, si no se indica, con `PROMPT_VERSION` (o `STRUCTURED_PROMPT_VERSION` en `/estimate/structured` y `RENDERED_PROMPT_VERSION` en `/estimate/rendered`). El endpoint solo acepta versiones publicadas (un directorio `vN/` con `system.j2`); el valor nunca llega a armar una ruta de archivo sin haberse comparado antes con esa lista.

Lo que vive **fuera** de la versión (en código): el contrato (`EstimationRequest`), el switch de versión, el wrapper y la aritmética de los ejemplos (`loader.py`). Todo lo demás (rol del modelo, reglas, ejemplos, tarifas, formatos de salida, niveles de detalle) vive en `v1/`. Si para cambiar el comportamiento del modelo hay que tocar Python, la separación está rota.

## Llamada al LLM, caché y trazabilidad

**Tres capas.** El router (`app/routers/estimations.py`) solo traduce HTTP: la versión pedida en la query y los errores a 422/502/504. El servicio (`app/services/estimation.py`, `EstimationService`) es el pipeline: guardrails de entrada → caché exacta → caché semántica (WU10, opt-in) → render → LLM → guardrails de salida → escritura en las cachés, y deja los eventos de negocio y los avisos. El wrapper (`app/services/llm_wrapper.py`) solo habla con el proveedor: no guarda caché.

**Guardrails de entrada (WU9 A).** Sobre la descripción, en este orden, en los tres endpoints (`app/guardrails/input.py`; los detectores puros están en `app/guardrails/detectors.py`):

1. **Inyección de prompt** (regex, local): instrucciones dirigidas al modelo («ignora las instrucciones anteriores»), pedidos de revelar el prompt, cambios de rol y etiquetas o marcadores del propio prompt (`</project_description>`). **Rechaza** con 422; la regla que saltó queda en el log.
2. **Datos personales** (regex y dígitos de control, local): email, teléfono, IBAN, tarjeta (Luhn), DNI/NIE, CUIT/CUIL. **Los oculta** con un marcador (`[EMAIL]`) y la estimación sigue, con un aviso en `warnings`. El prompt, la caché y el log ven solo el texto ya oculto.
3. **Moderación** de OpenAI (el endpoint no se cobra): se consulta con el texto ya oculto y **rechaza** con 422 lo que marca. Necesita `OPENAI_API_KEY`; sin clave queda apagada y el resto sigue. **Falla abierta**: si OpenAI no responde en `MODERATION_TIMEOUT`, la descripción sigue y queda `moderacion_no_disponible` en el log.

Son heurísticas, no una garantía: una inyección redactada de otra forma o en otro idioma pasa, y una descripción legítima que contenga una frase como «ignora las reglas anteriores» se rechaza. `entrada_rechazada` deja la regla para medir ambos errores. Los límites de cada regla están en el docstring del módulo.

**Guardrails de salida (WU9 B).** Sobre la estimación, en los endpoints estructurados:

1. **Horas contra el equipo (rechaza y re-pregunta).** Ninguna fase, ni el total, puede pasar de 40 h por persona y semana con todo el equipo trabajando (`team` suma las personas; las horas productivas de los prompts son 32). Es un validador del schema (`_hours_fit_team`): Instructor le devuelve el error al modelo, y agotados los intentos sale el 502 de siempre. Una entrada de la caché que ya no lo cumple cuenta como fallo y se vuelve a pedir.
2. **Confianza baja (avisa).** Con confianza global entre 30 % (umbral de rechazo) y 70 %, la estimación se devuelve con un aviso en `warnings` y el evento `confianza_baja`. El aviso se calcula siempre, también con un acierto de caché (`app/guardrails/output.py`). El umbral es `LOW_CONFIDENCE_WARN_PCT` y se apaga con `OUTPUT_GUARDRAILS_ENABLED=false` (solo avisos y eventos; las reglas del schema siguen).
3. **Datos personales y prompt volcado (descarta, 502).** Sobre todos los textos de la salida (en los dos endpoints estructurados, `rendered` incluido, y en el de texto): si traen un email, teléfono, IBAN, tarjeta o documento, o las etiquetas de las plantillas (`<rules>`, `</scope>`…) o frases fijas de las instrucciones, la respuesta se descarta con un 502 de mensaje fijo, sin re-preguntar (con el mismo prompt el riesgo no cambia y cada intento se paga) y sin escribir en la caché. Corre antes de guardar, así que una entrada de la caché ya pasó la comprobación. Son heurísticas: un dato con otro formato o instrucciones parafraseadas pasan (`app/guardrails/output.py`, detectores en `app/guardrails/detectors.py`).
4. **Alucinación de bajo nivel (rechaza y re-pregunta).** Tres validadores del schema, todos locales: el coste por hora de cada fase tiene que caer entre las tarifas de los prompts (50 y 62,50 EUR/h, con 5 % de margen por el redondeo); la duración total no puede ser mayor que la suma de las fases ni menor que la fase más larga (5 % de margen); y las cifras con unidad que cita un resumen (`390 horas`, `23.950 EUR`, `6 semanas`) tienen que ser las de las fases o los totales. Cada mensaje lleva el valor que esperaba, para que el modelo lo corrija. Los resúmenes con cifras sin unidad (`3 sprints`, `15 %`) o con otra unidad no se tocan. Si el modelo repite la misma respuesta en los reintentos, se agotan y sale el 502 de siempre. `OUTPUT_GUARDRAILS_ENABLED` no los apaga: viven en el schema.
5. **Respuesta cortada (`IncompleteJSONError`).** Si el proveedor corta la respuesta por `max_tokens`, no se reintenta: con el mismo presupuesto se cortaría en el mismo punto y cada intento se paga. Sale un 502 con mensaje propio y el log lleva `max_tokens` y el coste gastado. Se arregla subiendo `LLM_MAX_TOKENS`.

**Política de reintento.** Cada fallo tiene un dueño: una respuesta que no cumple el schema la re-pregunta Instructor (`STRUCTURED_MAX_RETRIES`); un fallo del proveedor lo reintenta el Router y, si no alcanza, responde el respaldo (`LLM_MAX_RETRIES`); una respuesta cortada no se reintenta. Nada se escribe en la caché si no fue una respuesta válida.

**Router con respaldo.** `llm_wrapper.py` arma un Router de LiteLLM con dos deployments, `PRIMARY_MODEL` y `FALLBACK_MODEL`, cada uno con la clave de su proveedor. Las llamadas van al primario; si falla tras `LLM_MAX_RETRIES` reintentos, responde el de respaldo. El Router es el único dueño de reintentos y respaldo (PLAN.md §2). El wrapper se construye en la primera request, no al arrancar, para que el servicio levante aunque falte una clave. Si falta la clave del respaldo, el Router se arma solo con el primario y cada respuesta lleva el aviso en `warnings`.

**Caché semántica (WU10).** Encendida por defecto (`SEMANTIC_CACHE_ENABLED=false` la apaga). Sirve la estimación de una descripción muy parecida a otra ya estimada, sin llamar al modelo. Solo en `/estimate/structured` y `/estimate/rendered`: son las que se validan, y a otras descripciones solo se reparte lo que pasó la validación; `/estimate` (texto libre) usa solo la exacta. Se busca solo si la exacta falló, y un acierto exige tres cosas: el mismo espacio de búsqueda (modelo, versión del prompt, tipo, nivel de detalle y, en v4, formato de salida), una similitud coseno por encima de `CACHE_SEMANTIC_THRESHOLD` y las mismas cifras en la descripción (los embeddings casi no distinguen «20.000 usuarios» de «200.000», y la estimación cambia mucho; «B2B» o «i18n» no cuentan como cifras). **Qué se guarda:** solo estimaciones sanas: no un rechazo, con confianza por encima de `LOW_CONFIDENCE_WARN_PCT` y con los totales cuadrados; una estimación dudosa se devuelve a quien la pidió con su aviso, pero no se reparte a las parecidas. Un acierto se copia a la caché exacta de esa descripción y devuelve un aviso (`Estimación reutilizada de una consulta muy parecida (similitud N %)…`); `estimacion_completada` lleva `tipo_cache=semantica`. Los vectores van en Redis plano (un hash por espacio con el vector normalizado y las cifras, y una clave por estimación), sin RediSearch: la similitud se calcula en la app, y `SEMANTIC_CACHE_MAX_ENTRIES` acota el coste. Cada miss de la exacta paga un embedding antes de llamar al modelo. Fail soft: un fallo del embedding o de Redis es un miss. Eventos: `cache_semantica_acierto`, `cache_semantica_fallo` (con la mejor similitud), `cache_semantica_cifras_distintas`, `cache_semantica_solo_log`, `cache_semantica_no_guardada`, `cache_semantica_no_disponible`, `cache_semantica_activada` y `cache_semantica_desactivada`; ninguno lleva la descripción. **Para encenderla con criterio:** empieza con `SEMANTIC_CACHE_LOG_ONLY=true`, mira las similitudes de `cache_semantica_solo_log` con tráfico real y ajusta el umbral antes de servir aciertos.

**Caché exact-match.** Vive en el servicio y se consulta después de los guardrails y antes del render y de la llamada. La clave es un hash del modelo, la versión del prompt, los parámetros de la request (tipo, nivel de detalle y, en texto y en v4, el formato de salida) y la descripción ya filtrada; no depende del prompt renderizado, porque la búsqueda va antes del render. Las versiones publicadas son inmutables, así que la versión invalida la caché: editar una plantilla sin subir de versión ya no cambia la clave. Un acierto no renderiza, no llama al proveedor y no vuelve a escribir; emite `estimacion_completada` con `desde_cache` y los mismos campos. Una entrada incompleta o que ya no cumple el schema cuenta como fallo y se sobrescribe. Se escribe solo después de una respuesta válida. Hay un único cliente de Redis, el que crea el lifespan (`app/cache.py`), y es el que se cierra al apagar. Si Redis está caído, lento o tiene una entrada corrupta, la estimación sale igual sin caché (fail soft, con timeouts de 1 s).

**Trazabilidad.** Cada estimación deja un evento en el log, emitido con `app/tracing.py`:

| Evento | Nivel | Campos |
|---|---|---|
| `estimacion_completada` | `info` | `modelo`, `proveedor`, `uso_respaldo`, `desde_cache`, `tokens_prompt`, `tokens_completion`, `coste_usd`, `coste_evitado_usd`, `latencia_ms`, `prompt_version`. En la salida estructurada y la renderizada, además `salida` (`estructurada` o `renderizada`) e `intentos` (0 si salió de la caché); tokens y coste suman todos los intentos |
| `estimacion_fallida` | `error` | `codigo_http` (502/504), `tipo_error`, `detalle` (el mensaje real del proveedor). En la salida estructurada y la renderizada, además `salida` (`estructurada` o `renderizada`) y, si ningún intento cumplió el schema o la respuesta salió cortada, `intentos` y `coste_usd` (y `max_tokens` en la cortada) |
| `confianza_baja` | `warning` | `prompt_version`, `modelo`, `confianza_pct`. La estimación se devolvió con un aviso: confianza entre 30 y 70 % |
| `salida_rechazada` | `warning` | `regla` (`datos_personales` o `prompt_filtrado`), `prompt_version`, `modelo` y `tipos` (datos personales) o `detalle` (`etiqueta` o `frase`). Nunca lleva el texto ni el dato. La respuesta se descartó con un 502 |
| `totales_no_cuadran` | `warning` | `prompt_version`, `modelo`, `detalle` (los mismos avisos que recibe el usuario) |
| `estimacion_fuera_de_alcance` | `info` | `prompt_version`, `modelo`, `confianza_pct`. El modelo rechazó estimar porque la descripción no alcanza |
| `entrada_rechazada` | `warning` | `regla` (`ignorar_instrucciones`, `revelar_prompt`, `cambio_de_rol`, `etiquetas_del_prompt`, `marcadores_de_rol` o `moderacion`), `longitud` y, en la moderación, `categorias`. Nunca el texto |
| `datos_personales_ocultados` | `info` | `tipos` (`email`, `telefono`, `iban`, `tarjeta`, `documento`) |
| `moderacion_no_disponible` | `warning` | `tipo_error`, `detalle`. La moderación falló y la descripción siguió |
| `moderacion_desactivada` / `guardrails_desactivados` | `warning` | `motivo`. Una vez, al armar los guardrails |
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
| `PROMPT_VERSION` | `v2` | Versión de la plantilla de prompt. Entra en la clave de la caché exacta: es su mecanismo de invalidación (y también del semántico: es parte de su espacio de búsqueda) |
| `STRUCTURED_PROMPT_VERSION` | `v3` | Versión por defecto de `/estimate/structured`. Tiene que ser de salida estructurada; si no, ese endpoint responde 503 |
| `STRUCTURED_MAX_RETRIES` | `2` | Veces que se vuelve a preguntar al modelo si la respuesta no cumple el schema. Cada intento se paga. Vale también para `/estimate/rendered` |
| `RENDERED_PROMPT_VERSION` | `v4` | Versión por defecto de `/estimate/rendered`. Tiene que ser de salida renderizada; si no, ese endpoint responde 503 |
| `GUARDRAILS_ENABLED` | `true` | `false` apaga todos los guardrails de entrada (deja `guardrails_desactivados` en el log). Para medir sin ellos, no para producción |
| `OUTPUT_GUARDRAILS_ENABLED` | `true` | `false` apaga los guardrails de salida: el descarte por datos personales o prompt volcado y los avisos y eventos (totales, confianza baja, fuera de alcance). No apaga las reglas del schema, que rechazan y re-preguntan |
| `LOW_CONFIDENCE_WARN_PCT` | `70` | Confianza global (0-100) por debajo de la cual se avisa. `0` lo desactiva |
| `MODERATION_ENABLED` | `true` | Moderación de OpenAI. Solo actúa si hay `OPENAI_API_KEY`; sin clave queda apagada y se avisa una vez en el log |
| `MODERATION_MODEL` | `omni-moderation-latest` | Modelo de moderación |
| `MODERATION_TIMEOUT` | `3.0` | Segundos que se espera a la moderación (reintentos incluidos). Pasado ese tiempo la descripción sigue |
| `REDIS_URL` | `redis://localhost:6379/0` | Vacío = caché desactivada. En Docker lo fija el compose: `redis://redis:6379/0` |
| `CACHE_TTL` | `86400` | Segundos (también de la caché semántica) |
| `SEMANTIC_CACHE_ENABLED` | `true` | Caché semántica. Necesita Redis y `OPENAI_API_KEY` (clave del proveedor del modelo de embeddings); sin alguno queda apagada con un aviso en el log |
| `CACHE_SEMANTIC_THRESHOLD` | `0.95` | Similitud coseno mínima, en (0, 1]. Más alto = más estricto |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Modelo de embeddings (LiteLLM) |
| `EMBEDDING_TIMEOUT` | `3.0` | Segundos que se espera al embedding; pasado ese tiempo, la búsqueda cuenta como miss |
| `SEMANTIC_CACHE_LOG_ONLY` | `false` | Busca y deja la similitud en el log, pero nunca sirve un acierto: para calibrar el umbral |
| `SEMANTIC_CACHE_MAX_ENTRIES` | `200` | Entradas por espacio de búsqueda; se desaloja la más antigua. La similitud se calcula en la app y su coste crece con este número |
| `DESCRIPTION_MIN_CHARS` / `DESCRIPTION_MAX_CHARS` | `20` / `2000` | Techo del operador; solo puede estrechar el contrato. Los nombres anteriores, `DESCRIPCION_MIN_CHARS` / `DESCRIPCION_MAX_CHARS`, se siguen leyendo; si están los dos, manda el nuevo |
| `APP_ENV` | `local` | Se muestra en `/health` |
| `LOG_LEVEL` | `INFO` | Nivel de los logs propios. Las librerías HTTP, el SDK, LiteLLM e Instructor quedan fijas en `WARNING`: ningún nivel escribe el prompt |
| `APP_PORT` | `8001` | Puerto de `python -m app` (fuera de Docker; el compose usa 8001) |
| `ESTIMATOR_API_BASE_URL` | `http://localhost:8001` | Lo lee el cliente Streamlit. En Docker lo fija el compose: `http://api:8001` |

`get_settings()` es un singleton cacheado con `lru_cache`: cualquier cambio en `.env` requiere reiniciar el servicio (`docker compose up -d`).

---

> Este proyecto forma parte del **Master en AI Engineering** y es la base sobre la que se construye en directo el resto de la Sesión 04 (guardrails, cache semántico).
