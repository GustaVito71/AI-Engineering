# lidr_4 — Estimador estructurado sobre LiteLLM

Quinta capa del curso. Evoluciona el estimador de `../lidr_3/` para controlar la
variabilidad de la salida. `../ganttly/` queda como backlog/playground.

Este documento es el plan de construcción: el porqué de las decisiones y el orden
de las unidades de trabajo. **El `README.md` es la referencia del estado actual**
(estructura, cómo levantar y testar, reparto de responsabilidades en los prompts,
variables de entorno). Si algo de este plan lo contradice, manda el README.

---

## 1. Objetivo

Cinco capas, en este orden de dependencia:

| # | Capa | Qué resuelve |
|---|------|--------------|
| 1 | Formulario tipado | El usuario no escribe un prompt |
| 2 | Plantillas Jinja2 versionadas | El prompt es un artefacto revisable, no un string en código |
| 3 | Salida estructurada validada | La variabilidad del LLM deja de ser un problema de parseo |
| 4 | Guardrails | Entrada y salida dejan de ser campos de confianza |
| 5 | Caché semántico | Consultas equivalentes no vuelven a llamar al modelo |

Las capas 1 y 2 eliminan variabilidad *a la entrada*. La 3 la elimina *a la
salida*. La 4 la acota. La 5 la evita.

---

## 2. Decisiones tomadas

Cada decisión lleva el porqué, porque varias parecen arbitrarias hasta que se
conoce el problema que resuelven.

| Decisión | Por qué |
|---|---|
| `EstimationRequest` alimenta la UI; el contrato del LLM aparece en WU8 | La UI puede cambiar sin tocar el prompt ni la salida del modelo |
| Los few-shot se versionan con el prompt: datos en `vN/examples.yaml`, presentación en `vN/examples.j2`, que `system.j2` incluye | Una sola fuente por versión. Los ejemplos se maquetan según el `output_format` pedido, así que se elimina la clase de bugs "el ejemplo 3 enseña un formato que el prompt ya no pide" |
| La versión por defecto es `v2`: la estimación sale en castellano | La convención del README (mensajes al usuario en castellano) alcanza también a la estimación. `v1` (inglés) sigue publicada e inmutable; `v2` usa los mismos datos y la misma aritmética |
| `total_horas` y `duracion_semanas` los calcula el modelo | Es lo que hace `../session_4/estimator`. Obliga al modelo a comprometerse con un total. Ver §3.4 |
| Todo corre en Docker: Redis, API y frontend (`docker-compose.yml`, una imagen) | Un solo modo de arranque. Redis queda dentro de la red de Docker; la API y el frontend se publican solo en `127.0.0.1`. Los tests corren fuera, con `uv` |
| LiteLLM `Router` es el **único** dueño de retry y fallback | Dos niveles de retry se multiplican y su composición es imposible de razonar |
| Gateway **async** con `router.acompletion` | El endpoint es async; un cliente sync bloquea el event loop |
| Coste con `completion_cost()`, sin tabla de precios propia | La tabla se desactualiza en silencio. Ver §8 |
| Provider real vía `response._hidden_params["model_id"]` | No adivinar por el nombre del modelo |
| `prompt_version` explícito en la clave del caché semántico | El caché semántico **no** se invalida solo al cambiar el prompt. Ver §8 |
| Cachear **solo después de validar** | El coste se conoce al cierre del stream, no al inicio. Ver §8 |

---

## 3. Esquemas

Dos tipos distintos, y la separación es deliberada: uno es lo que el modelo
promete, el otro es lo que el sistema entrega.

### 3.1 `EstimationRequest` — capa 1, contrato HTTP

**Implementado en WU2.** Cuatro campos, en `app/schemas/estimation.py`. El modelo
nunca ve este tipo.

```python
class ProjectType(str, Enum):
    MOBILE_APP = "mobile_app"
    WEB_SAAS = "web_saas"
    INTERNAL_TOOL = "internal_tool"
    DATA_PIPELINE = "data_pipeline"


class DetailLevel(str, Enum):
    SUMMARY = "summary"
    MEDIUM = "medium"
    DETAILED = "detailed"


class OutputFormat(str, Enum):
    PHASES_TABLE = "phases_table"
    LINE_ITEMS = "line_items"
    NARRATIVE = "narrative"


class EstimationRequest(BaseModel):
    description: str = Field(min_length=20, max_length=2000)
    project_type: ProjectType
    detail_level: DetailLevel
    output_format: OutputFormat


class EstimationResponse(BaseModel):
    text: str
    prompt_version: str
```

Los nombres van en inglés aunque la documentación esté en español: son el contrato
de la API y viajan en el cable, donde un cliente en otro idioma tiene que poder
mapearlos. `prompt_version` viaja en la respuesta y no solo en los headers,
porque es lo que permite correlacionar una estimación con la plantilla que la
generó.

**Lo que se cayó respecto del diseño original de nueve campos:** `estado_codigo`,
`stack`, `requisitos_no_funcionales`, `integraciones`, `tamano_equipo`,
`deadline_semanas`, `nivel_calidad` y `tipo_proyecto` (que se funde en
`project_type`). No es una simplificación de esta unidad sino una decisión de
partida: cuatro campos obligan al modelo a inferir el resto desde la
descripción, que es el ejercicio que se quiere medir. El plan original ya
decía que el riesgo no puede ser un campo explícito, y que el modelo lo infiere
de `estado_codigo`, `requisitos_no_funcionales` e `integraciones`. Con cuatro
campos, esa inferencia es total. Si el dominio de WU8 necesita esos datos, se
agregan como salida, no como entrada.

**Sobre el riesgo:** sigue sin ser un campo. Igual que en el diseño original, el
modelo lo infiere. Pedirle al usuario que cuantifique algo que todavía no sabe
sería cambiar la pregunta.

**Sobre los límites de `description`: dos capas, y no es redundancia.**

| Capa | Dónde | Quién lo cambia | Para qué |
|------|-------|-----------------|----------|
| Contrato | `Field(20, 2000)` en el schema | nadie, requiere redeploy | La promesa de la API. Es lo que documenta el JSON Schema |
| Techo del operador | `Settings.descripcion_min/max_chars` | `.env` | Bajar el coste sin redeploy: *"el tamaño de la entrada es la factura"* |

El techo solo puede **estrechar** el contrato, nunca hacerlo más laxo. Con
`min < 20` o `max > 2000` el `Field` sigue cortando pero el parche de OpenAPI
escribe el número laxo, así que Swagger prometería un rango que el servicio no
acepta. `Settings.validate_description_ceiling` rechaza esas dos direcciones al
arrancar: es un `.env` mal puesto, y su síntoma como 422 en producción sería
inexplicable.

El default es 20/2000, o sea el contrato mismo. Con 50/5000 —el valor de WU1—
el `min_length=20` del schema no se cumpliría nunca y nadie se enteraría.

`CONTRACT_MIN_CHARS` y `CONTRACT_MAX_CHARS` están duplicados en `app/config.py`
porque config no puede importar el schema sin ciclo (el schema importa
`get_settings`). Un test vigila que no se desincronicen.

**Espacios y campos extra (11 de octubre de 2026).** `EstimationRequest` lleva
`model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")`:

- **Descripción en blanco:** los espacios, saltos de línea y tabulaciones de los
  extremos se recortan *antes* de medir el largo, así que veinte espacios, o un
  texto de relleno alrededor de pocas letras, son un 422 como cualquier
  descripción corta y no llegan al modelo. El contrato (`Field`) y el techo del
  operador miden el texto recortado, y el prompt y la clave de caché lo reciben
  así. El formulario ya recortaba; ahora también la API, para cualquier cliente.
- **Campos fuera del contrato:** son un 422 que nombra el campo, en los tres
  endpoints. Antes se descartaban sin aviso: un `prompt_version` mandado en el
  cuerpo en vez de en la query se ignoraba y la estimación salía con la versión
  predeterminada. No era un riesgo de seguridad (el loader solo usa los cuatro
  campos), sino un error del cliente que nadie veía.

### 3.2 La salida hasta WU8 es texto libre

**Hasta WU7 no hay contrato de salida del LLM.** Lo que el sistema devuelve es un
`text: str`, igual que en `../session_4/estimator`. No hay schema, no hay campos que
validar y no hay estructura que parsear.

Esto es una decisión de secuencia, no una limitación. La capa 3 del objetivo —"salida
estructurada validada"— llega en WU8, y el diseño de esa capa no está escrito a
propósito: `SolicitudEstimacion` no tiene forma
hasta que el prompt de WU4 exista.

La estructura de prompts la fija el README y parte de la de `../session_4/estimator`:
`app/prompts/<use_case>/<version>/` con `system.j2`, `user.j2`, `examples.j2` y `examples.yaml`,
más un `loader.py` con `render_estimation_prompt(request, version="v1")`.
La versión viaja en el path, no en el nombre del archivo: un `v2/` al lado de `v1/`
entra sin tocar router ni schemas.

**Diferencia con la referencia:** `session_4/estimator` tiene un `examples.j2` con
los ejemplos ya escritos como tabla. Ese archivo tenía tres problemas: el `include`
llevaba `v1` escrito en la ruta, los ejemplos mostraban una tabla aunque se pidiera
`narrative` o `line_items`, y los costes no cuadraban con las tarifas de `<scope>`.
Acá `examples.j2` se mantiene (y `system.j2` lo incluye con `{% include %}`, como
pide la consigna), pero sin esos problemas: la ruta del `include` usa la versión
como variable, los datos están en `examples.yaml` y los calcula `loader.py`, y
`examples.j2` solo los muestra en el formato pedido. Hasta el 4 de octubre de 2026
la presentación vivía en macros dentro de `system.j2`; moverla a `examples.j2` no
cambió ningún prompt (los 72 renders de `v1` y `v2` son idénticos byte a byte). Qué
contiene y qué decide cada archivo está en el README, sección *Versionado de
prompts*, que es la referencia.

**Lo que cambia cuando llegue WU8, y es más de lo que parece.** No alcanza con
agregar clases: la forma de la respuesta pasa de `{text: str}` a
`{tareas: [...], equipo: [...], ...}`, y eso toca el schema HTTP de WU2, el contrato
de eventos SSE de §4, y el frontend de WU6. Por eso WU8 está marcada como riesgo
**Alto**: es la unidad que rompe los contratos que las anteriores escribieron.

**Los supuestos de negocio no desaparecen, cambian de dueño.** `HORAS_SEMANALES` y
`FRACCION_ENFOQUE` estaban en `app/domain/estimacion.py` como constantes testeables.
Con el cálculo del modelo, van al prompt:

```jinja
- Work in calendar weeks; assume one developer week = 32 productive hours.
- Be conservative: round hours up to the nearest 5h.
```

Es el mismo contenido con menos garantías. Una constante en Python se testea; una
línea en un `.j2` se lee. El intercambio es deliberado —`session_4/estimator` no tiene
cálculo en código— pero conviene saber que se pierde: un cambio de `0.8` a `1.0`
ya no rompe ningún test, solo cambia el prompt y hay que notarlo en el diff.

**Matiz desde WU4:** las tarifas, las horas productivas y el redondeo ya no son
líneas sueltas del `.j2`. Viven en `examples.yaml` y alimentan tanto `<scope>` como
los ejemplos, y `test_prompts.py` comprueba que los ejemplos cuadren con ellos. Lo
que sigue sin test es que **el modelo** respete esas reglas en su respuesta.

### 3.3 `StructuredResult` — capa 3, contrato del LLM (WU8)

**Implementado el 5 de octubre de 2026, sin streaming.** Vive en
`app/schemas/structured_estimation.py` y lo produce el prompt `v3`. La forma final
no es la de tareas con `depende_de` que se había esbozado: es la del desglose que
ya producían `v1` y `v2`, por fases.

- **Forma:** `phases` (nombre, resumen, semanas, horas, coste, confianza,
  supuestos y riesgos con su mitigación), `team` (rol y personas), `totals`
  (horas, coste y semanas), `summary` y `confidence_pct` globales. `phases` va
  primero, por la reserva 1 de WU0 (§7); el resumen y la confianza, al final.
- **Cómo se obtiene:** `StructuredResult` (Pydantic) → Instructor → Router de
  LiteLLM. Instructor envía el schema como `response_format` (modo `JSON_SCHEMA`)
  y, si la respuesta no cumple, re-pregunta con el error de validación hasta
  `STRUCTURED_MAX_RETRIES` veces. Va sobre `Router.acompletion`, así que el
  Router sigue siendo el único dueño del respaldo y de los reintentos ante
  fallos del proveedor (§2). Instructor envuelve esos fallos en su propia
  excepción y el wrapper los desenvuelve, para que el endpoint siga
  distinguiendo 504 de 502.
- **Validaciones que re-preguntan:** rangos con mínimo y tope (entre 1 y 8 fases,
  hasta 52 semanas y 1.000.000 EUR por fase, entre otros), nombres de fase únicos,
  y la coherencia del rechazo. Los IDs y `depende_de` desaparecieron con las
  tareas.
- **Rechazo («Fuera de alcance:»):** con confianza menor que 30, `summary` empieza
  con ese prefijo y la estimación es una sola fase «Sin estimar» en cero. Le da
  al modelo una salida estructurada para una descripción que no alcanza, en vez
  de inventar números. Es un 200 con el evento `estimacion_fuera_de_alcance`.
  El prompt toma el umbral, el prefijo y el nombre de la fase de las constantes
  del schema, así que no pueden discrepar.
- **Tomado de `session_4_live/estimator`:** el resumen general y por fase, la
  confianza global con el rechazo, los topes, el bloque `<totals>` del prompt
  (decidir las fases, sumar y comprobar) y `cached` en la respuesta. **No
  tomado:** Instructor sobre `litellm.completion` (se saltea el Router y pierde
  el respaldo), `MODEL_COSTS` (§8), `max_retries=6`, re-preguntar por el total,
  ejemplos en Markdown dentro de un prompt que pide JSON, y la duración total
  como suma obligatoria de las fases.
- **La decisión que quedaba abierta (el total frente a la suma):** término medio
  con tolerancia de **5 %** (`TOTALS_MAX_DEVIATION_PCT`, 11 de octubre de 2026).
  Hasta 5 % se **acepta y se avisa**: la estimación se devuelve tal cual, con el
  aviso en `warnings` y el evento `totales_no_cuadran`. Por encima de 5 % (horas o
  coste) es un error del schema y se re-pregunta; agotados los intentos, 502. La
  duración total no se compara (las fases pueden solaparse). Se descartó
  re-preguntar siempre (paga una llamada completa por una diferencia mínima), no
  tener tolerancia alguna (dejaba pasar totales muy errados) y recalcular en
  código (revertiría §3.4).
- **Coste:** cada intento se paga, y el coste y los tokens suman todos los
  intentos; un error por intentos agotados lleva al log el coste de los
  fallidos.
- **Convivencia:** `/estimate` (texto, `v1`/`v2`) no cambió. Cada versión
  declara su tipo de salida en `examples.yaml` (`output`), y cada endpoint
  acepta solo las de su tipo.

**Salida renderizada (`v4`, 6 de octubre de 2026).** Con `v3`, `output_format` no
entra al prompt y la presentación la arma el frontend. `v4` lo pone en el prompt,
porque el formato de salida es un parámetro del usuario como el nivel de detalle,
y porque lo que ve el usuario tiene que poder validarse (guardrail de salida):

- **Dónde va el texto:** en un campo propio, `rendered`, de `RenderedResult`
  (subclase de `StructuredResult`, al final del schema). El prompt pide la tabla,
  la lista o la prosa *en ese campo*; así no compite con el schema, que es lo
  que le pasa a `session_4_live`, donde el prompt pide una tabla Markdown como
  respuesta y a la vez un objeto estructurado.
- **Cómo se valida:** el validador recibe `output_format` por el contexto de
  validación de Instructor y comprueba la forma pedida y que las horas y el coste
  de cada fase y de los totales aparezcan en `rendered` con las cifras de los
  campos. Si no cumple, re-pregunta; agotados los intentos, 502 (no un aviso: un
  texto que contradice los datos no se le muestra al usuario).
- **Regla:** `output_format` cambia el texto, nunca las cifras. Los totales mal
  sumados hasta 5 % siguen siendo un aviso, y por encima se re-pregunta (se
  compara `rendered` con los campos, no con la suma).
- **Convivencia:** `v3` queda igual. `v4` declara `output: rendered` y tiene su
  endpoint, `POST /api/v1/estimate/rendered`, con `RENDERED_PROMPT_VERSION`. En el
  formulario, un segundo interruptor, excluyente con «Salida estructurada».

**Lo que queda de WU8:** el streaming con `ijson` y los eventos por fase del §4.

### 3.4 El exchange de alcance, y por qué está documentado

**El 29 de septiembre de 2026 se revirtió una decisión de WU3.** La implementación inicial calculaba `total_horas` y `duracion_semanas` en código, con la regla de que
"lo derivado se calcula en código" y tolerancia de error exactamente 0.

El motivo del cambio: el objetivo es replicar `../session_4/estimator`, donde **el
modelo calcula y el sistema no toca el número**. Ese servicio no tiene ni una línea
de aritmética —ni `calcular_total`, ni `calcular_semanas`, ni un schema de salida que
validar— y la respuesta es texto libre.

Lo que se pierde con el cambio, y queda registrado para no descubrirlo en WU8:

| | Se calculaba en código | Lo calcula el modelo |
|---|---|---|
| Tolerancia de error en el total | 0 | la del modelo |
| Coste de una suma mal hecha | ninguno | una llamada completa |
| Un cambio de supuesto | rompe un test | cambia un prompt |
| Detección de la incoherencia | imposible que ocurra | `model_validator` en WU8 |

Lo que se gana: el modelo se compromete con un total antes de emitirlo, y eso lo
obliga a razonar sobre las horas de cada fase en lugar de soltarlas sin más. Es el
único punto en que el diseño de la referencia es superior al nuestro, y es la razón
del cambio.

**Evidencia de la prueba contra la API real (3 de octubre de 2026).** El costo
que acepta esta decisión apareció en la primera estimación real: con `v1` y
`gpt-4o-mini`, las fases sumaban 29.750 € y el modelo escribió un total de
29.850 €. El sistema lo devolvió tal cual, como prevé la tabla de arriba; la
detección sigue siendo el `model_validator` de WU8.

En la misma prueba, el modelo copió en su respuesta la etiqueta `</estimation>`
que envuelve los ejemplos del prompt. Eso sí se corrige en código: es
estructura del prompt, no un número del modelo, y el endpoint la quita antes de
devolver el texto (`app/routers/estimations.py`).

**Una confusión que hubo que corregir antes de decidir:** la referencia de `session_4`
y la de `session_4_live` no son lo mismo. `session_4_live` sí tiene schema estructurado
con `Phase.Estimate`, `total_cost_estimate` y un `model_validator` que comprueba que
las fases sumen —con Instructor y `max_retries=3` para re-preguntar—. `session_4`, que
es la que replica `lidr_4`, no valida nada porque no tiene nada que validar. La
comparación inicial las mezcló.
---

## 4. Contrato de eventos (SSE)

El modelo emite JSON. `ijson` lo consume incrementalmente y cada elemento
completado se convierte en un evento. La UI pinta filas mientras llegan.

| Evento | Payload | Cuándo |
|--------|---------|--------|
| `meta` | `request_id`, `prompt_version`, `cached`, `provider`, `model` | Al abrir el stream |
| `tarea` | `indice`, `total`, `tarea: Tarea` | Cada elemento de `tareas` cierra su JSON |
| `equipo` | `indice`, `rol_equipo: RolEquipo` | Cada elemento de `equipo` |
| `resumen` | `resumen: str` | Al cerrar `resumen` |
| `totals` | `total_horas`, `duracion_semanas` | Tras validar el objeto completo |
| `error` | `code`, `message`, `recoverable` | En cualquier momento |
| `done` | — | Siempre al final |

**El orden de las claves del JSON no es cosmético.** WU0 lo midió (§7): con
`solicitud` primero, la primera tarea aparece en el 91% del stream y las tres
caen en 0.82s (OpenAI) o 0.61s (Anthropic). Incremental en el papel, inútil en
la pantalla. Por
eso el schema pone `tareas` primero y `solicitud` al final: las filas empiezan
a aparecer antes, y el eco de la solicitud normalizada llega al final, que es
donde estorba menos. Es gratis, y solo visible si se midió (§7).

Reglas:

- `meta` es siempre el primero. La UI nunca muestra filas sin saber qué versión
  de prompt las produjo.
- `totals` **solo** se emite tras validar con Pydantic. Nunca antes.
- `error` con `recoverable: false` cierra el stream. `done` no llega.
- Un error de parseo `ijson` durante el stream **nunca** se cachea (§8).

---

## 5. Qué heredamos de `lidr_3`

`lidr_4` no arranca de cero, pero tampoco es un port: **una app nueva que
hereda un esqueleto probado.** `lidr_3` son 4606 líneas de Python. De ellas,
~1750 se descartan, ~400 se heredan casi enteras, el resto es nuevo.

**Se hereda:**

| De `lidr_3` | Líneas | Veredicto |
|---|---|---|
| `app/config.py` | 156 | Casi entero. Se van `llm_provider` y `llm_fallback`; la filosofía queda intacta |
| `app/main.py` | 130 | Casi entero: `create_app()`, lifespan, `_complete_openapi_schema()` |
| `app/cache.py` | 75 | El plomaje Redis y el TTL. Se reemplaza el keying exacto por semántico |
| `app/tracing.py` | 33 | Entero |
| `.env.example`, `pyproject.toml` | ~95 | Convenciones |

**Se descarta:**

| De `lidr_3` | Líneas | Por qué |
|---|---|---|
| `app/providers/` | 600 | Lo reemplaza el `Router` |
| `app/services/llm_service.py` | 612 | Casi todo es el dispatch que se elimina |
| `app/services/pricing.py` | 68 | Pricing a mano. Ver §8 |
| `app/context/examples.py` | 190 | Prompt hardcodeado → Jinja2 versionado |
| `app/schemas/estimation.py` | 57 | `transcription: str` → `EstimationRequest` (WU2, §3.1) |
| `streamlit_app.py` | 221 | Un textarea → nueve campos |

`config.py` y `tracing.py` valen más que su tamaño. Los dos resuelven
problemas invisibles al leer el código:

- **`config.py`** hace que las API keys sean opcionales al arrancar y se exijan
  en el punto de uso. Si la validación ocurre al importar, el proceso muere
  antes de que exista la aplicación: no hay `/health`, no hay nada que le diga
  al orquestador qué falta. *Un health check tiene que sobrevivir a la avería
  que diagnostica.* Además `apply_defaults` normaliza un `.env` copiado del
  `.env.example` sin romper, mientras que un `LOG_LEVEL` inválido sí falla al
  arrancar.
- **`tracing.py`** evita que un logger module-level se congele en su primer uso.
  Si un test toca el módulo antes del lifespan, el logger queda atado al backend
  `print` y la captura de los tests siguientes pasa a depender del orden de la
  suite.

**Sobre los tests: no se hereda ninguno, y por qué importa.** `lidr_3` tiene
~2300 líneas de tests (101 funciones) que nunca corrieron en CI, porque el
workflow solo disparaba `lidr_2/**`. Es tentador tratarlo como deuda a pagar
copiando la suite. No: esos tests están atados a `llm_provider`,
`build_cache_key`, `transcription` y a un contrato de streaming que no es el
nuestro, así que portarlos es reescribirlos con pasos extra. Se escriben
nuevos, en la unidad donde aparece el código.

Del `lidr_3` se heredan cuatro archivos, no una suite. Y lo que realmente se
hereda son dos **decisiones documentadas**, que viven en los docstrings de
`config.py` y `tracing.py` y que ningún test transmite mejor que la prosa:

---

## 6. Unidades de trabajo

Cada unidad es commiteable y revisable por separado. Ninguna depende de una posterior.

| Unidad | Contenido | Riesgo |
|--------|-----------|--------|
| ~~**WU0**~~ | Spike de LiteLLM. §7. **Cerrado:** las 3 preguntas dan sí | Resuelto |
| ~~**WU1**~~ | Heredar el esqueleto de `lidr_3` (§5): copiar `config.py`, `main.py`, `tracing.py`, `cache.py` y la infra. CI en matriz. **Cerrado** | Resuelto |
| ~~**WU2**~~ | Contrato de entrada: `EstimationRequest`/`EstimationResponse` en `app/schemas/estimation.py`, límites en dos capas (§3.1). 31 tests. **Cerrado** | Resuelto |
| ~~**WU3**~~ | ~~Dominio con los totales calculados en código~~ **Revertido**: el modelo calcula. `app/domain/` borrado. Ver §3.4 | — |
| ~~**WU4**~~ | `app/prompts/estimation/v1/{system.j2,user.j2,examples.yaml}` + `app/prompts/loader.py`. `render_estimation_prompt(request, version="v1")`. Ejemplos como datos, maquetados por `output_format`; roles, tarifas y redondeo en el YAML. 19 tests. **Cerrado** | Resuelto |
| ~~**WU5**~~ | Gateway async: `Router` con primario y respaldo, claves por proveedor, coste con `completion_cost()`, provider vía `_hidden_params["model_id"]`. Respaldo opcional: sin su clave el servicio sigue con el primario y avisa al usuario (`warnings` en la respuesta y en `/health`). Caché exact-match fail soft sobre un único cliente de Redis. Errores 503 (falta la clave del primario), 502 y 504 con mensaje limpio. Trazabilidad con `estimacion_completada` / `estimacion_fallida`; los logs de LiteLLM quedan en `WARNING` con el formato de structlog, sin prompt ni claves. Devuelve `text: str` sin parsear. 61 tests (wrapper, caché, endpoint, logs), independientes del `.env` y del proveedor primario. **Cerrado** | Resuelto |
| ~~**WU6**~~ | `EstimationRequest` en Streamlit → `POST /api/v1/estimate`. Validación de longitud en el formulario sin `max_chars` (Streamlit descartaba entero un pegado de más de 2000 caracteres). 15 tests con transporte mockeado. **Cerrado y probado contra la API real** el 3 de octubre de 2026, con todo en Docker: estimación con OpenAI, caché en Redis (8,3 s → 0,5 ms en la repetición), trazabilidad completa y prompt `v2`. Ver §3.4 para lo que mostró la salida del modelo | Resuelto |
| ~~**WU7**~~ | Slice vertical end-to-end con `mock_response`: `test/test_e2e.py` recorre formulario de Streamlit → API (lifespan, caché) → prompt `v2` → Router de LiteLLM → caché → formulario, con solo el proveedor y Redis simulados. El armado del cuerpo del formulario pasa a `_build_payload()` para que el test use el mismo que la interfaz. 11 tests. **Cerrado** | Resuelto |
| **WU8** | Structured output + `ijson` + eventos `tarea`. **Parcial** (5 de octubre de 2026): `StructuredResult` con Instructor sobre el Router, prompt `v3` (tipo de salida declarado en `examples.yaml`, ejemplo de rechazo), `POST /api/v1/estimate/structured` (con `cached`), interruptor «Salida estructurada» en el formulario; totales que no cuadran como aviso (§3.3). 155 tests nuevos (377 en total). **Ampliado** (6 de octubre de 2026): salida renderizada, con `RenderedResult`, prompt `v4` (`output_format` en el prompt, presentación en `rendered` validada contra las cifras), `POST /api/v1/estimate/rendered` e interruptor «Salida renderizada», excluyente con el anterior (§3.3). 127 tests nuevos (505 en total). **Ajuste** (11 de octubre de 2026): tolerancia de totales del 5 % (hasta ahí aviso, por encima se re-pregunta); 526 tests en total. **Capa de servicio** (11 de octubre de 2026): `EstimationService` (`app/services/estimation.py`) saca del router el render, la llamada y los avisos, sin cambiar el comportamiento; 14 tests nuevos (540 en total). **Pendiente:** `ijson` y eventos por fase | **Alto** — depende de WU0 |
| **WU9** | Guardrails entrada/salida + `IncompleteJSONError` + política de reintento. Se parte en dos (12 de octubre de 2026). **WU9 A, hecho:** guardrails de entrada en `EstimationService` (`app/guardrails/input.py`), antes del render y de la caché del wrapper, en los tres endpoints: inyección de prompt por regex (rechaza, 422), datos personales (los oculta y avisa en `warnings`) y moderación de OpenAI (rechaza, 422; falla abierta; apagada sin `OPENAI_API_KEY`). Configuración `GUARDRAILS_ENABLED`, `MODERATION_ENABLED`, `MODERATION_MODEL`, `MODERATION_TIMEOUT`; eventos `entrada_rechazada`, `datos_personales_ocultados`, `moderacion_no_disponible`, `moderacion_desactivada`, `guardrails_desactivados`. 84 tests nuevos (624 en total), 14 mutaciones muertas. **Refactor de la caché exacta** al servicio (12 de octubre): 633 tests en total, 10 mutaciones sobre la lógica movida muertas. **WU9 B, hecho (12 de octubre de 2026):** guardrail de salida, `IncompleteJSONError` y política de reintento, sin esperar al streaming de WU8. Regla de horas contra el equipo como validador del schema (re-pregunta; 40 h por persona y semana, `_hours_fit_team`), aviso de confianza baja entre 30 y 70 % (`app/guardrails/output.py`, evento `confianza_baja`), y `IncompleteJSONError` para la respuesta cortada por `max_tokens` (no se reintenta; 502 con mensaje propio). 17 tests nuevos (650 en total). La primera medición con el modelo real (12 de octubre, `medir_v4.py -n 1`) tuvo un caso que agotó los intentos con «totales en cero» sin que el mensaje dijera cuál; ahora el mensaje nombra el campo (`totals.duration_weeks`) y, en las fases, cuál de `duration_weeks`, `hours` o `cost_eur` está en cero (6 tests más, 656 en total). Con el mensaje nuevo y la frase del tope de 40 h en la descripción de `hours`, la medición (`-n 3`) tuvo 2 fallos de 27: uno con totales en cero (el mensaje ya nombraba los campos y el modelo repitió la misma respuesta en los 3 intentos) y otro con `totals.cost_eur` a más de 5 % de la suma. Sin esa frase (quitada el mismo día; el tope sigue como validador): 0 fallos en 36 ejecuciones (`-n 1` más `-n 3`), 5 de 36 con algún reintento por totales que se aleja de la suma y todos recuperados. 3 fallos de 36 frente a 0 de 36 no es significativo (p ≈ 0,24, dos colas), pero la frase no aporta nada que el validador no haga, así que se queda fuera. La regla de horas no rechazó ninguna de las 72 ejecuciones medidas. Sigue pendiente, como límite conocido, que un modelo que repite la misma respuesta en los reintentos agota los intentos y da 502 **Pendiente de WU8:** si llega el streaming, la regla de horas tendrá que poder aplicarse por fase a medida que salen | Medio |
| **WU10** | Caché semántico + guarda anti-envenenamiento + `prompt_version`. **Hecho (13 de octubre de 2026)**, encendida por defecto (`SEMANTIC_CACHE_ENABLED=true`; sin Redis o sin clave de embeddings queda apagada con aviso): `app/services/semantic_cache.py`, embeddings `text-embedding-3-small` por LiteLLM (el mismo de la referencia `session_4_live`), Redis plano con coseno en la app (la referencia usa redis-stack + redisvl; acá el compose sigue en `redis:7-alpine`, a cambio de un tope de entradas por espacio, `SEMANTIC_CACHE_MAX_ENTRIES`). Espacio de búsqueda = las partes de la clave exacta sin la descripción (modelo, `prompt_version`, tipo, nivel de detalle, y formato en v4) más el modelo de embeddings. Guardas: solo estimaciones sanas (no rechazo, confianza ≥ `LOW_CONFIDENCE_WARN_PCT`, totales cuadrados), mismas cifras en la descripción, y escribir después de los guardrails de salida. Modo `SEMANTIC_CACHE_LOG_ONLY` (de la referencia) para calibrar el umbral. Solo en los endpoints estructurado y renderizado. Un acierto se copia a la exacta y lleva un aviso. 67 tests nuevos; 801 en total. **Umbral calibrado el 13/10 con `test/calibrar_cache.py` (23 pares, `text-embedding-3-small`):** a 0,92 servía 5 de 10 paráfrasis y dejaba un falso acierto («con» y «sin pagos», 0,937); el piso sin falsos aciertos es 0,94 (2 de 10 paráfrasis), y el defecto es 0,95. Muestra chica y escrita a mano: falta afinar con tráfico real (`SEMANTIC_CACHE_LOG_ONLY`). Sin medir el ahorro real de llamadas | Medio |

**Renumeración.** El WU2 original ("dominio") pasó a ser WU3 y el actual WU2 es
otra cosa: el contrato de entrada que definiste antes de arrancar. No es un
desvío del plan, es un prerrequisito que el plan daba por hecho — el §3 ya
describía la capa 1 como un schema Pydantic, pero sin los límites en dos capas
de hoy. Lo que cambia con la renumeración: `SolicitudForm` pasa a llamarse
`EstimationRequest` y sus cuatro campos son los que ya valida el schema. El
dominio de WU3 sigue igual salvo por ese nombre.

**WU1 incluyó un arreglo que no es de esta entrega:** el CI solo corría
`lidr_2/**`, así que los tests de `lidr_3` nunca se ejecutaron. Pasarlos a una
matriz los hace correr por primera vez. El primer push ya ocurrió: el job
`lidr_4` pasó los cuatro comandos (`uv sync --locked`, `pytest`, `ruff check`,
`ruff format --check`) verificados en local antes de commitear.

**Sobre los tests de `lidr_3`: no se porta ninguno.** Escribirlos de nuevo contra
el código nuevo, en la unidad donde aparece. La alternativa —copiar las ~2300
líneas de `test/`— es reescribirlas con pasos extra, porque están atadas a
`llm_provider`, `build_cache_key`, `transcription` y a un contrato de streaming
que ya no existe. Lo único que vale de esa suite son dos insights, y ya están
en los docstrings de `config.py` y `tracing.py` que heredamos.

**No hay unidad de tests.** Los tests son parte de la unidad que escribe el
código, nunca una tarea aparte: WU2 los del contrato (puros, sin I/O, sin red),
WU4 los de la plantilla, WU5 los del gateway, WU6 los del frontend, WU10 los del
keying. Si una unidad agrega comportamiento, agrega su test en el mismo commit.

El más importante de todos va a ser el de WU4, y es el que verifica que los
few-shot renderizados producen algo con la forma que el prompt dice producir. La
versión original de este plan lo formulara con
`SolicitudEstimacion.model_validate_json`, y ya no se puede escribir así: no hay
schema de salida hasta WU8. Lo que reemplaza a esa aserción es **comparar los
few-shot renderizados contra la respuesta real del modelo**, para que los ejemplos
no se queden viejos en silencio cuando cambia el prompt.

**El contrato (WU2) va antes que la infraestructura (WU5)** porque no necesita red
para testearse, y es la parte que más cambios de requisitos va a tolerar. El dominio
ya no está en esa posición: pertenece a WU8, y va después de la infraestructura
porque necesita un prompt y un modelo reales para poder diseñar su forma con
evidencia.

---

## 7. WU0 — resultado del spike

Ejecutado contra OpenAI y Anthropic con el schema real (3 tareas, strict).
Las tres preguntas quedan respondidas: **sí, sí, sí.** La capa 3 se puede
diseñar con eventos de fila. Pero con dos reserve que cambian WU8 y WU10.

### Las 3 preguntas

| # | Pregunta | Respuesta |
|---|----------|-----------|
| 1 | ¿`stream` + `response_format=json_schema` da deltas utilizables? | **Sí.** JSON válido y parseable en ambos, sin texto de relleno ni markdown |
| 2 | ¿Funciona a través de `Router`? | **Sí.** 217 chunks, JSON válido, y acá **sí** aparece `_hidden_params["model_id"]` |
| 3 | ¿Los deltas alimentan `ijson` sin acumular el stream? | **Sí.** Tareas emitidas por separado mientras el modelo sigue escribiendo |

### Lo que sale de medir

**Separación entre tareas: ~0.3s en ambos providers** (OpenAI 0.34/0.35,
Anthropic 0.30/0.28). Tan simétrica que parece marcar el provider, no el
modelo. Es lo que hace viable el evento por fila.

**`_hidden_params["model_id"]` solo existe con `Router`.** En `acompletion`
directo viene `None`. O sea que la identidad del deployment **depende de pasar
por el Router**, que ya era la decisión de diseño; queda confirmado que no es
un atajo opcional. `provider_response_model` sí viene siempre y sirve de red
de seguridad (`gpt-4o-mini-2024-07-18`, `claude-haiku-4-5-20251001`).

**`usage` llega en ambos, en el último chunk**, con `content=None`. Hay que
filtrar los chunks sin `delta.content` (3 en OpenAI, 2 en Anthropic) o el
ensamblado revienta. `usage.cost` ya viene calculado; `completion_cost()`
coincide con ese valor, así que sirve de verificación cruzada, no de fuente.

### Reserva 1 — el 91% del stream es espera

La **primera** tarea aparece en el 91% (OpenAI, 8.09s de 8.91s) y el 95%
(Anthropic, 12.18s de 12.79s) del stream. Las tres filas caen en una ventana de
**0.82s** (OpenAI) y **0.61s** (Anthropic), entre el 7% y el 9% del stream.

Es incremental, pero perceptualmente no: el usuario mira un spinner 8–12
segundos y después ve todo de golpe. La causa es el **orden de las claves del
schema**: `solicitud` va antes que `tareas`, y el modelo razona sobre la
solicitud antes de emitir la primera tarea.

**Decisión para WU8: `tareas` va primero en el schema, `solicitud` al final.**
Invierte la espera: las filas empiezan a aparecer antes y el eco de la
solicitud llega al final, que es donde molesta menos. Es gratis.

### Reserva 2 — Anthropic cuesta 10.2x con el mismo schema

| | prompt_tokens | completion_tokens | coste |
|---|---|---|---|
| `gpt-4o-mini` | 175 | 224 | $0.00016 |
| `claude-haiku-4-5` | 536 | 220 | $0.00164 |

3.1x más prompt tokens por el mismo schema. La causa es que LiteLLM implementa
`json_schema` en Anthropic como **tool use**, así que el schema viaja como
definición de herramienta. Y esto es con **cero few-shot**: con los 3 ejemplos
de §4 la brecha se agranda.

Consecuencia directa: **el fallback a Anthropic no es gratis**, y fingir que lo
es porque "ambos dan JSON válido" esconde una decisión de costo de 10x.
`usage` de Anthropic trae `cache_read_input_tokens` y `cache_write_tokens`, así
que **prompt caching aplica**, y el prefijo estático (schema + ejemplos) es el
candidato ideal. Eso le da a WU10 más peso del que tenía y es la razón de que
`PROMPT_VERSION` tenga que ser explícito.

**Criterio de decisión, ya ejecutado:** si la 1 fallaba, la capa 3 se rediseñaba
con eventos de token. No hizo falta.

---

## 8. Decisiones que parecen extrañas y no lo son

Documentadas aquí para que nadie las "corrija" después.

**La caché semántica está encendida por defecto y no sirve texto libre (13/10).** Devuelve la estimación de *otra* descripción; por eso el operador puede apagarla con `SEMANTIC_CACHE_ENABLED=false`, y `SEMANTIC_CACHE_LOG_ONLY=true` permite calibrar el umbral con tráfico real sin servir nada. Solo actúa en los endpoints estructurado y renderizado porque son los que pasan por el schema y los guardrails de salida; el de texto libre no se valida y repartirlo a descripciones parecidas sería repartir lo que no se comprobó. Las cifras de la descripción tienen que coincidir porque los embeddings casi no distinguen «20.000» de «200.000». No usa RediSearch (el compose es `redis:7-alpine`): la similitud se calcula en la app sobre un hash por espacio, con un tope de entradas. Límite conocido: dos descripciones con las mismas cifras y muy parecidas pero con un matiz que cambia el alcance («sin pagos» / «con pagos») pueden acertar entre sí si el embedding las ve cerca; el umbral y el modo `log_only` son la defensa, y la calibración (`test/calibrar_cache.py`, 13/10) es con una muestra chica escrita a mano: a 0,92 ese caso acertaba (0,937); con el umbral por defecto de 0,95 no, a costa de servir pocas paráfrasis.

**Los guardrails viven en `app/guardrails/` (13/10).** `input.py` (inyección, datos personales, moderación), `output.py` (avisos y eventos sobre la estimación) y `detectors.py` (detectores puros que comparten ambos). Las reglas de salida que rechazan siguen como validadores del schema: Instructor solo re-pregunta si un validador lanza, y el schema no puede importar `app.guardrails` (ciclo: `output` importa el schema). `output.py` documenta ese mapa y solo emite avisos. Interruptores: `GUARDRAILS_ENABLED` (entrada), `OUTPUT_GUARDRAILS_ENABLED` (avisos y eventos de salida, no las reglas del schema) y `LOW_CONFIDENCE_WARN_PCT` (70). Los eventos ahora salen con el logger `app.guardrails.*`. 20 tests nuevos de salida y detectores; 676 en total. **Paso 2 (13/10):** `OutputGuardrails.inspect_*` descarta con 502, sin re-preguntar ni cachear, la salida con datos personales (mismos detectores que la entrada) o con el prompt volcado (etiquetas de las plantillas salvo `<estimation>`, y frases fijas cuya presencia en las plantillas vigila un test); evento `salida_rechazada`. Corre en los tres endpoints; se apaga con `OUTPUT_GUARDRAILS_ENABLED=false`. Una entrada de la caché con datos personales no puede existir (se comprueba antes de escribir). **Paso 3 (13/10):** chequeos de alucinación de bajo nivel como validadores del schema, que re-preguntan: coste por hora entre las tarifas de los prompts (50 y 62,50 EUR/h, 5 % de margen; un test vigila que sigan siendo las de las plantillas), duración total entre la fase más larga y la suma de las fases (5 %), y cifras con unidad del resumen (`horas`, `EUR`, `semanas`) que no están en los campos. Riesgo conocido: más re-preguntas y algún 502 si el modelo repite la respuesta; medido con `medir_v4.py -n 3` (gpt-4o-mini, v4): 27 de 27 válidas, 3 con reintento (11 %, 1,15 intentos de media, 0,0547 USD) y los tres por la regla de totales de antes; ninguna de las tres nuevas rechazó una respuesta. Un caso repitió la misma respuesta en dos reintentos y se salvó en el tercer intento (el límite de §9). 734 tests en total.

**El router no tiene lógica de estimación (desde el 11/10).** La referencia de
`session_4_live` tiene `app/services/estimation.py`, un orquestador donde el
router solo traduce errores a HTTP. En `lidr_4` esa lógica estaba repartida entre
`_estimate_validated()` del router y el wrapper, porque no había nada más que
orquestar. Con WU9 sí: los guardrails de entrada tienen que correr antes de la
caché, y la caché exacta tiene que consultarse antes del render; con la caché en
el wrapper ninguna de las dos cosas era posible. `EstimationService` (render,
llamada, eventos y avisos) quedó entre los dos. Los errores no se traducen en el
servicio: llegan al router tal cual. Un prompt que no se puede armar es un error
nuestro y sale como `PromptRenderError` (500), no como un 502 atribuido al
proveedor. La caché exacta se movió del wrapper al servicio (12 de octubre de 2026) para
respetar el orden guardrails → exacta → semántica → render → LLM → guardrail de
salida → escritura en ambas cachés. Como la búsqueda va antes del render, la
clave ya no es el hash de los dos prompts: es el hash de modelo, versión del
prompt, parámetros de la request y descripción filtrada (la referencia también
incluye el modelo). `output_format` entra solo en texto y en v4; en v3 no cambia
la respuesta. Contrapartida: editar una plantilla sin subir de versión no
invalida la caché, y las versiones publicadas son inmutables (§3.3). El servicio
emite `estimacion_completada` en los aciertos y el wrapper solo en las llamadas
reales al proveedor.

**Los guardrails de entrada ocultan los datos personales y rechazan la inyección (desde el 12/10).**
Las dos reglas no son iguales. Una descripción con un email o un teléfono sigue
siendo una descripción de proyecto: rechazarla tiraría casos legítimos y le pediría
al usuario que reescriba algo que no estaba mal. Se oculta con un marcador
(`[EMAIL]`) antes del render, de la caché y del log, y la respuesta lleva un aviso.
Una inyección no tiene versión legítima que salvar: se rechaza con 422 y un mensaje
fijo por regla, sin repetir lo escrito. Decisión mía, no confirmada por la referencia:
`session_4_live` no deja ver (en la página que se pudo leer) qué hace con los datos
personales. Si se prefiere rechazarlos, es cambiar el `return` de una regla.

**La moderación falla abierta (desde el 12/10).** Es un filtro extra sobre un tercero
(OpenAI), no la única defensa: las reglas locales corren antes. Si no responde, no
tiene clave o tarda más de `MODERATION_TIMEOUT` (3 s, reintentos del cliente
incluidos), la descripción sigue y queda `moderacion_no_disponible` en el log. Fallar
cerrado dejaría el servicio sin estimar por una avería ajena. Se consulta con el texto
ya oculto, para no mandar datos personales a un tercero. El endpoint no se cobra
(documentación de OpenAI, verificada el 12/10) pero tiene límites de tasa por tier de
la cuenta (250 peticiones por minuto y 5.000 por día en el gratuito).

**Los guardrails de entrada son heurísticas (desde el 12/10).** Las reglas de
inyección buscan frases concretas en castellano e inglés, y se escribieron para no
saltar con descripciones normales: «mostrar el system prompt configurado a los
administradores» o «detectar jailbreak en móviles» pasan (están en los tests). A
cambio, una inyección redactada de otra forma pasa. `entrada_rechazada` deja la regla
que saltó, que es lo que permite medir los falsos positivos con tráfico real antes de
tocar un patrón. Un IBAN o una tarjeta con el dígito de control mal no se reconocen
como tales, pero sus grupos de dígitos los oculta igual la regla de teléfonos: se
prefiere ocultar de más.

**Los totales los declara el modelo (desde el 29/09, ver §3.4).** La referencia de
`session_3` los pide y después los verifica con `abs(suma - declarado) <= 1` para
las horas y `<= 2%` para el coste. Esas tolerancias son una admisión de que la
aritmética del modelo no es confiable. El diseño original evitaba el problema
calculando los totales en código (tolerancia 0); al revertirlo, la tolerancia pasa a
ser la del modelo y la decisión de cómo validarla queda abierta para WU8 (§3.3).

**Sin `MODEL_COSTS` propia.** La referencia mantiene una tabla de precios
hardcodeada, y desactualizada. Eso contradice el motivo principal de adoptar
LiteLLM. Se usa `completion_cost()`.

Y no es solo un defecto de la referencia: `lidr_3/app/services/pricing.py`
—68 líneas— ya hacía pricing a mano. El defecto viene de la línea de
descendencia, así que el riesgo real es **heredarlo sin mirar**. Está en la
lista de descartes de §5 justamente por eso.

**`llmprice-kit` no se usa.** Se evaluó y se descartó: es un reempaquetado del
`model_prices_and_context_window.json` de LiteLLM, o sea una copia de datos que
ya tenemos en el árbol de dependencias. Su argumento de venta ("LiteLLM es
pesado, si solo querés precios usá esto") es al revés de nuestro caso. Además
su último release es de abril de 2026 y promete releases semanales. Solo como
herramienta de desarrollo, vía `uvx llmprice get gpt-4o`, para contrastar a mano
lo que reporta `completion_cost()`.

**Sin `_provider_from_model` por string matching.** La referencia deduce el
provider del nombre (`claude` → anthropic, `gpt|o1|o3` → openai, resto
`unknown`). Agregar un provider y el tracking degrada en silencio. Se usa
`model_id`.

**El wrapper es async.** La referencia usa `litellm.completion` sync, que
bloquea el event loop en un FastAPI async. Eso es una regresión respecto de
`lidr_3`.

**`prompt_version` va en la clave del caché.** La referencia invalida su caché
gratis porque su clave es SHA-256 del system prompt completo. El caché semántico
embebe **solo la consulta del usuario**: subir a `v2` sin tocar la clave
devolvería estimaciones de `v1` sin avisar. Output silenciosamente incorrecto.

**`output_format` entra al prompt en `v4` y no en `v3` (6 de octubre de 2026).**
En `v3` el servicio devuelve datos y el cliente decide la presentación. En `v4`
la presentación la escribe el modelo y la valida el servicio, porque el prompt
compone parámetros del usuario y un guardrail de salida necesita saber qué se
pidió. Conviven: `v3` es inmutable y una versión nueva es la forma de cambiar el
comportamiento (§3.3). El texto va en `rendered` y no como respuesta entera, para
que el formato y el schema no compitan.

**Cachear después de validar, siempre.** Con async el coste no existe hasta que
el stream cierra. Cachear al inicio significaría cachear un coste unknowable y un
JSON sin verificar. Es la política, no una preferencia.

**La regla de horas rechaza; la confianza baja solo avisa (12 de octubre de 2026).**
Unas horas que el equipo no puede trabajar (más de 40 por persona y semana) no
describen una estimación, y se re-preguntan en el schema, como los totales que se
alejan más de 5 %. Una confianza entre 30 y 70 % sí describe una estimación, con
la duda declarada: devolver un error la tiraría, y el usuario necesita verla. El
tope es 40 y no las 32 horas productivas de los prompts porque las horas se
redondean hacia arriba y un tope justo rechazaría estimaciones correctas. Se
compara con el equipo entero (`team`), no con el de cada fase: el schema no dice
qué roles trabajan en cada una, y así la regla nunca rechaza por una suposición.
La regla vive en el schema y no en el servicio porque ahí Instructor la ejecuta,
le devuelve el mensaje al modelo y la caché la vuelve a aplicar al leer.

**Una respuesta cortada no se reintenta (12 de octubre de 2026).** Instructor ya
relanza `IncompleteOutputException` sin re-preguntar; sin tratamiento salía como
un 502 genérico y sin el coste gastado. `IncompleteJSONError` le da mensaje propio
y deja `max_tokens` y el coste en el log. No se reintentó con más presupuesto: el
coste se duplicaría y el límite lo decide la configuración (`LLM_MAX_TOKENS`),
que es lo que hay que subir si el evento se repite.

---

## 9. Riesgos abiertos

| Riesgo | Impacto | Mitigación |
|--------|---------|------------|
| `response_format` + `stream` no soportado por el provider | Rompe la capa 3 | WU0 lo mide antes de WU8 |
| El modelo inventa IDs de tarea en `depende_de` | `T3` depende de una tarea inexistente | **No aplica**: el contrato de WU8 es por fases, sin dependencias (§3.3) |
| `horas` incoherentes con el tamaño de equipo | Estimación absurda | **Mitigado en WU9 B**: validador del schema (`_hours_fit_team`) que re-pregunta. Riesgo nuevo: un tope de 40 h por persona y semana que rechace una estimación razonable cuesta un reintento y, si se repite, un 502; se mide con `estimacion_fallida` y su `detalle` |
| Una regla de inyección rechaza una descripción legítima | Un 422 que el cliente no entiende: tiene que reescribir algo que estaba bien (p. ej. «el motor ignora las reglas anteriores cuando…») | **Sin medir**: los patrones exigen verbo + objeto + «anteriores/previas» y las pruebas incluyen casos normales parecidos. Medir con `entrada_rechazada` (campo `regla`) con tráfico real antes de ampliar patrones |
| Una inyección que las reglas no ven | Instrucciones del usuario dentro del prompt (otro idioma, otra redacción, letras de otro alfabeto) | **Límite conocido**: las reglas son frases concretas. La defensa de fondo es el schema (la salida se valida) y el guardrail de salida de WU9 B |
| Datos personales que la regex no reconoce | Nombres propios, direcciones y números con otro formato llegan al proveedor | **Límite conocido**: solo se ocultan email, teléfono, IBAN, tarjeta y documentos con formato. No hay NER. Quien pegue una transcripción con nombres sigue enviándolos |
| La moderación depende de OpenAI | Sin `OPENAI_API_KEY` no hay moderación; con la cuenta en el tier gratuito hay tope de 250 peticiones por minuto | **Mitigado**: falla abierta con evento, apagada sin clave con evento. No se probó contra la API real (la sesión de desarrollo no tenía clave ni salida a `api.openai.com`): el adaptador se probó contra un transporte HTTP falso con la forma de respuesta de OpenAI. Primera ejecución real: comprobar que `moderacion_no_disponible` no aparece |
| El total no cuadra con la suma de las fases | Ya ocurrió en la prueba real (29.850 € frente a 29.750 €). En `/estimate` llega así al usuario | **Mitigado en `/estimate/structured`**: se detecta; hasta 5 % llega como aviso y por encima se re-pregunta (§3.3). **Medido** (11 de octubre de 2026, `v4`, 27 ejecuciones): aviso en 23, re-pregunta en 4, y las 4 se corrigieron en el segundo intento. En `/estimate` sigue sin detectarse: el texto no se parsea |
| La salida estructurada o renderizada no cumple el schema en ningún intento | 502 con `STRUCTURED_MAX_RETRIES + 1` llamadas pagadas | El coste de los intentos queda en el log (`estimacion_fallida`) |
| El validador de `rendered` rechaza una respuesta correcta | El modelo escribe una cifra en una forma que el validador no acepta (`34 050`, `34k`, `34.050,00`): cada rechazo es un reintento pagado y, si se repite, un 502. Con «Detallado» y narrativa, `rendered` alarga la respuesta y puede acercarse a `LLM_MAX_TOKENS` (4000) | **Medido** (11 de octubre de 2026, `test/medir_v4.py`, `gpt-4o-mini`, los 9 casos × 3 repeticiones): 27 de 27 válidas, ningún rechazo del validador de `rendered` y ninguna respuesta cerca de `LLM_MAX_TOKENS`. Los 4 reintentos fueron por totales desviados más de 5 %, no por `rendered`. Muestra pequeña: no descarta formas de cifra que no aparecieron. Repetir al cambiar de modelo o de prompt (§3.3) |
| La estimación varía mucho entre ejecuciones del mismo caso | Dos peticiones iguales dan totales distintos (en la medición, hasta 39 % de rango en horas y 41 % en coste; otros casos, 0 %). No se debe prometer una cifra estable | **Límite conocido, sin mitigar**: la temperatura ya es 0,1. La caché devuelve la misma estimación a la misma petición. No se midió si `output_format` mueve las cifras: el ruido de cada caso es del tamaño de la diferencia entre formatos |
| El modelo no redondea el coste al múltiplo de 50 EUR | En la medición, 25 % de las fases (las horas sí salen redondeadas) | **Aceptado**: es de estilo, no cambia el resultado. No se valida ni se avisa |
| Coste por hora de una fase fuera de las tarifas del prompt | En la medición, 9 de 135 fases, 7 de ellas la misma (195 h por 12.250 EUR = 62,82 EUR/h, un 0,5 % sobre la tarifa máxima: el modelo calculó con 196 h y mostró 195) | ~~No aplica~~ **Reabierto y hecho (13/10, WU9 B paso 3)**: validador del schema que re-pregunta, con 5 % de margen (la fase de 62,82 EUR/h de la medición queda dentro). Medido (13/10, `medir_v4.py -n 3`, gpt-4o-mini, v4): 27 de 27 válidas, 0 fases fuera de tarifa |
| El modelo copia etiquetas del prompt (`</estimation>`) | Texto sucio para el usuario | **Mitigado**: el endpoint las quita (§3.4) |
| Caché semántico sirve una estimación de otro dominio | Resultado plausible pero fuera de tema | Namespace por `prompt_version` + umbral alto, WU10 |
| Deriva de precios del modelo | Coste reportado ≠ coste real | `completion_cost()` se recalcula, no se cachea el precio |

---

## 10. Fuera de alcance

- Proxy LiteLLM propio (SDK embebido es suficiente aquí)
- Persistencia de estimaciones en base de datos
- Autenticación y multi-tenancy
- Reemplazar `ganttly`
- Portar los tests de `lidr_3` que cubren los providers y el fallback custom

---

## 11. Referencias

- Implementación de referencia: `https://github.com/LIDR-academy/ai-engineering/tree/session_3`
- `../lidr_3/` — base a evolucionar
- `../ganttly/README.md` — patrones de ambigüedad y cálculos deterministas
