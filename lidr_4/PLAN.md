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
| Los few-shot se versionan con el prompt: datos en `v1/examples.yaml`, presentación en macros de `v1/system.j2` | Una sola fuente por versión. Los ejemplos se maquetan según el `output_format` pedido, así que se elimina la clase de bugs "el ejemplo 3 enseña un formato que el prompt ya no pide" |
| `total_horas` y `duracion_semanas` los calcula el modelo | Es lo que hace `../session_4/estimator`. Obliga al modelo a comprometerse con un total. Ver §3.4 |
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
acepta. `Settings.validar_techo_descripcion` rechaza esas dos direcciones al
arrancar: es un `.env` mal puesto, y su síntoma como 422 en producción sería
inexplicable.

El default es 20/2000, o sea el contrato mismo. Con 50/5000 —el valor de WU1—
el `min_length=20` del schema no se cumpliría nunca y nadie se enteraría.

`CONTRATO_MIN_CHARS` y `CONTRATO_MAX_CHARS` están duplicados en `app/config.py`
porque config no puede importar el schema sin ciclo (el schema importa
`get_settings`). Un test vigila que no se desincronicen.

### 3.2 La salida hasta WU8 es texto libre

**Hasta WU7 no hay contrato de salida del LLM.** Lo que el sistema devuelve es un
`text: str`, igual que en `../session_4/estimator`. No hay schema, no hay campos que
validar y no hay estructura que parsear.

Esto es una decisión de secuencia, no una limitación. La capa 3 del objetivo —"salida
estructurada validada"— llega en WU8, y el diseño de esa capa no está escrito a
propósito: `SolicitudEstimacion` no tiene forma
hasta que el prompt de WU4 exista.

La estructura de prompts la fija el README y parte de la de `../session_4/estimator`:
`app/prompts/<use_case>/<version>/` con `system.j2`, `user.j2` y `examples.yaml`,
más un `loader.py` con `render_estimation_prompt(request, version="v1")`.
La versión viaja en el path, no en el nombre del archivo: un `v2/` al lado de `v1/`
entra sin tocar router ni schemas.

**Diferencia con la referencia:** `session_4/estimator` tiene un `examples.j2` con
los ejemplos ya escritos como tabla. Acá se reemplazó por datos (`examples.yaml`) y
macros de presentación en `system.j2`, por tres problemas que tenía el `.j2` fijo:
el `include` llevaba `v1` escrito en la ruta, los ejemplos mostraban una tabla aunque
se pidiera `narrative` o `line_items`, y los costes no cuadraban con las tarifas de
`<scope>`. Qué contiene y qué decide cada archivo (`examples.yaml`, `system.j2`,
`loader.py`) está en el README, sección *Versionado de prompts*, que es la referencia.

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

### 3.3 `SOLICITUD_ESTIMACION` — capa 3, contrato del LLM (WU8)

**Todavía no implementado.** Se diseña en WU8, cuando exista el prompt de WU4 que
tiene que producir esta forma.

Lo que se sabe hoy, y no es mucho:

- La forma tiene que admitir el desglose que el prompt de `session_4/estimator`
  produce: fases con duración y coste, equipo, supuestos y riesgos.
- `total_horas` y `duracion_semanas` los **emite el modelo**, y el sistema no los
  recalcula. Es el cambio de alcance que se decidió después de implementar
  lo contrario.
- Las validaciones que hacen falta ya se conocen por haber estado escritas:
  `horas` y `cantidad` con `gt=0`, IDs únicos, `depende_de` sin huérfanas ni ciclos.
  Esas tres se implementan en WU8, no antes.

Lo que **no** se decide ahora: si el total se valida contra la suma de las tareas, si
se acepta y se loguea la discrepancia, o si se re-pregunta al modelo. Esa decisión
depende de lo que WU4 muestre sobre la calidad de las estimaciones que produce el
prompt, y tomarla ahora sería fijarla sin evidencia.

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
| `app/main.py` | 130 | Casi entero: `create_app()`, lifespan, `_completar_schema_openapi()` |
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
  que diagnostica.* Además `aplicar_defaults` normaliza un `.env` copiado del
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
| ~~**WU5**~~ | Gateway async: `Router` con primario y respaldo, claves por proveedor, coste con `completion_cost()`, provider vía `_hidden_params["model_id"]`. Caché exact-match fail soft sobre un único cliente de Redis. Errores 503 (falta clave), 502 y 504 con mensaje limpio. Trazabilidad con `estimacion_completada` / `estimacion_fallida`. Devuelve `text: str` sin parsear. 34 tests (wrapper, caché, endpoint). **Cerrado** | Resuelto |
| ~~**WU6**~~ | `EstimationRequest` en Streamlit → `POST /api/v1/estimate`. 9 tests con transporte mockeado. **Cerrado** (todavía sin probar contra la API real) | Resuelto |
| **WU7** | Slice vertical end-to-end con `mock_response` | Medio |
| **WU8** | Structured output + `ijson` + eventos `tarea` | **Alto** — depende de WU0 |
| **WU9** | Guardrails entrada/salida + `IncompleteJSONError` + política de reintento | Medio |
| **WU10** | Caché semántico + guarda anti-envenenamiento + `prompt_version` | Medio |

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

**Cachear después de validar, siempre.** Con async el coste no existe hasta que
el stream cierra. Cachear al inicio significaría cachear un coste unknowable y un
JSON sin verificar. Es la política, no una preferencia.

---

## 9. Riesgos abiertos

| Riesgo | Impacto | Mitigación |
|--------|---------|------------|
| `response_format` + `stream` no soportado por el provider | Rompe la capa 3 | WU0 lo mide antes de WU8 |
| El modelo inventa IDs de tarea en `depende_de` | `T3` depende de una tarea inexistente | Validación cruzada de referencias en WU8 |
| `horas` incoherentes con el tamaño de equipo | Estimación absurda | Regla en guardrails de salida, WU9 |
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
