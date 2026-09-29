# lidr_4 — Estimador estructurado sobre LiteLLM

Quinta capa del curso. Evoluciona el estimador de `../lidr_3/` para controlar la
variabilidad de la salida. `../ganttly/` queda como backlog/playground.

Este documento es el plan de construcción. El `README.md` del proyecto se escribe
al final, cuando el código exista.

---

## 1. Objetivo

Cinco capas, en este orden de dependencia:

| # | Capa | Qué resuelve |
|---|------|--------------|
| 1 | Formulario tipado | El usuario no escribe un prompt |
| 2 | Plantillas Jinja2 versionadas | El prompt es un artefacto revisable, no un string en código |
| 3 | Salida estructurada validada | La variabilidad del LLM deja de ser un problema de parseo |
| 4 | Guardrails | Entrada y salida dejan de ser campos de confianza |
| 5 | Caché semántico | Consultos equivalentes no vuelven a llamar al modelo |

Las capas 1 y 2 eliminan variabilidad *a la entrada*. La 3 la elimina *a la
salida*. La 4 la acota. La 5 la evita.

---

## 2. Decisiones tomadas

Cada decisión lleva el porqué, porque varias parecen arbitrarias hasta que se
conoce el problema que resuelven.

| Decisión | Por qué |
|---|---|
| `SolicitudEstimacion` es el contrato canónico; `SolicitudForm` solo alimenta la UI | La UI puede cambiar sin tocar el dominio ni el prompt |
| Los few-shot se renderizan con la **misma** plantilla que la salida | Los ejemplos quedan válidos contra el schema **por construcción**. Se elimina la clase de bugs "el ejemplo 3 enseña un formato que el schema no permite" |
| `total_horas` y `duracion_semanas` los calcula el código, no el modelo | Tolerancia de error **exactamente 0**. Ver §8 |
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

### 3.1 `SolicitudForm` — capa 1, solo UI

Los nueve campos acordados. El modelo nunca ve este tipo.

```python
class TipoProyecto(StrEnum):
    WEB = "web"
    MOVIL = "movil"
    BACKEND = "backend"
    INTEGRACION = "integracion"
    DATOS = "datos"
    INFRAESTRUCTURA = "infraestructura"


class EstadoCodigo(StrEnum):
    NUEVO = "nuevo"
    EXISTENTE_ESTABLE = "existente_estable"
    LEGACY = "legacy"
    CRITICO = "critico"


class NivelCalidad(StrEnum):
    BASICO = "basico"
    ESTANDAR = "estandar"
    ALTO = "alto"


class SolicitudForm(BaseModel):
    tipo_proyecto: TipoProyecto
    estado_codigo: EstadoCodigo
    stack: list[str]
    nivel_calidad: NivelCalidad
    requisitos_no_funcionales: list[str]
    integraciones: list[str]
    tamano_equipo: int
    deadline_semanas: int | None  # None = sin deadline
    descripcion: str
```

**Sobre el riesgo:** no es un campo. El modelo lo infiere de
`estado_codigo`, `requisitos_no_funcionales` e `integraciones`. Añadirlo como
campo explícito sería pedirle al usuario que cuantifique algo que todavía no sabe.

### 3.2 `SolicitudEstimacion` — capa 3, contrato del LLM

Es literalmente el `response_format` de structured output. **No contiene
totales.**

```python
class Riesgo(StrEnum):
    BAJO = "bajo"
    MEDIO = "medio"
    ALTO = "alto"


class Tarea(BaseModel):
    id: str  # "T1", "T2"... para referencias entre tareas
    titulo: str
    descripcion: str
    horas: int  # > 0
    depende_de: list[str]  # ids de tareas previas
    riesgo: Riesgo


class RolEquipo(BaseModel):
    rol: str
    cantidad: int
    foco: str


class Supuesto(BaseModel):
    texto: str
    impacto: str  # qué pasa si el supuesto es falso


class SolicitudEstimacion(BaseModel):
    resumen: str
    tareas: list[Tarea]
    equipo: list[RolEquipo]
    supuestos: list[Supuesto]
    riesgos: list[str]
    advertencias: list[str]
```

### 3.3 `EstimacionCompleta` — lo que sale del sistema

```python
class EstimacionCompleta(SolicitudEstimacion):
    total_horas: int  # calculado
    duracion_semanas: int  # calculado
```

Se valida con un modelo base, y los totales se calculan **después**, en código:

```python
def calcular_total(estimacion: SolicitudEstimacion) -> int:
    return sum(t.horas for t in estimacion.tareas)


def calcular_semanas(estimacion: SolicitudEstimacion) -> int:
    # capacidad = horas del equipo ajustadas por enfoque (≈0.8, no todos programan a la vez)
    return max(1, ceil(total_horas / (tamano_equipo * 40 * 0.8)))
```

El `0.8` documenta su propio supuesto: el equipo no dedica el 100% del tiempo a
esta estimación. Es una decisión de negocio, no del modelo, y por eso vive en
código donde se puede discutir y testear.

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
| `app/schemas/estimation.py` | 57 | `transcription: str` → `SolicitudForm` |
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
| **WU1** | Heredar el esqueleto de `lidr_3` (§5): copiar `config.py`, `main.py`, `tracing.py`, `cache.py` y la infra. CI para `lidr_4` **y `lidr_3`**, `ruff format --check` | Bajo |
| **WU2** | Dominio: enums, `SolicitudForm`, `SolicitudEstimacion`, `calcular_total()`, `calcular_semanas()`. Tests puros, sin I/O | Bajo |
| **WU3** | `prompts/estimacion.v1.j2` versionado. Los few-shot salen de la misma plantilla | Bajo |
| **WU4** | Gateway async: `Router`, dispatch, tracing, pre-arranque. Coste con `completion_cost()` | Medio |
| **WU5** | `SolicitudForm` en Streamlit → dominio | Bajo |
| **WU6** | Slice vertical end-to-end con `mock_response` | Medio |
| **WU7** | Structured output + `ijson` + eventos `tarea` | **Alto** — depende de WU0 |
| **WU8** | Guardrails entrada/salida + `IncompleteJSONError` + política de reintento | Medio |
| **WU9** | Caché semántico + guarda anti-envenenamiento + `prompt_version` | Medio |

**WU1 incluye un arreglo que no es de esta entrega:** el CI actual solo corre
`lidr_2/**`, así que los tests de `lidr_3` nunca se ejecutaron. Pasarlos a una
matriz los hace correr por primera vez. El primer push es el diagnóstico: si
`lidr_3` aparece rojo, se decide con el dato a la vista (`continue-on-error`
anotando la deuda, o un fix), no antes.

**Sobre los tests de `lidr_3`: no se porta ninguno.** Escribirlos de nuevo contra
el código nuevo, en la unidad donde aparece. La alternativa —copiar las ~2300
líneas de `test/`— es reescribirlas con pasos extra, porque están atadas a
`llm_provider`, `build_cache_key`, `transcription` y a un contrato de streaming
que ya no existe. Lo único que vale de esa suite son dos insights, y ya están
en los docstrings de `config.py` y `tracing.py` que heredamos.

**No hay unidad de tests.** Los tests son parte de la unidad que escribe el
código, nunca una tarea aparte: WU2 los de dominio (puros, sin I/O, sin red),
WU3 los de la plantilla, WU4 los del gateway, WU9 los del keying. Si una unidad
agrega comportamiento, agrega su test en el mismo commit.

El más importante de todos es el de WU3, y no se puede escribir antes: extraer
los few-shot renderizados y pasarlos por `SolicitudEstimacion.model_validate_json`.
Es el único que demuestra que "los ejemplos salen de la misma plantilla que la
salida" funciona de verdad, y es el que la suite de `lidr_3` no podía tener
porque no había schema contra el cual fallar.

Orden deliberado: el dominio (WU2) va antes que la infraestructura (WU4)
porque las funciones de cálculo no necesitan red para testearse, y son la parte
que más cambios de requisitos va a tolerar.

---

## 7. WU0 — resultado del spike

Ejecutado contra OpenAI y Anthropic con el schema real (3 tareas, strict).
Las tres preguntas quedan respondidas: **sí, sí, sí.** La capa 3 se puede
diseñar con eventos de fila. Pero con dos reserve que cambian WU7 y WU9.

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

**Decisión para WU7: `tareas` va primero en el schema, `solicitud` al final.**
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
candidato ideal. Eso le da a WU9 más peso del que tenía y es la razón de que
`PROMPT_VERSION` tenga que ser explícito.

**Criterio de decisión, ya ejecutado:** si la 1 fallaba, la capa 3 se rediseñaba
con eventos de token. No hizo falta.

---

## 8. Decisiones que parecen extrañas y no lo son

Documentadas aquí para que nadie las "corrija" después.

**Los totales no los declara el modelo.** La referencia de `session_3` los pide
y después los verifica con `abs(suma - declarado) <= 1` para las horas y `<= 2%`
para el coste. Esas tolerancias son una admisión de que la aritmética del modelo
no es confiable. Acá la tolerancia es 0 porque no hay nada que validar.

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
| `response_format` + `stream` no soportado por el provider | Rompe la capa 3 | WU0 lo mide antes de WU7 |
| El modelo inventa IDs de tarea en `depende_de` | `T3` depende de una tarea inexistente | Validación cruzada de referencias en WU2 |
| `horas` incoherentes con el tamaño de equipo | Estimación absurda | Regla en guardrails de salida, WU8 |
| Caché semántico sirve una estimación de otro dominio | Resultado plausible pero fuera de tema | Namespace por `prompt_version` + umbral alto, WU9 |
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
