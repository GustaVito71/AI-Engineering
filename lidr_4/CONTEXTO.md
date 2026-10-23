# lidr_4 — Contexto para retomar el proyecto en otra sesión

> Resumen escrito el 13/10/2026 al cerrar la sesión de WU10 (caché semántica). Pensado para que una sesión nueva entienda de qué va el proyecto, qué se hizo, por qué, cómo se trabaja con el usuario y qué falta. Las fuentes de verdad siguen siendo `README.md` (estado actual, variables, estructura) y `PLAN.md` (el porqué de las decisiones, §8 "Decisiones que parecen extrañas y no lo son", §9 riesgos). Si este resumen contradice a esos dos, mandan ellos.

## 1. Qué es

`lidr_4` es un servicio **FastAPI** que estima proyectos de software con un LLM. Es la quinta capa de un curso (Master en AI Engineering, programa LIDR): evoluciona `../lidr_3/` para **controlar la variabilidad de la salida del modelo**. El usuario (Gustavo) lo construye paso a paso, por unidades de trabajo (WU0–WU10), y pide a Claude que lo implemente, revise y mida.

Cinco capas, en orden de dependencia: (1) formulario tipado, (2) plantillas Jinja2 versionadas, (3) salida estructurada validada, (4) guardrails de entrada y salida, (5) caché semántica.

Repo: `GustaVito71/AI-Engineering` (raíz de git = la carpeta padre `AI Engineering`; el proyecto vive en `lidr_4/`). Referencia del curso: `LIDR-academy/ai-engineering`, ramas `session_3` y `session_4_live` (`estimator/app/...`). Hay también `lidr4_arquitectura.pdf` en la raíz del proyecto.

## 2. Cómo trabajar con el usuario

- **Idioma:** respuestas en español, **cortas y concisas** (preferencia explícita). Mensajes al usuario, comentarios y docstrings en **español**; identificadores (variables, funciones, clases) en **inglés**. Los nombres de eventos y campos de log quedan en español (`estimacion_completada`, `cache_semantica_acierto`).
- **Git:** se trabaja en la rama `entrega/lidr_4`; se commitea, se hace `git push origin entrega/lidr_4`, y luego `git checkout main && git merge --ff-only entrega/lidr_4 && git push origin main`. Los commits terminan con las dos líneas de atribución que indique el sistema (`Co-Authored-By: ...` y `Claude-Session: ...`). El usuario ejecuta los comandos él mismo y pega la salida; usar `git add -A <rutas>` con pathspec (una vez commiteó mal por no usarlo; el `similitudes.csv` de calibración no debe entrar).
- **Entorno de la sesión de desarrollo:** Claude trabaja en una copia (worktree) en un contenedor sin salida a `api.openai.com`; **no puede medir con el modelo real**. Las mediciones (`test/medir_v4.py`, `test/calibrar_cache.py`) las corre el usuario en su máquina y pega la salida. Los archivos se entregan a la carpeta del usuario (`/home/gustavo/AI Engineering/lidr_4`) por el puente de dispositivo.
- **Verificación antes de entregar:** `pytest` (801 pasan), `ruff check .` y `ruff format --check .` limpios. Se usaron mutaciones manuales para comprobar que los tests muerden.
- **Preguntas:** el usuario suele contestar "[No preference]" y deja los valores recomendados; conviene decidir con un buen defecto y decirlo.

## 3. Stack y ejecución

Python ≥3.11, FastAPI, Pydantic v2, **LiteLLM `Router`** (único dueño de retry y fallback), **Instructor** (salida estructurada sobre el Router), Jinja2 + YAML (prompts versionados), Redis (`redis:7-alpine`), Streamlit (frontend), structlog, `uv` para dependencias, `ijson` declarado pero **aún sin usar**. Modelo primario `openai/gpt-4o-mini`, respaldo `anthropic/claude-haiku-4-5`.

- Todo corre en Docker (`docker-compose.yml`): `redis`, `api` (127.0.0.1:8001) y `frontend` (127.0.0.1:8501), una sola imagen. El código se **copia** a la imagen: tras cambiar código, `docker compose up --build -d`. Un cambio solo en `.env` pide `docker compose up -d --force-recreate api`. `get_settings()` está cacheado: un cambio de configuración exige reiniciar.
- Tests y herramientas corren fuera de Docker: `uv run pytest -q`, `uv run ruff check . && uv run ruff format --check .`.
- Scripts de medición con el modelo real (cuestan centavos, no son tests, pytest no los recoge): `uv run python test/medir_v4.py -n 3` y `uv run python test/calibrar_cache.py --csv similitudes.csv`.

## 4. Arquitectura: el pipeline

`router` (solo traduce HTTP y errores) → `EstimationService` (`app/services/estimation.py`) → `LLMWrapper` (`app/services/llm_wrapper.py`).

Orden pedido por el usuario y vigente en los endpoints estructurado y renderizado:

**guardrails de entrada → caché exacta → caché semántica → render del prompt → LLM (Instructor re-pregunta si el schema rechaza) → guardrails de salida → escribir caché exacta y semántica → avisos (`warnings`)**

Puntos clave:
- La caché exacta vive en el servicio (antes estaba en el wrapper). Su clave es el hash de modelo + versión del prompt + parámetros + descripción ya filtrada; **no** el hash de los prompts. Editar una plantilla sin subir de versión no invalida la caché; las versiones publicadas son **inmutables**.
- Se cachea **solo después de validar**.
- El servicio emite `estimacion_completada` también en los aciertos de caché (con `desde_cache=True`, `tipo_cache=semantica` si fue semántico); el wrapper solo en llamadas reales.
- Los errores no se traducen en el servicio: llegan al router (`PromptRenderError` → 500, 503 sin clave del primario, 502/504 del proveedor, 422 de guardrails de entrada).

## 5. Endpoints y versiones de prompt

| Endpoint | Salida | Prompt por defecto |
|---|---|---|
| `POST /api/v1/estimate` | texto libre (`text`), sin parsear; solo caché exacta | `v2` (`PROMPT_VERSION`) |
| `POST /api/v1/estimate/structured` | `StructuredResult` validado (+ `cached`, `warnings`) | `v3` |
| `POST /api/v1/estimate/rendered` | `RenderedResult` = estructurado + `rendered` (la presentación del `output_format`, validada contra las cifras) | `v4` |
| `GET /health` | estado y avisos de configuración | — |

Entrada (`EstimationRequest`): `description` (20–2000 caracteres tras recortar espacios; `extra="forbid"`), `project_type` (`mobile_app`, `web_saas`, `internal_tool`, `data_pipeline`), `detail_level` (`summary`, `medium`, `detailed`), `output_format` (`phases_table`, `line_items`, `narrative`). `?prompt_version=` elige otra versión publicada.

Prompts en `app/prompts/estimation/v1..v4/` (`system.j2`, `user.j2`, `examples.j2`, `examples.yaml`). `v1` inglés, `v2` castellano (mismos datos), `v3` estructurado (el cliente decide la presentación), `v4` renderizado (`output_format` entra al prompt). Los few-shot se versionan con el prompt, maquetados según `output_format`; tarifas, roles y redondeo viven en el YAML.

Frontend Streamlit (`streamlit_app.py`): formulario con dos interruptores excluyentes, "Salida estructurada" y "Salida renderizada".

## 6. El contrato de salida y sus validadores

`StructuredResult` (`app/schemas/structured_estimation.py`): `phases` (nombre, resumen, semanas, horas, coste EUR, confianza, supuestos, riesgos con mitigación), `team` (rol + cantidad), `totals` (horas, coste, semanas), `summary`, `confidence_pct`, y el caso de **rechazo/fuera de alcance**. Los totales **los declara el modelo** (decisión del 29/09: se intentó calcularlos en código y se revirtió; ver PLAN §3.4).

**Regla de diseño:** lo que *rechaza* es un validador del schema (Instructor solo re-pregunta si un validador lanza; cada intento se paga; tras `STRUCTURED_MAX_RETRIES`=2 re-preguntas sin éxito → 502). Lo que *avisa* va a `warnings`. El schema **no puede importar `app.guardrails`** (ciclo).

Validadores que rechazan y re-preguntan, en orden: nombres de fase únicos → confianza coherente con el rechazo → forma coherente con el resultado → **horas caben en el equipo** (≤40 h por persona y semana, `_hours_fit_team`) → **totales a ≤5 %** de la suma de fases (`TOTALS_MAX_DEVIATION_PCT`; por debajo del 5 % pero distintos, aviso) → **coste/hora dentro de las tarifas** (50–62,5 EUR/h con 5 % de margen) → **duración total coherente** (tolerancia 0,5 semanas) → **cifras del resumen existentes** (el resumen no puede inventar cifras). Las tres últimas (paso 3 de WU9 B) no se aplican a estimaciones fuera de alcance. `RenderedResult` añade `_rendered_matches_format`: la presentación debe corresponder al `output_format` y a las cifras.

## 7. Guardrails (`app/guardrails/`)

- `input.py`: **inyección de prompt** por regex (castellano e inglés; rechaza con 422, mensaje fijo por regla, sin repetir lo escrito), **datos personales** (email, teléfono, IBAN, tarjeta, documentos con formato: se **ocultan** con `[EMAIL]` etc. antes del render, la caché y el log, y se avisa en `warnings`; no se rechaza porque la descripción sigue siendo válida) y **moderación de OpenAI** (rechaza con 422; **falla abierta**; apagada sin `OPENAI_API_KEY`; recibe el texto ya oculto). Son heurísticas, con límites documentados en PLAN §8–§9.
- `output.py`: avisos y eventos sobre la estimación (totales, `confianza_baja` por debajo de `LOW_CONFIDENCE_WARN_PCT`=70, fuera de alcance) y **descarte con 502** (sin re-preguntar ni cachear) si la salida trae **datos personales** (`datos_personales`) o **filtra el prompt** (`prompt_filtrado`: etiquetas del prompt salvo `estimation` y frases como "system prompt"). Evento `salida_rechazada` sin texto ni datos.
- `detectors.py`: detectores puros compartidos (`normalize_text`, `find_prompt_leak`, PII...).
- Interruptores: `GUARDRAILS_ENABLED` (entrada), `OUTPUT_GUARDRAILS_ENABLED` (descarte, avisos y eventos de salida; **no** apaga las reglas del schema), `MODERATION_ENABLED`.
- `IncompleteJSONError`: respuesta cortada por `max_tokens` → no se reintenta, 502 con mensaje propio y coste en el log (se sube `LLM_MAX_TOKENS`).

## 8. Caché semántica (WU10, 13/10/2026)

Archivo: `app/services/semantic_cache.py`. Se activa en los endpoints **estructurado y renderizado** (no en texto libre: ahí no hay validación y repartir lo no comprobado a otras descripciones sería peligroso).

- **Embeddings:** `text-embedding-3-small` por `litellm.aembedding` (API externa, decisión del usuario; el mismo modelo que la referencia `session_4_live`, que usa redisvl + redis-stack). Aquí se usa **Redis plano** (el compose sigue en `redis:7-alpine`): un hash por espacio de búsqueda con el vector normalizado (float32, base64) y una clave por entrada; el coseno se calcula en la app; tope `SEMANTIC_CACHE_MAX_ENTRIES` (200) que desaloja la más antigua. TTL = `CACHE_TTL`.
- **Espacio de búsqueda:** las partes de la clave exacta sin la descripción (modelo, `prompt_version`, tipo, nivel de detalle, formato en v4) + modelo de embeddings. `prompt_version` es parte del espacio porque la caché semántica no se invalida sola al cambiar el prompt.
- **Un acierto exige:** misma búsqueda, similitud ≥ `CACHE_SEMANTIC_THRESHOLD` (**0,95** por defecto) y **mismas cifras** en la descripción (`extract_numbers`: «20.000», «20,000» y «20000» son iguales; «B2B»/«i18n» no cuentan; los embeddings casi no distinguen 20.000 de 200.000). Recorre las candidatas de mayor a menor similitud y se queda con la primera que cumple ambas.
- **Qué se guarda:** solo estimaciones sanas: no rechazo/fuera de alcance, confianza ≥ `LOW_CONFIDENCE_WARN_PCT`, totales cuadrados. Se escribe después de los guardrails de salida. El evento `cache_semantica_no_guardada` lleva `motivos` (`fuera_de_alcance`, `confianza_baja`, `totales_no_cuadran`).
- **Al acertar:** copia la estimación a la caché exacta, añade el aviso «Estimación reutilizada de una consulta muy parecida (similitud N %)…» y el evento lleva `tipo_cache=semantica`.
- **Fail soft:** un fallo del embedding o de Redis cuenta como miss. Ningún evento lleva la descripción.
- **Encendida por defecto** (`SEMANTIC_CACHE_ENABLED=true`, decisión del usuario). Sin Redis o sin clave del proveedor de embeddings queda apagada con `cache_semantica_desactivada`. `SEMANTIC_CACHE_LOG_ONLY=true` busca y registra la similitud sin servir nada (para calibrar con tráfico real).
- **Eventos:** `cache_semantica_activada/desactivada/acierto/fallo` (con `entradas` y `mejor_similitud`)`/cifras_distintas/solo_log/no_guardada/no_disponible`.
- **Variables:** `SEMANTIC_CACHE_ENABLED`, `CACHE_SEMANTIC_THRESHOLD`, `EMBEDDING_MODEL`, `EMBEDDING_TIMEOUT` (3 s), `SEMANTIC_CACHE_LOG_ONLY`, `SEMANTIC_CACHE_MAX_ENTRIES`.

## 9. Qué se midió con el modelo real (gpt-4o-mini)

- **`v4`, `medir_v4.py` (9 casos × 3 reps, tras el paso 3):** 27/27 válidas, 3 reintentos (todos por totales a más de 5 %, regla antigua), 0 fases fuera de tarifa, 0,0547 USD en total. Antes (11/10): 4 reintentos por totales; ningún rechazo del validador de `rendered`; ninguna respuesta cerca de `LLM_MAX_TOKENS`. El coste por llamada ronda 0,0016 USD (~10.000 tokens de prompt, ~1.500 de salida, 11–14 s).
- **Variación entre ejecuciones del mismo caso:** hasta 39 % de rango en horas y 41 % en coste en algunos casos (con temperatura 0,1). No se debe prometer una cifra estable. El coste no se redondea al múltiplo de 50 en ~25 % de las fases (aceptado, es de estilo).
- **Calibración del umbral (`calibrar_cache.py`, 23 pares escritos a mano):** a 0,92 se servían 5 de 10 paráfrasis y había **un falso acierto** («con» vs «sin pagos», 0,937). El piso sin falsos aciertos es 0,94 (sirve 2 de 10); por eso el defecto es **0,95**. Las "cifras distintas" tienen las similitudes más altas (0,95–0,97): sin la guarda se servirían estimaciones equivocadas. Muestra chica: falta afinar con tráfico real.
- **Prueba de punta a punta de la caché semántica (13/10):** acierto con similitud 0,9956 en ~0,9 s (frente a 12–14 s) y 0,0016 USD evitados; la guarda de cifras descartó candidatas con similitud 0,987–0,997; recorre las entradas hasta hallar una con las mismas cifras. Con `gpt-4o-mini` la confianza global suele salir en 65–70 para descripciones cortas: con el 70 por defecto, nada se guardaba. En el `.env` local del usuario `LOW_CONFIDENCE_WARN_PCT=60` (también afecta al aviso de baja confianza).

## 10. Historial de las unidades de trabajo

| WU | Estado | Contenido |
|---|---|---|
| WU0 | Cerrado | Spike de LiteLLM: structured output, stream y fallback (las 3 preguntas dan sí). |
| WU1 | Cerrado | Esqueleto heredado de `lidr_3` (config, main, tracing, cache, infra, CI en matriz). |
| WU2 | Cerrado | Contrato de entrada `EstimationRequest`/`EstimationResponse`, límites en dos capas. |
| WU3 | Revertido | Dominio con totales calculados en código; se revirtió: el modelo declara los totales. |
| WU4 | Cerrado | Prompts Jinja2 versionados + `loader.py`. |
| WU5 | Cerrado | Gateway async con Router, respaldo opcional, caché exacta, errores 503/502/504, trazabilidad. |
| WU6 | Cerrado | Frontend Streamlit. |
| WU7 | Cerrado | Slice end-to-end con `mock_response`. |
| WU8 | **Parcial** | Hecho: salida estructurada y renderizada (Instructor, `v3`/`v4`, endpoints, interruptores), tolerancia de totales, capa de servicio. **Pendiente: streaming con `ijson` y eventos por fase** (§4 del PLAN). |
| WU9 A | Hecho | Guardrails de entrada; la caché exacta pasó al servicio. |
| WU9 B | Hecho | Pasos 1–3: regla de horas, aviso de confianza baja, `IncompleteJSONError`, reorganización de `app/guardrails/` (input/output/detectors), descarte por PII/prompt filtrado (502), validadores anti-alucinación (coste/hora, duración, cifras del resumen). |
| WU10 | Hecho | Caché semántica (sección 8), calibrada. 801 tests en total. |

Commits recientes en `main`: `715faa4` (reorganización de guardrails), `285f90f` (descarte de salida), paso 3 (subido), `51c13cd` (WU10, caché semántica encendida por defecto con umbral 0,95). El último cambio entregado, el campo `motivos` del evento `cache_semantica_no_guardada` (+3 tests), puede no estar commiteado: comprobar con `git status`.

## 11. Pendientes y próximos pasos

1. **WU8 — streaming con `ijson`** (riesgo alto). Implica pedir `stream=True` al proveedor y leer el JSON incremental, emitir eventos SSE por fase (§4 del PLAN) y adaptar Streamlit a pintar fase por fase. Complicaciones ya identificadas: las reglas que miran el conjunto (totales, duración, resumen) solo se pueden comprobar al final; una re-pregunta de Instructor o un descarte de guardrail de salida después de haber enviado fases obliga a que la UI descarte lo pintado; los aciertos de caché se sirven de golpe, no en streaming; la regla de horas habría que aplicarla por fase. Primer paso sugerido: verificar con el modelo real que el proveedor soporta `response_format` + `stream`, y decidir cómo tratar los rechazos tardíos.
2. **Afinar el umbral de la caché semántica con tráfico real** (`SEMANTIC_CACHE_LOG_ONLY=true` unos días y mirar `cache_semantica_solo_log`). Posible mejora: separar la guarda de confianza de la caché en `SEMANTIC_CACHE_MIN_CONFIDENCE_PCT` si 70 resulta demasiado alto para `gpt-4o-mini`.
3. Medir `ijson`/streaming y el ahorro real de llamadas de la caché semántica.

## 12. Límites conocidos (resumen de PLAN §9)

- Respuestas idénticas repetidas entre re-preguntas de Instructor terminan en 502.
- La caché semántica no distingue matices de alcance entre descripciones muy parecidas con las mismas cifras («sin pagos»/«con pagos» a 0,937); el umbral 0,95 y el modo `log_only` son la defensa. Cada miss de la exacta paga un embedding.
- Los guardrails de entrada son heurísticas (regex, sin NER): inyecciones con otra redacción o nombres propios pueden pasar.
- La moderación depende de OpenAI (tope de tasa en el tier gratuito) y falla abierta.
- `/estimate` (texto libre) no valida totales ni se beneficia de la caché semántica.
- El modelo varía entre ejecuciones del mismo caso; la caché devuelve la misma estimación a la misma petición.

## 13. Mapa de archivos

```
lidr_4/
  app/
    main.py            lifespan (Redis, caché semántica), logging
    config.py          Settings (pydantic-settings), validaciones, constantes
    dependencies.py    inyección: wrapper, guardrails, cachés, servicio
    routers/estimations.py        los tres endpoints + traducción de errores
    services/estimation.py        EstimationService (el pipeline)
    services/llm_wrapper.py       Router de LiteLLM + Instructor, coste, errores
    services/cache.py             caché exacta (Redis, fail soft)
    services/semantic_cache.py    caché semántica
    schemas/estimation.py         EstimationRequest / respuesta de texto
    schemas/structured_estimation.py  StructuredResult, RenderedResult y validadores
    guardrails/{input,output,detectors}.py
    prompts/estimation/v1..v4/    plantillas + examples.yaml; prompts/loader.py
    tracing.py         eventos estructurados
  streamlit_app.py
  test/               801 tests (+ medir_v4.py, calibrar_cache.py: scripts, no tests)
  docker-compose.yml, Dockerfile, pyproject.toml, uv.lock, .env.example
  README.md (estado y variables), PLAN.md (porqué), CONTEXTO.md (este archivo)
```
