# PLAN — lidr_5

Plan de implementación derivado de `ENUNCIADO.md`. Para las decisiones y su motivo, ver [CONTEXTO.md](CONTEXTO.md).

## Objetivo

Convertir el estimador de lidr_4 (transaccional) en una conversación iterativa: el servicio recuerda el proyecto en curso entre turnos y acepta adjuntos PDF y Word.

Por decisión del responsable, el objetivo de lidr_5 incluye los tres mecanismos para enriquecer en tiempo de ejecución el contexto de un sistema CAG, sin pasar todavía a RAG: adjuntos (Pasos 3 y 4), búsqueda web (Paso 11) y consultas a la BBDD del backend de negocio por function calling (Paso 12). Reglas: el contexto dinámico es input y no programa (se delimita, nunca se interpreta como instrucciones), tiene costo en tokens por petición y suma latencia. Se activa con criterio, no por defecto.

## Decisiones de referencia

- **Adjuntos:** camino B (extracción local con pypdf y python-docx). Motivos y costo: ver CONTEXTO.md.
- **Memoria de sesiones:** diccionario en memoria del proceso, sin BBDD ni Redis. Requiere `uvicorn --workers=1`.
- **`project_metadata`:** extraído con un LLM. Primario: Gemini. Respaldo: DeepSeek (`EXTRACTOR_FALLBACK`).
- **Ventana deslizante:** `MAX_TURNS = 6` por defecto, configurable.
- **Caché:** exact-match y semántica heredadas de lidr_4, sobre Redis plano (`redis:7-alpine`): la semántica calcula la similitud en la app y no usa RediSearch.
- **Puertos:** API en 8001; Streamlit en 8002.
- **Búsqueda web (sin paso asignado):** servicio independiente expuesto al LLM como función. Tavily principal, Firecrawl de respaldo (`SEARCH_DEFAULT`, `SEARCH_FALLBACK`); Exa descartada por bloqueo de la cuenta. Amplía el ENUNCIADO y su paso se diseñará cuando el responsable termine de detallar el mecanismo. Reglas ya fijadas con mediciones: Firecrawl con `scrapeOptions={}`, `snippet` truncado por resultado (del orden de 1.500 caracteres) y máximo de 3 resultados por búsqueda. Motivos y riesgos: ver CONTEXTO.md.
- **Funcionalidad de lidr_4:** no cambia, salvo lo que el ENUNCIADO pide modificar. Lo nuevo se suma a lo existente. Lo que el enunciado modifica: el template del system prompt (Paso 4) y el cliente Streamlit (Paso 6).
- **ACB (Actor-Critic-Boss):** entra en el alcance por decisión del responsable, aunque el ENUNCIADO lo deja para el directo. Se construye sin tier dinámico ni compresión (Paso 8) y deja puntos de extensión para sumarlos en la etapa final (Paso 9).
- **Tier:** fijo en `default` (configurable con `DEFAULT_TIER`) hasta el Paso 9. Valores del enum, tomados de `session_05_live`: `executive`, `pm`, `developer`, `default`.

## Pasos

### Paso 1 — Modelar el estado de la sesión

- Crear `app/sessions/models.py` (o `sessions.py`, según la estructura final) con:
  - `ConversationHistory`: lista limitada de mensajes con ventana deslizante. Conserva siempre el system prompt.
  - `ProjectMetadata`: modelo Pydantic con `project_name`, `assumed_team_size`, `mentioned_technologies` (lista) y `agreed_scope` (texto libre).
  - `Session`: agrupa historial y metadata, indexado por `session_id`.
- Documentar en docstrings por qué se acepta la volatilidad de la memoria.
- **Hecho cuando:** los modelos existen y tienen tests unitarios.
- **Estado (2026-10-18): aplicado.** `app/sessions/` (`models.py`, `store.py`), `max_turns` en `Settings` (`MAX_TURNS`, 6 por defecto, mínimo 1) y `test/test_sessions_models.py` (30 tests). Detalle en CONTEXTO.md.

### Paso 2 — Crear sesiones

- `POST /sessions` devuelve `{"session_id": "<uuid4>"}`.
- **Hecho cuando:** el endpoint responde y la sesión queda registrada en el store.
- **Estado (2026-10-18): aplicado.** `app/routers/sessions.py` (responde 201), `app/schemas/session.py` (`CreateSessionResponse`), `get_session_store` en `app/dependencies.py` y el almacén creado en el `lifespan` de `app/main.py`. Tests: `test/test_sessions_endpoint.py` (9). No hay `GET /sessions`: conocer el identificador es lo único que da acceso a una sesión.

### Paso 3 — Adjuntos en el endpoint principal

- `POST /sessions/{session_id}/estimate` acepta `multipart/form-data`:
  - `transcript`: string.
  - `attachments`: lista opcional de `UploadFile`.
- Los endpoints de lidr_4 (`POST /api/v1/estimate`, `/structured`, `/rendered`) siguen disponibles. Ninguno lo quita el enunciado.
- Camino B: extraer texto de cada adjunto y concatenarlo al transcript con el separador `--- attachment: <filename> ---`.
- Adjunto sin texto (por ejemplo, un PDF escaneado, para el que `pypdf` devuelve texto vacío): la respuesta lleva un aviso explícito por archivo en `warnings` («no se pudo extraer texto de x.pdf; puede estar escaneado») y el archivo no se concatena al prompt. El OCR queda fuera de alcance.
- Documentar en el README el camino elegido y el motivo.
- Límites: transcripción de 20 a 80.000 caracteres. Adjuntos `.pdf`, `.docx`, `.txt` y `.md`; otros formatos responden 415. Máximo 10 MB por archivo y 5 archivos por petición. El texto extraído se trunca a 60.000 caracteres por archivo y a 150.000 entre todos (tope total decidido el 2026-10-20: se recorta el que no cabe y no entran los que quedan sin cupo); la respuesta incluye un aviso en cada caso.
- **Hecho cuando:** un PDF y un Word se convierten en texto y llegan al prompt.
- **Estado (2026-10-18): aplicado.** `app/services/attachments.py` (extracción y bloque `<attachments>`), `EstimationService.estimate_conversation` y el endpoint en `app/routers/sessions.py`. Códigos: 404 sesión inexistente, 422 transcripción fuera de rango, versión de prompt inexistente o más de 5 archivos, 415 formato, 413 tamaño. Sin caché; aún no guarda el historial (Paso 5). Tests: `test/test_attachments.py` (23) y `test/test_sessions_estimate_endpoint.py` (18). El aviso de recorte de un PDF indica hasta qué página llegó el texto y cuál es la última con texto.

### Paso 4 — Inyección de `project_metadata` en el system prompt

- Modificar el template Jinja2 del system prompt para incluir un bloque `<project_metadata>`. Si la metadata está vacía (primera llamada), el bloque va vacío.
- El bloque `<project_metadata>` va en una versión nueva del prompt: **v5**, en `app/prompts/estimation/v5/`. Las versiones v1 a v4 no cambian.
- Metadata extractor: `app/prompts/metadata_extraction/v1/`.
- v5 incluye también un bloque `<audience>` dirigido por el tier, que se renderiza solo si el tier no es `default`. Mientras el tier sea fijo en `default`, no aparece y v5 sigue idéntico a la base.
- Sin metadata, el prompt v5 renderizado debe ser idéntico al de la versión sobre la que se construye; los tests de prompts de lidr_4 pasan sin cambios.
- Tras cada respuesta del LLM, actualizar la metadata con un extractor LLM (una llamada extra por turno).
- Justificar la elección en el README: el extractor es más robusto que una heurística y el benchmark ([CONTEXTO.md](CONTEXTO.md#resultados-del-benchmark-sintético-n--12-scorer-v1)) favorece a Gemini por latencia.
- **Hecho cuando:** la metadata se actualiza entre turnos y aparece en el system prompt.
- **Estado (2026-10-19): aplicado.** Decisiones: (1) `ProjectMetadata` suma `explicit_constraints` y `rejected_options` para poder olvidar con criterio (política 1: revisión explícita del usuario). (2) El extractor recibe la metadata actual y el último turno y devuelve la metadata **completa**, no un parche, para poder quitar y mover hechos; `ProjectMetadata.normalized` aplica por código que lo descartado salga de las tecnologías vigentes. (3) v5 conserva `<project_description>` (no `<transcript>`): así, sin metadata y con tier `default`, su prompt es idéntico al de v3, como exige este paso. (4) El endpoint de sesiones usa `CONVERSATION_PROMPT_VERSION` (v5 por defecto). (5) Si el extractor falla o no está configurado, la estimación se entrega, la memoria anterior se conserva y hay un aviso en `warnings`. (6) El extractor reutiliza `LLMWrapper` con `Settings.for_extractor()`. (7) Publicar v5 amplía a «v3, v5» las versiones estructuradas que listan `/health` y los 422 de lidr_4: se actualizaron las aserciones de tests que enumeraban solo `v3`. (8) Las etiquetas nuevas (`project_metadata`, `audience`, `attachments`, etc.) se suman a `PROMPT_TAGS` del guardrail de entrada. Tests: `test/test_prompts_v5.py` (12), `test/test_metadata_extractor.py` (18), `test/test_sessions_memory_endpoint.py` (9).

### Paso 5 — Ventana deslizante

- Implementar la ventana en `ConversationHistory`:
  - El system prompt es un invariante.
  - Un turno es un par usuario + asistente.
  - Al superar `MAX_TURNS`, se descartan los pares más antiguos.
  - `to_messages_list()` devuelve el array `messages` listo para el LLM, con el system prompt regenerado desde la metadata actual.
- **Hecho cuando:** ningún envío al LLM supera `MAX_TURNS` pares.
- **Estado (2026-10-19): aplicado.** `ConversationHistory` ya existía (Paso 2); ahora el endpoint la usa: pasa `session.history.to_history_messages()` al servicio y al wrapper (`history=`, entre el system prompt y el mensaje actual) y, tras una estimación exitosa, anota el turno (mensaje del usuario recortado a 4.000 caracteres + resumen de la estimación; los adjuntos no se guardan). Un turno fallido no se anota. Se suman las políticas de olvido 2 y 3: las sesiones caducan tras `SESSION_TTL_SECONDS` sin uso (24 h; 404) y hay un tope `MAX_SESSIONS` (1.000; al llenarse se desaloja la menos usada); el reset sigue siendo `POST /sessions`. Tests: `test/test_sessions_history.py` (14) y uno en `test_llm_wrapper_structured.py`.

### Paso 6 — Cliente Streamlit

- Crear una sesión al cargar la página y guardar el `session_id` en `session_state`.
- Campo de texto para la transcripción y selector múltiple de archivos.
- Panel lateral o expandible con la `project_metadata` actual.
- Botón «Nueva conversación» que vuelve a llamar a `POST /sessions` y reinicia el estado.
- La pestaña «Estimación Dinámica» envía `project_type`, `detail_level` y `output_format`, y muestra la estimación estructurada con las vistas de lidr_4.
- Interruptor «Revisar con crítico (ACB)» en esa pestaña: si está activo, el envío va a `POST /sessions/{session_id}/estimate-acb` y la estimación se acompaña de un expander con la traza (decisión final, iteraciones, veredicto y confianza del crítico, y problemas por ronda). Apagado, usa `/estimate`.
- Funcionalidad de lidr_4 que se conserva en el cliente:
  - Panel de conexión: URL de la API editable, «Probar conexión» y avisos de falta de API key y de respaldo no disponible.
  - Modos de salida: texto libre, estructurada y renderizada, con las mismas vistas (métricas, tabla de fases, partidas o narrativa, equipo, supuestos y riesgos, aviso de fuera de alcance).
  - Selector de versión del prompt, leído desde `/health`.
  - Validación de longitud de la descripción (20 a 2000 caracteres), sin cambios en su pestaña. La transcripción nueva define sus propias reglas (pendiente).
  - Indicador de caché y manejo de errores (422 legibles y fallo de conexión con `docker compose up -d`).
- Propuesta pendiente de confirmar: la interfaz de lidr_4 queda intacta en una pestaña «Estimación», y la conversación va en una pestaña nueva «Estimación Dinámica». Prototipo navegable del resultado: https://claude.ai/artifact/TprjnSK75VUekykxyeWKQd (no cubre el código).
- **Hecho cuando:** un usuario puede mantener una conversación de tres turnos desde el navegador, y las funciones de lidr_4 listadas arriba siguen funcionando.
- **Estado (2026-10-19): aplicado; el interruptor ACB se añadió con el Paso 8 (2026-10-20). Falta la prueba manual en el navegador.** Decisiones: (1) dos pestañas, «Estimación» con la interfaz de lidr_4 intacta y «Estimación Dinámica»; la barra lateral (URL, conexión, salida y versión) sigue siendo de la pestaña de lidr_4. (2) Nuevo endpoint `GET /sessions/{id}` (turnos recordados, ventana y `project_metadata`, sin el texto de los mensajes) para el panel «Memoria del proyecto»; no estaba en el plan. (3) La pestaña dinámica muestra la salida estructurada con las vistas de lidr_4; los modos texto y renderizado no existen para sesiones. (4) Reglas de entrada: mensaje de 20 a 80.000 caracteres, hasta 5 adjuntos de 10 MB (.pdf, .docx, .txt, .md), validadas en el cliente antes de enviar. (5) Si la sesión caducó (404), se crea otra, se avisa y el usuario reenvía el mensaje. (6) Todo sigue en `streamlit_app.py` porque el Dockerfile solo copia ese archivo. (7) Los dos interruptores de salida y el selector de versión de lidr_4 se reemplazaron (pedido del responsable) por un único selector dentro de la pestaña «Estimación», que ofrece v2 (texto libre), v3 (estructurada) y v4 (renderizada); la versión elegida decide también el endpoint y siempre viaja explícita en `?prompt_version=`. v1 y v5 no se ofrecen ahí (v5 es de la pestaña dinámica; la API sigue aceptando v1). La barra lateral conserva URL, «Probar conexión» y la memoria del proyecto. Cambio de comportamiento respecto de lidr_4: ya no existe la opción «Predeterminada» (la del servidor) en esa pestaña. (8) El frontend se lanza con `--server.maxUploadSize=10` para que Streamlit anuncie el mismo tope de 10 MB que valida la API. Tests: `test/test_frontend_dynamic.py` (17, con `AppTest` de Streamlit y `httpx` simulado) y 3 de `GET /sessions/{id}`.

### Paso 7 — Tests

Tests de integración con pytest y `httpx.AsyncClient`:

1. Una sesión que enlaza dos peticiones: la `project_metadata` se actualiza.
2. Una petición con un PDF adjunto: un campo de la estimación cambia al añadir el adjunto (test cualitativo).
3. Ocho turnos en la misma sesión: el historial enviado al LLM nunca supera `MAX_TURNS`.

- Los tests de lidr_4 se conservan, incluido `test/test_frontend.py` (cliente Streamlit con `httpx.MockTransport`).
- Los tests del ACB se describen en el Paso 8.
- **Hecho cuando:** los tres tests nuevos y los de lidr_4 pasan en local.
- **Estado (2026-10-19): aplicado.** `test/test_sessions_integration.py` (3 tests, `httpx.AsyncClient` sobre la app real con solo el wrapper y el extractor falsos). Límite del segundo: con un modelo falso comprueba que el texto del PDF llega al prompt (y que sin adjunto no aparece), no que la estimación cambie; esa prueba cualitativa se hace a mano con el modelo real. Los tests de lidr_4, incluido `test_frontend.py`, siguen pasando.

### Paso 8 — ACB (Actor-Critic-Boss)

Referencia: `session_05_live`. El actor propone, el crítico evalúa y el jefe decide: aceptar, iterar o sintetizar.

- Endpoint `POST /sessions/{session_id}/estimate-acb`, con el mismo contrato multipart que `/estimate` (incluye `tier` opcional). Responde con `ACBResponse`: la estimación más un campo `acb` con la traza.
- Piezas nuevas:
  - `app/schemas/acb.py` (`ACBIteration`, `BossTrace`) y `app/schemas/critic.py`.
  - `app/services/critic.py` (función sin estado) y `app/services/boss.py` (orquestación).
  - Prompts del crítico en `app/prompts/critic/v1/`.
- Configuración: `BOSS_MAX_ITERATIONS` (3 en live). No hay `CRITIC_MODEL`: el crítico usa siempre un proveedor distinto del que produjo la estimación, para que no la revise el mismo modelo. Orden de intentos del crítico: (1) el proveedor distinto del que produjo la estimación (actor en `LLM_DEFAULT` → crítico en `LLM_FALLBACK`; actor en `LLM_FALLBACK` → crítico en `LLM_DEFAULT`); (2) si ese falla o no está disponible, el mismo proveedor del actor: se registra un aviso en el log y la traza marca la iteración como `same_model`; (3) si también falla, no hay revisión (`unavailable`). Cada iteración es una llamada del actor y una del crítico: hasta 6 llamadas al LLM por petición.
- Estado de la revisión: `ACBIteration` gana el campo `review_status` (`independent`, `same_model` o `unavailable`). El crítico devuelve el veredicto junto con su estado, en vez de fabricar un `accept` con confianza 0 como en live.
- Reglas del jefe según el estado de la revisión:
  - `independent`: decide como en live (`accept`, `needs_iteration` o `reject`).
  - `same_model`: el veredicto es real y se respeta, salvo que un `accept` baja a `needs_iteration` (el jefe itera mientras queden iteraciones). Los veredictos `needs_iteration` y `reject` se mantienen. Como el crítico seguirá siendo del mismo modelo, el bucle normalmente agota `BOSS_MAX_ITERATIONS` y termina en `synthesize`.
  - `unavailable`: el jefe trata el caso como `reject` y sintetiza de inmediato, sin iterar. Devuelve el borrador del actor con una nota al inicio del resumen que dice que la estimación no pudo ser revisada. La confianza baja a 30, el mínimo que el schema permite sin el prefijo «Fuera de alcance:» (si ya era menor, no se modifica). En la traza, el veredicto del crítico es `unavailable` y la confianza es 0.
  - El texto de la síntesis dice «revisión independiente» solo cuando el estado es `independent`; en los demás casos dice «revisión» a secas (en live siempre dice «independent»).
- Cliente: el expander de la traza muestra la etiqueta «Revisión no independiente» para `same_model` y «Sin revisión del crítico» para `unavailable`.
- El actor usa el prompt de estimación v5 y la misma ventana deslizante que `/estimate`.
- **Decisiones del responsable (2026-10-20):** (a) el crítico recibe el mismo bloque `<attachments>` que el actor, además de la transcripción y `project_metadata`, para poder detectar contradicciones con los metadatos y componentes mencionados que faltan. (b) Categorías de issues = unión de la referencia y del enunciado (9): `math_error`, `hallucination`, `scope_mismatch`, `phase_imbalance`, `missing_assumption`, `unrealistic_estimate`, `tier_mismatch`, `missing_component`, `inconsistency_with_metadata`. (c) Síntesis como en la referencia: último borrador del actor con las observaciones abiertas al inicio del resumen y confianza reducida (mínimo 30), sin llamada extra al LLM; el jefe es código, no una llamada.
- **Puntos de extensión** (para el Paso 9):
  - `app/sessions/tier_resolver.py`: enum `Tier` y `resolve_tier(...)`, con la misma firma que en live. En esta etapa devuelve el override o `DEFAULT_TIER`.
  - `app/sessions/compression.py`: `apply_compression(...)` que por ahora no hace nada. La firma exacta se toma de live al implementar.
- Tests (por definir con el mismo criterio que el Paso 7): crítico, orquestación del jefe y endpoint ACB. Incluyen los tres estados de revisión: crítico independiente, crítico del mismo modelo (con `accept` bajado a `needs_iteration`) y crítico no disponible (`reject` y síntesis con nota).
- **Hecho cuando:** el endpoint devuelve la estimación con la traza, no supera `BOSS_MAX_ITERATIONS` y el interruptor del cliente muestra la traza.

- **Estado (2026-10-20): implementado y probado con falsos (1033 tests, ruff limpio). Falta la prueba real con claves y la prueba manual en el navegador.** Lo construido y lo que se decidió al construirlo:
  - Archivos: `app/schemas/critic.py`, `app/schemas/acb.py`, `app/services/critic.py`, `app/services/boss.py`, `app/prompts/critic/v1/`, `app/sessions/tier_resolver.py`, `app/sessions/compression.py`; cambios en `routers/sessions.py` (endpoint y helpers comunes `_prepare_turn` / `_record_turn`), `services/estimation.py` (`critic_feedback`), `prompts/loader.py` (`render_critic_prompt`, `critic_feedback`), `prompts/estimation/v5/user.j2` (bloque `<critic_feedback>`, sin feedback idéntico a v3), `services/llm_wrapper.py` (`use_fallback`), `config.py` (`BOSS_MAX_ITERATIONS`, `Settings.for_provider`), `dependencies.py` (`get_critic`) y `streamlit_app.py`.
  - Cuarto estado de revisión `skipped`: si el actor rechaza la estimación («Fuera de alcance:», confianza menor que 30), no hay nada que auditar y se entrega tal cual. Sin esto, el estado `unavailable` habría tratado un rechazo legítimo como una revisión fallida.
  - El crítico usa un `LLMWrapper` por proveedor **sin respaldo** (`use_fallback=False`): con un respaldo silencioso no se sabría quién respondió y se perdería la independencia. Cada intento del crítico es un solo proveedor; el orden `independent` → `same_model` → `unavailable` lo hace `Critic.review`.
  - Etiquetas nuevas en `PROMPT_TAGS` (`critic_feedback`, `original_transcript`, `audience_tier`, `requested_detail_level`, `estimation_under_review`): una descripción que las traiga se rechaza, como las demás. Cambia el guardrail de entrada heredado solo para esas etiquetas.
  - La traza se muestra en un recuadro dentro del turno, no en un desplegable: Streamlit no permite un `expander` dentro de otro.
  - Si el actor falla en una vuelta posterior a la primera, se entrega el borrador anterior con salvedades (producción antes que perfección).
  - Límites conocidos: (1) los guardrails de entrada corren en cada vuelta del actor (con moderación activa, una llamada extra por vuelta); (2) con un crítico `same_model` que acepta sin problemas, la vuelta siguiente repite el prompt del actor sin feedback nuevo, tal como se acordó; (3) el `tier` no tiene selector en Streamlit, solo en la API.
  - **Ajustes posteriores, tras la prueba manual (2026-10-20), con tests; el crítico real sigue sin verificarse:**
    - Crítico cortado: con OpenAI como actor, el crítico de Anthropic fallaba con `IncompleteJSONError` (JSON cortado; causa probable `max_tokens`, no confirmada) y el jefe caía a `same_model`. `CRITIC_MAX_TOKENS` pasa de 2.000 a 4.000, el prompt `critic/v1` suma la regla «Sé breve» (máximo 5 problemas, una frase de unas 25 palabras cada uno) y el evento `critico_fallo` registra el `detalle` del error.
    - Streamlit: mientras se estima, «Probar conexión», «Nueva conversación» y «Enviar» están desactivados (tocar un widget corta el script y la petición); si el servicio responde con error, el mensaje se muestra y los botones vuelven a estar activos. Tipo de proyecto, detalle, formato e interruptor ACB se conservan entre turnos (solo se vacían el mensaje y los adjuntos). La pestaña «Estimación Dinámica» no ofrece selector de versión del prompt: siempre v5.
    - Síntesis con cifras (500 en producción): las salvedades del crítico citan cifras que no están en la estimación y el schema rechazaba el resumen al armar la respuesta. `ReviewedResult` (en `app/schemas/acb.py`) aplica la comprobación de cifras solo al resumen del modelo, sin la nota; el schema de lidr_4 no cambia.
    - Crítico y totales (decisión del responsable, 2026-10-20): el log `jefe_vuelta` mostró `critical:math_error@totals.cost_eur` en las tres vueltas (gpt-4o-mini no suma exacto; diferencias de 0,4 % a 1,6 %), así que el bucle nunca convergía. El prompt `critic/v1` marca `math_error` solo si horas o coste totales se desvían más de `TOTALS_MAX_DEVIATION_PCT` (5 %) de la suma de las fases, y la duración solo si supera la suma o queda por debajo de la fase más larga (las fases pueden solaparse). Los totales mostrados siguen sin recalcularse por código. El evento `jefe_vuelta` registra además `categorias`. Sin verificar con claves reales.
    - Etiqueta de la traza: si todas las vueltas son `skipped` (el actor rechazó la estimación) el recuadro dice «sin revisión (estimación rechazada)» en vez de «aceptada en 1 vuelta(s)».

### Paso 9 — Etapa final: tier dinámico y compresión

- **Estado (2026-10-20): implementado y probado con falsos (1097 tests, ruff limpio). Falta la prueba real del resumidor con claves y la prueba manual en el navegador.** Decisiones del responsable: anclas por heurística en español e inglés (modo `llm` opcional por `ANCHOR_DETECTION_MODE`); tier dinámico en `/estimate` y `/estimate-acb`; anclas sin tope. Construido: `app/sessions/tier_resolver.py` (cadena de reglas, también en español), `app/sessions/compression.py` (`AnchorDetector`, `ConversationSummarizer`, `apply_compression`, `build_compression_tools`), `app/prompts/conversation_summary/v1/`, `ConversationHistory` con `anchors` y `summary` (`append` devuelve lo expulsado), `GET /sessions/{id}` con `anchors`, `has_summary`, `tier` y `tier_rule`, y el panel de Streamlit. El resumidor reutiliza el proveedor del extractor (sin `COMPRESSION_MODEL`).
Evolución a la réplica de `session_05_live`. Se hace al final y sin tocar el jefe ni el crítico.

- Tier dinámico: reemplazar el cuerpo de `resolve_tier(...)` por la cadena de reglas de live (`nda_detected`, `regulatory_context`, `technical_audience`, `low_budget_pm`; el override siempre gana). Devuelve el tier y el nombre de la regla.
- Prompt: no hay versión nueva. El bloque `<audience>` ya está en v5 desde el Paso 4 y solo se renderiza si el tier no es `default`; aquí empieza a recibir tiers distintos de `default`.
- Compresión: reemplazar `apply_compression(...)` por anclas más resumen acumulativo, con sus prompts (`conversation_summary`) y configuración.
- Tests: resolución de tier, anclas y política de compresión.
- **Hecho cuando:** el tier se deriva de la conversación, el historial se comprime y los tests de los Pasos 7 y 8 siguen pasando.

### Paso 10 — Bucle agéntico, presupuesto de tokens y trazas

Base común de los Pasos 11 y 12. Hoy no hay function calling en `app/`: cada petición es una llamada al LLM.

- Bucle agéntico en el wrapper: el modelo pide una herramienta, el servicio la ejecuta, devuelve el resultado y repite hasta la respuesta final. Tope de iteraciones (valor a fijar). Con el ACB, que ya hace hasta 6 llamadas, hay que fijar un tope combinado.
- Failover a mitad del bucle: si el primario falla en una iteración, el respaldo continúa con el mismo historial de herramientas. A verificar antes: que OpenAI y Anthropic acepten el mismo formato de herramientas a través de LiteLLM.
- Resultados de herramientas como datos: entran delimitados y marcados como datos, nunca como instrucciones.
- Presupuesto de tokens por turno (valor a fijar): suma de system prompt, transcripción, adjuntos, resultados de búsqueda y de BBDD, medida con `litellm.token_counter`. Si se supera, se trunca con un orden de prioridad (a definir).
- Trazabilidad: `structlog` con `emit` (heredado), más un `request_id` en todos los eventos de un turno (con `structlog.contextvars`). Un evento por herramienta: nombre, duración, tamaño devuelto y si se truncó. Sin Logfire ni Langfuse.
- Costo por turno: un acumulador suma costo y tokens de todas las llamadas del turno (bucle, reintentos y ACB) y los registra en el evento con el `request_id`. Cada llamada ya informa su costo (`LLMCallResult.cost_usd`).
- Activación: cada mecanismo se enciende o apaga por configuración (por ejemplo `WEB_SEARCH_ENABLED` y `BUSINESS_DB_ENABLED`) y la descripción de cada herramienta indica al modelo cuándo usarla.
- **Valores fijados (2026-10-21, aprobados por el responsable):** `AGENT_MAX_ITERATIONS=4`, `TURN_TOKEN_BUDGET=60000`, `AGENT_LOOP_ENABLED=false` por defecto. Con el ACB el bucle corre una sola vez antes de la primera vuelta del actor; las vueltas siguientes reutilizan lo recogido. Prioridad del presupuesto: sistema, transcripción, historial, adjuntos, resultados de herramientas (se recorta primero lo último).
- **Hecho cuando:** el bucle respeta el tope de iteraciones y el presupuesto, los eventos se pueden correlacionar por `request_id` y los tests de los Pasos 7 y 8 siguen pasando.
- **Estado (2026-10-22):** implementado y VERIFICADO con claves reales: `test/verificar_herramientas.py` confirma el formato de herramientas en ambos proveedores, y en el servicio real el cambio a mitad del bucle (clave de OpenAI inválida) mostró `uso_respaldo=True` en los tres `agente_llamada` del mismo `request_id`, con el evento `turno_costo` final (`llamadas_herramienta=2`). `X-Request-ID` también confirmado.

### Paso 11 — Búsqueda web como herramienta

- Servicio independiente expuesto al LLM como función: Tavily principal, Firecrawl de respaldo (`SEARCH_DEFAULT`, `SEARCH_FALLBACK`, distintos entre sí; se validan en `Settings`). Claves `WEB_TAVILY_KEY` y `WEB_FIRECRAWL_KEY` como `SecretStr`.
- Reglas medidas (ver CONTEXTO.md): Firecrawl siempre con `scrapeOptions={}`, `snippet` truncado por resultado (del orden de 1.500 caracteres), máximo 3 resultados por búsqueda, tope de búsquedas por petición.
- **Valores fijados (2026-10-21, aprobados por el responsable):** `SEARCH_MAX_PER_TURN=2`, `SEARCH_MAX_SNIPPET_CHARS=1500`, `SEARCH_TIMEOUT_SECONDS=15` por servicio. De la consulta solo se registra el largo; con `SEARCH_LOG_QUERIES=true`, el texto truncado a 80 caracteres. La herramienta `web_search` se registra solo con `AGENT_LOOP_ENABLED=true` y al menos una clave. Si fallan los dos servicios devuelve un texto de «no disponible» y no rompe la petición. Llamadas por `litellm.asearch`.
- **Estado (2026-10-22):** implementado y VERIFICADO con claves reales (`test/verificar_busqueda_agente.py`): el modelo pide `web_search` ante una pregunta que depende de datos actuales y no la pide si no hace falta; además, en una corrida con el respaldo forzado, el tope de `SEARCH_MAX_PER_TURN=2` se aplicó (la tercera petición de búsqueda recibió el aviso de límite y el modelo siguió sin ella).
- Cuándo se usa: la pregunta es sensible al tiempo o pide datos posteriores al corte de entrenamiento del modelo.
- **Hecho cuando:** el modelo invoca la búsqueda, el resultado llega truncado y delimitado, el respaldo toma el relevo si el principal falla y el evento de traza registra la duración.

### Paso 12 — Consultas a la BBDD del negocio por function calling

- El servicio IA no accede a la BBDD. Expone al LLM la herramienta `find_similar_projects`; su implementación hace una llamada HTTP autenticada al backend de negocio, que resuelve la consulta y devuelve un payload limpio. El servicio IA no conoce el esquema de proyectos.
- Backend de negocio simulado (decisión del responsable): servicio aparte dentro del repo, con su propio proceso y contenedor en `docker-compose` (puerto 8003 propuesto), endpoint REST con clave y datos de ejemplo de proyectos históricos (horas reales, desviaciones, riesgos materializados). El servicio IA solo conoce `BUSINESS_API_URL` y `BUSINESS_API_KEY`; cambiar al backend real exige que el contrato REST coincida.
- Cuándo se usa: la respuesta debe basarse en datos propios de la empresa (histórico, catálogo, clientes).
- **Hecho cuando:** el modelo invoca la herramienta, el payload llega acotado y delimitado, un fallo del backend no tumba la estimación (se estima sin esa información y se avisa) y el evento de traza registra la duración.
- **Valores fijados (2026-10-22, aprobados por el responsable):** puerto 8003, servicio `business_backend/` con autenticación por cabecera `X-API-Key`, `BUSINESS_MAX_PER_TURN=2`, `BUSINESS_TIMEOUT_SECONDS=5`. Datos de ejemplo: 18 proyectos sintéticos generados con un criterio razonable (tipos y desviaciones variadas), en `business_backend/data.json`.
- **Estado (2026-10-22):** implementado y VERIFICADO con el backend simulado real (`test/verificar_negocio_agente.py`): la herramienta consultó el backend (41 ms) y trajo 3 proyectos, y el modelo pidió `find_similar_projects` ante una pregunta que se beneficia del histórico.
- Igual que la búsqueda web, `find_similar_projects` solo se registra con `AGENT_LOOP_ENABLED=true` y `BUSINESS_API_URL` + `BUSINESS_API_KEY` configuradas.

Interfaz (Paso 6, ampliación): las herramientas invocadas se muestran en la pestaña «Estimación Dinámica», en un expander como el del ACB.

El orden de ejecución entre el Paso 9 y los Pasos 10 a 12 no está decidido.

## Criterios de hecho (ENUNCIADO)

- [ ] `POST /sessions` crea una sesión y devuelve un `session_id`.
- [ ] `POST /sessions/{session_id}/estimate` acepta transcripción y adjuntos opcionales, y devuelve una estimación que respeta el schema Pydantic existente.
- [ ] Tras varios turnos, el LLM mantiene el nombre del proyecto.
- [ ] La `project_metadata` se actualiza entre turnos.
- [ ] El historial respeta la ventana deslizante.
- [ ] README con: cómo se levanta, cómo se ejecutan los tests, camino de adjuntos y motivo, y cómo se extrae la `project_metadata`.
- [ ] Los tests del Paso 7 pasan en local.
- [ ] Opcional: captura o GIF de una conversación de al menos tres turnos con el panel de metadata visible.
- [ ] (Decisión del responsable) `POST /sessions/{session_id}/estimate-acb` devuelve la estimación con la traza, y el interruptor del cliente la muestra.
- [x] (Etapa final) Tier dinámico y compresión según el Paso 9 (verificado con claves reales el 2026-10-20).
- [x] (Decisión del responsable) Bucle agéntico con presupuesto de tokens y trazas por `request_id` (Paso 10, verificado con claves reales), [x] búsqueda web (Paso 11, verificado con claves reales) y [x] `find_similar_projects` contra un backend de negocio simulado (Paso 12, verificado con el backend simulado real).

## Fuera de alcance

- Persistencia de la memoria entre reinicios.
- OCR de adjuntos escaneados y lectura de contenido visual (diagramas, imágenes).
- `GET /sessions/{id}` (el enunciado no lo pide).
- Streaming de la salida estructurada con `ijson` y eventos por fase (WU8 pendiente en lidr_4). Paso posterior al ACB, con diseño propio: afecta el contrato de `/estimate` y, con el ACB, la interfaz tendría que descartar borradores rechazados.

El ENUNCIADO deja fuera el resumen acumulativo con anclas, el tier dinámico y el ACB. Por decisión del responsable entran como Pasos 8 y 9. Lo mismo ocurre con la búsqueda web y el function calling a la BBDD del negocio, que el ENUNCIADO deja fuera: entran como Pasos 11 y 12, sobre el bucle del Paso 10.

## Pendientes antes de cerrar

- Dependencias de `pyproject.toml`: se conservan las de lidr_4, incluida `ijson` (declarada y sin uso, a la espera del streaming de WU8). Se suman `openai` (los tests lo importan directo), `python-multipart`, `pypdf` y `python-docx`. `python-dotenv` se mantiene: ningún archivo lo importa, pero lo requieren `litellm`, `pydantic-settings` y `uvicorn`. `redisvl` no es dependencia de lidr_4: la caché semántica usa `redis` y `litellm`.
- Cliente Streamlit: antes de tocar `streamlit_app.py`, listar los nombres que importan los seis archivos de tests que lo cargan (por ejemplo `OUT_OF_SCOPE_PREFIX`); `test_frontend_rendered.py` ejecuta el archivo completo.
- `.env.example` fusionado y modelos migrados a `LLM_DEFAULT`, `LLM_FALLBACK`, `OPENAI_MODEL` y `ANTHROPIC_MODEL` (sin `PRIMARY_MODEL` ni `FALLBACK_MODEL`; ver CONTEXTO.md). Resuelto (opción B): el wrapper pone la temperatura en cada deployment y la omite con modelos que LiteLLM no la admite (`claude-haiku-5-5`); ver CONTEXTO.md.
- Caché: decidido, el endpoint conversacional no usa caché (la respuesta depende del historial y la clave de lidr_4 solo cubre la descripción). Se aplica también al endpoint ACB.
- Modelo por llamada: verificado (ver CONTEXTO.md). El wrapper ya informa modelo, proveedor y costo por llamada, pero no permite elegir el modelo. Se agrega un parámetro interno (solo entre `LLM_DEFAULT` y `LLM_FALLBACK`) que usa el crítico del ACB en el Paso 8; el usuario nunca elige modelo.
- Dataset real etiquetado para el benchmark.
- Repetir el benchmark con el scorer v2.
