En lidr_4 el estimator es un sistema transaccional: una transcripción entra, una estimación sale. En lidr_5 el estimador deberá evolucionar para convertirse en una conversación iterativa donde el cliente refina el alcance, añade nueva información, sube documentos complementarios, y espera que el sistema recuerde sobre qué proyecto estamos hablando.

Debemos relizar dos cambios concretos. Primero, el estimator va a mantener memoria conversacional dentro de una misma sesión: el contexto del proyecto en curso se preserva entre turnos sin reenviar todo el historial bruto en cada llamada. Segundo, va a aceptar adjuntos: PDFs y documentos Word que enriquecen la transcripción con especificaciones técnicas, propuestas previas o diagramas de arquitectura. Todo ello basado en compresión de memoria con anclas, tier dinámico y el patrón Actor-Critic-Boss.

Dentro del alcance:

    Sesión conversacional con identificador. Un nuevo endpoint POST /sessions que crea una sesión vacía y devuelve un session_id. Las peticiones de estimación llevarán este identificador.

    Memoria conversacional con ventana deslizante. El servicio IA mantiene en memoria del proceso (un diccionario, sin BBDD) el historial de mensajes asociado a cada session_id, conservando los últimos N turnos.

    project_metadata separado del historial. Un diccionario por sesión que captura los hechos relevantes del proyecto en curso (nombre, equipo asumido, tecnologías mencionadas, alcance acordado). Se inyecta en el system prompt en cada turno, vive aparte del historial.

    Endpoint multi-turno. Un nuevo endpoint POST /sessions/{session_id}/estimate que acepta una transcripción más una lista opcional de adjuntos, y devuelve la estimación. El historial y el project_metadata se actualizan automáticamente con cada llamada.

    Adjuntos. Soporte para PDFs y documentos Word adjuntos en multipart/form-data. Implementas uno de los dos caminos a tu elección:

        Camino A (multimodal directo): subir el PDF directamente al LLM usando la Files API de OpenAI o Anthropic. Más simple, menos código, acoplado al proveedor multimodal.

        Camino B (extracción local): extraer texto localmente con pypdf o PyMuPDF para PDFs y python-docx para Word, y enviar el texto extraído como parte del prompt. Más control, independiente del proveedor, prepara el terreno para el chunking de RAG en el módulo 3.

Fue de alcance:

Estrategia de resumen acumulativo o estrategia híbrida con anclas — se construyen en el directo.

Tier dinámico derivado de contexto en runtime — se construye en el directo.

Persistencia de la memoria entre reinicios del servicio — un diccionario en memoria del proceso es suficiente para esta fase.

Búsqueda web integrada y function calling para BBDD del backend de negocio — aparecen en el material teórico pero no se implementan aquí.

Cualquier pieza del patrón Actor-Critic-Boss — segunda hora del directo.

Pasos guiados

Cada paso enuncia el objetivo. Una solución completa a modo de guía está en un repositorio en GitHub por si sugiera alguna duda.

Paso 1 — Modelar el estado de la sesión

Crear un módulo sessions.py en el servicio IA con dos estructuras:

    - ConversationHistory: una lista limitada de mensajes con la lógica de ventana deslizante. Cuando supera N turnos, descarta los más antiguos preservando siempre el system prompt.

    - ProjectMetadata: un Pydantic model con campos como project_name, assumed_team_size, mentioned_technologies (lista), agreed_scope (texto libre).

Ambas estructuras viven dentro de una clase Session indexada por session_id en un diccionario en memoria del proceso. El estado de sesión no usa BBDD ni Redis: se pierde al reiniciar el servicio. Esta restricción aplica solo a la memoria de sesiones; el resto del servicio conserva lo heredado de lidr_4, incluida la caché de respuestas en Redis. Documentar brevemente en docstrings por qué aceptas esa volatilidad en esta fase.

Paso 2 — Endpoint para crear sesiones

Añadir POST /sessions que devuelve {"session_id": "..."}. El identificador puede ser un UUID v4. Si el cliente quiere reutilizar memoria entre páginas, el session_id viaja en cada petición posterior.

Paso 3 — Soporte de adjuntos en el endpoint principal

Implementar el camino que yo acabe eligiendo (A o B). Definir el endpoint POST /sessions/{session_id}/estimate aceptando multipart/form-data con dos campos:

    - transcript: el texto de la transcripción (string).

    - attachments: lista opcional de UploadFile con la documentación complementaria.

Si opto por A, hay que usar la Files API del proveedor que se prefiera y referenciar los archivos en el bloque de contenido del mensaje. Si opto por B, hay que extraer el texto en el servicio IA y concaténalo al transcript con un separador claro (--- attachment: filename.pdf ---) antes de pasarlo al prompt.

Documenta en el README qué camino se eligió y por qué.

Paso 4 — Inyección de project_metadata en el system prompt

Modificar el template Jinja2 del system prompt para que reciba un bloque <project_metadata> con los hechos conocidos del proyecto. Si el project_metadata está vacío (primera llamada de la sesión), el bloque también lo está.

Después de cada respuesta del LLM, actualizar el project_metadata extrayendo los hechos relevantes de la nueva interacción. Para esta fase se puede hacer de dos maneras:

    - Heurística simple: parsea la respuesta del LLM buscando patrones (regex o lógica básica) para extraer nombre del proyecto, tecnologías mencionadas, etc.

    - LLM extractor: una segunda llamada al LLM con un prompt específico que devuelve un JSON con los campos del ProjectMetadata.

Cualquiera de las dos es válida. La heurística es más barata y rápida; el extractor es más robusto pero añade una llamada por turno. Hay que justificar la decisión en el README.

Paso 5 — Gestión del historial con ventana deslizante

Implementar la lógica de ventana deslizante en ConversationHistory. Sugerencias:

    - Mantener el system prompt como invariante (siempre presente).

    - Definir MAX_TURNS = 6 como valor por defecto (ajustable por configuración). Un turno es un par user+assistant.

    - Cuando se supera el límite, descarta los pares más antiguos.

    - Expober un método to_messages_list() que devuelva el array messages listo para pasar a la API del LLM, con el system prompt regenerado a partir del project_metadata actual.

Paso 6 — Adaptar el cliente

Ajustar el cliente Streamlit (o el stack en uso) para:

    - Crear una sesión al cargar la página y guardar el session_id en la session_state.

    - Mostrar un campo de texto para la transcripción y un selector múltiple de archivos.

    - Mostrar el project_metadata actual en un panel lateral o expandible (útil para debugging y para que el alumno vea la separación entre memoria y historial).

    - Mantener un botón "Nueva conversación" que llame de nuevo a POST /sessions y resetee el estado.

Paso 7 — Tests mínimos

Añadir dos o tres tests de integración con pytest y httpx.AsyncClient:

    - Una sesión que enlaza dos peticiones y verificar que el project_metadata se actualiza correctamente.

    - Una petición con un PDF adjunto que verifica que el contenido del documento influye en la estimación (puede ser un test cualitativo, como por ejemplo que comprueba que un campo concreto del output cambia cuando se añade el adjunto).

    - Un test que envía 8 turnos a la misma sesión y verifica que el historial efectivo enviado al LLM nunca supera los MAX_TURNS configurados.

Criterios de "hecho"

El proyecto está completo cuando:

    POST /sessions crea una sesión y devuelve un session_id.

    POST /sessions/{session_id}/estimate acepta multipart/form-data con transcripción y adjuntos opcionales, y devuelve una estimación que respeta el schema Pydantic existente.

    Tras varios turnos en la misma sesión, el LLM responde con coherencia respecto al proyecto en curso (no "se olvida" del nombre del proyecto entre turnos).

    El project_metadata se actualiza visiblemente entre turnos.

    El historial respeta el límite de la ventana deslizante.
    
Puedes leer primero CONTEXTO.md, luego el README y, para el porqué de una decisión, PLAN.md

    README breve indicando cómo se levanta, cómo se ejecutan los tests, qué camino de adjuntos se escogió y por qué, cómo se extrae el project_metadata.

    Los tests del Paso 7 pasan en local.

    Captura o GIF de la nueva interfaz mostrando una conversación de al menos tres turnos con el panel de project_metadata visible (opcional).
    
Modelos de ejemplo: 

    https://github.com/LIDR-academy-temp/ai-engineering/tree/session_5

    https://github.com/LIDR-academy-temp/ai-engineering/tree/session_05_live
    
