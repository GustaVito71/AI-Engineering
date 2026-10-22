"""Configuración de la aplicación (pydantic-settings, leída de .env/entorno).

Heredada de lidr_3, con los cambios que impone LiteLLM. Los tres principios
del original se conservan intactos:

1. Las API keys son opcionales en la construcción (`SecretStr | None`) y se
   exigen en el punto de uso: al construir el wrapper LLM, en la primera
   request que lo necesita (ver app/dependencies.py). Si la validación ocurre al
   importar el módulo, el proceso muere ANTES de que exista la aplicación:
   no hay /health, no hay /docs, no hay nada que le diga al orquestador qué
   falta. Un health check tiene que sobrevivir a la avería que diagnostica.

2. `apply_defaults` normaliza lo que pydantic-settings no puede: un `X=`
   vacío se parsea como `""`, no como `None`, así que el "no configurado" se
   detecta con falsy y no con `is None`. Un `.env` copiado de `.env.example`
   no debe romper el arranque. Un valor presente pero inválido sí lo rompe:
   fail fast.

3. El render del logging no vive acá, vive en main.configure_logging.

QUÉ CAMBIÓ RESPECTO DE LIDR_3 Y POR QUÉ

- Desaparecen `LLM_PROVIDER` y `PRIMARY_MODEL`/`FALLBACK_MODEL`. Ya no hay un
  switch de proveedor ni modelos escritos como "proveedor/modelo": se elige el
  PROVEEDOR del servicio con `LLM_DEFAULT` y `LLM_FALLBACK`, y el modelo de cada
  proveedor sale de su variable `<PROVEEDOR>_MODEL` (`OPENAI_MODEL`,
  `ANTHROPIC_MODEL`). El `Router` de LiteLLM recibe un deployment por modelo (el
  primario y, si tiene clave, el de respaldo) y decide el failover. Escribir
  el retry a mano sería tener dos dueños de la misma decisión.

- `primary_model` y `fallback_model` siguen existiendo, pero como propiedades de
  solo lectura ("openai" + "gpt-4o-mini" -> "openai/gpt-4o-mini", el formato que
  pide LiteLLM). El wrapper, /health y la caché no cambian.

- El provider ya no se parsea de un nombre de modelo: es lo que el operador
  escribió en `LLM_DEFAULT`/`LLM_FALLBACK`. Ojo con la diferencia respecto de la
  referencia session_3, que adivina el provider del NOMBRE del modelo con
  `startswith("claude")`. Acá no se adivina nada. Qué deployment *respondió* no
  se infiere del nombre: el wrapper lo lee de `response._hidden_params["model_id"]`.

- `PROMPT_VERSION` es obligatoria y no decorativa: elige la plantilla
  (prompts/estimation/vN/) y viaja en cada respuesta y en cada evento de
  trazabilidad. También invalida la caché: entra en la clave de la caché
  exacta (modelo, versión, parámetros y descripción; no el prompt renderizado,
  porque la búsqueda va antes del render) y será el mecanismo del caché
  semántico de WU10. Sin ella, cambiar la
  plantilla seguiría sirviendo respuestas viejas en silencio.
"""

from functools import lru_cache
from typing import Annotated, Literal

from pydantic import AliasChoices, BeforeValidator, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

API_KEY_VARIABLES = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}

# Proveedores que puede usar el extractor de `project_metadata` (EXTRACTOR_DEFAULT y
# EXTRACTOR_FALLBACK): los del servicio más Gemini y DeepSeek. El servicio de
# estimación (LLM_DEFAULT / LLM_FALLBACK) sigue limitado a `API_KEY_VARIABLES`.
EXTRACTOR_KEY_VARIABLES = {
    **API_KEY_VARIABLES,
    "gemini": "GEMINI_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
}

# Niveles de audiencia del prompt v5 (bloque `<audience>`). `default` no agrega nada.
TIER_VALUES = ("executive", "pm", "developer", "default")

# `log_level` es el único campo con dominio cerrado que además decide algo:
# main.configure_logging lo pasa a logging.setLevel, así que un valor inválido
# muere al arrancar. `Literal` no cambia ese comportamiento —la stdlib ya
# rechazaba con `ValueError: Unknown level`— pero mueve el error al lado de
# Pydantic, que nombra el campo y lista los valores válidos en vez de dejar
# `Unknown level: 'info'` a secas.
#
# El `BeforeValidator` no es decorativo: sin él, un `LOG_LEVEL=` vacío rompería
# el arranque, porque `Literal` rechaza `""` antes de que `apply_defaults`
# pueda convertirlo en el default. Eso rompería la regla heredada de `lidr_3`
# —cadena vacía = "no configurado"— que el punto 2 del docstring declara como
# principio. Se normaliza antes de validar el dominio: vacío y sin espacios van
# al default, y cualquier otra cosa se pasa a `Literal` sin tocar, para que un
# `info` en minúscula siga siendo un error en vez de un(DEFAULT silencioso.
#
# `app_env` sigue siendo `str` a propósito: hoy no ramifica en ningún lado, se
# solo muestra en /health. Tiparlo sería validar un valor del que nadie depende.
LOG_LEVELS = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


def _normalize_log_level(value: str) -> str:
    """Vacío = "no configurado" = default, igual que el resto de Settings.

    Solo contempla el caso vacío. No hace `.upper()` a propósito: un `LOG_LEVEL=info`
    es un `.env` mal puesto, y silenciarlo a `INFO` esconde el error detrás de un
    default. Que muera al arrancar es el comportamiento que se busca."""
    return value or "INFO"


# --- Input guardrails -------------------------------------------------------
def _empty_is_true(value: object) -> object:
    """Vacío = "no configurado" = activado, igual que el resto de Settings.

    pydantic-settings parsea `GUARDRAILS_ENABLED=` como `""`, y `""` no es un bool
    válido: un `.env` a medio completar rompería el arranque. Cualquier otro valor
    pasa tal cual, así que un `GUARDRAILS_ENABLED=nunca` sigue siendo un error."""
    return True if value == "" else value


# Confianza global por debajo de la cual la estimación lleva un aviso (guardrails de
# salida). El umbral de rechazo (30) vive en el schema: este va por encima.
DEFAULT_LOW_CONFIDENCE_WARN_PCT = 70


def _empty_is_default_confidence(value: object) -> object:
    """Vacío = "no configurado" = el umbral por defecto, como `_empty_is_true`."""
    return DEFAULT_LOW_CONFIDENCE_WARN_PCT if value == "" else value


# --- Sesiones conversacionales --------------------------------------------------
DEFAULT_MAX_TURNS = 6
DEFAULT_SESSION_TTL_SECONDS = 86_400
DEFAULT_MAX_SESSIONS = 1_000
# Máximo de vueltas actor + crítico del jefe (ACB, Paso 8). Cada vuelta son dos llamadas.
DEFAULT_BOSS_MAX_ITERATIONS = 3
# Bucle agéntico (Paso 10). Máximo de rondas modelo→herramientas por turno, y presupuesto de
# tokens del contexto de un turno (transcripción, historial, adjuntos y resultados).
DEFAULT_AGENT_MAX_ITERATIONS = 4
DEFAULT_TURN_TOKEN_BUDGET = 60_000
# Búsqueda web (Paso 11). Tope de búsquedas por petición, largo máximo del texto de cada
# resultado (en caracteres) y segundos máximos de espera a cada servicio de búsqueda.
DEFAULT_SEARCH_MAX_PER_TURN = 2
DEFAULT_SEARCH_MAX_SNIPPET_CHARS = 1_500
DEFAULT_SEARCH_TIMEOUT_SECONDS = 15.0
# Servicios de búsqueda que el código sabe usar.
SUPPORTED_SEARCH_PROVIDERS = ("tavily", "firecrawl")
# Consulta al backend de negocio (Paso 12). Tope por petición y timeout por request.
DEFAULT_BUSINESS_MAX_PER_TURN = 2
DEFAULT_BUSINESS_TIMEOUT_SECONDS = 5.0


def _empty_is_default_max_turns(value: object) -> object:
    """Vacío = "no configurado" = la ventana por defecto, como el resto de Settings."""
    return DEFAULT_MAX_TURNS if value == "" else value


def _empty_is_default_session_ttl(value: object) -> object:
    """Vacío = "no configurado" = 24 horas."""
    return DEFAULT_SESSION_TTL_SECONDS if value == "" else value


def _empty_is_default_max_sessions(value: object) -> object:
    """Vacío = "no configurado" = 1.000 sesiones."""
    return DEFAULT_MAX_SESSIONS if value == "" else value


def _empty_is_default_boss_iterations(value: object) -> object:
    """Vacío = "no configurado" = 3 vueltas."""
    return DEFAULT_BOSS_MAX_ITERATIONS if value == "" else value


def _empty_is_default_search_max_per_turn(value: object) -> object:
    """Vacío = "no configurado" = 2 búsquedas por petición."""
    return DEFAULT_SEARCH_MAX_PER_TURN if value == "" else value


def _empty_is_default_search_snippet_chars(value: object) -> object:
    """Vacío = "no configurado" = 1.500 caracteres por resultado."""
    return DEFAULT_SEARCH_MAX_SNIPPET_CHARS if value == "" else value


def _empty_is_default_search_timeout(value: object) -> object:
    """Vacío = "no configurado" = 15 segundos."""
    return DEFAULT_SEARCH_TIMEOUT_SECONDS if value == "" else value


def _empty_is_default_business_max_per_turn(value: object) -> object:
    """Vacío = "no configurado" = 2 consultas por petición."""
    return DEFAULT_BUSINESS_MAX_PER_TURN if value == "" else value


def _empty_is_default_business_timeout(value: object) -> object:
    """Vacío = "no configurado" = 5 segundos."""
    return DEFAULT_BUSINESS_TIMEOUT_SECONDS if value == "" else value


def _empty_is_default_agent_iterations(value: object) -> object:
    """Vacío = "no configurado" = 4 rondas."""
    return DEFAULT_AGENT_MAX_ITERATIONS if value == "" else value


def _empty_is_default_token_budget(value: object) -> object:
    """Vacío = "no configurado" = 60.000 tokens."""
    return DEFAULT_TURN_TOKEN_BUDGET if value == "" else value


# --- Caché semántica (WU10) ---------------------------------------------------
DEFAULT_SEMANTIC_THRESHOLD = 0.95
DEFAULT_SEMANTIC_MAX_ENTRIES = 200
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"


def _empty_is_false(value: object) -> object:
    """Vacío = "no configurado" = apagado: la caché semántica es opt-in."""
    return False if value == "" else value


def _empty_is_default_threshold(value: object) -> object:
    """Si la variable llega vacía (como en un `.env` copiado sin completar), usa el umbral
    por defecto de la caché semántica.
    """
    return DEFAULT_SEMANTIC_THRESHOLD if value == "" else value


def _empty_is_default_max_entries(value: object) -> object:
    """Si la variable llega vacía, usa el máximo de entradas por defecto de la caché
    semántica.
    """
    return DEFAULT_SEMANTIC_MAX_ENTRIES if value == "" else value


# El contrato de `description` vive en el Field de
# app.schemas.estimation.EstimationRequest. Se replica acá porqueSettings no
# puede importar al schema (sería un ciclo: el schema importa get_settings) y
# el validador de abajo necesita compararse contra estos números. Si se cambia
# uno hay que cambiar el otro: el test de WU2 falla si se desincronizan.
CONTRACT_MIN_CHARS = 20
CONTRACT_MAX_CHARS = 2000


class LLMConfigurationError(Exception):
    """Falta configuración para usar el LLM (no es un fallo del proveedor).

    Diferente de un fallo de llamada: aquí el problema es local y se sabe
    exactamente qué falta. Nombra la variable en el mensaje."""


def provider_from_model(model: str) -> str:
    """'openai/gpt-4o-mini' -> 'openai'.

    Un modelo sin prefijo asume 'openai', que es el default de LiteLLM.
    Solo se usa para elegir contra qué API key construir el deployment, en
    tiempo de configuración. Nunca para inferir qué deployment respondió.
    """
    return model.split("/", 1)[0] if "/" in model else "openai"


class Settings(BaseSettings):
    """Configuración de la aplicación cargada desde variables de entorno
    y el archivo .env.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- API keys. Opcionales acá, exigidas en el punto de uso.
    openai_api_key: SecretStr | None = None
    anthropic_api_key: SecretStr | None = None
    # Claves del extractor de project_metadata (ver EXTRACTOR_DEFAULT / EXTRACTOR_FALLBACK).
    gemini_api_key: SecretStr | None = None
    deepseek_api_key: SecretStr | None = None

    # --- LiteLLM. Un deployment por modelo; el Router decide el failover.
    # Ver el docstring del módulo.
    # Proveedor del servicio y del respaldo, y el modelo de cada proveedor. Tienen
    # que ser proveedores distintos (ver `validate_models`).
    llm_default: str = "openai"
    llm_fallback: str = "anthropic"
    openai_model: str = "gpt-4o-mini"
    anthropic_model: str = "claude-haiku-5-5"

    # --- Extractor de project_metadata (Paso 4). Una segunda llamada, después de cada
    # estimación, que actualiza los hechos del proyecto. Tiene su propio primario y
    # respaldo, distintos entre sí, y su modelo sale de `<PROVEEDOR>_MODEL`.
    extractor_default: str = "gemini"
    extractor_fallback: str = "deepseek"
    gemini_model: str = "gemini-3.1-flash-lite"
    deepseek_model: str = "deepseek-flash"

    # --- Prompt de las sesiones conversacionales (Paso 4): versión por defecto de
    # POST /sessions/{id}/estimate. Tiene que ser de salida estructurada. Es v5, la que
    # inyecta `<project_metadata>`; las versiones anteriores ignoran la memoria.
    conversation_prompt_version: str = "v5"
    # Audiencia del bloque `<audience>` de v5 cuando ninguna regla del tier dinámico (Paso 9)
    # ni el override de la petición la cambian. `default` = sin bloque.
    default_tier: str = "default"

    llm_timeout: float = 30.0
    llm_max_retries: int = 2
    # 4000 y no los 2000 de lidr_3: la salida estructurada (resumen + tareas
    # + equipo + supuestos + riesgos) es bastante más larga que el markdown
    # equivalente, y un max_tokens corto produce JSON truncado.
    llm_max_tokens: int = 4000

    # --- Capa 2. Qué plantilla se usa. Ver el docstring sobre su relación
    # con la caché.
    prompt_version: str = "v2"

    # --- Structured response ----------------------------------------------------
    # Versión de prompt por defecto de POST /api/v1/estimate/structured. Tiene
    # que ser una versión de salida estructurada (`output: structured` en su
    # examples.yaml); PROMPT_VERSION sigue siendo la de /estimate, de texto libre.
    structured_prompt_version: str = "v3"
    # Cuántas veces Instructor vuelve a preguntar al modelo cuando la respuesta no
    # cumple el schema. Cada reintento es una llamada completa que se paga. Es
    # independiente de LLM_MAX_RETRIES, que reintenta los fallos del proveedor.
    structured_max_retries: int = 2

    # --- Rendered response ------------------------------------------------------
    # Versión de prompt por defecto de POST /api/v1/estimate/rendered. Tiene que
    # ser de salida renderizada (`output: rendered` en su examples.yaml). Usa los
    # mismos reintentos que la salida estructurada (STRUCTURED_MAX_RETRIES).
    rendered_prompt_version: str = "v4"

    # --- Caché. REDIS_URL vacío = caché desactivado, sin tocar código.
    redis_url: str = "redis://localhost:6379/0"
    cache_ttl: int = 86400

    # --- Techo del operador sobre `description`. El contrato con el cliente
    # (20/2000) vive en el Field de app.schemas.estimation.EstimationRequest y
    # no se toca acá: estos dos valores solo pueden RESTRINGIR ese rango, nunca
    # ampliarlo, porque el `max` es lo que protege el coste (el tamaño de la
    # entrada es la factura) y un techo que necesita redeploy es un techo blando.
    #
    # El default es el contrato mismo (20/2000), no un valor más permisivo: si
    # fuera 50/5000, el `min_length=20` del schema nunca se cumpliría y el
    # operador ni se enteraría. Configurar esto más estricto es deliberado y
    # `validate_description_ceiling` lo acepta; configurarlo más laxo es un error
    # de configuración y muere al arrancar.
    #
    # Se llamaban DESCRIPCION_MIN_CHARS y DESCRIPCION_MAX_CHARS: esos nombres se
    # siguen leyendo, para que un .env anterior no vuelva en silencio al default.
    description_min_chars: int = Field(
        default=20, validation_alias=AliasChoices("description_min_chars", "descripcion_min_chars")
    )
    description_max_chars: int = Field(
        default=2000,
        validation_alias=AliasChoices("description_max_chars", "descripcion_max_chars"),
    )

    # --- Input guardrails -------------------------------------------------------
    # Reglas que corren sobre la descripción antes del render y de la caché (ver
    # app/guardrails/input.py). GUARDRAILS_ENABLED=false las apaga todas, con
    # un evento en el log; sirve para medir sin ellas, no para producción.
    guardrails_enabled: Annotated[bool, BeforeValidator(_empty_is_true)] = True
    # Moderación de OpenAI (el endpoint no se cobra). Necesita OPENAI_API_KEY: sin
    # clave queda apagada y el resto de las reglas sigue. Falla abierta: si no
    # responde en MODERATION_TIMEOUT segundos, la descripción sigue su camino.
    moderation_enabled: Annotated[bool, BeforeValidator(_empty_is_true)] = True
    moderation_model: str = "omni-moderation-latest"
    moderation_timeout: float = 3.0

    # --- Output guardrails ------------------------------------------------------
    # Avisos y eventos sobre la estimación ya validada (app/guardrails/output.py):
    # totales que no cuadran, confianza baja, fuera de alcance. OUTPUT_GUARDRAILS_ENABLED=false
    # los apaga, con un evento en el log. NO apaga las reglas del schema que rechazan y
    # re-preguntan (horas contra el equipo, totales a más de 5 %...): esas viven en el
    # contrato de la respuesta, no son un interruptor.
    output_guardrails_enabled: Annotated[bool, BeforeValidator(_empty_is_true)] = True
    # Por debajo de esta confianza global (y por encima del umbral de rechazo, 30) la
    # estimación se devuelve con un aviso. 0 desactiva solo este aviso.
    low_confidence_warn_pct: Annotated[int, BeforeValidator(_empty_is_default_confidence)] = (
        DEFAULT_LOW_CONFIDENCE_WARN_PCT
    )

    # --- Caché semántica (WU10) ---------------------------------------------------
    # Sirve la estimación de una descripción muy parecida a otra ya estimada, sin llamar
    # al modelo (app/services/semantic_cache.py). Apagada por defecto: devuelve la
    # estimación de OTRA descripción, y eso es una decisión del operador. Necesita
    # Redis (REDIS_URL) y la clave del proveedor del modelo de embeddings (OpenAI para
    # el de por defecto); sin alguno de los dos queda apagada con un evento en el log.
    semantic_cache_enabled: Annotated[bool, BeforeValidator(_empty_is_true)] = True
    # Similitud coseno mínima, en (0, 1]. Más alto = más estricto. 0.95 es el piso sin falsos
    # aciertos medidos con test/calibrar_cache.py (a 0.92 «con y sin pagos» acertaba).
    cache_semantic_threshold: Annotated[float, BeforeValidator(_empty_is_default_threshold)] = (
        DEFAULT_SEMANTIC_THRESHOLD
    )
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    embedding_timeout: float = 3.0
    # Con SEMANTIC_CACHE_LOG_ONLY=true busca y deja la similitud en el log, pero nunca
    # sirve un acierto: sirve para calibrar el umbral con tráfico real antes de encenderla.
    semantic_cache_log_only: Annotated[bool, BeforeValidator(_empty_is_false)] = False
    # Tope de entradas por espacio de búsqueda (modelo, versión, parámetros): la
    # similitud se calcula en la app, y el coste de cada búsqueda crece con este número.
    semantic_cache_max_entries: Annotated[int, BeforeValidator(_empty_is_default_max_entries)] = (
        DEFAULT_SEMANTIC_MAX_ENTRIES
    )

    # --- Sesiones conversacionales. Cuántos turnos (pares usuario + asistente) conserva la
    # ventana deslizante del historial de cada sesión. Ver app/sessions/models.py.
    max_turns: Annotated[int, BeforeValidator(_empty_is_default_max_turns)] = DEFAULT_MAX_TURNS
    # Segundos sin uso tras los cuales una sesión caduca, y tope de sesiones vivas.
    session_ttl_seconds: Annotated[int, BeforeValidator(_empty_is_default_session_ttl)] = (
        DEFAULT_SESSION_TTL_SECONDS
    )
    max_sessions: Annotated[int, BeforeValidator(_empty_is_default_max_sessions)] = (
        DEFAULT_MAX_SESSIONS
    )
    # --- ACB (Paso 8). Tope de vueltas actor + crítico por petición: acota coste y latencia.
    boss_max_iterations: Annotated[int, BeforeValidator(_empty_is_default_boss_iterations)] = (
        DEFAULT_BOSS_MAX_ITERATIONS
    )

    # --- Bucle agéntico y presupuesto de tokens (Paso 10). `agent_loop_enabled` enciende el
    # bucle en el que el modelo pide herramientas (solo hace algo si hay herramientas
    # registradas: las traen los Pasos 11 y 12). `agent_max_iterations` es el tope de rondas
    # modelo→herramientas por turno; `turn_token_budget`, el techo de tokens del contexto.
    agent_loop_enabled: bool = False
    agent_max_iterations: Annotated[int, BeforeValidator(_empty_is_default_agent_iterations)] = (
        DEFAULT_AGENT_MAX_ITERATIONS
    )
    turn_token_budget: Annotated[int, BeforeValidator(_empty_is_default_token_budget)] = (
        DEFAULT_TURN_TOKEN_BUDGET
    )

    # --- Búsqueda web como herramienta (Paso 11). `search_default` es el servicio principal y
    # `search_fallback` el respaldo (tienen que ser distintos). Las claves son opcionales: sin
    # ninguna, la herramienta `web_search` no se registra. `search_max_per_turn` acota las
    # búsquedas por petición; `search_max_snippet_chars`, el texto de cada resultado;
    # `search_timeout_seconds`, la espera a cada servicio. `search_log_queries` permite
    # registrar (truncado) el texto de la consulta; apagado, solo se registra su largo.
    search_default: str = "tavily"
    search_fallback: str = "firecrawl"
    web_tavily_key: SecretStr | None = None
    web_firecrawl_key: SecretStr | None = None
    search_max_per_turn: Annotated[int, BeforeValidator(_empty_is_default_search_max_per_turn)] = (
        DEFAULT_SEARCH_MAX_PER_TURN
    )
    search_max_snippet_chars: Annotated[
        int, BeforeValidator(_empty_is_default_search_snippet_chars)
    ] = DEFAULT_SEARCH_MAX_SNIPPET_CHARS
    search_timeout_seconds: Annotated[float, BeforeValidator(_empty_is_default_search_timeout)] = (
        DEFAULT_SEARCH_TIMEOUT_SECONDS
    )
    search_log_queries: bool = False

    # --- Consulta a la BBDD del negocio por function calling (Paso 12). El servicio de IA no
    # conoce el esquema del negocio: solo la URL y la clave del backend (en este repo, el
    # simulado de `business_backend/`). Sin URL o sin clave, la herramienta
    # `find_similar_projects` no se registra, igual que la búsqueda web sin claves.
    business_api_url: str | None = None
    business_api_key: SecretStr | None = None
    business_max_per_turn: Annotated[
        int, BeforeValidator(_empty_is_default_business_max_per_turn)
    ] = DEFAULT_BUSINESS_MAX_PER_TURN
    business_timeout_seconds: Annotated[
        float, BeforeValidator(_empty_is_default_business_timeout)
    ] = DEFAULT_BUSINESS_TIMEOUT_SECONDS

    # --- Compresión del historial (Paso 9). Cómo se reconocen los turnos con compromisos
    # («anclas») que se conservan textuales: `heuristic` (expresiones regulares, sin coste)
    # o `llm` (un clasificador por cada turno que sale de la ventana, más robusto ante
    # paráfrasis). Vacío = heuristic.
    anchor_detection_mode: str = "heuristic"

    # --- Frontend. URL base del API para Streamlit (cliente HTTP separado).
    estimator_api_base_url: str = "http://localhost:8001"

    # --- Servicio.
    app_env: str = "local"
    log_level: Annotated[LOG_LEVELS, BeforeValidator(_normalize_log_level)] = "INFO"
    app_port: int = 8001

    @model_validator(mode="after")
    def apply_defaults(self) -> "Settings":
        """Normaliza lo opcional y falla rápido ante lo inválido.

        Se conserva la regla de lidr_3: cadena vacía = "no configurado" =
        se usa el default. Un valor presente pero inválido es un error de
        configuración y muere acá, no a mitad de una request.
        """
        self.app_env = self.app_env or "local"
        self.llm_default = self.llm_default or "openai"
        self.llm_fallback = self.llm_fallback or "anthropic"
        self.openai_model = self.openai_model or "gpt-4o-mini"
        self.anthropic_model = self.anthropic_model or "claude-haiku-5-5"
        self.extractor_default = self.extractor_default or "gemini"
        self.extractor_fallback = self.extractor_fallback or "deepseek"
        self.gemini_model = self.gemini_model or "gemini-3.1-flash-lite"
        self.deepseek_model = self.deepseek_model or "deepseek-flash"
        self.conversation_prompt_version = self.conversation_prompt_version or "v5"
        self.default_tier = self.default_tier or "default"
        self.anchor_detection_mode = self.anchor_detection_mode or "heuristic"
        self.moderation_model = self.moderation_model or "omni-moderation-latest"
        # pydantic-settings parsea un `PROMPT_VERSION=` vacío como "", y "" no
        # es un nombre de archivo válido: un .env a medio completar rompería el
        # arranque en el peor momento. Acá se valida el dominio a mano porque
        # `log_level` sí es un Literal y no necesita esto: su campo normaliza
        # el vacío en un BeforeValidator, que corre antes de validar el dominio.
        if not self.prompt_version:
            raise ValueError("PROMPT_VERSION no puede estar vacío")
        return self

    @model_validator(mode="after")
    def validate_models(self) -> "Settings":
        """Los dos deployments del Router tienen que ser distinguibles.

        Cada proveedor tiene un solo modelo (`<PROVEEDOR>_MODEL`), así que dos
        proveedores iguales son dos modelos iguales: el "fallback" sería un
        no-op que igual paga una llamada extra y una latencia. Y un proveedor
        sin API key ni modelo conocidos no puede armar un deployment. Los dos son
        errores de configuración, no casos degenerados aceptables.
        """
        for variable, proveedor in (
            ("LLM_DEFAULT", self.llm_default),
            ("LLM_FALLBACK", self.llm_fallback),
        ):
            if proveedor not in API_KEY_VARIABLES:
                validos = ", ".join(sorted(API_KEY_VARIABLES))
                raise ValueError(
                    f"{variable}='{proveedor}' no es un proveedor soportado. Valores válidos: "
                    f"{validos}."
                )
        if self.llm_default == self.llm_fallback:
            raise ValueError(
                f"LLM_DEFAULT y LLM_FALLBACK son iguales ('{self.llm_default}'). "
                "El fallback no podría nunca atender una request."
            )
        return self

    @model_validator(mode="after")
    def validate_description_ceiling(self) -> "Settings":
        """Settings solo puede hacer el contrato MÁS ESTRICTO, nunca más laxo.

        La intersección de los dos límites es la que se aplica, y el `Field`
        (20/2000) siempre está: no se puede relajar. Entonces hay dos
        direcciones, y solo una rompe la documentación:

        - Settings más estricto (min >= 20, max <= 2000): manda Settings, y el
          parche escribe ese mismo número. Swagger coincide con el servicio.
          Acá es donde vive la reducción de coste sin redeploy.
        - Settings más laxo (min < 20, max > 2000): el `Field` sigue cortando,
          pero el parche escribe el número laxo, así que Swagger promete un
          rango que el servicio no acepta. Es la misma mentira que tenía WU1,
          solo que con un default que la dispara.

        Los dos casos laxos son un `.env` mal puesto y mueren acá, al arrancar.
        Si se dejaran pasar, el síntoma aparecería como un 422 inexplicable
        en producción, que es el peor lugar y el peor momento."""
        if self.description_min_chars < CONTRACT_MIN_CHARS:
            raise ValueError(
                f"DESCRIPTION_MIN_CHARS={self.description_min_chars} es menor que el "
                f"mínimo del contrato ({CONTRACT_MIN_CHARS}). Este valor solo puede "
                "estrechar el rango: el `min_length` del schema seguiría rechazando "
                "texto más corto y Swagger documentaría un mínimo que el servicio "
                "no acepta. Para subir el mínimo no hay que tocar esto."
            )
        if self.description_max_chars > CONTRACT_MAX_CHARS:
            raise ValueError(
                f"DESCRIPTION_MAX_CHARS={self.description_max_chars} es mayor que el "
                f"máximo del contrato ({CONTRACT_MAX_CHARS}). Este valor solo puede "
                "estrechar el rango: el `max_length` del schema seguiría rechazando "
                "texto más largo y Swagger documentaría un máximo que el servicio "
                f"no acepta. Para BAJAR el techo de coste, ponelo en {CONTRACT_MAX_CHARS} o menos."
            )
        if self.description_min_chars > self.description_max_chars:
            raise ValueError(
                f"DESCRIPTION_MIN_CHARS={self.description_min_chars} es mayor que "
                f"DESCRIPTION_MAX_CHARS={self.description_max_chars}: ningún texto "
                "sería aceptado."
            )
        return self

    @model_validator(mode="after")
    def validate_extractor(self) -> "Settings":
        """Primario y respaldo del extractor: proveedores conocidos y distintos; tier válido."""
        for variable, proveedor in (
            ("EXTRACTOR_DEFAULT", self.extractor_default),
            ("EXTRACTOR_FALLBACK", self.extractor_fallback),
        ):
            if proveedor not in EXTRACTOR_KEY_VARIABLES:
                validos = ", ".join(sorted(EXTRACTOR_KEY_VARIABLES))
                raise ValueError(
                    f"{variable}='{proveedor}' no es un proveedor soportado. Valores válidos: "
                    f"{validos}."
                )
        if self.extractor_default == self.extractor_fallback:
            raise ValueError(
                f"EXTRACTOR_DEFAULT y EXTRACTOR_FALLBACK son iguales ('{self.extractor_default}'). "
                "El respaldo no podría nunca atender una llamada."
            )
        if self.default_tier not in TIER_VALUES:
            raise ValueError(
                f"DEFAULT_TIER='{self.default_tier}' no es válido. Valores válidos: "
                f"{', '.join(TIER_VALUES)}."
            )
        return self

    # --- Structured response ----------------------------------------------------
    @model_validator(mode="after")
    def validate_structured_output(self) -> "Settings":
        """Mismas reglas que PROMPT_VERSION y un número de reintentos posible.

        Si la versión existe y es de salida estructurada lo comprueba el endpoint,
        contra los directorios publicados: Settings no conoce el loader.
        """
        if not self.structured_prompt_version:
            raise ValueError("STRUCTURED_PROMPT_VERSION no puede estar vacío")
        if not self.rendered_prompt_version:
            raise ValueError("RENDERED_PROMPT_VERSION no puede estar vacío")
        if self.structured_max_retries < 0:
            raise ValueError(
                f"STRUCTURED_MAX_RETRIES={self.structured_max_retries} no puede ser negativo."
            )
        return self

    # --- Input guardrails -------------------------------------------------------
    @model_validator(mode="after")
    def validate_input_guardrails(self) -> "Settings":
        """Un timeout de moderación que no espera nada rechazaría todo en silencio."""
        if self.moderation_timeout <= 0:
            raise ValueError(
                f"MODERATION_TIMEOUT={self.moderation_timeout} tiene que ser mayor que cero."
            )
        return self

    # --- Output guardrails -------------------------------------------------------
    @model_validator(mode="after")
    def validate_output_guardrails(self) -> "Settings":
        """Un umbral fuera de 0 a 100 no es una confianza: mejor romper al arrancar."""
        if not 0 <= self.low_confidence_warn_pct <= 100:
            raise ValueError(
                f"LOW_CONFIDENCE_WARN_PCT={self.low_confidence_warn_pct} tiene que estar "
                "entre 0 y 100."
            )
        return self

    # --- Sesiones -----------------------------------------------------------------
    @model_validator(mode="after")
    def validate_sessions(self) -> "Settings":
        """Una ventana sin turnos no recordaría nada: mejor romper al arrancar."""
        if self.max_turns < 1:
            raise ValueError(f"MAX_TURNS={self.max_turns} tiene que ser al menos 1.")
        if self.session_ttl_seconds < 1:
            raise ValueError(
                f"SESSION_TTL_SECONDS={self.session_ttl_seconds} tiene que ser al menos 1."
            )
        if self.max_sessions < 1:
            raise ValueError(f"MAX_SESSIONS={self.max_sessions} tiene que ser al menos 1.")
        if self.boss_max_iterations < 1:
            raise ValueError(
                f"BOSS_MAX_ITERATIONS={self.boss_max_iterations} tiene que ser al menos 1."
            )
        if self.agent_max_iterations < 1:
            raise ValueError(
                f"AGENT_MAX_ITERATIONS={self.agent_max_iterations} tiene que ser al menos 1."
            )
        if self.turn_token_budget < 1_000:
            raise ValueError(
                f"TURN_TOKEN_BUDGET={self.turn_token_budget} es demasiado bajo: mínimo 1.000."
            )
        for name, value in (
            ("SEARCH_DEFAULT", self.search_default),
            ("SEARCH_FALLBACK", self.search_fallback),
        ):
            if value not in SUPPORTED_SEARCH_PROVIDERS:
                raise ValueError(
                    f"{name}='{value}' no es válido: usa uno de {list(SUPPORTED_SEARCH_PROVIDERS)}."
                )
        if self.search_default == self.search_fallback:
            raise ValueError(
                "SEARCH_DEFAULT y SEARCH_FALLBACK tienen que ser distintos "
                f"(ambos son '{self.search_default}')."
            )
        if self.search_max_per_turn < 1:
            raise ValueError(
                f"SEARCH_MAX_PER_TURN={self.search_max_per_turn} tiene que ser al menos 1."
            )
        if self.search_max_snippet_chars < 100:
            raise ValueError(
                f"SEARCH_MAX_SNIPPET_CHARS={self.search_max_snippet_chars} es demasiado bajo: "
                "mínimo 100."
            )
        if self.search_timeout_seconds <= 0:
            raise ValueError(
                f"SEARCH_TIMEOUT_SECONDS={self.search_timeout_seconds} tiene que ser mayor que 0."
            )
        if self.business_max_per_turn < 1:
            raise ValueError(
                f"BUSINESS_MAX_PER_TURN={self.business_max_per_turn} tiene que ser al menos 1."
            )
        if self.business_timeout_seconds <= 0:
            raise ValueError(
                f"BUSINESS_TIMEOUT_SECONDS={self.business_timeout_seconds} tiene que ser mayor "
                "que 0."
            )
        if self.anchor_detection_mode not in ("heuristic", "llm"):
            raise ValueError(
                f"ANCHOR_DETECTION_MODE='{self.anchor_detection_mode}' no es válido: "
                "usa 'heuristic' o 'llm'."
            )
        return self

    # --- Caché semántica -------------------------------------------------------
    @model_validator(mode="after")
    def validate_semantic_cache(self) -> "Settings":
        """Un umbral fuera de (0, 1] o un tope sin entradas no son una caché: romper al arrancar."""
        if not 0 < self.cache_semantic_threshold <= 1:
            raise ValueError(
                f"CACHE_SEMANTIC_THRESHOLD={self.cache_semantic_threshold} tiene que estar "
                "en (0, 1]."
            )
        if self.embedding_timeout <= 0:
            raise ValueError(
                f"EMBEDDING_TIMEOUT={self.embedding_timeout} tiene que ser mayor que cero."
            )
        if self.semantic_cache_max_entries < 1:
            raise ValueError(
                f"SEMANTIC_CACHE_MAX_ENTRIES={self.semantic_cache_max_entries} tiene que ser "
                "al menos 1."
            )
        self.embedding_model = self.embedding_model or DEFAULT_EMBEDDING_MODEL
        return self

    def _model_of(self, provider: str) -> str:
        """'openai' -> 'openai/gpt-4o-mini' (el modelo sale de `<PROVEEDOR>_MODEL`)."""
        return f"{provider}/{getattr(self, f'{provider}_model')}"

    @property
    def primary_model(self) -> str:
        """Modelo del deployment primario, en el formato de LiteLLM."""
        return self._model_of(self.llm_default)

    @property
    def fallback_model(self) -> str:
        """Modelo del deployment de respaldo, en el formato de LiteLLM."""
        return self._model_of(self.llm_fallback)

    @property
    def primary_provider(self) -> str:
        """Provider del deployment primario (LLM_DEFAULT)."""
        return self.llm_default

    @property
    def fallback_provider(self) -> str:
        """Provider del deployment de respaldo (LLM_FALLBACK)."""
        return self.llm_fallback

    def for_extractor(self) -> "Settings":
        """Copia de la configuración con el primario y el respaldo del EXTRACTOR.

        Permite reutilizar `LLMWrapper` (Router de LiteLLM, reintentos, respaldo y coste)
        para el extractor sin duplicarlo: el wrapper lee `llm_default` y `llm_fallback`, y
        en esta copia valen `extractor_default` y `extractor_fallback`. No se vuelve a
        validar (`model_copy`), porque `validate_models` solo admite los proveedores del
        servicio de estimación; los del extractor ya los validó `validate_extractor`.
        """
        return self.model_copy(
            update={"llm_default": self.extractor_default, "llm_fallback": self.extractor_fallback}
        )

    def for_provider(self, provider: str) -> "Settings":
        """Copia de la configuración con `provider` como único proveedor primario.

        La usa el crítico del ACB para construir un `LLMWrapper` que llame a un proveedor
        concreto (el distinto del actor, o el mismo si el otro falla). No se vuelve a
        validar (`model_copy`): los proveedores ya los validó `validate_models`.
        """
        return self.model_copy(update={"llm_default": provider})

    @property
    def is_configured(self) -> bool:
        """True si hay API key para el deployment primario.

        Reporta el primario y no ambos a propósito: /health debe describir
        si el camino de una request normal funciona, no si están todos los
        cables conectados. El fallback se valida en su propio momento, cuando
        se usa.
        """
        return self.active_api_key(self.primary_provider) is not None

    def active_api_key(self, provider: str | None = None) -> str | None:
        """La key del provider indicado (SecretStr -> str), sin lanzar.

        provider=None usa el primario. Devuelve None si falta la key.
        """
        provider = provider or self.primary_provider
        if provider not in EXTRACTOR_KEY_VARIABLES:
            return None
        raw = getattr(self, f"{provider}_api_key", None)
        return raw.get_secret_value() if raw else None


@lru_cache
def get_settings() -> Settings:
    """Devuelve la configuración de la app, leída de las variables de entorno y del `.env`.

    Está envuelta con una caché (ver `lru_cache` arriba): la configuración se lee una
    sola vez y todos comparten el mismo objeto.
    """
    return Settings()
