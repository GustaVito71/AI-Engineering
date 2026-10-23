"""Dependencias de FastAPI: el LLMWrapper compartido entre requests.

El wrapper se construye perezoso, en la primera request que lo necesita, y se
guarda en app.state para reutilizarlo. Perezoso a propósito: si falta una API
key, el servicio tiene que arrancar igual y /health tiene que poder decirlo
(ver app/main.py); construirlo en el lifespan lo haría morir al arrancar.

Los guardrails de entrada y de salida se arman igual: una vez, en la primera request, y se
guardan en app.state. Así el evento que avisa de una moderación apagada (por
falta de clave) sale una vez y no en cada request.

La caché NO se crea acá: el servicio usa la que armó el lifespan sobre el único
cliente de Redis de la app (el wrapper no la toca). Sin lifespan (una app montada
a mano en un test) no hay caché, y eso es el estado válido "sin caché", no un
error.
"""

from __future__ import annotations

from fastapi import Depends, Request

from app.agent.tools import ToolRegistry
from app.business.tool import build_tool_from_settings as build_business_tool_from_settings
from app.config import LLMConfigurationError, get_settings
from app.guardrails import (
    InputGuardrails,
    OutputGuardrails,
    create_input_guardrails,
    create_output_guardrails,
)
from app.search.tool import build_tool_from_settings as build_search_tool_from_settings
from app.services.cache import ExactCache, create_estimation_cache
from app.services.critic import Critic
from app.services.estimation import EstimationService
from app.services.llm_wrapper import LLMWrapper
from app.services.metadata_extractor import MetadataExtractor
from app.services.semantic_cache import SemanticCache
from app.sessions import SessionStore
from app.sessions.compression import CompressionTools, build_compression_tools
from app.tracing import emit


def get_llm_wrapper(request: Request) -> LLMWrapper:
    """Devuelve el cliente de LLM compartido, creándolo la primera vez que alguien lo pide.

    Se guarda en `app.state` para reutilizarlo en todas las peticiones.
    """
    state = request.app.state
    wrapper = getattr(state, "llm_wrapper", None)
    if wrapper is None:
        wrapper = LLMWrapper(settings=get_settings())
        state.llm_wrapper = wrapper
    return wrapper


def get_metadata_extractor(request: Request) -> MetadataExtractor | None:
    """El extractor de `project_metadata`, o None si no está configurado.

    Se crea la primera vez que se pide y se guarda en `app.state`. Si falta la clave
    del proveedor primario del extractor (por ejemplo `GEMINI_API_KEY`) devuelve None
    en vez de fallar: la estimación sigue funcionando y el endpoint avisa de que la
    memoria no se actualizó. El detalle (qué variable falta) queda en el log.
    """
    state = request.app.state
    extractor = getattr(state, "metadata_extractor", None)
    if extractor is None:
        try:
            extractor = MetadataExtractor.from_settings(get_settings())
        except LLMConfigurationError as exc:
            emit(__name__, "extractor_no_disponible", level="warning", detalle=str(exc))
            return None
        state.metadata_extractor = extractor
    return extractor


def get_compression_tools(request: Request) -> CompressionTools:
    """El detector de anclas y el resumidor del Paso 9, creados la primera vez y compartidos.

    Nunca falla: sin clave del proveedor del extractor no hay resumidor y la compresión
    solo conserva las anclas (ver `build_compression_tools`).
    """
    state = request.app.state
    tools = getattr(state, "compression_tools", None)
    if tools is None:
        tools = build_compression_tools(get_settings())
        state.compression_tools = tools
    return tools


def get_critic(request: Request) -> Critic:
    """El crítico del ACB (Paso 8), creado la primera vez que se pide y compartido después.

    No necesita claves al crearse: cada proveedor se prepara cuando el crítico lo usa, y si
    falta su clave esa revisión falla y el crítico pasa al siguiente intento (ver
    `app/services/critic.py`). Por eso esta dependencia nunca devuelve None.
    """
    state = request.app.state
    critic = getattr(state, "critic", None)
    if critic is None:
        critic = Critic(settings=get_settings())
        state.critic = critic
    return critic


def get_tool_registry(request: Request) -> ToolRegistry:
    """El registro de herramientas del bucle agéntico (Paso 10), creado la primera vez.

    Se llena con las herramientas disponibles, todas condicionadas a `agent_loop_enabled`:
    la búsqueda web (Paso 11) si hay alguna clave de búsqueda, y la consulta al backend de
    negocio (Paso 12) si hay URL y clave configuradas. Mientras esté vacío, el bucle no hace
    nada.
    """
    state = request.app.state
    registry = getattr(state, "tool_registry", None)
    if registry is None:
        registry = ToolRegistry()
        settings = get_settings()
        if settings.agent_loop_enabled:
            web_search = build_search_tool_from_settings(settings)
            if web_search is not None:
                registry.register(web_search)
            find_similar_projects = build_business_tool_from_settings(settings)
            if find_similar_projects is not None:
                registry.register(find_similar_projects)
        state.tool_registry = registry
    return registry


# --- Exact cache ------------------------------------------------------------


def get_estimation_cache(request: Request) -> ExactCache:
    """La caché del lifespan, o una desactivada si la app se montó sin lifespan."""
    return getattr(request.app.state, "estimation_cache", None) or create_estimation_cache(
        None, get_settings().cache_ttl
    )


# --- Semantic cache ---------------------------------------------------------


def get_semantic_cache(request: Request) -> SemanticCache | None:
    """La caché semántica del lifespan, o None si está apagada o no hay lifespan."""
    return getattr(request.app.state, "semantic_cache", None)


# --- Sesiones conversacionales ----------------------------------------------


def get_session_store(request: Request) -> SessionStore:
    """El almacén de sesiones, compartido por todas las peticiones de la app.

    Tiene que ser UN solo objeto durante toda la vida del servicio: si cada petición
    creara el suyo, una sesión creada en una petición no existiría en la siguiente.
    Por eso se guarda en `app.state` (el espacio de la aplicación donde viven los
    objetos compartidos) y se reutiliza.

    Lo normal es que lo cree el arranque de la app (ver `lifespan` en app/main.py).
    Si no existe todavía (una app montada a mano en un test, sin arranque), se crea
    acá la primera vez. No hace falta protegerlo contra dos peticiones simultáneas:
    la app corre en un solo hilo y no hay pausas entre mirar y guardar.
    """
    state = request.app.state
    store = getattr(state, "session_store", None)
    if store is None:
        settings = get_settings()
        store = SessionStore(
            max_turns=settings.max_turns,
            ttl_seconds=settings.session_ttl_seconds,
            max_sessions=settings.max_sessions,
        )
        state.session_store = store
    return store


# --- Input guardrails -------------------------------------------------------

# None también es un valor válido ("apagados"), así que hace falta otro centinela
# para distinguir "todavía no se armaron" de "se armaron y están apagados".
_NOT_BUILT = object()


def get_input_guardrails(request: Request) -> InputGuardrails | None:
    """Devuelve las reglas que revisan la descripción antes de llamar al modelo (o None si
    están apagadas).

    Se arman la primera vez y se reutilizan.
    """
    state = request.app.state
    guardrails = getattr(state, "input_guardrails", _NOT_BUILT)
    if guardrails is _NOT_BUILT:
        guardrails = create_input_guardrails(get_settings())
        state.input_guardrails = guardrails
    return guardrails


# --- Output guardrails ------------------------------------------------------


def get_output_guardrails(request: Request) -> OutputGuardrails | None:
    """Devuelve las reglas que revisan la respuesta del modelo antes de entregarla (o None
    si están apagadas).

    Se arman la primera vez y se reutilizan.
    """
    state = request.app.state
    guardrails = getattr(state, "output_guardrails", _NOT_BUILT)
    if guardrails is _NOT_BUILT:
        guardrails = create_output_guardrails(get_settings())
        state.output_guardrails = guardrails
    return guardrails


def get_estimation_service(
    llm_wrapper: LLMWrapper = Depends(get_llm_wrapper),
    input_guardrails: InputGuardrails | None = Depends(get_input_guardrails),
    cache: ExactCache = Depends(get_estimation_cache),
    output_guardrails: OutputGuardrails | None = Depends(get_output_guardrails),
    semantic_cache: SemanticCache | None = Depends(get_semantic_cache),
) -> EstimationService:
    """El pipeline de la estimación sobre el wrapper, los guardrails y la caché.

    No guarda estado, así que se arma en cada request. Depende de
    `get_llm_wrapper`, y por eso un test que reemplaza el wrapper con
    `dependency_overrides` reemplaza también el que usa el servicio.
    """
    return EstimationService(
        llm_wrapper, input_guardrails, cache, output_guardrails, semantic_cache
    )
