"""Dependencias de FastAPI: el LLMWrapper compartido entre requests.

El wrapper se construye perezoso, en la primera request que lo necesita, y se
guarda en app.state para reutilizarlo. Perezoso a propósito: si falta una API
key, el servicio tiene que arrancar igual y /health tiene que poder decirlo
(ver app/main.py); construirlo en el lifespan lo haría morir al arrancar.

Los guardrails de entrada se arman igual: una vez, en la primera request, y se
guardan en app.state. Así el evento que avisa de una moderación apagada (por
falta de clave) sale una vez y no en cada request.

La caché NO se crea acá: el wrapper usa la que armó el lifespan sobre el único
cliente de Redis de la app. Sin lifespan (una app montada a mano en un test) no
hay caché, y eso es el estado válido "sin caché", no un error.
"""

from __future__ import annotations

from fastapi import Depends, Request

from app.config import get_settings
from app.services.cache import create_estimation_cache
from app.services.estimation import EstimationService
from app.services.guardrails import InputGuardrails, create_input_guardrails
from app.services.llm_wrapper import LLMWrapper


def get_llm_wrapper(request: Request) -> LLMWrapper:
    state = request.app.state
    wrapper = getattr(state, "llm_wrapper", None)
    if wrapper is None:
        settings = get_settings()
        cache = getattr(state, "estimation_cache", None) or create_estimation_cache(
            None, settings.cache_ttl
        )
        wrapper = LLMWrapper(settings=settings, cache=cache)
        state.llm_wrapper = wrapper
    return wrapper


# --- Input guardrails -------------------------------------------------------

# None también es un valor válido ("apagados"), así que hace falta otro centinela
# para distinguir "todavía no se armaron" de "se armaron y están apagados".
_NOT_BUILT = object()


def get_input_guardrails(request: Request) -> InputGuardrails | None:
    state = request.app.state
    guardrails = getattr(state, "input_guardrails", _NOT_BUILT)
    if guardrails is _NOT_BUILT:
        guardrails = create_input_guardrails(get_settings())
        state.input_guardrails = guardrails
    return guardrails


def get_estimation_service(
    llm_wrapper: LLMWrapper = Depends(get_llm_wrapper),
    input_guardrails: InputGuardrails | None = Depends(get_input_guardrails),
) -> EstimationService:
    """El pipeline de la estimación sobre el wrapper y los guardrails compartidos.

    No guarda estado, así que se arma en cada request. Depende de
    `get_llm_wrapper`, y por eso un test que reemplaza el wrapper con
    `dependency_overrides` reemplaza también el que usa el servicio.
    """
    return EstimationService(llm_wrapper, input_guardrails)
