"""Dependencias de FastAPI: el LLMWrapper compartido entre requests.

El wrapper se construye perezoso, en la primera request que lo necesita, y se
guarda en app.state para reutilizarlo. Perezoso a propósito: si falta una API
key, el servicio tiene que arrancar igual y /health tiene que poder decirlo
(ver app/main.py); construirlo en el lifespan lo haría morir al arrancar.

La caché NO se crea acá: el wrapper usa la que armó el lifespan sobre el único
cliente de Redis de la app. Sin lifespan (una app montada a mano en un test) no
hay caché, y eso es el estado válido "sin caché", no un error.
"""

from __future__ import annotations

from fastapi import Depends, Request

from app.config import get_settings
from app.services.cache import create_estimation_cache
from app.services.estimation import EstimationService
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


def get_estimation_service(llm_wrapper: LLMWrapper = Depends(get_llm_wrapper)) -> EstimationService:
    """El pipeline de la estimación sobre el wrapper compartido.

    No guarda estado, así que se arma en cada request. Depende de
    `get_llm_wrapper`, y por eso un test que reemplaza el wrapper con
    `dependency_overrides` reemplaza también el que usa el servicio.
    """
    return EstimationService(llm_wrapper)
