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

from fastapi import Request

from app.config import get_settings
from app.services.cache import crear_estimation_cache
from app.services.llm_wrapper import LLMWrapper


def get_llm_wrapper(request: Request) -> LLMWrapper:
    state = request.app.state
    wrapper = getattr(state, "llm_wrapper", None)
    if wrapper is None:
        settings = get_settings()
        cache = getattr(state, "estimation_cache", None) or crear_estimation_cache(
            None, settings.cache_ttl
        )
        wrapper = LLMWrapper(
            openai_api_key=settings.openai_api_key,
            anthropic_api_key=settings.anthropic_api_key,
            primary_model=settings.primary_model,
            fallback_model=settings.fallback_model,
            timeout=settings.llm_timeout,
            num_retries=settings.llm_max_retries,
            model_group=settings.llm_model_group,
            cache=cache,
        )
        state.llm_wrapper = wrapper
    return wrapper
