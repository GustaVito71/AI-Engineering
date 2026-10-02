"""FastAPI dependency factories for shared singletons (cache and LLM wrapper)."""

from __future__ import annotations

from functools import lru_cache

from app.config import get_settings
from app.services.cache import EstimationCache
from app.services.llm_wrapper import LLMWrapper


@lru_cache
def get_cache() -> EstimationCache:
    settings = get_settings()
    return EstimationCache.from_url(settings.redis_url, ttl=settings.cache_ttl)


@lru_cache
def get_llm_wrapper() -> LLMWrapper:
    settings = get_settings()
    return LLMWrapper(
        openai_api_key=settings.openai_api_key,
        anthropic_api_key=settings.anthropic_api_key,
        primary_model=settings.primary_model,
        fallback_model=settings.fallback_model,
        timeout=settings.llm_timeout,
        num_retries=settings.llm_max_retries,
        model_group=settings.llm_model_group,
        cache=get_cache(),
    )
