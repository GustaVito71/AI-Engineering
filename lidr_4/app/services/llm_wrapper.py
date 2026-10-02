"""Wrapper del LiteLLM Router con fallback automático y tracking de coste.

Diseñado para ser instanciado una sola vez (singleton vía get_llm_wrapper)
y reutilizado en todas las requests. El Router maneja el failover entre
el deployment primario y el fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from litellm import Router, completion_cost

from app.config import LLMConfigurationError, get_settings


@dataclass
class LLMCallResult:
    """Resultado de una llamada al LLM con metadata para caché y observabilidad."""

    content: str
    model: str
    provider: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    prompt_version: str


class LLMWrapper:
    """Wrapper alrededor del LiteLLM Router con fallback y tracking de coste."""

    def __init__(
        self,
        *,
        openai_api_key: str | None,
        anthropic_api_key: str | None,
        primary_model: str,
        fallback_model: str | None,
        timeout: float,
        num_retries: int,
        model_group: str,
        cache: Any,  # EstimationCache
    ) -> None:
        self._cache = cache
        self._primary_model = primary_model
        self._fallback_model = fallback_model
        self._timeout = timeout
        self._num_retries = num_retries
        self._model_group = model_group

        settings = get_settings()

        # Resolve API keys using Settings.active_api_key
        primary_provider = primary_model.split("/", 1)[0] if "/" in primary_model else "openai"
        fallback_provider = (
            (fallback_model or "").split("/", 1)[0]
            if fallback_model and "/" in fallback_model
            else "openai"
        )

        primary_key = settings.active_api_key(primary_provider)
        fallback_key = settings.active_api_key(fallback_provider) if fallback_model else None

        if not primary_key:
            raise LLMConfigurationError(
                f"Missing API key for primary model {primary_model} (provider: {primary_provider})"
            )
        if fallback_model and not fallback_key:
            raise LLMConfigurationError(
                f"Missing API key for fallback model {fallback_model} "
                f"(provider: {fallback_model.split('/', 1)[0] if '/' in fallback_model else 'openai'})"
            )

        # Build deployments for the Router
        deployments = [
            {
                "model_name": primary_model,
                "litellm_params": {"model": primary_model, "api_key": primary_key},
            },
        ]
        if fallback_model:
            deployments.append(
                {
                    "model_name": fallback_model,
                    "litellm_params": {"model": fallback_model, "api_key": fallback_key},
                }
            )

        self._router = Router(
            model_list=deployments,
            fallbacks=[{primary_model: [fallback_model]}] if fallback_model else [],
            num_retries=num_retries,
            timeout=timeout,
        )

        # El Router asigna a cada deployment un id interno (un hash) y lo devuelve
        # en response._hidden_params["model_id"]. Este mapa lo traduce al nombre
        # del modelo ("openai/gpt-4o-mini"), que es lo que se expone y se cachea.
        self._modelo_por_id = {
            d["model_info"]["id"]: d["litellm_params"]["model"] for d in self._router.model_list
        }
        self._max_tokens = settings.llm_max_tokens

    async def estimate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        prompt_version: str,
        max_tokens: int | None = None,
    ) -> LLMCallResult:
        """Ejecuta la estimación con fallback automático y cache."""
        # Check cache first
        cache_key = self._cache.make_key(system_prompt, user_prompt)
        cached = await self._cache.get(cache_key)
        if cached:
            return LLMCallResult(
                content=cached["content"],
                model=cached["model"],
                provider=cached["provider"],
                prompt_tokens=cached["prompt_tokens"],
                completion_tokens=cached["completion_tokens"],
                cost_usd=cached["cost_usd"],
                prompt_version=prompt_version,
            )

        response = await self._router.acompletion(
            model=self._primary_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max_tokens or self._max_tokens,
            temperature=0.1,
        )

        content = response.choices[0].message.content

        # Qué deployment respondió (primario o respaldo), traducido a su nombre.
        deployment_id = (getattr(response, "_hidden_params", None) or {}).get("model_id")
        model = self._modelo_por_id.get(deployment_id, response.model)
        provider = model.split("/", 1)[0] if "/" in model else "unknown"

        usage = response.usage
        prompt_tokens = usage.prompt_tokens if usage else 0
        completion_tokens = usage.completion_tokens if usage else 0

        # Un modelo sin precio en la tabla de LiteLLM no debe convertir una
        # respuesta válida en un error: el coste es observabilidad, no negocio.
        try:
            cost_usd = completion_cost(response)
        except Exception:  # noqa: BLE001
            cost_usd = 0.0

        result = LLMCallResult(
            content=content,
            model=model,
            provider=provider,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=cost_usd,
            prompt_version=prompt_version,
        )
        await self._cache.set(
            cache_key,
            {
                "content": result.content,
                "model": result.model,
                "provider": result.provider,
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
                "cost_usd": result.cost_usd,
            },
        )
        return result
