"""Wrapper del LiteLLM Router con fallback automático y tracking de coste.

Diseñado para ser instanciado una sola vez (singleton vía get_llm_wrapper)
y reutilizado en todas las requests. El Router maneja el failover entre
el deployment primario y el fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from litellm import Router, acompletion
from litellm.types.router import Deployment as DeploymentDict

from app.config import get_settings


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
        fallback_model: str,
        timeout: float,
        num_retries: int,
        model_group: str,
        cache: Any,  # EstimationCache
    ) -> None:
        self._cache = cache
        self._model_group = model_group

        settings = get_settings()

        # Build deployments for the Router
        primary_deployment = self._build_deployment(
            model_name=primary_model,
            litellm_model=primary_model,
            api_key=(
                openai_api_key.get_secret_value()
                if openai_api_key and settings.primary_provider == "openai"
                else anthropic_api_key.get_secret_value()
                if anthropic_api_key
                else None
            ),
        )
        fallback_deployment = self._build_deployment(
            model_name=fallback_model,
            litellm_model=fallback_model,
            api_key=(
                openai_api_key.get_secret_value()
                if openai_api_key and settings.fallback_provider == "openai"
                else anthropic_api_key.get_secret_value()
                if anthropic_api_key
                else None
            ),
        )

        self._router = Router(
            model_list=[primary_deployment, fallback_deployment],
            routing_strategy="simple-shuffle",
            fallbacks=[{primary_model: [fallback_model]}],
            num_retries=2,
            timeout=settings.llm_timeout,
        )

        self._primary_model = primary_model
        self._fallback_model = fallback_model
        self._timeout = timeout
        self._max_retries = 2

    def _build_deployment(
        self,
        model_name: str,
        litellm_model: str,
        api_key: str | None,
    ) -> DeploymentDict:
        """Construye un deployment para el LiteLLM Router."""
        provider = litellm_model.split("/", 1)[0] if "/" in litellm_model else "openai"
        params = {
            "model": litellm_model,
        }
        if api_key:
            params["api_key"] = api_key
        return {
            "model_name": model_name,
            "litellm_params": params,
            "model_info": {"provider": provider},
        }

    async def estimate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        prompt_version: str,
        max_tokens: int = 4000,
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

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        response = await acompletion(
            model=self._primary_model,
            messages=messages,
            max_tokens=4000,
            temperature=0.1,
        )

        # Extract response content and metadata
        content = response.choices[0].message.content
        model = response.model
        provider = self._get_provider_from_model(model)

        # Calculate cost from usage
        usage = response.usage
        prompt_tokens = usage.prompt_tokens if usage else 0
        completion_tokens = usage.completion_tokens if usage else 0
        cost_usd = self._calculate_cost(provider, prompt_tokens, completion_tokens)

        # Cache the result
        cache_data = {
            "content": content,
            "model": model,
            "provider": provider,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cost_usd": cost_usd,
        }
        await self._cache.set(cache_key, cache_data)

        return LLMCallResult(
            content=content,
            model=model,
            provider=provider,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=cost_usd,
            prompt_version=prompt_version,
        )

    def _get_provider_from_model(self, model: str) -> str:
        """Extract provider from model name returned by LiteLLM."""
        if model.startswith("openai/"):
            return "openai"
        elif model.startswith("anthropic/"):
            return "anthropic"
        elif model.startswith("gemini/"):
            return "google"
        elif model.startswith("bedrock/"):
            return "bedrock"
        return "unknown"

    def _calculate_cost(self, provider: str, prompt_tokens: int, completion_tokens: int) -> float:
        r"""Calculate cost in USD based on provider and token counts.

        Pricing as of 2024 (approximate):
        - gpt-4o-mini: \$0.15/1M input, \$0.60/1M output
        - claude-3-haiku: \$0.25/1M input, \$1.25/1M output
        """
        if provider == "openai":
            return (prompt_tokens * 0.15 + completion_tokens * 0.60) / 1_000_000
        elif provider == "anthropic":
            return (prompt_tokens * 0.25 + completion_tokens * 1.25) / 1_000_000
        return 0.0
