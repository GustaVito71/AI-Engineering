"""Wrapper del LiteLLM Router con fallback automático y tracking de coste.

Diseñado para ser instanciado una sola vez (singleton vía get_llm_wrapper)
y reutilizado en todas las requests. El Router maneja el failover entre
el deployment primario y el fallback.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from litellm import Router, completion_cost

from app.config import VARIABLE_DE_API_KEY, LLMConfigurationError, Settings, provider_de_modelo
from app.tracing import emitir


def _variable_de_clave(provider: str) -> str:
    """'openai' -> 'OPENAI_API_KEY'. Un provider sin variable conocida se nombra tal cual."""
    return VARIABLE_DE_API_KEY.get(provider, f"la API key del proveedor '{provider}'")


def aviso_sin_respaldo(settings: Settings) -> str | None:
    """Aviso para el usuario si el modelo de respaldo no tiene API key, o None.

    Sin la clave del respaldo el servicio funciona igual con el primario, pero
    pierde la protección ante una caída del proveedor. Lo usan el wrapper (que
    lo devuelve en cada estimación) y /health, para que los dos digan lo mismo.
    """
    # Settings garantiza que FALLBACK_MODEL nunca está vacío (aplicar_defaults).
    fallback_model = settings.fallback_model
    provider = settings.fallback_provider
    if settings.active_api_key(provider):
        return None
    return (
        f"El modelo de respaldo {fallback_model} no está disponible porque falta "
        f"{_variable_de_clave(provider)}. Las estimaciones usan solo el modelo "
        "primario: si ese proveedor falla, la estimación no se podrá completar."
    )


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
    # True si la respuesta salió de la caché: no hubo llamada al proveedor.
    cached: bool = False


class LLMWrapper:
    """Wrapper alrededor del LiteLLM Router con fallback y tracking de coste."""

    def __init__(self, *, settings: Settings, cache: Any) -> None:
        """Arma el Router con la configuración de `settings`.

        `cache` es una EstimationCache (o la caché nula si Redis está
        desactivado): ver app/services/cache.py.
        """
        self._cache = cache
        primary_model = settings.primary_model
        fallback_model = settings.fallback_model
        self._primary_model = primary_model

        # Cada deployment recibe la clave de su propio proveedor.
        primary_provider = settings.primary_provider
        primary_key = settings.active_api_key(primary_provider)
        fallback_key = settings.active_api_key(settings.fallback_provider)

        # Sin la clave del primario no hay servicio: el mensaje nombra la
        # variable y llega tal cual al cliente en el 503 (ver app/main.py).
        if not primary_key:
            raise LLMConfigurationError(
                f"Falta {_variable_de_clave(primary_provider)} "
                f"para el modelo primario {primary_model}."
            )

        # Sin la clave del respaldo el servicio sigue con el primario solo,
        # igual que /health (Settings.is_configured mira solo el primario). El
        # operador se entera por el log; el usuario, por `avisos` en cada
        # respuesta.
        aviso = aviso_sin_respaldo(settings)
        self.avisos: tuple[str, ...] = (aviso,) if aviso else ()
        if aviso:
            emitir(
                __name__,
                "respaldo_no_disponible",
                nivel="warning",
                modelo_respaldo=fallback_model,
                detalle=aviso,
            )

        # Deployments del Router: el primario y, si tiene clave, el de respaldo.
        deployments = [
            {
                "model_name": primary_model,
                "litellm_params": {"model": primary_model, "api_key": primary_key},
            },
        ]
        if fallback_key:
            deployments.append(
                {
                    "model_name": fallback_model,
                    "litellm_params": {"model": fallback_model, "api_key": fallback_key},
                }
            )

        self._router = Router(
            model_list=deployments,
            fallbacks=[{primary_model: [fallback_model]}] if fallback_key else [],
            num_retries=settings.llm_max_retries,
            timeout=settings.llm_timeout,
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
        inicio = time.perf_counter()

        # Primero la caché: un acierto evita la llamada al proveedor.
        cache_key = self._cache.make_key(system_prompt, user_prompt)
        cached = await self._cache.get(cache_key)
        if cached:
            result = LLMCallResult(
                content=cached["content"],
                model=cached["model"],
                provider=cached["provider"],
                prompt_tokens=cached["prompt_tokens"],
                completion_tokens=cached["completion_tokens"],
                cost_usd=cached["cost_usd"],
                prompt_version=prompt_version,
                cached=True,
            )
            self._registrar(result, inicio)
            return result

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
        provider = provider_de_modelo(model)

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
        self._registrar(result, inicio)
        return result

    def _registrar(self, result: LLMCallResult, inicio: float) -> None:
        """Evento de trazabilidad de una estimación completada.

        Con acierto de caché no se paga nada: `coste_usd` es 0 y lo que costó la
        respuesta original va a `coste_evitado_usd`, que es lo que la caché ahorró.
        `uso_respaldo` marca las respuestas del modelo de respaldo, que pueden
        costar bastante más que el primario (ver PLAN.md §7).
        """
        emitir(
            __name__,
            "estimacion_completada",
            modelo=result.model,
            proveedor=result.provider,
            uso_respaldo=result.model != self._primary_model,
            desde_cache=result.cached,
            tokens_prompt=result.prompt_tokens,
            tokens_completion=result.completion_tokens,
            coste_usd=0.0 if result.cached else result.cost_usd,
            coste_evitado_usd=result.cost_usd if result.cached else 0.0,
            latencia_ms=round((time.perf_counter() - inicio) * 1000, 1),
            prompt_version=result.prompt_version,
        )
