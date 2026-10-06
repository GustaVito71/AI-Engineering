"""Wrapper del LiteLLM Router con fallback automático y tracking de coste.

Diseñado para ser instanciado una sola vez (singleton vía get_llm_wrapper)
y reutilizado en todas las requests. El Router maneja el failover entre
el deployment primario y el fallback.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import instructor
from instructor.core import InstructorRetryException
from litellm import Router, completion_cost

from app.config import API_KEY_VARIABLES, LLMConfigurationError, Settings, provider_from_model
from app.schemas.structured_estimation import StructuredResult
from app.tracing import emit


def _key_variable(provider: str) -> str:
    """'openai' -> 'OPENAI_API_KEY'. Un provider sin variable conocida se nombra tal cual."""
    return API_KEY_VARIABLES.get(provider, f"la API key del proveedor '{provider}'")


def fallback_warning(settings: Settings) -> str | None:
    """Aviso para el usuario si el modelo de respaldo no tiene API key, o None.

    Sin la clave del respaldo el servicio funciona igual con el primario, pero
    pierde la protección ante una caída del proveedor. Lo usan el wrapper (que
    lo devuelve en cada estimación) y /health, para que los dos digan lo mismo.
    """
    # Settings garantiza que FALLBACK_MODEL nunca está vacío (apply_defaults).
    fallback_model = settings.fallback_model
    provider = settings.fallback_provider
    if settings.active_api_key(provider):
        return None
    return (
        f"El modelo de respaldo {fallback_model} no está disponible porque falta "
        f"{_key_variable(provider)}. Las estimaciones usan solo el modelo "
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
                f"Falta {_key_variable(primary_provider)} para el modelo primario {primary_model}."
            )

        # Sin la clave del respaldo el servicio sigue con el primario solo,
        # igual que /health (Settings.is_configured mira solo el primario). El
        # operador se entera por el log; el usuario, por `warnings` en cada
        # respuesta.
        warning = fallback_warning(settings)
        self.warnings: tuple[str, ...] = (warning,) if warning else ()
        if warning:
            emit(
                __name__,
                "respaldo_no_disponible",
                level="warning",
                modelo_respaldo=fallback_model,
                detalle=warning,
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
        self._model_by_id = {
            d["model_info"]["id"]: d["litellm_params"]["model"] for d in self._router.model_list
        }
        self._max_tokens = settings.llm_max_tokens

        # --- Structured response ----------------------------------------------------
        self._structured_max_retries = settings.structured_max_retries

    async def estimate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        prompt_version: str,
        max_tokens: int | None = None,
    ) -> LLMCallResult:
        """Ejecuta la estimación con fallback automático y cache."""
        start = time.perf_counter()

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
            self._log_completion(result, start)
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
        model = self._model_by_id.get(deployment_id, response.model)
        provider = provider_from_model(model)

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
        self._log_completion(result, start)
        return result

    def _log_completion(self, result: LLMCallResult, start: float, **extra: object) -> None:
        """Evento de trazabilidad de una estimación completada.

        Con acierto de caché no se paga nada: `coste_usd` es 0 y lo que costó la
        respuesta original va a `coste_evitado_usd`, que es lo que la caché ahorró.
        `uso_respaldo` marca las respuestas del modelo de respaldo, que pueden
        costar bastante más que el primario (ver PLAN.md §7). `extra` agrega
        campos propios de la salida estructurada (tipo de salida e intentos).
        """
        emit(
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
            latencia_ms=round((time.perf_counter() - start) * 1000, 1),
            prompt_version=result.prompt_version,
            **extra,
        )

    # --- Structured response ----------------------------------------------------
    async def estimate_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        prompt_version: str,
        max_tokens: int | None = None,
    ) -> StructuredCallResult:
        """Estimación validada contra `StructuredResult`, con reintentos y caché.

        Instructor va sobre `self._router.acompletion`, así que el respaldo, los
        reintentos ante fallos del proveedor y el `model_id` del deployment siguen
        siendo los del Router. Lo que agrega Instructor es el schema como
        `response_format` y, si la respuesta no lo cumple, una nueva pregunta con
        el error de validación, hasta `STRUCTURED_MAX_RETRIES` veces.

        Cada intento se paga: el coste y los tokens suman todas las respuestas,
        no solo la que se aceptó. En la caché se guarda el JSON validado como
        `content`, con la misma forma que las estimaciones de texto.

        Errores:
        - `StructuredOutputError`: ningún intento cumplió el schema.
        - Las excepciones de LiteLLM (Timeout, APIConnectionError...) llegan
          tal cual: Instructor las envuelve y acá se desenvuelven, para que el
          endpoint siga respondiendo 504 a un timeout y 502 al resto.
        """
        start = time.perf_counter()

        cache_key = self._cache.make_key(system_prompt, user_prompt)
        cached = await self._cache.get(cache_key)
        estimation = _cached_estimation(cached)
        if estimation is not None:
            result = StructuredCallResult(
                content=cached["content"],
                model=cached["model"],
                provider=cached["provider"],
                prompt_tokens=cached["prompt_tokens"],
                completion_tokens=cached["completion_tokens"],
                cost_usd=cached["cost_usd"],
                prompt_version=prompt_version,
                cached=True,
                estimation=estimation,
                attempts=0,
            )
            self._log_completion(result, start, salida="estructurada", intentos=result.attempts)
            return result

        # Un cliente por llamada: el hook junta las respuestas de ESTA estimación,
        # y con un cliente compartido se mezclarían las de requests concurrentes.
        responses: list[Any] = []
        client = instructor.from_litellm(self._router.acompletion, mode=instructor.Mode.JSON_SCHEMA)
        client.on("completion:response", responses.append)
        try:
            estimation, response = await client.chat.completions.create_with_completion(
                model=self._primary_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_model=StructuredResult,
                max_retries=self._structured_max_retries,
                max_tokens=max_tokens or self._max_tokens,
                temperature=0.1,
            )
        except InstructorRetryException as exc:
            cause = exc.__cause__
            if cause is not None and not isinstance(cause, ValueError):
                # Fallo del proveedor, no de la respuesta: lo decide el Router.
                raise cause from None
            raise StructuredOutputError(
                attempts=exc.n_attempts,
                cost_usd=_total_cost(responses),
                last_error=str(cause or exc),
            ) from None

        # Instructor devuelve una subclase propia de StructuredResult (le agrega
        # sus métodos). Se reconstruye la clase exacta para que una estimación
        # nueva y una de la caché sean el mismo tipo y se comparen iguales.
        estimation = StructuredResult.model_validate(estimation.model_dump())

        deployment_id = (getattr(response, "_hidden_params", None) or {}).get("model_id")
        model = self._model_by_id.get(deployment_id, response.model)
        result = StructuredCallResult(
            content=estimation.model_dump_json(),
            model=model,
            provider=provider_from_model(model),
            prompt_tokens=sum(_usage(r, "prompt_tokens") for r in responses),
            completion_tokens=sum(_usage(r, "completion_tokens") for r in responses),
            cost_usd=_total_cost(responses),
            prompt_version=prompt_version,
            estimation=estimation,
            attempts=len(responses),
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
        self._log_completion(result, start, salida="estructurada", intentos=result.attempts)
        return result


@dataclass
class StructuredCallResult(LLMCallResult):
    """Resultado de `estimate_structured`: el de siempre más la estimación validada."""

    estimation: StructuredResult | None = None
    # Respuestas pedidas al modelo: 1 si la primera cumplió el schema, 0 si salió
    # de la caché.
    attempts: int = 1


class StructuredOutputError(Exception):
    """El modelo no devolvió una estimación válida en ninguno de los intentos."""

    def __init__(self, *, attempts: int, cost_usd: float, last_error: str) -> None:
        super().__init__(
            f"Sin estimación válida tras {attempts} intentos. Último error: {last_error}"
        )
        self.attempts = attempts
        self.cost_usd = cost_usd
        self.last_error = last_error


def _cached_estimation(cached: dict | None) -> StructuredResult | None:
    """La estimación guardada en la caché, o None si no hay o ya no cumple el schema.

    Una entrada que no valida (por ejemplo, porque el schema cambió) se trata
    como un fallo de caché: se vuelve a pedir y se sobrescribe.
    """
    if not cached:
        return None
    try:
        return StructuredResult.model_validate_json(cached["content"])
    except (KeyError, ValueError):
        return None


def _usage(response: Any, field: str) -> int:
    usage = getattr(response, "usage", None)
    return getattr(usage, field, 0) or 0


def _total_cost(responses: list[Any]) -> float:
    """Coste de todas las respuestas."""
    return sum(_cost(r) for r in responses)


def _cost(response: Any) -> float:
    """Coste de una respuesta. Un modelo sin precio cuenta 0, como en `estimate`."""
    try:
        return completion_cost(response)
    except Exception:  # noqa: BLE001 - el coste es observabilidad, no negocio
        return 0.0
