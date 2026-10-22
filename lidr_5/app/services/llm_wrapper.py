"""Wrapper del LiteLLM Router con fallback automático y tracking de coste.

Diseñado para ser instanciado una sola vez (singleton vía get_llm_wrapper)
y reutilizado en todas las requests. El Router maneja el failover entre
el deployment primario y el fallback.

No guarda caché: cada llamada a `estimate` o `estimate_structured` va al
proveedor. La caché exacta y la semántica las consulta y las escribe
`EstimationService` (app/services/estimation.py), que decide si hace falta llamar.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import instructor
from instructor.core import IncompleteOutputException, InstructorRetryException
from litellm import Router, UnsupportedParamsError, completion_cost, get_optional_params
from pydantic import BaseModel

from app.config import EXTRACTOR_KEY_VARIABLES, LLMConfigurationError, Settings, provider_from_model
from app.observability import record_llm_call
from app.schemas.critic import CriticFeedback
from app.schemas.structured_estimation import RenderedResult, StructuredResult
from app.sessions.models import ExtractedMetadata
from app.tracing import emit


def _key_variable(provider: str) -> str:
    """'openai' -> 'OPENAI_API_KEY'. Un provider sin variable conocida se nombra tal cual."""
    return EXTRACTOR_KEY_VARIABLES.get(provider, f"la API key del proveedor '{provider}'")


# Temperatura de las estimaciones: baja, para que el mismo encargo dé resultados parecidos.
# (La temperatura es el número que controla cuánto varía la respuesta de un modelo entre
# una llamada y otra: bajo = parecidas, alto = más variadas.)
TEMPERATURE = 0.1


def supports_temperature(model: str) -> bool:
    """False si LiteLLM rechaza `temperature=TEMPERATURE` para ese modelo.

    En palabras simples: dice si se le puede pedir a ese modelo la temperatura baja.

    Algunos modelos solo admiten su temperatura por defecto (claude-haiku-5-5
    lanza `UnsupportedParamsError`). Se pregunta a LiteLLM y no a una lista de
    nombres, para que un modelo nuevo no obligue a tocar el código. Es una
    comprobación local, sin llamada al proveedor.
    """
    provider = provider_from_model(model)
    try:
        get_optional_params(
            model=model.split("/", 1)[-1], custom_llm_provider=provider, temperature=TEMPERATURE
        )
    except UnsupportedParamsError:
        return False
    return True


def instructor_mode(model: str) -> instructor.Mode:
    """Cómo le pide Instructor al modelo que responda con el formato del schema.

    En palabras simples: casi todos los proveedores aceptan que se les mande el schema
    como `response_format` (JSON_SCHEMA). DeepSeek lo rechaza ("This response_format type
    is unavailable now"); a ese se le pide JSON simple (`json_object`) y Instructor
    agrega el schema al mensaje de sistema. El modo se elige según el modelo primario;
    si el respaldo es de otro tipo, el modo del primario es el que se usa.
    """
    if provider_from_model(model) == "deepseek":
        return instructor.Mode.JSON
    return instructor.Mode.JSON_SCHEMA


def _deployment(model: str, api_key: str) -> dict[str, Any]:
    """Deployment del Router. La temperatura va en el deployment y no en la llamada:
    ante un failover cada modelo usa la suya, y el que no la admite no la recibe."""
    params: dict[str, Any] = {"model": model, "api_key": api_key}
    if supports_temperature(model):
        params["temperature"] = TEMPERATURE
    return {"model_name": model, "litellm_params": params}


def fallback_warning(settings: Settings) -> str | None:
    """Aviso para el usuario si el modelo de respaldo no tiene API key, o None.

    Sin la clave del respaldo el servicio funciona igual con el primario, pero
    pierde la protección ante una caída del proveedor. Lo usan el wrapper (que
    lo devuelve en cada estimación) y /health, para que los dos digan lo mismo.
    """
    # Settings garantiza que el modelo de respaldo nunca está vacío (apply_defaults).
    fallback_model = settings.fallback_model
    provider = settings.fallback_provider
    if settings.active_api_key(provider):
        return None
    return (
        f"El modelo de respaldo {fallback_model} no está disponible porque falta "
        f"{_key_variable(provider)}. Las estimaciones usan solo el modelo "
        "primario: si ese proveedor falla, la estimación no se podrá completar."
    )


def emit_completion(
    result: LLMCallResult,
    start: float,
    *,
    primary_model: str,
    event: str = "estimacion_completada",
    **extra: object,
) -> None:
    """Evento de trazabilidad de una estimación completada.

    Lo emiten el wrapper, cuando hubo llamada al proveedor, y el servicio, cuando
    la respuesta salió de la caché. Con acierto de caché no se paga nada: `coste_usd`
    es 0 y lo que costó la respuesta original va a `coste_evitado_usd`, que es lo
    que la caché ahorró. `uso_respaldo` marca las respuestas del modelo de respaldo,
    que pueden costar bastante más que el primario (ver PLAN.md §7). `extra` agrega
    campos propios de la salida estructurada (tipo de salida e intentos). `event` es el
    nombre del evento: el extractor de metadata usa uno propio para no mezclar su coste
    con el de las estimaciones.
    """
    # Suma la llamada a la hucha de coste de la petición (Paso 10). Un acierto de caché no
    # llamó al proveedor: no cuenta como llamada ni como gasto.
    if not result.cached:
        record_llm_call(
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            cost_usd=result.cost_usd,
        )
    emit(
        __name__,
        event,
        modelo=result.model,
        proveedor=result.provider,
        uso_respaldo=result.model != primary_model,
        desde_cache=result.cached,
        tokens_prompt=result.prompt_tokens,
        tokens_completion=result.completion_tokens,
        coste_usd=0.0 if result.cached else result.cost_usd,
        coste_evitado_usd=result.cost_usd if result.cached else 0.0,
        latencia_ms=round((time.perf_counter() - start) * 1000, 1),
        prompt_version=result.prompt_version,
        **extra,
    )


# Nombre del evento de trazabilidad según el tipo de salida: el extractor y el crítico usan
# uno propio para no mezclar su coste con el de las estimaciones.
_COMPLETION_EVENTS = {"metadata": "metadata_extraida", "critico": "critico_completado"}


def output_label(response_model: type[BaseModel]) -> str:
    """El valor del campo `salida` de los eventos: `renderizada`, `estructurada` o `metadata`.

    `metadata` es la llamada del extractor de `project_metadata` (Paso 4) y `critico` la
    del crítico del ACB (Paso 8): las dos reutilizan este wrapper con otro schema.
    """
    if issubclass(response_model, ExtractedMetadata):
        return "metadata"
    if issubclass(response_model, CriticFeedback):
        return "critico"
    return "renderizada" if issubclass(response_model, RenderedResult) else "estructurada"


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


@dataclass
class ToolCompletion:
    """Resultado de una ronda del bucle agéntico (`LLMWrapper.complete_with_tools`).

    `message` es el mensaje `assistant` tal como lo devolvió el proveedor (se reenvía en la
    ronda siguiente); `tool_calls` son las herramientas que pidió (vacío si respondió con
    texto) y `content` el texto, que puede venir vacío cuando solo pide herramientas.
    """

    message: Any
    tool_calls: list[Any]
    content: str
    model: str
    provider: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float


class LLMWrapper:
    """Wrapper alrededor del LiteLLM Router con fallback y tracking de coste."""

    def __init__(self, *, settings: Settings, use_fallback: bool = True) -> None:
        """Arma el Router con la configuración de `settings`.

        `use_fallback=False` arma el Router solo con el primario, sin respaldo y sin el
        aviso de respaldo: lo usa el crítico del ACB, que necesita saber con certeza
        qué proveedor respondió (un respaldo silencioso rompería su independencia).
        """
        primary_model = settings.primary_model
        fallback_model = settings.fallback_model
        self._primary_model = primary_model

        # Cada deployment recibe la clave de su propio proveedor.
        primary_provider = settings.primary_provider
        primary_key = settings.active_api_key(primary_provider)
        fallback_key = settings.active_api_key(settings.fallback_provider) if use_fallback else None

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
        warning = fallback_warning(settings) if use_fallback else None
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
        deployments = [_deployment(primary_model, primary_key)]
        if fallback_key:
            deployments.append(_deployment(fallback_model, fallback_key))

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

    @property
    def primary_model(self) -> str:
        """El modelo primario del Router: con él se arma la clave de caché y `uso_respaldo`."""
        return self._primary_model

    async def estimate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        prompt_version: str,
        max_tokens: int | None = None,
    ) -> LLMCallResult:
        """Ejecuta la estimación con fallback automático. Siempre llama al proveedor."""
        start = time.perf_counter()

        response = await self._router.acompletion(
            model=self._primary_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max_tokens or self._max_tokens,
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
        emit_completion(result, start, primary_model=self._primary_model)
        return result

    # --- Llamada con herramientas (bucle agéntico, Paso 10) ----------------------
    async def complete_with_tools(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        tool_choice: str = "auto",
        max_tokens: int | None = None,
    ) -> ToolCompletion:
        """Una ronda del bucle agéntico: el modelo responde con texto o pide herramientas.

        `messages` es la conversación completa de la ronda en formato OpenAI (incluye los
        mensajes `assistant` con `tool_calls` y los `tool` con sus resultados); `tools` son
        las definiciones de las herramientas disponibles. Va por el mismo Router que el resto
        de llamadas, así que el respaldo y los reintentos ante fallos del proveedor siguen
        siendo los del Router: si el primario cae en una ronda, el respaldo recibe el mismo
        historial de herramientas. `tool_choice="none"` obliga a responder sin herramientas
        (se usa para cerrar el bucle al agotar las rondas).

        Cada ronda se suma a la hucha de coste de la petición y emite `agente_llamada`.
        """
        start = time.perf_counter()
        response = await self._router.acompletion(
            model=self._primary_model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            max_tokens=max_tokens or self._max_tokens,
            # LiteLLM 1.106 intenta cargar su pasarela MCP cada vez que se envían `tools`, y
            # esa pasarela exige el paquete `mcp` (no es una dependencia nuestra): sin esta
            # marca la llamada falla con «No module named 'mcp'» y se queda reintentando.
            # Nuestras herramientas son funciones propias, no servidores MCP.
            # `test_llm_wrapper_tools.py` vigila que la marca siga funcionando al actualizar.
            _skip_mcp_handler=True,
        )
        message = response.choices[0].message
        deployment_id = (getattr(response, "_hidden_params", None) or {}).get("model_id")
        model = self._model_by_id.get(deployment_id, response.model)
        result = ToolCompletion(
            message=message,
            tool_calls=list(getattr(message, "tool_calls", None) or []),
            content=getattr(message, "content", None) or "",
            model=model,
            provider=provider_from_model(model),
            prompt_tokens=_usage(response, "prompt_tokens"),
            completion_tokens=_usage(response, "completion_tokens"),
            cost_usd=_cost(response),
        )
        record_llm_call(
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            cost_usd=result.cost_usd,
        )
        emit(
            __name__,
            "agente_llamada",
            modelo=result.model,
            proveedor=result.provider,
            uso_respaldo=result.model != self._primary_model,
            tokens_prompt=result.prompt_tokens,
            tokens_completion=result.completion_tokens,
            coste_usd=result.cost_usd,
            herramientas_pedidas=[call.function.name for call in result.tool_calls],
            latencia_ms=round((time.perf_counter() - start) * 1000, 1),
        )
        return result

    # --- Structured response ----------------------------------------------------
    async def estimate_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        prompt_version: str,
        max_tokens: int | None = None,
        response_model: type[BaseModel] = StructuredResult,
        context: dict[str, Any] | None = None,
        history: list[dict[str, str]] | None = None,
    ) -> StructuredCallResult:
        """Estimación validada contra `StructuredResult`, con reintentos.

        `history` son los mensajes previos de la conversación (pares usuario +
        asistente, ya recortados por la ventana); van entre el system prompt y el
        mensaje actual.

        Instructor va sobre `self._router.acompletion`, así que el respaldo, los
        reintentos ante fallos del proveedor y el `model_id` del deployment siguen
        siendo los del Router. Lo que agrega Instructor es el schema como
        `response_format` y, si la respuesta no lo cumple, una nueva pregunta con
        el error de validación, hasta `STRUCTURED_MAX_RETRIES` veces.

        Cada intento se paga: el coste y los tokens suman todas las respuestas,
        no solo la que se aceptó. `content` es el JSON validado, con la misma forma
        que las estimaciones de texto.

        `response_model` es el schema que se le exige al modelo: `StructuredResult`
        (v3) o `RenderedResult` (v4). `context` llega a los validadores (en v4, el
        `output_format` de la request).

        Política de reintento (cada caso tiene un dueño y un tope):
        - La respuesta no cumple el schema (incluidas las reglas del guardrail de
          salida, como las horas contra el equipo): se vuelve a pedir con el error,
          hasta `STRUCTURED_MAX_RETRIES` veces. Agotadas, `StructuredOutputError`.
        - La respuesta se cortó por `max_tokens`: NO se reintenta. Con el mismo
          presupuesto el modelo se corta en el mismo punto, y cada intento se
          paga. Sale `IncompleteJSONError` de inmediato, con lo gastado.
        - El proveedor falla (timeout, conexión, 5xx): lo reintenta el Router y,
          si no alcanza, responde el modelo de respaldo (`LLM_MAX_RETRIES`).
        - Nada de esto se escribe en la caché: lo hace el servicio, solo con una
          respuesta válida.

        Errores:
        - `StructuredOutputError`: ningún intento cumplió el schema.
        - `IncompleteJSONError`: la respuesta llegó cortada por `max_tokens`.
        - Las excepciones de LiteLLM (Timeout, APIConnectionError...) llegan
          tal cual: Instructor las envuelve y acá se desenvuelven, para que el
          endpoint siga respondiendo 504 a un timeout y 502 al resto.
        """
        start = time.perf_counter()

        output_kind = output_label(response_model)

        # Un cliente por llamada: el hook junta las respuestas de ESTA estimación,
        # y con un cliente compartido se mezclarían las de requests concurrentes.
        responses: list[Any] = []
        client = instructor.from_litellm(
            self._router.acompletion, mode=instructor_mode(self._primary_model)
        )
        client.on("completion:response", responses.append)
        try:
            estimation, response = await client.chat.completions.create_with_completion(
                model=self._primary_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    # Turnos anteriores de la conversación (la ventana deslizante), si hay.
                    *(history or []),
                    {"role": "user", "content": user_prompt},
                ],
                response_model=response_model,
                context=context,
                max_retries=self._structured_max_retries,
                max_tokens=max_tokens or self._max_tokens,
            )
        except IncompleteOutputException as exc:
            # Instructor no re-pregunta una respuesta cortada; sin esto saldría como
            # un fallo genérico del proveedor y sin el coste que ya se gastó.
            completions = responses or [exc.last_completion]
            raise IncompleteJSONError(
                attempts=len(completions),
                cost_usd=_total_cost(completions),
                max_tokens=max_tokens or self._max_tokens,
            ) from None
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

        # Instructor devuelve una subclase propia del schema (le agrega sus
        # métodos). Se reconstruye la clase exacta para que una estimación nueva
        # y una de la caché sean el mismo tipo y se comparen iguales.
        estimation = response_model.model_validate(estimation.model_dump(), context=context)

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
        emit_completion(
            result,
            start,
            primary_model=self._primary_model,
            event=_COMPLETION_EVENTS.get(output_kind, "estimacion_completada"),
            salida=output_kind,
            intentos=result.attempts,
        )
        return result


@dataclass
class StructuredCallResult(LLMCallResult):
    """Resultado de `estimate_structured`: el de siempre más la estimación validada."""

    estimation: Any = None
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


class IncompleteJSONError(Exception):
    """La respuesta del modelo se cortó por `max_tokens` antes de cerrar el JSON.

    No es un fallo de validación: re-preguntar con el mismo presupuesto cortaría
    la respuesta en el mismo punto. La arregla quien configura `LLM_MAX_TOKENS`
    (o un prompt que pida menos), por eso es un error propio y no un 502 genérico.
    """

    def __init__(self, *, attempts: int, cost_usd: float, max_tokens: int) -> None:
        super().__init__(
            f"Respuesta cortada por max_tokens={max_tokens} tras {attempts} "
            "intento(s); el JSON quedó incompleto."
        )
        self.attempts = attempts
        self.cost_usd = cost_usd
        self.max_tokens = max_tokens


def _usage(response: Any, field: str) -> int:
    """Lee de la respuesta del proveedor un dato de consumo (por ejemplo, tokens de
    entrada). Devuelve 0 si no viene.
    """
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
