"""Orquestador de la estimación: lo que pasa entre el router y el LLMWrapper.

Tres capas, cada una con una sola responsabilidad:

- el router traduce HTTP: la versión pedida en la query y los errores que se
  convierten en 4xx/5xx (app/routers/estimations.py);
- el wrapper habla con el proveedor: Router de LiteLLM, Instructor, reintentos y
  respaldo (app/services/llm_wrapper.py). No guarda caché: cada llamada va al
  proveedor;
- este servicio es el pipeline de la estimación: guardrails de entrada, cachés,
  render del prompt, la llamada y lo que se hace con el resultado (guardrail de
  salida, eventos de negocio y avisos al usuario).

Pipeline, en este orden:

1. guardrails de entrada (WU9 A, app/guardrails/input.py): rechazan la
   inyección de prompt y el contenido que marca la moderación, y ocultan los
   datos personales. Corren antes de cualquier caché: una descripción rechazada
   no llega al proveedor ni se sirve desde la caché, y las cachés y el prompt
   solo ven el texto ya oculto;
2. caché exacta (app/services/cache.py): la clave es un hash del modelo, la
   versión del prompt, los parámetros de la request y la descripción filtrada.
   No incluye el prompt renderizado, así que se consulta antes de armarlo;
3. caché semántica (WU10, todavía no existe): solo si la exacta no acertó, porque
   necesita un embedding. Se busca por la descripción;
4. render del prompt (plantilla versionada);
5. llamada al wrapper (LLM, validación contra el schema);
6. guardrail de salida (WU9 B): lo que rechaza vive en el schema y lo valida el
   wrapper con re-pregunta, así que acá solo llega una estimación válida;
7. escritura en las cachés. Va después del guardrail de salida: no se guarda una
   estimación que se rechazó ("cachear después de validar", PLAN.md §8);
8. avisos y eventos de la estimación (`OutputGuardrails`, app/guardrails/output.py):
   fuera de alcance, totales que no cuadran, confianza baja.

Un acierto de caché salta de 3 a 8: no hay render, llamada ni escritura, y el
servicio deja el evento `estimacion_completada` con `desde_cache=True`. Las
estimaciones que vuelven de la caché ya pasaron el guardrail de salida cuando se
guardaron.

Los errores no se traducen acá. Los del proveedor (`litellm.Timeout`, etc.) y
`StructuredOutputError` llegan al router tal cual, y él decide el código HTTP y
qué se deja en el log. Un guardrail de entrada que rechaza lanza
`InputGuardrailViolation`, y el router la traduce a 422. Un fallo al armar el
prompt es un error nuestro, no del proveedor: sale como `PromptRenderError`, que
el router no convierte en 502. La caché nunca hace fallar una estimación: si
Redis falla, la consulta es un fallo de caché y la escritura se omite.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass

from app.guardrails import InputGuardrails, OutputGuardrails
from app.prompts.loader import render_estimation_prompt
from app.schemas.estimation import EstimationRequest
from app.schemas.structured_estimation import RenderedResult, StructuredResult
from app.services.cache import ExactCache, create_estimation_cache
from app.services.llm_wrapper import (
    LLMCallResult,
    LLMWrapper,
    StructuredCallResult,
    emit_completion,
    output_label,
)

# Los ejemplos del prompt envuelven cada estimación en <estimation>…</estimation>,
# y el modelo a veces copia esas etiquetas en su respuesta. Son estructura del
# prompt, no contenido: se quitan antes de devolver el texto al cliente.
_PROMPT_TAGS = re.compile(r"</?estimation>")


class PromptRenderError(Exception):
    """No se pudo armar el prompt (plantilla o datos de la versión rotos).

    Es un fallo del servicio, no del proveedor: el router lo deja pasar como
    error interno en vez de atribuirlo al proveedor con un 502.
    """


@dataclass(frozen=True)
class TextEstimation:
    """Resultado de una estimación en texto libre."""

    text: str
    prompt_version: str
    warnings: list[str]


@dataclass(frozen=True)
class StructuredEstimation:
    """Resultado de una estimación estructurada o renderizada.

    `call` es el resultado de la llamada o de la caché (la estimación validada, el
    coste, los intentos); `warnings` son los avisos para el usuario.
    """

    call: StructuredCallResult
    warnings: list[str]


class EstimationService:
    """El pipeline de la estimación. No guarda estado entre llamadas."""

    def __init__(
        self,
        llm_wrapper: LLMWrapper,
        input_guardrails: InputGuardrails | None = None,
        cache: ExactCache | None = None,
        output_guardrails: OutputGuardrails | None = None,
    ) -> None:
        """Sin `input_guardrails` corre sin reglas de entrada; sin `cache`, sin caché; sin
        `output_guardrails`, sin avisos ni eventos sobre la estimación (las reglas del
        schema que rechazan corren igual: viven en el wrapper)."""
        self._llm_wrapper = llm_wrapper
        self._input_guardrails = input_guardrails
        self._output_guardrails = output_guardrails
        self._cache = cache if cache is not None else create_estimation_cache(None, 0)

    async def estimate_text(self, request: EstimationRequest, version: str) -> TextEstimation:
        """Estimación en texto libre (prompts de texto: v1, v2)."""
        # --- Input guardrails -------------------------------------------------------
        # WU9 A: antes de las cachés, del render y de la llamada. Un rechazo sale como
        # InputGuardrailViolation, sin pagar nada.
        request, guardrail_warnings = await self._check_input(request)
        # ----------------------------------------------------------------------------
        start = time.perf_counter()
        cache_key = self._exact_key(request, version)
        result = await self._cached_text(cache_key, version, start)
        if result is None:
            system_prompt, user_prompt = _render(request, version)
            result = await self._llm_wrapper.estimate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                prompt_version=version,
            )
            self._inspect_text(result, version)
            await self._cache.set(cache_key, _cache_entry(result))
        return TextEstimation(
            text=_strip_prompt_tags(result.content),
            prompt_version=result.prompt_version,
            warnings=[*guardrail_warnings, *self._llm_wrapper.warnings],
        )

    async def estimate_structured(
        self,
        request: EstimationRequest,
        version: str,
        response_model: type[StructuredResult],
        *,
        context: dict | None = None,
    ) -> StructuredEstimation:
        """Estimación validada contra `response_model` (v3: structured, v4: rendered).

        `context` llega a los validadores del schema: en v4, el `output_format`
        de la request.

        Los avisos son los de los guardrails de entrada (datos personales ocultados),
        los del wrapper (respaldo no disponible) y los de salida (`OutputGuardrails`):
        totales que no cuadran con la suma de las fases, hasta la tolerancia del
        schema, y confianza baja.
        """
        # --- Input guardrails -------------------------------------------------------
        # WU9 A: antes de las cachés (exacta y semántica), del render y de la llamada. Un
        # rechazo sale como InputGuardrailViolation, sin pagar nada. Las cachés y el
        # prompt ven solo la descripción ya oculta.
        request, guardrail_warnings = await self._check_input(request)
        # ----------------------------------------------------------------------------

        start = time.perf_counter()
        cache_key = self._exact_key(request, version, response_model=response_model)
        result = await self._cached_structured(cache_key, version, response_model, context, start)
        if result is None:
            # --- Semantic cache -----------------------------------------------------
            # WU10: la búsqueda va acá, solo si la exacta no acertó, antes del render y
            # de la llamada. Un acierto devuelve la estimación guardada.

            # ------------------------------------------------------------------------
            system_prompt, user_prompt = _render(request, version)
            result = await self._llm_wrapper.estimate_structured(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                prompt_version=version,
                response_model=response_model,
                context=context,
            )

            # --- Output guardrails --------------------------------------------------
            # WU9 B: lo que rechaza una estimación (horas contra el equipo, totales,
            # rechazo coherente) ya lo validó el schema dentro del wrapper, con
            # re-pregunta: acá solo llega una estimación válida. La escritura en las
            # cachés, exacta y semántica, va después y es la única: no se guarda nada
            # que el guardrail rechazó. Los avisos (`OutputGuardrails`) se calculan
            # después, para que valgan también cuando la estimación sale de la caché.

            # Datos personales y prompt volcado (`inspect_estimation`): se descartan
            # sin re-preguntar, antes de escribir en las cachés.
            self._inspect_estimation(result, version)
            # ------------------------------------------------------------------------
            await self._cache.set(cache_key, _cache_entry(result))

        output_warnings = (
            self._output_guardrails.check(
                result.estimation, prompt_version=version, model=result.model
            )
            if self._output_guardrails is not None
            else []
        )

        return StructuredEstimation(
            call=result,
            warnings=[
                *guardrail_warnings,
                *self._llm_wrapper.warnings,
                *output_warnings,
            ],
        )

    def _inspect_text(self, result: LLMCallResult, version: str) -> None:
        if self._output_guardrails is not None:
            self._output_guardrails.inspect_text(
                result.content, prompt_version=version, model=result.model
            )

    def _inspect_estimation(self, result: StructuredCallResult, version: str) -> None:
        if self._output_guardrails is not None:
            self._output_guardrails.inspect_estimation(
                result.estimation, prompt_version=version, model=result.model
            )

    async def _check_input(self, request: EstimationRequest) -> tuple[EstimationRequest, list[str]]:
        """La request con la descripción que pasó los guardrails, y sus avisos."""
        if self._input_guardrails is None:
            return request, []
        checked = await self._input_guardrails.check(request.description)
        if checked.description == request.description:
            return request, checked.warnings
        # model_copy no vuelve a correr los validadores: el texto ocultado puede ser más
        # corto que el mínimo de la descripción, y eso no es un error del cliente.
        return request.model_copy(update={"description": checked.description}), checked.warnings

    def _exact_key(
        self,
        request: EstimationRequest,
        version: str,
        *,
        response_model: type[StructuredResult] | None = None,
    ) -> str:
        """La clave de la caché exacta: lo que determina la respuesta, sin armar el prompt.

        Modelo, versión del prompt y los parámetros que entran al prompt. En la salida
        estructurada (`StructuredResult`, v3) `output_format` no entra al prompt ni
        cambia la respuesta (PLAN.md §3.3), así que no entra a la clave: pedir la misma
        estimación con otro formato es un acierto. En el texto y en la renderizada sí.
        """
        uses_format = response_model is None or issubclass(response_model, RenderedResult)
        return self._cache.make_key(
            self._llm_wrapper.primary_model,
            version,
            request.project_type.value,
            request.detail_level.value,
            request.output_format.value if uses_format else "",
            request.description,
        )

    async def _cached_text(
        self, cache_key: str, version: str, start: float
    ) -> LLMCallResult | None:
        """El resultado guardado en la caché exacta, o None si no hay (o no sirve)."""
        entry = await self._cache.get(cache_key)
        if not _is_valid_entry(entry):
            return None
        result = LLMCallResult(**_entry_fields(entry), prompt_version=version, cached=True)
        emit_completion(result, start, primary_model=self._llm_wrapper.primary_model)
        return result

    async def _cached_structured(
        self,
        cache_key: str,
        version: str,
        response_model: type[StructuredResult],
        context: dict | None,
        start: float,
    ) -> StructuredCallResult | None:
        """La estimación guardada en la caché exacta, o None si no hay o ya no cumple el schema.

        Una entrada que no valida (por ejemplo, porque el schema cambió) se trata
        como un fallo de caché: se vuelve a pedir y se sobrescribe.
        """
        entry = await self._cache.get(cache_key)
        if not _is_valid_entry(entry):
            return None
        try:
            estimation = response_model.model_validate_json(entry["content"], context=context)
        except ValueError:
            return None
        result = StructuredCallResult(
            **_entry_fields(entry),
            prompt_version=version,
            cached=True,
            estimation=estimation,
            attempts=0,
        )
        emit_completion(
            result,
            start,
            primary_model=self._llm_wrapper.primary_model,
            salida=output_label(response_model),
            intentos=0,
        )
        return result


# Lo que se guarda en la caché exacta de cada respuesta.
_ENTRY_KEYS = ("content", "model", "provider", "prompt_tokens", "completion_tokens", "cost_usd")


def _cache_entry(result: LLMCallResult) -> dict:
    return {key: getattr(result, key) for key in _ENTRY_KEYS}


def _is_valid_entry(entry: object) -> bool:
    """Una entrada con todos sus campos. Una incompleta cuenta como fallo de caché."""
    return isinstance(entry, dict) and all(key in entry for key in _ENTRY_KEYS)


def _entry_fields(entry: dict) -> dict:
    return {key: entry[key] for key in _ENTRY_KEYS}


def _render(request: EstimationRequest, version: str) -> tuple[str, str]:
    try:
        return render_estimation_prompt(request, version)
    except Exception as exc:
        raise PromptRenderError(
            f"No se pudo armar el prompt {version}: {type(exc).__name__}: {exc}"
        ) from exc


def _strip_prompt_tags(text: str) -> str:
    """Quita las etiquetas <estimation> que el modelo copia de los ejemplos del prompt."""
    return _PROMPT_TAGS.sub("", text).strip()
