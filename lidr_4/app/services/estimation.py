"""Orquestador de la estimación: lo que pasa entre el router y el LLMWrapper.

Tres capas, cada una con una sola responsabilidad:

- el router traduce HTTP: la versión pedida en la query y los errores que se
  convierten en 4xx/5xx (app/routers/estimations.py);
- el wrapper habla con el proveedor: Router de LiteLLM, Instructor, reintentos,
  respaldo y caché exacta (app/services/llm_wrapper.py);
- este servicio es el pipeline de la estimación: render del prompt, la llamada
  y lo que se hace con el resultado (eventos de negocio y avisos al usuario).

Pipeline de hoy:

1. render del prompt (plantilla versionada);
2. llamada al wrapper (caché exacta, LLM, validación contra el schema);
3. eventos y avisos de la estimación (fuera de alcance, totales que no cuadran).

Dónde entran las etapas que faltan, sin tocar el router ni el wrapper:

- guardrails de entrada (WU9): antes del paso 1, y por lo tanto antes de la caché
  del wrapper, así una descripción rechazada nunca se sirve desde la caché;
- caché semántica (WU10): después de los guardrails de entrada y antes del paso 2;
- guardrail de salida (WU9): después del paso 2.

Los errores no se traducen acá. Los del proveedor (`litellm.Timeout`, etc.) y
`StructuredOutputError` llegan al router tal cual, y él decide el código HTTP y
qué se deja en el log. Un guardrail que rechace una entrada tendrá su propia
excepción, y el router la traducirá igual. Un fallo al armar el prompt es un
error nuestro, no del proveedor: sale como `PromptRenderError`, que el router
no convierte en 502.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.prompts.loader import render_estimation_prompt
from app.schemas.estimation import EstimationRequest
from app.schemas.structured_estimation import StructuredResult
from app.services.llm_wrapper import LLMWrapper, StructuredCallResult
from app.tracing import emit

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

    `call` es el resultado del wrapper (la estimación validada, el coste, los
    intentos); `warnings` son los avisos para el usuario.
    """

    call: StructuredCallResult
    warnings: list[str]


class EstimationService:
    """El pipeline de la estimación. No guarda estado entre llamadas."""

    def __init__(self, llm_wrapper: LLMWrapper) -> None:
        self._llm_wrapper = llm_wrapper

    async def estimate_text(self, request: EstimationRequest, version: str) -> TextEstimation:
        """Estimación en texto libre (prompts de texto: v1, v2)."""
        system_prompt, user_prompt = _render(request, version)
        result = await self._llm_wrapper.estimate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            prompt_version=version,
        )
        return TextEstimation(
            text=_strip_prompt_tags(result.content),
            prompt_version=result.prompt_version,
            warnings=list(self._llm_wrapper.warnings),
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

        Los avisos son los del wrapper (respaldo no disponible) más los totales
        que no cuadran con la suma de las fases, hasta la tolerancia del schema.
        """
        system_prompt, user_prompt = _render(request, version)
        result = await self._llm_wrapper.estimate_structured(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            prompt_version=version,
            response_model=response_model,
            context=context,
        )

        if result.estimation.out_of_scope:
            # No es un fallo: el modelo dijo que la descripción no alcanza. Queda en el
            # log para medir cuántas descripciones llegan sin información suficiente.
            emit(
                __name__,
                "estimacion_fuera_de_alcance",
                prompt_version=version,
                modelo=result.model,
                confianza_pct=result.estimation.confidence_pct,
            )

        discrepancies = result.estimation.total_discrepancies()
        if discrepancies:
            emit(
                __name__,
                "totales_no_cuadran",
                level="warning",
                prompt_version=version,
                modelo=result.model,
                detalle=discrepancies,
            )

        return StructuredEstimation(
            call=result, warnings=[*self._llm_wrapper.warnings, *discrepancies]
        )


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
