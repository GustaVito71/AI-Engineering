"""Router de los endpoints de estimación.

POST /api/v1/estimate: genera una estimación en texto libre a partir del
formulario tipado (versiones de prompt de texto: v1, v2).

POST /api/v1/estimate/structured: la misma entrada, con la estimación como JSON
validado contra `StructuredResult` (versiones de salida estructurada: v3).

Errores que pueden devolver, además del 422 de validación de Pydantic:

- 422: `?prompt_version=` pide una versión que no existe, o que es del otro
  tipo de salida. El mensaje lista las versiones disponibles del endpoint.
- 503: falta configuración local (una API key). Lo resuelve el handler de
  LLMConfigurationError en app/main.py, no este router.
- 504: el proveedor de LLM no respondió a tiempo (agotados reintentos y respaldo).
- 502: cualquier otro fallo al llamar al proveedor. En /estimate/structured,
  también cuando ningún intento del modelo cumplió el schema.

En 502 y 504 el cliente recibe un mensaje genérico en español. El detalle real
(tipo de excepción y mensaje del proveedor) va solo al log: puede incluir
identificadores de cuenta, nombres de modelo o fragmentos de la petición, y no
es algo que el cliente pueda resolver.
"""

from __future__ import annotations

import re

import litellm
from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.config import LLMConfigurationError, get_settings
from app.dependencies import get_llm_wrapper
from app.prompts.loader import (
    OUTPUT_STRUCTURED,
    OUTPUT_TEXT,
    available_versions,
    render_estimation_prompt,
)
from app.schemas.estimation import EstimationRequest, EstimationResponse
from app.schemas.structured_estimation import StructuredEstimationResponse
from app.services.llm_wrapper import StructuredOutputError
from app.tracing import emit

router = APIRouter(prefix="/api/v1", tags=["estimation"])

# Los ejemplos del prompt envuelven cada estimación en <estimation>…</estimation>,
# y el modelo a veces copia esas etiquetas en su respuesta. Son estructura del
# prompt, no contenido: se quitan antes de devolver el texto al cliente.
_PROMPT_TAGS = re.compile(r"</?estimation>")

TIMEOUT_MESSAGE = (
    "El proveedor de IA tardó demasiado en responder. Volvé a intentarlo en unos minutos."
)
PROVIDER_FAILURE_MESSAGE = (
    "No se pudo generar la estimación porque el proveedor de IA falló. "
    "Volvé a intentarlo en unos minutos."
)


@router.post(
    "/estimate",
    response_model=EstimationResponse,
    status_code=status.HTTP_200_OK,
    summary="Generar una estimación de proyecto",
    description=(
        "Genera una estimación de proyecto a partir de una descripción y tres "
        "campos tipados. Devuelve el texto de la estimación y la versión de "
        "prompt que la produjo, para poder trazarla. Con `?prompt_version=v1` "
        "se elige otra versión publicada del prompt; sin el parámetro se usa la "
        "configurada en el servicio (PROMPT_VERSION)."
    ),
)
async def estimate(
    request: EstimationRequest,
    prompt_version: str | None = Query(
        default=None,
        description=(
            "Versión del prompt (por ejemplo `v1` o `v2`). Sin este parámetro se "
            "usa la configurada en el servicio."
        ),
        examples=["v1", "v2"],
    ),
    llm_wrapper=Depends(get_llm_wrapper),
) -> EstimationResponse:
    """Genera una estimación a partir del formulario tipado.

    EstimationRequest valida la entrada (descripción + 3 enums), el prompt sale
    de la plantilla Jinja2 versionada y el wrapper llama al LLM con respaldo y
    caché.
    """
    version = _resolve_version(prompt_version)

    system_prompt, user_prompt = render_estimation_prompt(request, version)

    try:
        result = await llm_wrapper.estimate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            prompt_version=version,
        )
    except litellm.Timeout as exc:
        # Timeout es subclase de APIConnectionError: tiene que ir antes que el
        # caso general para que no lo capture el 502.
        _log_failure(exc, status.HTTP_504_GATEWAY_TIMEOUT)
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT, detail=TIMEOUT_MESSAGE
        ) from None
    except Exception as exc:  # noqa: BLE001 - cualquier otro fallo del proveedor es un 502
        _log_failure(exc, status.HTTP_502_BAD_GATEWAY)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY, detail=PROVIDER_FAILURE_MESSAGE
        ) from None

    return EstimationResponse(
        text=_strip_prompt_tags(result.content),
        prompt_version=result.prompt_version,
        warnings=list(llm_wrapper.warnings),
    )


def _resolve_version(requested: str | None, output: str = OUTPUT_TEXT) -> str:
    """La versión de prompt a usar: la pedida en la query, o la configurada.

    La pedida se compara contra las versiones publicadas de este tipo de salida
    antes de usarla: el valor termina en una ruta de archivo, así que nada que
    no sea una versión existente llega al loader. Una versión que no existe es
    un error del cliente (422), con las versiones válidas en el mensaje. Si
    existe pero es del otro tipo de salida, el mensaje dice en qué endpoint
    pedirla.
    """
    if requested is None:
        if output == OUTPUT_TEXT:
            return get_settings().prompt_version
        return _configured_structured_version()
    available = available_versions(output)
    if requested not in available:
        other = OUTPUT_STRUCTURED if output == OUTPUT_TEXT else OUTPUT_TEXT
        if requested in available_versions(other):
            detail = (
                f"La versión de prompt '{requested}' es de {_OUTPUT_DESCRIPTIONS[other]}: "
                f"pedila en POST {_OUTPUT_ENDPOINTS[other]}. "
                f"Versiones disponibles en este endpoint: {', '.join(available)}."
            )
        else:
            detail = (
                f"La versión de prompt '{requested}' no existe. "
                f"Versiones disponibles: {', '.join(available)}."
            )
        # 422 a secas: el nombre de la constante cambió entre versiones de Starlette.
        raise HTTPException(status_code=422, detail=detail)
    return requested


def _strip_prompt_tags(text: str) -> str:
    """Quita las etiquetas <estimation> que el modelo copia de los ejemplos del prompt."""
    return _PROMPT_TAGS.sub("", text).strip()


def _log_failure(exc: Exception, status_code: int, **extra: object) -> None:
    """Deja en el log el detalle que el cliente no ve.

    `from None` en el raise corta la cadena de excepciones de la respuesta; el
    detalle completo queda acá, con el tipo de error para poder filtrarlo.
    `extra` agrega campos propios de la salida estructurada (intentos y coste).
    """
    emit(
        __name__,
        "estimacion_fallida",
        level="error",
        codigo_http=status_code,
        tipo_error=type(exc).__name__,
        detalle=str(exc),
        **extra,
    )


# --- Structured response ----------------------------------------------------

INVALID_OUTPUT_MESSAGE = (
    "El proveedor de IA no devolvió una estimación con el formato esperado. "
    "Volvé a intentarlo en unos minutos."
)

_OUTPUT_DESCRIPTIONS = {
    OUTPUT_TEXT: "texto libre",
    OUTPUT_STRUCTURED: "salida estructurada",
}
_OUTPUT_ENDPOINTS = {
    OUTPUT_TEXT: "/api/v1/estimate",
    OUTPUT_STRUCTURED: "/api/v1/estimate/structured",
}


@router.post(
    "/estimate/structured",
    response_model=StructuredEstimationResponse,
    status_code=status.HTTP_200_OK,
    summary="Generar una estimación de proyecto estructurada",
    description=(
        "La misma entrada que `/estimate`, pero la estimación llega como JSON "
        "validado: fases, equipo y totales. Si la respuesta del modelo no cumple "
        "el schema, se le vuelve a preguntar con el error (STRUCTURED_MAX_RETRIES). "
        "Si los totales no coinciden con la suma de las fases, la estimación se "
        "devuelve igual, con un aviso en `warnings`. Si la descripción no alcanza para "
        "estimar, `summary` empieza con «Fuera de alcance:» y la estimación va en cero. "
        "`cached` indica si salió de la caché. Con `?prompt_version=v3` se "
        "elige la versión; sin el parámetro se usa STRUCTURED_PROMPT_VERSION."
    ),
)
async def estimate_structured(
    request: EstimationRequest,
    prompt_version: str | None = Query(
        default=None,
        description=(
            "Versión del prompt de salida estructurada (por ejemplo `v3`). Sin este "
            "parámetro se usa la configurada en el servicio."
        ),
        examples=["v3"],
    ),
    llm_wrapper=Depends(get_llm_wrapper),
) -> StructuredEstimationResponse:
    """Genera una estimación estructurada a partir del formulario tipado.

    `output_format` no cambia la respuesta: el modelo siempre devuelve la misma
    estructura y el cliente decide cómo mostrarla.
    """
    version = _resolve_version(prompt_version, OUTPUT_STRUCTURED)

    system_prompt, user_prompt = render_estimation_prompt(request, version)

    try:
        result = await llm_wrapper.estimate_structured(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            prompt_version=version,
        )
    except StructuredOutputError as exc:
        # La respuesta llegó pero no sirve: es un fallo del proveedor visto
        # desde el cliente (502), con su propio mensaje. Los intentos ya se
        # pagaron, así que el coste va al log.
        _log_failure(
            exc,
            status.HTTP_502_BAD_GATEWAY,
            salida="estructurada",
            intentos=exc.attempts,
            coste_usd=exc.cost_usd,
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY, detail=INVALID_OUTPUT_MESSAGE
        ) from None
    except litellm.Timeout as exc:
        # Igual que en /estimate: Timeout antes que el caso general.
        _log_failure(exc, status.HTTP_504_GATEWAY_TIMEOUT, salida="estructurada")
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT, detail=TIMEOUT_MESSAGE
        ) from None
    except Exception as exc:  # noqa: BLE001 - cualquier otro fallo del proveedor es un 502
        _log_failure(exc, status.HTTP_502_BAD_GATEWAY, salida="estructurada")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY, detail=PROVIDER_FAILURE_MESSAGE
        ) from None

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

    return StructuredEstimationResponse(
        estimation=result.estimation,
        prompt_version=result.prompt_version,
        cached=result.cached,
        warnings=[*llm_wrapper.warnings, *discrepancies],
    )


def _configured_structured_version() -> str:
    """STRUCTURED_PROMPT_VERSION, comprobada contra las versiones publicadas.

    A diferencia de la versión pedida por el cliente, un valor inválido acá es
    un error de configuración del servicio: 503 con el nombre de la variable
    (handler de LLMConfigurationError), no un 422 que el cliente no puede
    resolver.
    """
    configured = get_settings().structured_prompt_version
    available = available_versions(OUTPUT_STRUCTURED)
    if configured not in available:
        raise LLMConfigurationError(
            f"STRUCTURED_PROMPT_VERSION='{configured}' no es una versión de salida "
            f"estructurada publicada. Versiones disponibles: {', '.join(available) or 'ninguna'}."
        )
    return configured
