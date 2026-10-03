"""Router del endpoint de estimación.

POST /api/v1/estimate: genera una estimación a partir del formulario tipado.

Errores que puede devolver, además del 422 de validación de Pydantic:

- 503: falta configuración local (una API key). Lo resuelve el handler de
  LLMConfigurationError en app/main.py, no este router.
- 504: el proveedor de LLM no respondió a tiempo (agotados reintentos y respaldo).
- 502: cualquier otro fallo al llamar al proveedor.

En 502 y 504 el cliente recibe un mensaje genérico en español. El detalle real
(tipo de excepción y mensaje del proveedor) va solo al log: puede incluir
identificadores de cuenta, nombres de modelo o fragmentos de la petición, y no
es algo que el cliente pueda resolver.
"""

from __future__ import annotations

import litellm
from fastapi import APIRouter, Depends, HTTPException, status

from app.config import get_settings
from app.dependencies import get_llm_wrapper
from app.prompts.loader import render_estimation_prompt
from app.schemas.estimation import EstimationRequest, EstimationResponse
from app.tracing import emitir

router = APIRouter(prefix="/api/v1", tags=["estimation"])

MENSAJE_TIMEOUT = (
    "El proveedor de IA tardó demasiado en responder. Volvé a intentarlo en unos minutos."
)
MENSAJE_FALLO_PROVEEDOR = (
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
        "prompt que la produjo, para poder trazarla."
    ),
)
async def estimate(
    request: EstimationRequest,
    llm_wrapper=Depends(get_llm_wrapper),
) -> EstimationResponse:
    """Genera una estimación a partir del formulario tipado.

    EstimationRequest valida la entrada (descripción + 3 enums), el prompt sale
    de la plantilla Jinja2 versionada y el wrapper llama al LLM con respaldo y
    caché.
    """
    settings = get_settings()

    system_prompt, user_prompt = render_estimation_prompt(request, settings.prompt_version)

    try:
        result = await llm_wrapper.estimate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            prompt_version=settings.prompt_version,
        )
    except litellm.Timeout as exc:
        # Timeout es subclase de APIConnectionError: tiene que ir antes que el
        # caso general para que no lo capture el 502.
        _registrar_fallo(exc, status.HTTP_504_GATEWAY_TIMEOUT)
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT, detail=MENSAJE_TIMEOUT
        ) from None
    except Exception as exc:  # noqa: BLE001 - cualquier otro fallo del proveedor es un 502
        _registrar_fallo(exc, status.HTTP_502_BAD_GATEWAY)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY, detail=MENSAJE_FALLO_PROVEEDOR
        ) from None

    return EstimationResponse(
        text=result.content,
        prompt_version=result.prompt_version,
    )


def _registrar_fallo(exc: Exception, codigo: int) -> None:
    """Deja en el log el detalle que el cliente no ve.

    `from None` en el raise corta la cadena de excepciones de la respuesta; el
    detalle completo queda acá, con el tipo de error para poder filtrarlo.
    """
    emitir(
        __name__,
        "estimacion_fallida",
        nivel="error",
        codigo_http=codigo,
        tipo_error=type(exc).__name__,
        detalle=str(exc),
    )
