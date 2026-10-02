"""Router for the estimation endpoint.

POST /api/v1/estimate - Generate an estimation from a structured request.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from app.config import get_settings
from app.dependencies import get_llm_wrapper
from app.prompts.loader import render_estimation_prompt
from app.schemas.estimation import EstimationRequest, EstimationResponse

router = APIRouter(prefix="/api/v1", tags=["estimation"])


@router.post(
    "/estimate",
    response_model=EstimationResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate project estimation",
    description=(
        "Generates a project estimation based on a structured description. "
        "Returns free-text estimation with prompt version for traceability."
    ),
)
async def estimate(
    request: EstimationRequest,
    llm_wrapper=Depends(get_llm_wrapper),
) -> EstimationResponse:
    """Generate an estimation from a typed form request.

    The request is validated by EstimationRequest (description + 3 enums).
    The prompt is rendered using the versioned Jinja2 template.
    The LLM is called with fallback and caching.
    """
    settings = get_settings()

    # Render the prompt using the versioned template
    system_prompt, user_prompt = render_estimation_prompt(request, settings.prompt_version)

    # Call LLM with fallback and caching
    try:
        result = await llm_wrapper.estimate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            prompt_version=settings.prompt_version,
        )
    except Exception as e:  # noqa: BLE001 - convert any LLM error to 502
        # TODO: better error handling with specific error types
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"LLM call failed: {e!s}",
        )

    return EstimationResponse(
        text=result.content,
        prompt_version=result.prompt_version,
    )
