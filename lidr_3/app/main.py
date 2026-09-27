"""Punto de entrada de la aplicación FastAPI.

/health NO depende de la configuración: el servicio arranca aunque falte la
API key y lo dice en la respuesta (`llm_configured: false`). Un health check
tiene que sobrevivir a la avería que está diagnosticando; si exige la key al
arrancar, muere antes que el problema y el orquestador ve un crashloop sin
forma de distinguir "falta un secreto" de "el código está roto".

No hay middleware CORS a propósito: no existe frontend que llame a esta API
desde un navegador, y CORS es una protección del navegador. Se agrega solo
cuando haya uno (con allow_origins explícito desde Settings)."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import structlog
from fastapi import Depends, FastAPI

from .config import Settings, get_settings
from .routers.estimations import router as estimations_router


def configure_logging(level: str = "INFO") -> None:
    """structlog como API de logging, stdlib como backend.

    El render (formato) queda centralizado acá. El nivel viene de Settings
    (LOG_LEVEL en .env). Un valor inválido lanza ValueError acá: el servicio
    falla al arrancar, no a mitad de jornada — fail fast.
    Quien quiera loguear usa structlog.get_logger(); si en el futuro se
    agregan variables de contexto (request_id, tenant, etc.) se agregan al
    pipeline de processors acá, sin tocar los puntos de logger."""
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.stdlib.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    handler = logging.StreamHandler()
    handler.setFormatter(
        structlog.stdlib.ProcessorFormatter(processor=structlog.dev.ConsoleRenderer())
    )
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)

    # Third-party loggers are pinned to WARNING on purpose.
    #
    # HTTP transport libraries (httpx, and the httpcore family) log full
    # request/response HEADERS at DEBUG, which is how account identifiers
    # (`openai-organization`) and Cloudflare `set-cookie` session values end up
    # written to the log file. The openai SDK adds its own request/response
    # trace lines at DEBUG for the same reason. None of that is the user's call:
    # only this project's own structlog events should follow LOG_LEVEL.
    #
    # WARNING and above still pass through, so genuine third-party warnings and
    # errors are preserved -- only the verbosity dump is suppressed.
    #
    # NOTE: pin the top-level namespace, not its children. Child loggers created
    # later (lazily, by a library that has not been imported yet) inherit from
    # the pinned parent. Pinning individual children such as "httpcore.http11"
    # silently fails the moment a transport library is renamed, because an
    # unlisted child of a NOTSET logger falls back to the DEBUG root level.
    for noisy in ("httpx", "httpcore", "httpcore2", "openai", "aiohttp", "h11"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging(get_settings().log_level)
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="Estimador CAG",
        description=(
            "Genera estimaciones de duración de proyectos a partir de "
            "transcripciones de reuniones usando CAG (Cache-Augmented Generation)."
        ),
        version="0.1.0",
        lifespan=lifespan,
    )
    app.include_router(estimations_router)

    @app.get("/")
    def root() -> dict[str, str]:
        return {"service": "estimador-cag", "status": "ok"}

    @app.get("/health")
    def health(settings: Settings = Depends(get_settings)) -> dict[str, object]:
        return {
            "status": "ok",
            "env": settings.app_env,
            "llm_configured": settings.is_configured,
            "provider": settings.llm_provider,
            "model": settings.llm_model,
        }

    _completar_schema_openapi(app)
    return app


def _completar_schema_openapi(app: FastAPI) -> None:
    """Swagger no puede quedar incompleto: el límite de longitud se valida en
    runtime contra Settings (model_validator del request), y Pydantic no puede
    volcar una validación que lee configuración a un JSON Schema estático.

    Se deriva el schema de la instancia activa de Settings: una sola fuente de
    verdad. Si el operador cambia ESTIMATION_MIN_CHARS en .env, la validación
    y la documentación cambian juntas; no hay un número mágico en el schema
    que pueda divergir del enforcement."""
    settings = get_settings()
    schema = app.openapi()
    transcription = schema["components"]["schemas"]["EstimationRequest"]["properties"][
        "transcription"
    ]
    transcription["minLength"] = settings.estimation_min_chars
    transcription["maxLength"] = settings.estimation_max_chars
    app.openapi_schema = schema


app = create_app()
