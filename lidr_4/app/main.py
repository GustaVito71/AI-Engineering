"""Punto de entrada de la aplicación FastAPI.

# /health NO depende de la configuración: el servicio arranca aunque falte la
# API key y lo dice en la respuesta (`llm_configured: false`). Un health check
# tiene que sobrevivir a la avería que está diagnosticando; si exige la key al
# arrancar, muere antes que el problema y el orquestador ve un crashloop sin
# forma de distinguir "falta un secreto" de "el código está roto".
#
# Lo mismo aplica al lifespan: crea el cliente de caché, pero NO el cliente
# LLM. El Router se construye perezoso en WU5, y fallar acá si falta una key
# reproduciría exactamente el bug que este diseño evita.

No hay middleware CORS a propósito: no existe frontend que llame a esta API
desde un navegador, y CORS es una protección del navegador. Se agrega solo
cuando haya uno (con allow_origins explícito desde Settings)."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import structlog
from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse

from .cache import cerrar_cliente_cache, crear_cliente_cache
from .config import LLMConfigurationError, Settings, get_settings
from .prompts.loader import versiones_disponibles
from .routers.estimations import router as estimations_router
from .services.cache import crear_estimation_cache
from .services.llm_wrapper import aviso_sin_respaldo

# Loggers propios de LiteLLM (los nombres llevan mayúsculas y espacios). En DEBUG
# escriben los parámetros completos de cada llamada, incluidos los mensajes: el
# prompt del sistema y la descripción del cliente. Se fijan en WARNING igual que
# los de transporte HTTP, y eso también pisa lo que pida LITELLM_LOG.
LOGGERS_DE_LITELLM = ("LiteLLM", "LiteLLM Router", "LiteLLM Proxy")


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

    # Los loggers de terceros se fijan en WARNING a propósito.
    #
    # Las librerías de transporte HTTP (httpx y la familia httpcore) escriben en
    # DEBUG los HEADERS completos de cada request y response. Por ahí terminan en
    # el log identificadores de cuenta (`openai-organization`) y valores de sesión
    # de Cloudflare (`set-cookie`). El SDK de openai agrega en DEBUG sus propias
    # trazas de request y response. Nada de eso debe depender de LOG_LEVEL: solo
    # los eventos structlog propios del proyecto siguen esa variable.
    #
    # WARNING y lo que está por encima siguen pasando, así que las advertencias y
    # los errores reales de terceros se conservan: solo se corta el volcado
    # detallado.
    #
    # OJO: se fija el espacio de nombres de primer nivel, no sus hijos. Los
    # loggers hijos que se crean después (cuando se importa una librería que
    # todavía no se cargó) heredan el nivel del padre fijado. Fijar hijos sueltos
    # como "httpcore.http11" deja de funcionar, sin aviso, en cuanto una librería
    # de transporte cambia de nombre: un hijo no listado de un logger NOTSET toma
    # el nivel DEBUG de la raíz.
    for noisy in (
        "httpx",
        "httpcore",
        "httpcore2",
        "openai",
        "aiohttp",
        "h11",
        *LOGGERS_DE_LITELLM,
    ):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _unificar_loggers_de_litellm()


def _unificar_loggers_de_litellm() -> None:
    """Que los mensajes de LiteLLM salgan una sola vez, con el formato de structlog.

    LiteLLM instala su propio handler en sus loggers y además deja que los
    mensajes sigan hasta la raíz, así que cada advertencia salía dos veces: una
    con su formato y otra con el de structlog. Se quita su handler y queda solo
    el de la raíz.

    Ese handler lleva los filtros que borran las claves de API de los mensajes,
    y un filtro de handler solo actúa en ese handler: quitarlo a secas dejaría
    pasar las claves a la raíz. Por eso antes se copian sus filtros a los
    loggers de LiteLLM (incluidos los hijos, como "LiteLLM Proxy.stdout"), que
    los aplican a cada mensaje antes de que llegue a cualquier handler.

    Se puede llamar más de una vez: la segunda ya no encuentra handlers.
    """
    raices = [logging.getLogger(nombre) for nombre in LOGGERS_DE_LITELLM]
    filtros = [f for logger in raices for handler in logger.handlers for f in handler.filters]
    for nombre in list(logging.root.manager.loggerDict):
        if any(nombre == r or nombre.startswith(f"{r}.") for r in LOGGERS_DE_LITELLM):
            for filtro in filtros:
                logging.getLogger(nombre).addFilter(filtro)  # addFilter no duplica
    for logger in raices:
        logger.handlers.clear()


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level)
    # Único cliente de Redis de la app. La caché de estimaciones se arma sobre
    # él y el wrapper LLM la toma de app.state (ver app/dependencies.py). Se
    # cierra en el `finally` para que el pool de conexiones no quede abierto si
    # el arranque falla a mitad o si hay un shutdown abrupto.
    cliente = await crear_cliente_cache(settings)
    app.state.cache_client = cliente
    app.state.estimation_cache = crear_estimation_cache(cliente, settings.cache_ttl)
    try:
        yield
    finally:
        await cerrar_cliente_cache(cliente)


def create_app() -> FastAPI:
    app = FastAPI(
        title="Estimador estructurado",
        description=(
            "Genera estimaciones de proyecto a partir de un formulario "
            "estructurado, devolviendo JSON validado contra un esquema y "
            "calculando los totales en código."
        ),
        version="0.4.0",
        lifespan=lifespan,
    )
    app.include_router(estimations_router)

    @app.exception_handler(LLMConfigurationError)
    async def llm_no_configurado(_request: Request, exc: LLMConfigurationError) -> JSONResponse:
        # Falta configuración local (una API key), no falló el proveedor: 503
        # "servicio no disponible" con el nombre de la variable, en vez de un
        # 500 genérico que obliga a leer el traceback. El mensaje lo arma el
        # wrapper y no contiene secretos, solo el nombre de la variable.
        return JSONResponse(status_code=503, content={"detail": str(exc)})

    @app.get("/")
    def root() -> dict[str, str]:
        return {"service": "estimador-estructurado", "status": "ok"}

    @app.get("/health")
    def health(settings: Settings = Depends(get_settings)) -> dict[str, object]:
        # Sin llamar al LLM ni exigirlo: describe el estado de la configuración.
        # `llm_configured` es lo que el orquestador mira para decidir si el
        # arranque sin secret es aceptable o no. Mira solo el primario: sin la
        # clave del respaldo el servicio funciona igual, y eso lo informan
        # `fallback_configured` y `avisos`, con el mismo texto que recibe el
        # usuario en cada estimación.
        aviso = aviso_sin_respaldo(settings)
        return {
            "status": "ok",
            "env": settings.app_env,
            "llm_configured": settings.is_configured,
            "fallback_configured": aviso is None,
            "primary_model": settings.primary_model,
            "fallback_model": settings.fallback_model,
            "prompt_version": settings.prompt_version,
            # Las que acepta `?prompt_version=` en el endpoint: el frontend arma
            # su selector con esta lista en vez de tener una escrita a mano.
            "prompt_versions": versiones_disponibles(),
            "cache_enabled": bool(settings.redis_url),
            "avisos": [aviso] if aviso else [],
        }

    _completar_schema_openapi(app)
    return app


def _completar_schema_openapi(app: FastAPI) -> None:
    """Swagger no puede quedar incompleto: el techo del operador se valida en
    runtime contra Settings (el `model_validator` de EstimationRequest), y
    Pydantic no puede volcar una validación que lee configuración a un JSON
    Schema estático.

    Sin esto, Swagger documentaría los límites del contrato (20/2000, los del
    `Field`) y no los que el servicio realmente tolera cuando el operador los
    ajustó. El `Field` no se toca: el contrato con el cliente no se reconfigura
    desde .env.

    Se deriva de la instancia activa de Settings: si el operador cambia
    DESCRIPCION_MAX_CHARS en .env, la documentación cambia con ella, y ambos
    números ya fueron validados contra el contrato al arrancar
    (Settings.validar_techo_descripcion), así que nunca prometen más de lo que
    el servicio acepta.

    Solo se completa `description`: los otros tres campos del formulario son
    enums y no tienen un techo configurable que valga la pena duplicar acá.
    """
    settings = get_settings()
    schema = app.openapi()
    description = schema["components"]["schemas"]["EstimationRequest"]["properties"]["description"]
    description["minLength"] = settings.descripcion_min_chars
    description["maxLength"] = settings.descripcion_max_chars
    app.openapi_schema = schema


app = create_app()
