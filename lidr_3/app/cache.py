"""Cache Redis de estimaciones (infraestructura, agnóstico del dominio).

Este módulo NO conoce EstimationResult ni LLMResponse: guarda y devuelve
dicts JSON-serializables. Quien usa el cache (llm_service) traduce entre
dict y resultado. Así el cache queda reutilizable para cualquier dato y el
dominio no se acopla a Redis.

Fail soft por diseño: un fallo de Redis (caído, timeout, entrada corrupta)
NUNCA tumba la estimación. Un cache es una optimización, no un requisito:
si no está disponible se loguea y se responde igual, generando sin cache.
En development no hay Redis corriendo; si el cache fuera obligatorio, cada
request moriría por una infraestructura que no es la del proveedor LLM.
"""

from __future__ import annotations

import hashlib
import json

import structlog
from fastapi import Request
from redis.asyncio import Redis
from redis.exceptions import RedisError

from .config import Settings
from .context.examples import build_system_prompt

logger = structlog.get_logger(__name__)

# Clave de app.state donde el lifespan publica el cliente. El router lo lee
# por dependency, no por importar el cliente: así los tests pueden inyectar un
# fakeredis (o None) sin conocer la clave.
CLIENTE_CACHE = "cache_client"


def build_cache_key(transcription: str, settings: Settings) -> str:
    """Clave determinista de la estimación (sha256 en hex).

    NO se hashea el prompt final tal como se envía: el delimitador de la
    transcripción es aleatorio POR LLAMADA (defensa anti-inyección) y
    rompería el hit. Se hashean las entradas y parámetros que determinan
    la respuesta: system prompt (cache CAG), transcripción cruda, modelo
    efectivo y max_tokens. Si cambia el CAG o el modelo, la clave cambia
    sola: el cache se invalida sin intervención."""
    material = "\x1f".join(
        (
            build_system_prompt(),
            transcription,
            settings.resolve_model(settings.llm_provider),
            str(settings.llm_max_tokens),
        )
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


async def get_cached_estimation(client: Redis, key: str) -> dict | None:
    """El resultado cacheado como dict, o None en miss o ante fallo.

    `dict` (no el resultado de dominio) a propósito: este módulo no
    conoce el dominio. Un JSON corrupto se descarta y se trata como miss."""
    try:
        payload = await client.get(key)
    except RedisError:
        # Servidor caído, timeout, red: NUNCA bloquear la estimación.
        logger.warning("Cache Redis no disponible en lectura; se sigue sin cache")
        return None
    if payload is None:
        return None
    try:
        return json.loads(payload)
    except (json.JSONDecodeError, TypeError):
        logger.warning("Entrada de cache corrupta; se descarta y se sigue sin cache")
        return None


async def set_cached_estimation(client: Redis, key: str, resultado: dict, ttl: int) -> None:
    """Guarda un resultado con TTL en segundos. Fail soft igual que la lectura."""
    try:
        await client.set(key, json.dumps(resultado), ex=ttl)
    except (RedisError, TypeError):
        logger.warning("Cache Redis no disponible en escritura; respuesta sin cachear")


# Un cache lento es peor que un cache ausente: si Redis está caído pero el
# socket no cierra, cada request cuelga hasta este timeout. 1s deja margen de
# sobra para un Redis sano en localhost y acota el daño cuando no lo está.
_TIMEOUT_SEGUNDOS = 1.0


async def crear_cliente_cache(settings: Settings) -> Redis | None:
    """Cliente Redis para el lifespan, o None si no hay cache.

    Devolver `None` (y no un cliente roto) cuando `REDIS_URL` está vacío es lo
    que permite desactivar el cache por configuración, sin código: es el
    escape hatch para desarrollo y para tests que no quieren Redis.

    Un `PING` con timeout AL ARRANQUE no se usa para decidir nada: `from_url`
    es lazy y no conecta, así que el fallo real aparecería en el primer request
    de cada usuario. Se hace para loguear el estado real de la infraestructura
    una vez, y su fallo es inocuo (warn) porque el servicio ya es fail-soft.
    """
    if not settings.redis_url:
        logger.info("Cache desactivado: REDIS_URL vacío")
        return None
    cliente = Redis.from_url(
        settings.redis_url,
        decode_responses=True,
        socket_connect_timeout=_TIMEOUT_SEGUNDOS,
        socket_timeout=_TIMEOUT_SEGUNDOS,
    )
    try:
        await cliente.ping()
    except RedisError as exc:
        # No se propaga: el cliente se entrega igual. Si Redis está caído ahora
        # pero vuelve, el cache se recupera solo en la request siguiente sin
        # reiniciar el servicio.
        logger.warning(
            "Redis no responde al PING de arranque; se sigue con fail soft", error=str(exc)
        )
    else:
        logger.info("Cache Redis conectado", redis_url=settings.redis_url)
    return cliente


async def cerrar_cliente_cache(cliente: Redis | None) -> None:
    """Cierra el pool de conexiones del cliente. Idempotente y fail-soft."""
    if cliente is None:
        return
    try:
        await cliente.aclose()
    except RedisError as exc:
        logger.warning("No se pudo cerrar el cliente de Redis limpiamente", error=str(exc))


async def get_cache_client(request: Request) -> Redis | None:
    """Dependency de FastAPI: el cliente del lifespan, o None si no hay cache.

    `getattr` y no `[...]`: sin lifespan (un test que monta la app con
    TestClient sin contexto, o una app construida a mano) no hay cliente, y eso
    es el estado válido "sin cache", no un error a propagar.
    """
    return getattr(request.app.state, CLIENTE_CACHE, None)
