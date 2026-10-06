"""Cache Redis (infraestructura, agnóstico del dominio).

Este módulo NO conoce la respuesta del LLM: guarda y devuelve dicts
JSON-serializables. Quien usa el cache (el gateway de WU5) traduce entre el dict
y lo que necesite el dominio. Así el cache queda reutilizable para cualquier
dato y el dominio no se acopla a Redis.

Fail soft por diseño: un fallo de Redis (caído, timeout, entrada corrupta)
NUNCA tumba la estimación. Un cache es una optimización, no un requisito:
si no está disponible se loguea y se responde igual, generando sin cache.
En development no hay Redis corriendo; si el cache fuera obligatorio, cada
request moriría por una infraestructura que no es la del proveedor LLM.

QUÉ SE SACÓ RESPECTO DE LIDR_3

`build_cache_key()` (sha256 del system prompt + transcripción + modelo +
max_tokens) no se hereda: era el keying EXACTO, y el de esta entrega es
semántico. Sobrevivió su mejor idea —que un cambio de prompt invalida el
caché sin intervención manual— pero su expresión era correcta solo porque
hasheaba el prompt entero. La versión semántica embebe únicamente la
consulta del usuario, así que esa propiedad NO es automática: pasa a ser
responsabilidad de la clave, y de `PROMPT_VERSION` en Settings.

Ver la nota sobre la clave en WU10 antes de escribir el keying.
"""

from __future__ import annotations

import json

import structlog
from redis.asyncio import Redis
from redis.exceptions import RedisError

from .config import Settings

logger = structlog.get_logger(__name__)


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


async def set_cached_estimation(client: Redis, key: str, result: dict, ttl: int) -> None:
    """Guarda un resultado con TTL en segundos. Fail soft igual que la lectura."""
    try:
        await client.set(key, json.dumps(result), ex=ttl)
    except (RedisError, TypeError):
        logger.warning("Cache Redis no disponible en escritura; respuesta sin cachear")


# Un cache lento es peor que un cache ausente: si Redis está caído pero el
# socket no cierra, cada request cuelga hasta este timeout. 1s deja margen de
# sobra para un Redis sano en localhost y acota el daño cuando no lo está.
_TIMEOUT_SECONDS = 1.0


async def create_cache_client(settings: Settings) -> Redis | None:
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
    redis_client = Redis.from_url(
        settings.redis_url,
        decode_responses=True,
        socket_connect_timeout=_TIMEOUT_SECONDS,
        socket_timeout=_TIMEOUT_SECONDS,
    )
    try:
        await redis_client.ping()
    except RedisError as exc:
        # No se propaga: el cliente se entrega igual. Si Redis está caído ahora
        # pero vuelve, el cache se recupera solo en la request siguiente sin
        # reiniciar el servicio.
        logger.warning(
            "Redis no responde al PING de arranque; se sigue con fail soft", error=str(exc)
        )
    else:
        logger.info("Cache Redis conectado", redis_url=settings.redis_url)
    return redis_client


async def close_cache_client(redis_client: Redis | None) -> None:
    """Cierra el pool de conexiones del cliente. Idempotente y fail-soft."""
    if redis_client is None:
        return
    try:
        await redis_client.aclose()
    except RedisError as exc:
        logger.warning("No se pudo cerrar el cliente de Redis limpiamente", error=str(exc))
