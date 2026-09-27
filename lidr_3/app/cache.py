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
from redis.asyncio import Redis
from redis.exceptions import RedisError

from .config import Settings
from .context.examples import build_system_prompt

logger = structlog.get_logger(__name__)


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
