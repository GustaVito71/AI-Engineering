"""Caché exact-match de respuestas del LLM, sobre Redis.

La clave es un hash determinista de las partes que le pasa `EstimationService`
(app/services/estimation.py), que es quien consulta y escribe la caché; el wrapper
no la toca. Hoy son el modelo, la versión del prompt, los parámetros de la request
y la descripción ya filtrada por los guardrails. Como la consulta ocurre antes de
armar el prompt, la clave no incluye el texto renderizado: lo que invalida la caché
al cambiar una plantilla es la versión del prompt (las versiones publicadas son
inmutables, PLAN.md §3.3), y el modelo entra para que cambiar PRIMARY_MODEL no sirva
respuestas de otro. TTL configurable con CACHE_TTL. REDIS_URL vacío desactiva la
caché sin tocar código.

Fail soft por diseño, igual que app/cache.py: si Redis está caído, lento o
devuelve una entrada corrupta, la lectura cuenta como miss y la escritura se
omite, con un warning en el log. La estimación nunca falla por la caché.

Este módulo no reimplementa el acceso a Redis ni crea conexiones: delega en las
funciones de app/cache.py y usa el único cliente de la app, el que crea el
lifespan con `create_cache_client` (timeouts, PING de arranque y cierre al
apagar). Así hay un solo pool de conexiones y es el que se cierra.
"""

from __future__ import annotations

import hashlib
import json
from typing import Protocol

from redis.asyncio import Redis

from app.cache import get_cached_estimation, set_cached_estimation


class ExactCache(Protocol):
    """Lo que `EstimationService` necesita de la caché exacta."""

    def make_key(self, *parts: str) -> str: ...

    async def get(self, key: str) -> dict | None: ...

    async def set(self, key: str, value: dict) -> None: ...


def create_estimation_cache(client: Redis | None, ttl: int) -> ExactCache:
    """La caché sobre el cliente del lifespan, o una desactivada si no hay cliente.

    `client` es None cuando REDIS_URL está vacío: es el modo "sin caché" de
    `create_cache_client`, y acá se traduce a una caché que siempre falla.
    """
    if client is None:
        return _NoOpCache()
    return EstimationCache(client, ttl)


class EstimationCache:
    """Caché exact-match en Redis que nunca hace fallar una estimación.

    No es dueña del cliente: lo crea y lo cierra el lifespan (app/main.py).
    """

    def __init__(self, client: Redis, ttl: int) -> None:
        self._client = client
        self._ttl = ttl

    def make_key(self, *parts: str) -> str:
        """Clave determinista a partir de las partes, en orden.

        Se serializan como una lista JSON y no unidas con un separador: así
        `("a|b", "c")` y `("a", "b|c")` dan claves distintas."""
        combined = json.dumps(parts, ensure_ascii=False)
        return f"estimator:{hashlib.sha256(combined.encode()).hexdigest()}"

    async def get(self, key: str) -> dict | None:
        """La respuesta cacheada, o None en miss, con Redis caído o entrada corrupta."""
        return await get_cached_estimation(self._client, key)

    async def set(self, key: str, value: dict) -> None:
        """Guarda la respuesta con TTL. Si Redis falla, se omite sin error."""
        await set_cached_estimation(self._client, key, value, self._ttl)


class _NoOpCache:
    """Caché desactivada (REDIS_URL vacío): siempre miss, nunca guarda."""

    def make_key(self, *_parts: str) -> str:
        return "noop"

    async def get(self, _key: str) -> None:
        return None

    async def set(self, _key: str, _value: dict) -> None:
        pass
