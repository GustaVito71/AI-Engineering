"""Caché exact-match de respuestas del LLM, sobre Redis.

La clave es un hash determinista del par (system_prompt, user_prompt): como el
prompt entero entra en la clave, cambiar la plantilla invalida la caché solo.
TTL configurable con CACHE_TTL. REDIS_URL vacío desactiva la caché sin tocar
código.

Fail soft por diseño, igual que app/cache.py: si Redis está caído, lento o
devuelve una entrada corrupta, la lectura cuenta como miss y la escritura se
omite, con un warning en el log. La estimación nunca falla por la caché.

Este módulo no reimplementa el acceso a Redis ni crea conexiones: delega en las
funciones de app/cache.py y usa el único cliente de la app, el que crea el
lifespan con `crear_cliente_cache` (timeouts, PING de arranque y cierre al
apagar). Así hay un solo pool de conexiones y es el que se cierra.
"""

from __future__ import annotations

import hashlib

from redis.asyncio import Redis

from app.cache import get_cached_estimation, set_cached_estimation


def crear_estimation_cache(client: Redis | None, ttl: int) -> EstimationCache | _NoOpCache:
    """La caché sobre el cliente del lifespan, o una desactivada si no hay cliente.

    `client` es None cuando REDIS_URL está vacío: es el modo "sin caché" de
    `crear_cliente_cache`, y acá se traduce a una caché que siempre falla.
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

    def make_key(self, system_prompt: str, user_prompt: str) -> str:
        """Clave determinista a partir del par de prompts."""
        combined = f"{system_prompt}|{user_prompt}"
        return f"estimator:{hashlib.sha256(combined.encode()).hexdigest()}"

    async def get(self, key: str) -> dict | None:
        """La respuesta cacheada, o None en miss, con Redis caído o entrada corrupta."""
        return await get_cached_estimation(self._client, key)

    async def set(self, key: str, value: dict) -> None:
        """Guarda la respuesta con TTL. Si Redis falla, se omite sin error."""
        await set_cached_estimation(self._client, key, value, self._ttl)


class _NoOpCache:
    """Caché desactivada (REDIS_URL vacío): siempre miss, nunca guarda."""

    def make_key(self, _system: str, _user: str) -> str:
        return "noop"

    async def get(self, _key: str) -> None:
        return None

    async def set(self, _key: str, _value: dict) -> None:
        pass
