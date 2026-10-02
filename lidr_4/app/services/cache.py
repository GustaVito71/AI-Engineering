"""Exact-match cache for LLM responses using Redis.

The cache key is a deterministic hash of the (system_prompt, user_prompt) pair.
TTL is configurable via Settings.CACHE_TTL (default 24h). An empty REDIS_URL
disables the cache entirely without code changes.
"""

from __future__ import annotations

import hashlib
import json

import redis.asyncio as redis


class EstimationCache:
    """Redis-backed exact-match cache for LLM responses."""

    def __init__(self, client: redis.Redis, ttl: int) -> None:
        self._client = client
        self._ttl = ttl

    @classmethod
    def from_url(cls, url: str, ttl: int) -> EstimationCache:
        """Create cache instance from Redis URL.

        Empty URL returns a no-op cache that always misses and never stores.
        """
        if not url:
            return _NoOpCache()
        client = redis.from_url(url, decode_responses=True)
        return cls(client, ttl)

    def _make_key(self, system_prompt: str, user_prompt: str) -> str:
        """Deterministic cache key from prompt pair."""
        combined = f"{system_prompt}|{user_prompt}"
        return f"estimator:{hashlib.sha256(combined.encode()).hexdigest()}"

    async def get(self, key: str) -> dict | None:
        """Retrieve cached response if present."""
        data = await self._client.get(key)
        if data:
            return json.loads(data)
        return None

    async def set(self, key: str, value: dict) -> None:
        """Store response in cache with TTL."""
        await self._client.setex(key, self._ttl, json.dumps(value))

    async def close(self) -> None:
        """Close Redis connection pool."""
        await self._client.close()

    def make_key(self, system_prompt: str, user_prompt: str) -> str:
        """Public method for external callers (e.g., LLMWrapper) to compute keys."""
        return self._make_key(system_prompt, user_prompt)


class _NoOpCache:
    """No-op cache implementation when Redis is disabled."""

    def make_key(self, _system: str, _user: str) -> str:
        return "noop"

    async def get(self, _key: str) -> None:
        return None

    async def set(self, _key: str, _value: dict) -> None:
        pass

    async def close(self) -> None:
        pass
