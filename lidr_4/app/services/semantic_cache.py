"""Caché semántica (WU10): sirve la estimación de una descripción muy parecida.

Dos descripciones cuentan como la misma consulta cuando se cumplen las tres cosas:

1. **Mismo espacio de búsqueda.** Modelo, versión del prompt, `project_type`,
   `detail_level` y (en texto y en v4) `output_format`: lo mismo que determina la clave
   de la caché exacta. Una descripción de otro espacio nunca acierta, por parecida que
   sea. Así subir de versión el prompt no devuelve estimaciones de la anterior
   (PLAN.md §8).
2. **Similitud coseno** de los embeddings de las descripciones por encima del umbral
   (`CACHE_SEMANTIC_THRESHOLD`, 0.95 por defecto).
3. **Mismas cifras.** Los embeddings casi no distinguen «20.000 usuarios» de «200.000
   usuarios», y la estimación cambia mucho. Un acierto exige que las dos descripciones
   traigan el mismo conjunto de números.

Lo que se guarda (guarda anti-envenenamiento): solo estimaciones sanas, es decir, que
pasaron el schema y los guardrails de salida (la escritura va después, igual que la
exacta) y además no son un rechazo («Fuera de alcance:»), tienen confianza por encima
del umbral de aviso y sus totales cuadran. Una estimación dudosa se puede servir a quien
la pidió con su aviso, pero no se reparte a las descripciones parecidas.

Dónde: Redis plano, sin RediSearch. Por espacio de búsqueda hay un hash con el vector
normalizado de cada entrada (float32 en base64) y sus cifras, y una clave aparte por
entrada con la estimación. La similitud se calcula en la app, así que el coste de una
búsqueda crece con las entradas del espacio; `SEMANTIC_CACHE_MAX_ENTRIES` lo acota
(se desaloja la más antigua). TTL: el de la caché exacta.

Fail soft, como la exacta: si falla el embedding o Redis, la búsqueda cuenta como miss y
la escritura se omite, con un warning en el log. Ningún evento lleva la descripción.
Cada miss de la exacta paga un embedding (unos milisegundos y una fracción de céntimo)
antes de llamar al modelo; el vector se reutiliza al guardar.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import math
import operator
import re
import time
from array import array
from dataclasses import dataclass
from typing import Protocol

import litellm
from redis.asyncio import Redis
from redis.exceptions import RedisError

from app.config import Settings, provider_from_model
from app.guardrails.detectors import normalize_text
from app.schemas.structured_estimation import StructuredResult
from app.tracing import emit

__all__ = [
    "Embedder",
    "LiteLLMEmbedder",
    "SemanticCache",
    "SemanticLookup",
    "SemanticRedisCache",
    "create_semantic_cache",
    "extract_numbers",
]


@dataclass(frozen=True)
class SemanticLookup:
    """Resultado de buscar una descripción.

    `entry` es la entrada de la caché exacta del acierto (o None). `similarity` es la
    del acierto o, en un miss, la mejor encontrada (None si no había nada que comparar).
    `embedding` se guarda para no calcularlo otra vez al escribir; None si falló.
    """

    entry: dict | None = None
    similarity: float | None = None
    embedding: list[float] | None = None


class SemanticCache(Protocol):
    """Lo que `EstimationService` necesita de la caché semántica."""

    async def lookup(self, namespace: str, description: str) -> SemanticLookup: ...

    async def store(
        self,
        namespace: str,
        description: str,
        lookup: SemanticLookup,
        entry: dict,
        estimation: StructuredResult,
    ) -> bool: ...


class Embedder(Protocol):
    async def embed(self, text: str) -> list[float]: ...


class LiteLLMEmbedder:
    """Embeddings por `litellm.aembedding`. `timeout` acota la llamada entera."""

    def __init__(
        self, *, api_key: str, model: str, timeout: float, client: object | None = None
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._timeout = timeout
        # Solo para los tests: un AsyncOpenAI con transporte falso, sin red.
        self._client = client

    async def embed(self, text: str) -> list[float]:
        extra = {"client": self._client} if self._client is not None else {}
        response = await asyncio.wait_for(
            litellm.aembedding(model=self._model, input=[text], api_key=self._api_key, **extra),
            timeout=self._timeout,
        )
        return [float(x) for x in response.data[0]["embedding"]]


# --- Cifras de la descripción -------------------------------------------------------

# Sin letra ni dígito delante: «B2B», «S3», «i18n» o «Web3» no son cifras del proyecto.
_NUMBER = re.compile(r"(?<![^\W_])\d+(?:[.,]\d+)*")
_THOUSANDS = re.compile(r"\d{1,3}(?:[.,]\d{3})+")


def extract_numbers(text: str) -> tuple[float, ...]:
    """Los números distintos del texto, ordenados. «20.000», «20,000» y «20000» son el mismo.

    Un separador seguido de exactamente tres dígitos es de miles; cualquier otro es un
    decimal («1,5» y «1.5»). Un número pegado a una letra («B2B», «i18n») no cuenta; sí «20k» o «5M». Los números
    escritos con palabras no se reconocen.
    """
    values: set[float] = set()
    for raw in _NUMBER.findall(text):
        if _THOUSANDS.fullmatch(raw):
            values.add(float(re.sub(r"[.,]", "", raw)))
            continue
        try:
            values.add(float(raw.replace(",", ".")))
        except ValueError:  # «1.2.3»: se comparan sus dígitos
            values.add(float(re.sub(r"[.,]", "", raw)))
    return tuple(sorted(values))


# --- Vectores ----------------------------------------------------------------------


def _normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vector))
    return [x / norm for x in vector] if norm else vector


def _pack(vector: list[float]) -> str:
    return base64.b64encode(array("f", vector).tobytes()).decode("ascii")


def _unpack(packed: str) -> list[float]:
    vector = array("f")
    vector.frombytes(base64.b64decode(packed))
    return list(vector)


def _dot(a: list[float], b: list[float]) -> float:
    return sum(map(operator.mul, a, b))


# --- Redis ------------------------------------------------------------------------


class SemanticRedisCache:
    """La caché semántica sobre el cliente de Redis del lifespan (no lo cierra)."""

    def __init__(
        self,
        client: Redis,
        embedder: Embedder,
        *,
        embedding_model: str,
        threshold: float,
        ttl: int,
        min_confidence_pct: int,
        max_entries: int,
        log_only: bool = False,
    ) -> None:
        self._client = client
        self._embedder = embedder
        self._embedding_model = embedding_model
        self._threshold = threshold
        self._ttl = ttl
        self._min_confidence_pct = min_confidence_pct
        self._max_entries = max_entries
        self._log_only = log_only

    # Claves. El modelo de embeddings entra al espacio: vectores de otro modelo no se comparan.
    def _vectors_key(self, namespace: str) -> str:
        digest = hashlib.sha256(f"{namespace}|{self._embedding_model}".encode()).hexdigest()
        return f"estimator:sem:v:{digest}"

    @staticmethod
    def _entry_id(namespace: str, description: str) -> str:
        digest = hashlib.sha256(f"{namespace}|{normalize_text(description)}".encode())
        return digest.hexdigest()[:32]

    @staticmethod
    def _entry_key(entry_id: str) -> str:
        return f"estimator:sem:e:{entry_id}"

    async def lookup(self, namespace: str, description: str) -> SemanticLookup:
        """La entrada más parecida que pasa el umbral y las cifras, o un miss. Nunca lanza."""
        try:
            embedding = await self._embedder.embed(description)
        except Exception as exc:  # noqa: BLE001 - cualquier fallo del proveedor es un miss
            self._unavailable("embedding", exc)
            return SemanticLookup()
        query = _normalize(embedding)
        try:
            stored = await self._client.hgetall(self._vectors_key(namespace))
        except RedisError as exc:
            self._unavailable("lectura", exc)
            return SemanticLookup(embedding=embedding)

        numbers = extract_numbers(description)
        scored: list[tuple[float, str, bool]] = []
        for entry_id, raw in stored.items():
            try:
                data = json.loads(raw)
                vector = _unpack(data["v"])
                same_numbers = tuple(data["n"]) == numbers
            except (ValueError, KeyError, TypeError):
                continue  # entrada corrupta: se ignora
            if len(vector) != len(query):
                continue
            scored.append((_dot(query, vector), entry_id, same_numbers))

        best = max((s for s, _, _ in scored), default=None)
        for similarity, entry_id, same_numbers in sorted(scored, reverse=True):
            if similarity < self._threshold:
                break
            if not same_numbers:
                emit(
                    __name__,
                    "cache_semantica_cifras_distintas",
                    similitud=round(similarity, 4),
                    umbral=self._threshold,
                )
                continue
            entry = await self._read_entry(namespace, entry_id)
            if entry is None:
                continue
            if self._log_only:
                emit(
                    __name__,
                    "cache_semantica_solo_log",
                    similitud=round(similarity, 4),
                    umbral=self._threshold,
                )
                return SemanticLookup(similarity=similarity, embedding=embedding)
            emit(
                __name__,
                "cache_semantica_acierto",
                similitud=round(similarity, 4),
                umbral=self._threshold,
            )
            return SemanticLookup(entry=entry, similarity=similarity, embedding=embedding)

        emit(
            __name__,
            "cache_semantica_fallo",
            mejor_similitud=None if best is None else round(best, 4),
            umbral=self._threshold,
            entradas=len(scored),
        )
        return SemanticLookup(similarity=best, embedding=embedding)

    async def _read_entry(self, namespace: str, entry_id: str) -> dict | None:
        """La entrada guardada, o None si venció o está corrupta (y se limpia su vector)."""
        try:
            payload = await self._client.get(self._entry_key(entry_id))
            if payload is not None:
                entry = json.loads(payload)
                if isinstance(entry, dict):
                    return entry
            await self._client.hdel(self._vectors_key(namespace), entry_id)
        except (RedisError, ValueError):
            self._unavailable("lectura", None)
        return None

    def is_storable(self, estimation: StructuredResult) -> bool:
        """Solo estimaciones sanas se reparten a otras descripciones (guarda anti-envenenamiento)."""
        return not self._unstorable_reasons(estimation)

    def _unstorable_reasons(self, estimation: StructuredResult) -> list[str]:
        """Por qué no se reparte: vacío si es sana. Solo nombres de reglas, nunca texto."""
        reasons = []
        if estimation.out_of_scope:
            reasons.append("fuera_de_alcance")
        if estimation.confidence_pct < self._min_confidence_pct:
            reasons.append("confianza_baja")
        if estimation.total_discrepancies():
            reasons.append("totales_no_cuadran")
        return reasons

    async def store(
        self,
        namespace: str,
        description: str,
        lookup: SemanticLookup,
        entry: dict,
        estimation: StructuredResult,
    ) -> bool:
        """Guarda la estimación para descripciones parecidas. True si se guardó. Nunca lanza."""
        if lookup.embedding is None:
            return False  # el embedding falló al buscar: no se reintenta ahora
        reasons = self._unstorable_reasons(estimation)
        if reasons:
            emit(
                __name__,
                "cache_semantica_no_guardada",
                motivos=reasons,
                confianza_pct=estimation.confidence_pct,
                fuera_de_alcance=estimation.out_of_scope,
            )
            return False
        entry_id = self._entry_id(namespace, description)
        vectors_key = self._vectors_key(namespace)
        value = json.dumps(
            {
                "v": _pack(_normalize(lookup.embedding)),
                "n": list(extract_numbers(description)),
                "t": time.time(),
            }
        )
        try:
            async with self._client.pipeline(transaction=False) as pipe:
                pipe.set(self._entry_key(entry_id), json.dumps(entry), ex=self._ttl)
                pipe.hset(vectors_key, entry_id, value)
                pipe.expire(vectors_key, self._ttl)
                await pipe.execute()
            await self._evict_oldest(vectors_key)
        except (RedisError, TypeError) as exc:
            self._unavailable("escritura", exc)
            return False
        return True

    async def _evict_oldest(self, vectors_key: str) -> None:
        """Deja como mucho `max_entries` entradas, quitando las más antiguas."""
        if await self._client.hlen(vectors_key) <= self._max_entries:
            return
        stored = await self._client.hgetall(vectors_key)
        dated: list[tuple[float, str]] = []
        for entry_id, raw in stored.items():
            try:
                dated.append((float(json.loads(raw)["t"]), entry_id))
            except (ValueError, KeyError, TypeError):
                dated.append((0.0, entry_id))
        for _, entry_id in sorted(dated)[: len(dated) - self._max_entries]:
            await self._client.hdel(vectors_key, entry_id)
            await self._client.delete(self._entry_key(entry_id))

    @staticmethod
    def _unavailable(operation: str, exc: Exception | None) -> None:
        emit(
            __name__,
            "cache_semantica_no_disponible",
            level="warning",
            operacion=operation,
            tipo_error=type(exc).__name__ if exc is not None else None,
        )


def create_semantic_cache(settings: Settings, client: Redis | None) -> SemanticRedisCache | None:
    """La caché semántica que pide la configuración, o None si está apagada o no puede.

    Encendida por defecto (`SEMANTIC_CACHE_ENABLED`). Encendida pero sin Redis o sin la clave
    del proveedor de embeddings, queda apagada con un warning: el servicio sigue sin ella.
    """
    if not settings.semantic_cache_enabled:
        return None
    if client is None:
        emit(
            __name__,
            "cache_semantica_desactivada",
            level="warning",
            motivo="REDIS_URL vacío",
        )
        return None
    provider = provider_from_model(settings.embedding_model)
    api_key = settings.active_api_key(provider)
    if api_key is None:
        emit(
            __name__,
            "cache_semantica_desactivada",
            level="warning",
            motivo=f"falta la clave de {provider}",
        )
        return None
    emit(
        __name__,
        "cache_semantica_activada",
        modelo=settings.embedding_model,
        umbral=settings.cache_semantic_threshold,
        solo_log=settings.semantic_cache_log_only,
    )
    return SemanticRedisCache(
        client,
        LiteLLMEmbedder(
            api_key=api_key, model=settings.embedding_model, timeout=settings.embedding_timeout
        ),
        embedding_model=settings.embedding_model,
        threshold=settings.cache_semantic_threshold,
        ttl=settings.cache_ttl,
        min_confidence_pct=settings.low_confidence_warn_pct,
        max_entries=settings.semantic_cache_max_entries,
        log_only=settings.semantic_cache_log_only,
    )
