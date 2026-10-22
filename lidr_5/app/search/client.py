"""Cliente de búsqueda web con servicio principal y de respaldo (Paso 11).

QUÉ HACE ESTE MÓDULO (en lenguaje llano)

Pregunta a un buscador de internet (Tavily o Firecrawl) y devuelve una lista corta de
resultados con título, dirección y un fragmento de texto. Si el buscador principal falla
(error, tiempo agotado o ninguna respuesta útil), prueba con el de respaldo. Si fallan los
dos, lanza `SearchUnavailableError` y quien lo llama decide qué hacer.

Las llamadas pasan por LiteLLM (`litellm.asearch`), la misma vía que ya se comprobó con
`test/verificar_busqueda.py`; no hay clientes HTTP propios.

REGLAS FIJADAS (medidas en CONTEXTO.md)

- Máximo 3 resultados por búsqueda.
- Firecrawl SIEMPRE con `scrapeOptions={}`: sin eso devuelve la página completa en markdown
  (unos 912.000 caracteres en la prueba) en lugar de unos 17.500.
- El fragmento de cada resultado se recorta a `max_snippet_chars`.
- Una clave nunca aparece en un mensaje de error: se sustituye por `***`.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

import litellm

from app.tracing import emit

# Máximo de resultados que se piden y se devuelven por búsqueda.
MAX_RESULTS = 3
# Marca que se añade al final de un fragmento recortado.
SNIPPET_TRUNCATION_MARK = "…"

# Firma de la función que hace la búsqueda real. Se puede sustituir en los tests.
SearchFunction = Callable[..., Awaitable[Any]]


class SearchUnavailableError(Exception):
    """Ningún servicio de búsqueda (ni el principal ni el respaldo) pudo responder."""


@dataclass(frozen=True)
class SearchResult:
    """Un resultado de búsqueda: título, dirección y un fragmento de texto."""

    title: str
    url: str
    snippet: str


@dataclass(frozen=True)
class SearchOutcome:
    """Lo que devuelve una búsqueda: resultados, quién respondió y cuánto tardó."""

    results: list[SearchResult]
    provider: str
    used_fallback: bool
    duration_ms: float


def _clean_error(error: Exception, secrets: list[str]) -> str:
    """Texto corto de un error, sin saltos de línea y con las claves tapadas."""
    text = str(error).replace("\n", " ")
    for secret in secrets:
        if secret:
            text = text.replace(secret, "***")
    return f"{type(error).__name__}: {text[:160]}"


def _truncate(text: str, max_chars: int) -> str:
    """Recorta `text` a `max_chars` caracteres (marca incluida) y quita espacios sobrantes."""
    text = " ".join((text or "").split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - len(SNIPPET_TRUNCATION_MARK)].rstrip() + SNIPPET_TRUNCATION_MARK


class WebSearchClient:
    """Busca en la web con un servicio principal y, si falla, con uno de respaldo."""

    def __init__(
        self,
        *,
        keys: dict[str, str],
        primary: str,
        fallback: str,
        timeout_seconds: float,
        max_snippet_chars: int,
        search_fn: SearchFunction | None = None,
    ) -> None:
        """`keys` mapea nombre de servicio → clave; solo se usan los servicios con clave.

        `search_fn` permite inyectar una función falsa en los tests; por defecto es
        `litellm.asearch`.
        """
        self._keys = {name: key for name, key in keys.items() if key}
        self._order = [name for name in (primary, fallback) if name in self._keys]
        self._timeout = timeout_seconds
        self._max_snippet_chars = max_snippet_chars
        self._search_fn: SearchFunction = search_fn or litellm.asearch

    @property
    def available(self) -> bool:
        """True si hay al menos un servicio con clave."""
        return bool(self._order)

    async def _search_one(self, provider: str, query: str) -> list[SearchResult]:
        """Una búsqueda contra un servicio. Lanza si falla o si no hay resultados."""
        extra: dict[str, Any] = {}
        if provider == "firecrawl":
            # Sin esto Firecrawl devuelve cada página completa (cientos de miles de caracteres).
            extra["scrapeOptions"] = {}
        response = await self._search_fn(
            query=query,
            search_provider=provider,
            max_results=MAX_RESULTS,
            api_key=self._keys[provider],
            timeout=self._timeout,
            **extra,
        )
        raw = list(getattr(response, "results", None) or [])[:MAX_RESULTS]
        results = [
            SearchResult(
                title=" ".join((getattr(r, "title", "") or "").split()),
                url=(getattr(r, "url", "") or "").strip(),
                snippet=_truncate(getattr(r, "snippet", "") or "", self._max_snippet_chars),
            )
            for r in raw
        ]
        if not results:
            raise ValueError("el servicio respondió sin resultados")
        return results

    async def search(self, query: str) -> SearchOutcome:
        """Busca en el principal y, si falla, en el respaldo.

        Lanza `SearchUnavailableError` si no hay servicios con clave o si todos fallan.
        """
        secrets = list(self._keys.values())
        errors: list[str] = []
        for index, provider in enumerate(self._order):
            started = time.perf_counter()
            try:
                results = await self._search_one(provider, query)
            except Exception as error:  # noqa: BLE001 - cualquier fallo pasa al respaldo
                duration = round((time.perf_counter() - started) * 1000, 1)
                detail = _clean_error(error, secrets)
                errors.append(f"{provider}: {detail}")
                emit(
                    __name__,
                    "busqueda_web_fallida",
                    level="warning",
                    proveedor=provider,
                    duracion_ms=duration,
                    error=detail,
                )
                continue
            return SearchOutcome(
                results=results,
                provider=provider,
                used_fallback=index > 0,
                duration_ms=round((time.perf_counter() - started) * 1000, 1),
            )
        if not self._order:
            raise SearchUnavailableError("no hay ningún servicio de búsqueda con clave")
        raise SearchUnavailableError("; ".join(errors))
