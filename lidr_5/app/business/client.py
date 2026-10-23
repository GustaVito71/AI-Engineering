"""Cliente HTTP al backend de negocio (Paso 12).

QUÉ HACE ESTE MÓDULO (en lenguaje llano)

Le pregunta por HTTP al backend de negocio (en este repo, el simulado de
`business_backend/`) qué proyectos pasados se parecen a uno nuevo. El servicio de IA no
conoce cómo el backend guarda esos datos: solo conoce la URL, la clave y este contrato REST
(`GET /projects/similar`). Si el backend no responde, está caído o devuelve un error, se
lanza `BusinessBackendUnavailableError` y quien llama decide qué hacer (la herramienta del
Paso 12 avisa al modelo y sigue sin esa información).
"""

from __future__ import annotations

from dataclasses import dataclass

import httpx

# Máximo de proyectos que se piden al backend.
MAX_RESULTS = 3


class BusinessBackendUnavailableError(Exception):
    """El backend de negocio no respondió, respondió con error o con datos inválidos."""


@dataclass(frozen=True)
class SimilarProject:
    """Un proyecto histórico parecido: lo que el backend de negocio devuelve de cada uno."""

    project_type: str
    hours_estimated: float
    hours_actual: float
    deviation_pct: float
    risks_materialized: list[str]


def _clean_error(error: Exception, secret: str) -> str:
    """Texto corto de un error, sin saltos de línea y con la clave tapada."""
    text = str(error).replace("\n", " ")
    if secret:
        text = text.replace(secret, "***")
    return f"{type(error).__name__}: {text[:160]}"


class BusinessBackendClient:
    """Pide al backend de negocio los proyectos históricos parecidos a uno nuevo."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        timeout_seconds: float,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        """`base_url` sin barra final esperada en el contrato (se normaliza igual).

        `transport` solo se usa en los tests (`httpx.MockTransport`), para no hacer red real.
        """
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout_seconds
        self._transport = transport

    async def find_similar(self, project_type: str) -> list[SimilarProject]:
        """Los proyectos más parecidos a `project_type`, o lanza si el backend no responde bien."""
        try:
            async with httpx.AsyncClient(
                timeout=self._timeout, transport=self._transport
            ) as http_client:
                response = await http_client.get(
                    f"{self._base_url}/projects/similar",
                    params={"query": project_type, "max_results": MAX_RESULTS},
                    headers={"X-API-Key": self._api_key},
                )
            response.raise_for_status()
            payload = response.json()
            raw = list(payload.get("results", []))[:MAX_RESULTS]
            return [
                SimilarProject(
                    project_type=str(item.get("project_type", "")),
                    hours_estimated=float(item.get("hours_estimated", 0)),
                    hours_actual=float(item.get("hours_actual", 0)),
                    deviation_pct=float(item.get("deviation_pct", 0)),
                    risks_materialized=list(item.get("risks_materialized", [])),
                )
                for item in raw
            ]
        except Exception as error:
            raise BusinessBackendUnavailableError(_clean_error(error, self._api_key)) from error
