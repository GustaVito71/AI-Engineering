"""Backend de negocio SIMULADO: histórico de proyectos ejecutados (Paso 12).

QUÉ ES ESTE SERVICIO (en lenguaje llano)

Es un servicio aparte, independiente del servicio de IA (lidr_5), que representa "el sistema
de negocio" de una empresa real: una base con proyectos ya ejecutados (horas estimadas y
reales, desviación, riesgos que se materializaron). El servicio de IA NO lo lee directo ni
conoce su esquema: le pide datos por HTTP, como lo haría con cualquier sistema externo real.

Por eso vive en su propio proceso y contenedor, con su propia clave de acceso. Sustituir este
simulado por el backend de negocio real de una empresa no debería exigir ningún cambio en el
servicio de IA, solo que el real responda el mismo contrato REST.

Los datos de `data.json` son sintéticos (generados con un criterio razonable, no reales).

ENDPOINT

    GET /projects/similar?query=<texto>&max_results=3
    Cabecera: X-API-Key: <clave>

    200: {"results": [{"project_type", "hours_estimated", "hours_actual",
                        "deviation_pct", "risks_materialized"}, ...]}
    401: clave ausente o incorrecta.
    422: falta `query`.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Header, HTTPException, Query

app = FastAPI(title="Backend de negocio simulado (lidr_5, Paso 12)")

# Tope absoluto de resultados, aunque el cliente pida más.
MAX_RESULTS_CAP = 5
DATA_PATH = Path(__file__).parent / "data.json"


def _load_projects() -> list[dict[str, Any]]:
    """Carga el histórico de proyectos desde `data.json` (una vez por proceso)."""
    with DATA_PATH.open(encoding="utf-8") as f:
        return json.load(f)


# Cargados al arrancar: son datos de ejemplo fijos, no hace falta releerlos por consulta.
_PROJECTS = _load_projects()


def _api_key() -> str:
    """La clave esperada, leída del entorno en cada request (simple: no hay Settings acá)."""
    return os.environ.get("BUSINESS_API_KEY", "")


def _score(project: dict[str, Any], query_words: list[str]) -> int:
    """Cuántas palabras de la consulta aparecen en el tipo de proyecto o en sus riesgos."""
    haystack = (project["project_type"] + " " + " ".join(project["risks_materialized"])).lower()
    return sum(1 for word in query_words if word in haystack)


@app.get("/health")
def health() -> dict[str, str]:
    """Para que `docker compose` y una persona puedan comprobar que el proceso arrancó."""
    return {"status": "ok"}


@app.get("/projects/similar")
def find_similar_projects(
    query: str = Query(..., min_length=1, description="Tipo de proyecto a buscar"),
    max_results: int = Query(3, ge=1, le=MAX_RESULTS_CAP),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> dict[str, Any]:
    """Los proyectos históricos más parecidos a `query`, por coincidencia de palabras."""
    expected = _api_key()
    if not expected or x_api_key != expected:
        raise HTTPException(status_code=401, detail="X-API-Key ausente o incorrecta")

    words = [w for w in query.lower().split() if w]
    ranked = sorted(_PROJECTS, key=lambda p: _score(p, words), reverse=True)
    # Si nada coincide, se devuelven los primeros de todos modos: es un simulado de ejemplo,
    # no hace falta un "sin resultados" más allá de la lista vacía con otra consulta.
    top = ranked[: min(max_results, MAX_RESULTS_CAP)]
    return {"results": top}
