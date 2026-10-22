"""Comprueba con llamadas reales que el modelo usa `find_similar_projects` (Paso 12).

Uso (desde la raíz del proyecto):

    1. Levanta el backend simulado (en otra terminal, desde `business_backend/`):
         BUSINESS_API_KEY=clave-de-ejemplo uvicorn main:app --port 8003
       o con Docker, si ya agregaste el servicio al docker-compose.yml (ver
       `business_backend/docker-compose.snippet.yml`).
    2. En el `.env` del servicio de IA, define:
         BUSINESS_API_URL=http://localhost:8003
         BUSINESS_API_KEY=clave-de-ejemplo   # la misma clave que el paso 1
    3. uv run python test/verificar_negocio_agente.py

QUÉ COMPRUEBA (cuesta unos céntimos: 2 o 3 llamadas cortas al modelo)

1. La herramienta `find_similar_projects` se construye con las claves del .env y consulta
   de verdad al backend simulado.
2. Con una pregunta que se beneficia de datos históricos, el modelo PIDE la herramienta
   dentro del bucle agéntico y recibe los resultados.

No imprime claves. Solo lee el .env a través de Settings, como el servicio.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.agent.loop import run_agent_loop
from app.agent.tools import ToolRegistry
from app.business.tool import build_tool_from_settings
from app.config import get_settings
from app.observability import start_turn_cost
from app.services.llm_wrapper import LLMWrapper

PREGUNTA_CON_HISTORICO = (
    "Estima un proyecto de migración de API. Antes de estimar, revisá si tenemos proyectos "
    "parecidos ya ejecutados para considerar su desviación histórica."
)


async def main() -> int:
    """Corre las dos comprobaciones y devuelve 0 si pasan."""
    settings = get_settings()
    herramienta = build_tool_from_settings(settings)
    if herramienta is None:
        print("SIN CONFIGURAR: define BUSINESS_API_URL y BUSINESS_API_KEY en el .env")
        return 1

    print("1. Consulta directa de la herramienta")
    texto = await herramienta.run({"project_type": "migración de API"})
    ok1 = texto.startswith("Proyectos históricos") or "no se encontraron" in texto.lower()
    print(f"   {'OK' if ok1 else 'FALLA'}: {texto[:160]!r}")
    if not ok1:
        print("   (revisa que el backend simulado esté levantado en BUSINESS_API_URL)")
        return 1

    registro = ToolRegistry()
    registro.register(herramienta)
    wrapper = LLMWrapper(settings=settings)
    start_turn_cost()
    print("2. Pregunta que se beneficia del histórico (debe pedir la herramienta)")
    resultado = await run_agent_loop(
        llm=wrapper,
        registry=registro,
        transcript=PREGUNTA_CON_HISTORICO,
        attachments_block="",
        max_iterations=settings.agent_max_iterations,
        token_budget=settings.turn_token_budget,
        model=settings.primary_model,
    )
    ok2 = any(run.name == "find_similar_projects" for run in resultado.runs)
    print(
        f"   {'OK' if ok2 else 'FALLA'}: parada={resultado.stopped_by}, "
        f"herramientas={[r.name for r in resultado.runs]}"
    )
    return 0 if (ok1 and ok2) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
