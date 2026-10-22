"""Comprueba con llamadas reales que el modelo usa la búsqueda web dentro del bucle (Paso 11).

Uso (desde la raíz del proyecto, con las claves del modelo y de búsqueda en el .env):

    uv run python test/verificar_busqueda_agente.py

QUÉ COMPRUEBA (cuesta unos céntimos: una búsqueda y 2 o 3 llamadas cortas al modelo)

1. La herramienta `web_search` se construye con las claves del .env y busca de verdad
   (servicio, número de resultados y tiempo).
2. Con una pregunta que depende de datos actuales, el modelo PIDE la búsqueda dentro del bucle
   agéntico y recibe los resultados.
3. Con una pregunta que no la necesita, el modelo NO la pide.

No imprime claves. Solo lee el .env a través de Settings, como el servicio.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.agent.loop import run_agent_loop
from app.agent.tools import ToolRegistry
from app.config import get_settings
from app.observability import start_turn_cost
from app.search.tool import build_tool_from_settings
from app.services.llm_wrapper import LLMWrapper

# Una consulta que necesita datos actuales y otra que no.
PREGUNTA_ACTUAL = (
    "Estima un proyecto que migra una API a la última versión estable de FastAPI. "
    "Necesito saber cuál es la versión estable actual antes de estimar."
)
PREGUNTA_SIN_BUSQUEDA = "Estima una página web de una sola pantalla con un formulario de contacto."


async def correr(wrapper: LLMWrapper, registro: ToolRegistry, texto: str, model: str):
    """Ejecuta el bucle con una pregunta y devuelve su resultado."""
    start_turn_cost()
    return await run_agent_loop(
        llm=wrapper,
        registry=registro,
        transcript=texto,
        attachments_block="",
        max_iterations=get_settings().agent_max_iterations,
        token_budget=get_settings().turn_token_budget,
        model=model,
    )


async def main() -> int:
    """Corre las tres comprobaciones y devuelve 0 si todas pasan."""
    settings = get_settings()
    herramienta = build_tool_from_settings(settings)
    if herramienta is None:
        print("SIN CLAVES: define WEB_TAVILY_KEY o WEB_FIRECRAWL_KEY en el .env")
        return 1
    registro = ToolRegistry()
    registro.register(herramienta)

    print("1. Búsqueda directa de la herramienta")
    texto = await herramienta.run({"query": "FastAPI latest stable version"})
    ok1 = texto.startswith("Resultados de búsqueda")
    print(f"   {'OK' if ok1 else 'FALLA'}: {texto[:120]!r}")

    wrapper = LLMWrapper(settings=settings)
    model = settings.primary_model
    print("2. Pregunta que depende de datos actuales (debe pedir la búsqueda)")
    r = await correr(wrapper, registro, PREGUNTA_ACTUAL, model)
    ok2 = any(run.name == "web_search" for run in r.runs)
    print(
        f"   {'OK' if ok2 else 'FALLA'}: parada={r.stopped_by}, herramientas={[x.name for x in r.runs]}"
    )

    print("3. Pregunta que no la necesita (no debe pedirla)")
    r = await correr(wrapper, registro, PREGUNTA_SIN_BUSQUEDA, model)
    ok3 = not any(run.name == "web_search" for run in r.runs)
    print(
        f"   {'OK' if ok3 else 'AVISO'}: parada={r.stopped_by}, herramientas={[x.name for x in r.runs]}"
    )

    return 0 if (ok1 and ok2) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
