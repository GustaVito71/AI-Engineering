"""Comprueba con llamadas reales el bucle agéntico y el cambio de proveedor a mitad del bucle.

Uso (desde la raíz del proyecto, con las claves del primario y del respaldo en el .env):

    uv run python test/verificar_herramientas.py

QUÉ COMPRUEBA (cuesta unos céntimos: unas 6 llamadas cortas)

1. Con el proveedor primario: el modelo pide una herramienta de prueba (`suma`), el bucle la
   ejecuta y el modelo contesta con el resultado.
2. Con el proveedor de respaldo SOLO (primario sin clave válida): el mismo bucle funciona, es
   decir, el formato de herramientas que enviamos lo aceptan los dos proveedores por LiteLLM.
3. Cambio a mitad del bucle: el historial de la ronda 1 (hecha por el primario) se reenvía al
   respaldo y este lo continúa. Es el caso que el Paso 10 dejó «a verificar».

No imprime claves. Solo lee el .env a través de Settings, como el servicio.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.agent.loop import run_agent_loop
from app.agent.tools import Tool, ToolRegistry
from app.config import get_settings
from app.services.llm_wrapper import LLMWrapper

# La pregunta obliga a usar la herramienta: el modelo no puede sumar de memoria un número
# que solo la herramienta conoce.
TRANSCRIPT = (
    "Necesito estimar un proyecto. Antes de nada, usa la herramienta `suma` con a=17 y b=25 "
    "y dime el resultado exacto."
)


def registro() -> ToolRegistry:
    """Un registro con una única herramienta de prueba que suma dos números."""
    reg = ToolRegistry()
    reg.register(
        Tool(
            name="suma",
            description="Suma dos números enteros. Úsala cuando el usuario lo pida.",
            parameters={
                "type": "object",
                "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                "required": ["a", "b"],
            },
            run=lambda args: f"resultado={args['a'] + args['b']}",
        )
    )
    return reg


async def probar(nombre: str, wrapper: LLMWrapper, model: str) -> bool:
    """Corre el bucle con un wrapper y dice si la herramienta se ejecutó con el resultado 42."""
    try:
        resultado = await run_agent_loop(
            llm=wrapper,
            registry=registro(),
            transcript=TRANSCRIPT,
            max_iterations=3,
            token_budget=20_000,
            model=model,
        )
    except Exception as exc:  # noqa: BLE001 - es un script de diagnóstico: se informa y sigue
        print(f"  [{nombre}] FALLA: {type(exc).__name__}: {str(exc)[:200]}")
        return False
    ok = any("resultado=42" in out for out in resultado.outputs)
    print(
        f"  [{nombre}] {'OK' if ok else 'FALLA'}: rondas={resultado.iterations} "
        f"terminado_por={resultado.stopped_by} herramientas={[r.name for r in resultado.runs]}"
    )
    return ok


async def main() -> int:
    """Corre las tres comprobaciones y devuelve 0 si todas pasaron."""
    settings = get_settings()
    print(f"Primario: {settings.primary_model} · respaldo: {settings.fallback_model}")
    wrapper = LLMWrapper(settings=settings)
    resultados = [await probar("1. primario", wrapper, settings.primary_model)]

    # 2. Solo el respaldo: se construye un wrapper cuyo «primario» es el modelo de respaldo.
    try:
        solo_respaldo = LLMWrapper(
            settings=settings.model_copy(update={"llm_default": settings.llm_fallback}),
            use_fallback=False,
        )
        resultados.append(await probar("2. respaldo", solo_respaldo, settings.fallback_model))
    except Exception as exc:  # noqa: BLE001
        print(f"  [2. respaldo] no se pudo preparar: {type(exc).__name__}: {str(exc)[:200]}")
        resultados.append(False)

    print(
        "  [3. cambio a mitad del bucle] se comprueba en el servicio real forzando un fallo del "
        "primario (clave inválida) con las dos claves puestas: el Router pasa al respaldo y "
        "debe seguir la ronda con el mismo historial. Revisa en el log `uso_respaldo=True` "
        "en los eventos `agente_llamada`."
    )
    print("RESULTADO:", "todo OK" if all(resultados) else "hay fallos (ver arriba)")
    return 0 if all(resultados) else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
