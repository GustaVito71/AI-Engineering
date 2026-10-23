"""Comprueba con llamadas reales si Anthropic acepta `temperature=0.1` por LiteLLM.

Uso (desde la raíz del proyecto, con ANTHROPIC_API_KEY en el .env):

    uv run python test/verificar_temperatura.py

Hace 2 llamadas de 5 tokens por modelo (con y sin `temperature`): coste despreciable.
No imprime la clave. Solo lee el .env a través de Settings, como el servicio.
"""

from __future__ import annotations

import sys
from pathlib import Path

import litellm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings

# Qué es la "temperatura": un número que controla cuánto varía la respuesta de un modelo
# entre una llamada y otra. Bajo (0.1) = respuestas parecidas; alto = más variadas.
# Los modelos que se comparan: el anterior (Haiku 4.5) y el actual (Haiku 5.5).
MODELOS = ("anthropic/claude-haiku-4-5", "anthropic/claude-haiku-5-5")


def probar(modelo: str, api_key: str, **extra: object) -> str:
    """Hace una llamada mínima al modelo y dice si la aceptó (`ACEPTA`) o no (`RECHAZA`).

    `extra` son parámetros opcionales de la llamada (aquí, `temperature`). Si el
    proveedor la rechaza, se devuelve el motivo en vez de interrumpir el script.
    """
    try:
        litellm.completion(
            model=modelo,
            api_key=api_key,
            messages=[{"role": "user", "content": "Responde solo: ok"}],
            max_tokens=5,
            **extra,
        )
        return "ACEPTA"
    except Exception as e:  # noqa: BLE001 - se muestra el tipo y el motivo
        return f"RECHAZA ({type(e).__name__}): {str(e).splitlines()[0][:140]}"


def main() -> None:
    """Prueba cada modelo con y sin `temperature` e imprime el resultado de cada caso."""
    # La clave se lee igual que en el servicio (a través de Settings) y nunca se imprime.
    clave = get_settings().active_api_key("anthropic")
    if not clave:
        raise SystemExit("Falta ANTHROPIC_API_KEY en el .env.")
    for modelo in MODELOS:
        print(modelo)
        print("  con temperature=0.1:", probar(modelo, clave, temperature=0.1))
        print("  sin temperature    :", probar(modelo, clave))


if __name__ == "__main__":
    main()
