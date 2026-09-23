"""Estimación del coste de una llamada LLM usando la base de precios LLMPrice.

¿De dónde salen los precios?
- Las APIs no los devuelven; cada proveedor publica precio por millón de tokens
  (#/1M) de entrada y salida.
- LLMPrice trae un snapshot local de esos precios (funciona offline). La versión
  del paquete codifica la fecha de ese snapshot (p. ej. 2026.4.3 = 3 de abril
  de 2026).

Misma implementación que lidr_1 (providers/pricing.py), con la misma firma
(coste, nota): la nota explica la fuente o, si coste es None, POR QUÉ no se
pudo estimar.
"""

import importlib.metadata  # para leer la versión instalada del paquete

from llmprice import LLMPrice

# Se crea UNA vez y se reutiliza (el snapshot se carga en memoria)
_price_db = LLMPrice()
_snapshot_version = importlib.metadata.version("llmprice-kit")

_MESES = [
    "enero", "febrero", "marzo", "abril", "mayo", "junio",
    "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre",
]


def _snapshot_label() -> str:
    """Fecha legible del snapshot, p. ej. '2026.4.3 (de abril)'."""
    try:
        _, mes, _ = _snapshot_version.split(".")  # formato AAAA.M.D
        return f"{_snapshot_version} (de {_MESES[int(mes) - 1]})"
    except (ValueError, IndexError):
        return _snapshot_version  # si el formato no es el esperado, solo la versión


def estimate_cost(model: str, usage: dict | None) -> tuple[float | None, str]:
    """Devuelve (coste_estimado_en_USD, nota).

    La nota explica la fuente o, si coste es None, POR QUÉ no se pudo estimar
    (modelo ausente del snapshot, o falta de metadatos de tokens).

    Fórmula: (input_tokens * precio_input + output_tokens * precio_output) / 1M.
    """
    if not usage:
        return None, "no hay metadatos de tokens en la respuesta"
    try:
        info = _price_db.get(model)  # excepción si el modelo no está en la base
    except KeyError:
        return None, f"LLMPrice no tiene datos para '{model}' en el snapshot {_snapshot_label()}"

    input_tokens = usage.get("input_tokens") or 0
    output_tokens = usage.get("output_tokens") or 0
    cost = (
        input_tokens * info.input_cost_per_1m
        + output_tokens * info.output_cost_per_1m
    ) / 1_000_000
    return cost, f"LLMPrice (snapshot {_snapshot_label()})"