"""Presupuesto de tokens por turno (Paso 10).

QUÉ HACE (en lenguaje llano)

Cada turno manda al modelo varias piezas de texto: el prompt del sistema, lo que dijo el
usuario, el historial, los adjuntos y los resultados de las herramientas. Si juntas son
demasiado grandes, la llamada sale cara y lenta (o falla). Este módulo mide cada pieza en
tokens y, si se pasa del presupuesto, recorta las MENOS importantes primero. El orden de
importancia, de más a menos, es:

1. Prompt del sistema (reserva fija; no se recorta).
2. Lo que acaba de escribir el usuario (nunca se recorta).
3. Historial de la conversación (ya viene acotado por la ventana deslizante; no se recorta).
4. Adjuntos del usuario.
5. Resultados de herramientas (lo primero que se recorta).

Los tokens se miden con `litellm.token_counter`. Si el contador falla para un modelo
desconocido, se usa una aproximación (4 caracteres por token) en lugar de romper la petición.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import litellm

# Tokens que se reservan para el prompt del sistema del actor (reglas, tarifas, ejemplos):
# se mide una vez a mano y se deja un margen, en vez de renderizarlo en cada turno.
SYSTEM_PROMPT_TOKEN_RESERVE = 6_000
# Aviso que se añade al final de un texto recortado, para que el modelo sepa que falta algo.
TRUNCATION_MARK = "\n[…recortado por el presupuesto de tokens]"
# Aproximación de último recurso cuando el contador no conoce el modelo.
CHARS_PER_TOKEN_FALLBACK = 4


def count_tokens(text: str, model: str | None = None) -> int:
    """Tokens de un texto según LiteLLM; sin contador disponible, una aproximación."""
    if not text:
        return 0
    try:
        return int(litellm.token_counter(model=model or "gpt-4o-mini", text=text))
    except Exception:  # noqa: BLE001 - medir tokens es observabilidad, no debe romper el turno
        return math.ceil(len(text) / CHARS_PER_TOKEN_FALLBACK)


def truncate_to_tokens(text: str, max_tokens: int, model: str | None = None) -> tuple[str, bool]:
    """Recorta un texto para que quepa en `max_tokens`; devuelve (texto, ¿se recortó?).

    Busca por bisección el trozo inicial más largo que, con el aviso de recorte, no pase del
    máximo. Un máximo de 0 o menos deja el texto vacío.
    """
    if count_tokens(text, model) <= max_tokens:
        return text, False
    if max_tokens <= 0:
        return "", True
    low, high = 0, len(text)
    while low < high:
        mid = (low + high + 1) // 2
        if count_tokens(text[:mid] + TRUNCATION_MARK, model) <= max_tokens:
            low = mid
        else:
            high = mid - 1
    return (text[:low] + TRUNCATION_MARK if low else ""), True


@dataclass(frozen=True)
class Block:
    """Una pieza de texto del contexto. `protected` = nunca se recorta."""

    name: str
    text: str
    protected: bool = False


@dataclass(frozen=True)
class FitResult:
    """Cómo quedó una pieza tras ajustar al presupuesto."""

    name: str
    text: str
    tokens: int
    truncated: bool


def fit_blocks(
    blocks: list[Block],
    *,
    budget: int,
    model: str | None = None,
    reserved: int = 0,
) -> list[FitResult]:
    """Ajusta las piezas al presupuesto, recortando primero las últimas de la lista.

    `blocks` va ordenado de MÁS a MENOS importante. Las piezas `protected` se conservan
    enteras y se descuentan primero (si solas ya superan el presupuesto, se conservan igual
    y las demás quedan vacías). Del resto se reparte lo que queda en orden: cada pieza toma
    lo que necesita o lo que quede. `reserved` son tokens que ya están gastados fuera de las
    piezas (el prompt del sistema). Devuelve los resultados en el mismo orden.
    """
    sizes = {b.name: count_tokens(b.text, model) for b in blocks}
    remaining = budget - reserved - sum(sizes[b.name] for b in blocks if b.protected)
    results: list[FitResult] = []
    for block in blocks:
        if block.protected:
            results.append(FitResult(block.name, block.text, sizes[block.name], False))
            continue
        text, truncated = truncate_to_tokens(block.text, max(remaining, 0), model)
        used = count_tokens(text, model) if truncated else sizes[block.name]
        remaining -= used
        results.append(FitResult(block.name, text, used, truncated))
    return results
