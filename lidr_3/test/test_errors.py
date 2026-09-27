"""Tests de la regla neutral de fallback (`LLMProviderError.can_fallback`).

La regla de oro del diseño: la decisión de fallback NO depende del SDK del
proveedor. Si un adaptador tradujo bien su error a `status_code`, la regla
funciona igual para OpenAI, Anthropic o un tercero. Por eso estos tests
construyen el error normalizado directamente (no lanzan excepciones de un SDK
concreto): prueban la regla, no la librería de un proveedor.

Caso por caso (ver el docstring de la propiedad):
- None  -> no hubo respuesta HTTP (red/timeout): otro proveedor puede
- 401   -> key del primario inválida/vencida: el fallback tiene OTRA key
- 429   -> límite/cuota del primario: otro proveedor procesa
- >=500 -> el servicio del primario está caído
- 4xx   -> el MISMO request va a fallar igual: no gastar el fallback
"""

from __future__ import annotations

import pytest

from app.providers import LLMProviderError


def _error(status_code: int | None) -> LLMProviderError:
    return LLMProviderError(
        provider="openai",
        detail="fallo de prueba",
        status_code=status_code,
    )


@pytest.mark.parametrize("status_code", [None, 401, 429, 500, 502, 503])
def test_disparan_fallback(status_code):
    assert _error(status_code).can_fallback is True


@pytest.mark.parametrize("status_code", [400, 403, 404, 422])
def test_no_disparan_fallback(status_code):
    assert _error(status_code).can_fallback is False


def test_sin_status_code_es_red_o_timeout_y_dispara_fallback():
    """`APIConnectionError` de ambos SDKs llega sin status HTTP: no hubo
    respuesta. Otro proveedor con red sana puede responder."""
    error = LLMProviderError(provider="openai", detail="se cayó la red")
    assert error.status_code is None
    assert error.can_fallback is True
