"""Tests de la elección de proveedores y modelos (LLM_DEFAULT, LLM_FALLBACK, *_MODEL).

`primary_model` y `fallback_model` son propiedades derivadas: el modelo de cada
proveedor sale de `<PROVEEDOR>_MODEL`. Sin red y sin `.env` (ver conftest.py).
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError


def test_defaults_derive_litellm_model_names(settings_predeterminada) -> None:
    """Sin configurar nada, el principal es OpenAI (gpt-4o-mini) y el respaldo es Anthropic (claude-haiku-5-5)."""
    s = settings_predeterminada
    assert (s.primary_provider, s.fallback_provider) == ("openai", "anthropic")
    assert s.primary_model == "openai/gpt-4o-mini"
    assert s.fallback_model == "anthropic/claude-haiku-5-5"


def test_model_comes_from_the_variable_of_its_provider(construir_settings) -> None:
    """El modelo de cada proveedor sale de su propia variable (OPENAI_MODEL o ANTHROPIC_MODEL), sea cual sea su rol."""
    s = construir_settings(
        llm_default="anthropic",
        llm_fallback="openai",
        anthropic_model="claude-x",
        openai_model="gpt-y",
    )
    assert s.primary_model == "anthropic/claude-x"
    assert s.fallback_model == "openai/gpt-y"


def test_read_from_the_environment(monkeypatch) -> None:
    """LLM_DEFAULT, LLM_FALLBACK y OPENAI_MODEL se leen del entorno."""
    from app.config import Settings

    monkeypatch.setenv("LLM_DEFAULT", "anthropic")
    monkeypatch.setenv("LLM_FALLBACK", "openai")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-z")
    s = Settings()
    assert s.primary_provider == "anthropic"
    assert s.fallback_model == "openai/gpt-z"


def test_empty_values_fall_back_to_defaults(monkeypatch) -> None:
    """Un `.env` copiado de `.env.example` con variables vacías no rompe el arranque."""
    from app.config import Settings

    for variable in ("LLM_DEFAULT", "LLM_FALLBACK", "OPENAI_MODEL", "ANTHROPIC_MODEL"):
        monkeypatch.setenv(variable, "")
    s = Settings()
    assert s.primary_model == "openai/gpt-4o-mini"
    assert s.fallback_model == "anthropic/claude-haiku-5-5"


def test_same_provider_in_both_roles_is_rejected(construir_settings) -> None:
    """Principal y respaldo con el mismo proveedor no dan respaldo real: el arranque lo rechaza."""
    with pytest.raises(ValidationError, match="LLM_DEFAULT y LLM_FALLBACK son iguales"):
        construir_settings(llm_default="openai", llm_fallback="openai")


@pytest.mark.parametrize("campo", ["llm_default", "llm_fallback"])
def test_unsupported_provider_is_rejected(construir_settings, campo) -> None:
    """Un proveedor que el servicio no soporta (aquí gemini) se rechaza, en cualquiera de los dos roles."""
    with pytest.raises(ValidationError, match="no es un proveedor soportado"):
        construir_settings(**{campo: "gemini"})
