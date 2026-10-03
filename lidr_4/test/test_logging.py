"""Tests de configure_logging con los loggers de LiteLLM.

- Ningún nivel escribe el prompt: LiteLLM, en DEBUG, vuelca los parámetros de
  cada llamada con los mensajes completos.
- Sus mensajes salen una sola vez, con el formato de structlog, y siguen sin
  mostrar claves de API.
"""

from __future__ import annotations

import logging

import litellm
import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import LOGGERS_DE_LITELLM, configure_logging, create_app

MARCADOR = "MARCADOR-DESCRIPCION-DEL-CLIENTE"
CLAVE_FALSA = "sk-proj-ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

BODY = {
    "description": f"{MARCADOR}: un SaaS B2B para gestionar préstamos de equipos.",
    "project_type": "web_saas",
    "detail_level": "medium",
    "output_format": "phases_table",
}


@pytest.fixture
def restaurar_logging():
    """configure_logging toca el logging global: se deja como estaba."""
    loggers = [logging.getLogger(), *(logging.getLogger(n) for n in LOGGERS_DE_LITELLM)]
    estado = [(lg, lg.level, lg.handlers[:], lg.filters[:]) for lg in loggers]
    yield
    for logger, nivel, handlers, filtros in estado:
        logger.setLevel(nivel)
        logger.handlers = handlers
        logger.filters = filtros


@pytest.mark.parametrize("nombre", LOGGERS_DE_LITELLM)
def test_litellm_loggers_are_pinned_to_warning(restaurar_logging, nombre) -> None:
    configure_logging("DEBUG")
    assert logging.getLogger(nombre).getEffectiveLevel() == logging.WARNING


def test_debug_never_writes_the_prompt(restaurar_logging, monkeypatch, capfd) -> None:
    """De punta a punta: una estimación con LOG_LEVEL=DEBUG no deja el prompt
    en stdout ni en stderr (LiteLLM escribe por los dos)."""
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-anthropic")
    monkeypatch.setenv("REDIS_URL", "")
    get_settings.cache_clear()

    original = litellm.Router.acompletion

    async def sin_red(self, **kwargs):
        return await original(self, **kwargs, mock_response="ok")

    monkeypatch.setattr(litellm.Router, "acompletion", sin_red)

    with TestClient(create_app()) as c:
        r = c.post("/api/v1/estimate", json=BODY)
    assert r.status_code == 200

    salida = capfd.readouterr()
    assert MARCADOR not in salida.out
    assert MARCADOR not in salida.err


def test_litellm_warning_is_written_once_in_structlog_format(restaurar_logging, capfd) -> None:
    configure_logging("INFO")
    # Sin handler propio: solo escribe el de la raíz. Se comprueba aparte porque
    # con LITELLM_LOG=ERROR o más alto el handler de LiteLLM calla las
    # advertencias y la salida no mostraría el duplicado.
    for nombre in LOGGERS_DE_LITELLM:
        assert logging.getLogger(nombre).handlers == []
    logging.getLogger("LiteLLM").warning("AVISO-DE-PRUEBA")

    salida = capfd.readouterr()
    todo = salida.out + salida.err
    assert todo.count("AVISO-DE-PRUEBA") == 1
    assert "LiteLLM:WARNING" not in todo  # el formato propio de LiteLLM


@pytest.mark.parametrize("nombre", [*LOGGERS_DE_LITELLM, "LiteLLM Proxy.stdout"])
def test_litellm_messages_still_hide_api_keys(restaurar_logging, capfd, nombre) -> None:
    """Sin el handler de LiteLLM, sus filtros siguen borrando las claves. Se
    configura dos veces, como cuando el lifespan corre más de una vez."""
    configure_logging("INFO")
    configure_logging("INFO")
    logging.getLogger(nombre).warning(f"fallo con api_key={CLAVE_FALSA}")

    salida = capfd.readouterr()
    todo = salida.out + salida.err
    assert "fallo con" in todo
    assert CLAVE_FALSA not in todo
