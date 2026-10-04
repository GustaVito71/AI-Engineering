"""Slice vertical de punta a punta (WU7): del formulario de Streamlit a la respuesta.

Recorre todas las piezas reales en un solo flujo:

    _armar_payload + _estimar (cliente de Streamlit)
      → FastAPI con su lifespan (cliente de Redis, caché)
      → EstimationRequest → render del prompt (versión por defecto, v2)
      → LLMWrapper → Router de LiteLLM (primario y respaldo)
      → caché exact-match → limpieza de etiquetas → EstimationResponse
      → _estimar → texto, versión y avisos para el formulario

Lo único simulado es el proveedor, con `mock_response` de LiteLLM (texto o
excepción), y Redis, con fakeredis detrás del `Redis.from_url` del lifespan. El
Router, la caché y el endpoint son los reales: si una pieza deja de encajar con
la siguiente, estos tests lo detectan aunque los de cada capa sigan pasando.
"""

from __future__ import annotations

import litellm
import pytest
from fakeredis import FakeAsyncRedis, FakeServer
from fastapi.testclient import TestClient
from redis.asyncio import Redis
from structlog.testing import capture_logs

from app.config import get_settings
from app.main import create_app
from app.routers.estimations import MENSAJE_FALLO_PROVEEDOR, MENSAJE_TIMEOUT
from streamlit_app import (
    _ApiError,
    _armar_payload,
    _detail_texto,
    _estimar,
    _leer_health,
    _opciones_de_version,
)

# `_estimar` pasa un `timeout` al cliente HTTP, que es lo correcto contra la API
# real; TestClient lo ignora y avisa en cada llamada. El aviso no aporta nada acá.
pytestmark = pytest.mark.filterwarnings("ignore:You should not use the 'timeout' argument")

API = "http://testserver"
DESCRIPCION = "App móvil para reservar turnos en el gimnasio municipal, con avisos push."
RESPUESTA_DEL_MODELO = "| Fase | Semanas | Coste (EUR) | Confianza (%) |\nTotales: 120 horas."


def _payload(descripcion: str = DESCRIPCION) -> dict:
    """Lo que envía el formulario de Streamlit, armado con su misma función."""
    return _armar_payload(descripcion, "mobile_app", "medium", "phases_table")


class _Proveedor:
    """Proveedor simulado detrás del Router real.

    Envuelve `Router.acompletion` para agregar `mock_response` y registrar con
    qué argumentos se llamó. `respuesta` puede ser un texto o una excepción.
    """

    def __init__(self) -> None:
        self.respuesta: str | Exception = RESPUESTA_DEL_MODELO
        self.llamadas: list[dict] = []

    def instalar(self, monkeypatch) -> None:
        original = litellm.Router.acompletion
        proveedor = self

        async def acompletion(router, **kwargs):
            proveedor.llamadas.append(kwargs)
            return await original(router, **kwargs, mock_response=proveedor.respuesta)

        monkeypatch.setattr(litellm.Router, "acompletion", acompletion)


@pytest.fixture
def proveedor(monkeypatch) -> _Proveedor:
    p = _Proveedor()
    p.instalar(monkeypatch)
    return p


@pytest.fixture
def stack(monkeypatch, proveedor):
    """Levanta la app real con su lifespan y devuelve una función que la arranca.

    Redis es un fakeredis compartido por todos los clientes que cree el
    lifespan. Sin reintentos para que los casos de fallo no esperen backoff.
    """
    servidor = FakeServer()
    monkeypatch.setattr(
        Redis,
        "from_url",
        lambda *_a, **_k: FakeAsyncRedis(server=servidor, decode_responses=True),
    )

    def _arrancar(**entorno: str) -> TestClient:
        base = {
            "OPENAI_API_KEY": "sk-openai",
            "ANTHROPIC_API_KEY": "sk-anthropic",
            "REDIS_URL": "redis://redis-e2e:6379/0",
            "LLM_MAX_RETRIES": "0",
        }
        for variable, valor in {**base, **entorno}.items():
            monkeypatch.setenv(variable, valor)
        get_settings.cache_clear()
        return TestClient(create_app())

    yield _arrancar
    get_settings.cache_clear()


def _eventos(logs: list[dict], nombre: str) -> list[dict]:
    return [e for e in logs if e["event"] == nombre]


# --- Camino normal -----------------------------------------------------------------


def test_del_formulario_al_proveedor_y_de_vuelta(stack, proveedor) -> None:
    proveedor.respuesta = f"{RESPUESTA_DEL_MODELO}\n</estimation>"

    with stack() as api, capture_logs() as logs:
        texto, version, avisos = _estimar(API, _payload(), api)

    # Lo que recibe el formulario: texto limpio, versión por defecto, sin avisos.
    assert texto == RESPUESTA_DEL_MODELO
    assert version == "v2"
    assert avisos == []

    # Lo que recibió el proveedor: el prompt v2 con la descripción del formulario.
    [llamada] = proveedor.llamadas
    assert llamada["model"] == "openai/gpt-4o-mini"
    sistema, usuario = (m["content"] for m in llamada["messages"])
    assert "Responde siempre en castellano" in sistema
    assert DESCRIPCION in usuario
    assert "Tipo de proyecto: mobile_app." in usuario

    # Lo que quedó en la trazabilidad.
    [evento] = _eventos(logs, "estimacion_completada")
    assert evento["prompt_version"] == "v2"
    assert evento["desde_cache"] is False
    assert evento["uso_respaldo"] is False
    assert evento["coste_usd"] > 0


def test_la_misma_estimacion_sale_de_la_cache(stack, proveedor) -> None:
    with stack() as api, capture_logs() as logs:
        primera = _estimar(API, _payload(), api)
        segunda = _estimar(API, _payload(), api)

    assert primera == segunda
    assert len(proveedor.llamadas) == 1  # la segunda no llegó al proveedor
    completadas = _eventos(logs, "estimacion_completada")
    assert [e["desde_cache"] for e in completadas] == [False, True]
    assert completadas[1]["coste_usd"] == 0.0
    assert completadas[1]["coste_evitado_usd"] == completadas[0]["coste_usd"] > 0


def test_otra_descripcion_vuelve_a_llamar_al_proveedor(stack, proveedor) -> None:
    with stack() as api:
        _estimar(API, _payload(), api)
        _estimar(API, _payload(DESCRIPCION + " Con panel web para el personal."), api)

    assert len(proveedor.llamadas) == 2


def test_la_cache_guarda_el_texto_crudo_y_se_limpia_al_responder(stack, proveedor) -> None:
    """La limpieza de etiquetas está en el endpoint: también se aplica a lo cacheado."""
    proveedor.respuesta = f"<estimation>\n{RESPUESTA_DEL_MODELO}\n</estimation>"

    with stack() as api:
        _estimar(API, _payload(), api)
        texto, _version, _avisos = _estimar(API, _payload(), api)

    assert texto == RESPUESTA_DEL_MODELO


def test_el_selector_de_version_llega_hasta_el_proveedor(stack, proveedor) -> None:
    """Las opciones salen del /health real, y la elegida decide el prompt que se envía."""
    with stack() as api:
        opciones = _opciones_de_version(_leer_health(API, api))
        assert opciones == [("Predeterminada (v2)", None), ("v1", "v1"), ("v2", "v2")]

        _etiqueta, elegida = opciones[1]
        _texto, version, _avisos = _estimar(API, _payload(), api, prompt_version=elegida)

    assert version == "v1"
    sistema = proveedor.llamadas[0]["messages"][0]["content"]
    assert "Always answer in English" in sistema


# --- Configuración ------------------------------------------------------------------


def test_sin_clave_de_respaldo_el_aviso_llega_al_formulario(stack, proveedor) -> None:
    with stack(ANTHROPIC_API_KEY="") as api:
        texto, _version, avisos = _estimar(API, _payload(), api)

    assert texto == RESPUESTA_DEL_MODELO
    [aviso] = avisos
    assert "ANTHROPIC_API_KEY" in aviso


def test_sin_clave_del_primario_el_formulario_recibe_el_503(stack, proveedor) -> None:
    with stack(OPENAI_API_KEY="") as api, pytest.raises(_ApiError) as error:
        _estimar(API, _payload(), api)

    assert error.value.status == 503
    assert "OPENAI_API_KEY" in _detail_texto(error.value.detail)
    assert proveedor.llamadas == []


def test_sin_redis_la_estimacion_sale_igual(stack, proveedor) -> None:
    with stack(REDIS_URL="") as api:
        primera = _estimar(API, _payload(), api)
        segunda = _estimar(API, _payload(), api)

    assert primera == segunda
    assert len(proveedor.llamadas) == 2  # sin caché, las dos van al proveedor


# --- Fallos -------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("excepcion", "codigo", "mensaje"),
    [
        (
            litellm.Timeout(message="detalle interno", model="gpt-4o-mini", llm_provider="openai"),
            504,
            MENSAJE_TIMEOUT,
        ),
        (
            litellm.APIConnectionError(
                message="detalle interno", llm_provider="openai", model="gpt-4o-mini"
            ),
            502,
            MENSAJE_FALLO_PROVEEDOR,
        ),
    ],
    ids=["timeout", "conexion"],
)
def test_un_fallo_del_proveedor_llega_al_formulario_en_castellano(
    stack, proveedor, excepcion, codigo, mensaje
) -> None:
    proveedor.respuesta = excepcion

    with stack() as api, capture_logs() as logs, pytest.raises(_ApiError) as error:
        _estimar(API, _payload(), api)

    assert error.value.status == codigo
    assert _detail_texto(error.value.detail) == mensaje
    assert "detalle interno" not in _detail_texto(error.value.detail)
    [fallo] = _eventos(logs, "estimacion_fallida")
    assert fallo["codigo_http"] == codigo
    assert "detalle interno" in fallo["detalle"]


def test_un_fallo_no_queda_en_la_cache(stack, proveedor) -> None:
    proveedor.respuesta = litellm.APIConnectionError(
        message="caído", llm_provider="openai", model="gpt-4o-mini"
    )
    with stack() as api:
        with pytest.raises(_ApiError):
            _estimar(API, _payload(), api)

        proveedor.respuesta = RESPUESTA_DEL_MODELO
        texto, _version, _avisos = _estimar(API, _payload(), api)

    assert texto == RESPUESTA_DEL_MODELO
    assert len(proveedor.llamadas) == 2


def test_una_entrada_invalida_no_llega_al_proveedor(stack, proveedor) -> None:
    """La API valida aunque el formulario no lo haga (otro cliente, por ejemplo)."""
    with stack() as api, pytest.raises(_ApiError) as error:
        _estimar(API, _payload("corta"), api)

    assert error.value.status == 422
    assert _detail_texto(error.value.detail)
    assert proveedor.llamadas == []
