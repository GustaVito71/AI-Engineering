"""Tests de `app/agent/context.py` (Paso 10): bucle + presupuesto dentro del flujo de un turno.

QUÉ SE COMPRUEBA

- Con el bucle apagado, sin herramientas o sin modelo, `gather_tool_context` no hace nada.
- Con el bucle encendido y una herramienta, los resultados salen en un bloque `<tool_results>`.
- `fit_turn_context` no toca nada si todo cabe; si no, recorta primero los resultados de
  herramientas y después los adjuntos, avisa al usuario y emite `presupuesto_recortado`.
"""

from __future__ import annotations

import pytest
from structlog.testing import capture_logs
from test_agent_loop import ModeloGuionado, pide, responde

from app.agent.context import TOOL_CONTEXT_HEADER, fit_turn_context, gather_tool_context
from app.agent.tools import Tool, ToolRegistry
from app.config import get_settings


@pytest.fixture
def ajustes(monkeypatch):
    """Una fábrica de `Settings` con las variables de entorno indicadas."""

    def _hacer(**entorno):
        """Fija el entorno y reconstruye la configuración."""
        for variable, valor in entorno.items():
            monkeypatch.setenv(variable, str(valor))
        get_settings.cache_clear()
        return get_settings()

    yield _hacer
    get_settings.cache_clear()


def _registro() -> ToolRegistry:
    """Un registro con una herramienta que devuelve un dato fijo."""
    registro = ToolRegistry()
    registro.register(
        Tool(
            name="precios",
            description="Consulta precios.",
            parameters={"type": "object", "properties": {}},
            run=lambda args: "tarifa media 55 EUR/h",
        )
    )
    return registro


async def test_con_el_bucle_apagado_no_hace_nada(ajustes) -> None:
    """`AGENT_LOOP_ENABLED=false` (por defecto): bloque vacío y el modelo no se toca."""
    modelo = ModeloGuionado(responde())
    bloque, resultado = await gather_tool_context(
        settings=ajustes(AGENT_LOOP_ENABLED="false"),
        registry=_registro(),
        llm=modelo,
        transcript="proyecto",
        attachments_block="",
    )
    assert (bloque, resultado) == ("", None)
    assert modelo.llamadas == []


async def test_sin_herramientas_o_sin_modelo_no_hace_nada(ajustes) -> None:
    """Encendido pero con registro vacío (o sin modelo): tampoco hay llamadas."""
    ajuste = ajustes(AGENT_LOOP_ENABLED="true")
    vacio = await gather_tool_context(
        settings=ajuste,
        registry=ToolRegistry(),
        llm=ModeloGuionado(responde()),
        transcript="p",
        attachments_block="",
    )
    sin_modelo = await gather_tool_context(
        settings=ajuste, registry=_registro(), llm=None, transcript="p", attachments_block=""
    )
    assert vacio == ("", None)
    assert sin_modelo == ("", None)


async def test_con_el_bucle_encendido_devuelve_los_resultados_como_bloque(ajustes) -> None:
    """Se ejecuta la herramienta y el bloque lleva la cabecera de datos y el resultado."""
    modelo = ModeloGuionado(pide(("precios", {})), responde("listo"))
    bloque, resultado = await gather_tool_context(
        settings=ajustes(AGENT_LOOP_ENABLED="true"),
        registry=_registro(),
        llm=modelo,
        transcript="Necesito precios",
        attachments_block="",
    )
    assert bloque.startswith(TOOL_CONTEXT_HEADER)
    assert "tarifa media 55 EUR/h" in bloque
    assert bloque.endswith("</tool_results>")
    assert [r.name for r in resultado.runs] == ["precios"]


async def test_si_el_modelo_no_pide_nada_el_bloque_queda_vacio(ajustes) -> None:
    """«Sin consultas necesarias»: no hay resultados que añadir al contexto."""
    bloque, resultado = await gather_tool_context(
        settings=ajustes(AGENT_LOOP_ENABLED="true"),
        registry=_registro(),
        llm=ModeloGuionado(responde("Sin consultas necesarias")),
        transcript="p",
        attachments_block="",
    )
    assert bloque == ""
    assert resultado.final_text == "Sin consultas necesarias"


def test_el_bucle_respeta_los_valores_de_configuracion(ajustes) -> None:
    """Los tres valores nuevos tienen sus defaults y se leen del entorno."""
    por_defecto = ajustes()
    assert por_defecto.agent_loop_enabled is False
    assert por_defecto.agent_max_iterations == 4
    assert por_defecto.turn_token_budget == 60_000
    cambiado = ajustes(AGENT_MAX_ITERATIONS="2", TURN_TOKEN_BUDGET="9000")
    assert (cambiado.agent_max_iterations, cambiado.turn_token_budget) == (2, 9000)


@pytest.mark.parametrize(
    ("variable", "valor"), [("AGENT_MAX_ITERATIONS", "0"), ("TURN_TOKEN_BUDGET", "500")]
)
def test_valores_invalidos_hacen_fallar_el_arranque(ajustes, variable, valor) -> None:
    """Un tope de rondas menor que 1 o un presupuesto absurdo no arrancan."""
    with pytest.raises(ValueError, match=variable):
        ajustes(**{variable: valor})


def test_vacio_vale_el_valor_por_defecto(ajustes) -> None:
    """Como el resto de la configuración, una variable vacía es «no configurada»."""
    ajuste = ajustes(AGENT_MAX_ITERATIONS="", TURN_TOKEN_BUDGET="")
    assert (ajuste.agent_max_iterations, ajuste.turn_token_budget) == (4, 60_000)


def test_si_todo_cabe_no_se_toca_nada(ajustes) -> None:
    """Contexto pequeño: mismos textos y ningún aviso."""
    ajuste = ajustes()
    ajustado = fit_turn_context(
        settings=ajuste,
        transcript="proyecto",
        history_text="historial",
        attachments_block="adjunto corto",
        tools_block="resultado corto",
    )
    assert ajustado.attachments_block == "adjunto corto"
    assert ajustado.tools_block == "resultado corto"
    assert ajustado.warnings == []


def test_si_no_cabe_se_recortan_primero_las_herramientas_y_se_avisa(ajustes) -> None:
    """Con presupuesto justo, los adjuntos caben y las herramientas se recortan."""
    ajuste = ajustes(TURN_TOKEN_BUDGET="7000")  # 6.000 de reserva del sistema + 1.000
    with capture_logs() as logs:
        ajustado = fit_turn_context(
            settings=ajuste,
            transcript="proyecto",
            history_text="",
            attachments_block="adjunto " * 200,
            tools_block="dato " * 5000,
        )
    assert ajustado.attachments_block == "adjunto " * 200
    assert "recortado por el presupuesto" in ajustado.tools_block
    assert len(ajustado.warnings) == 1
    assert "resultados de herramientas" in ajustado.warnings[0]
    evento = next(e for e in logs if e["event"] == "presupuesto_recortado")
    assert evento["recortados"] == ["herramientas"]


def test_si_aun_asi_no_cabe_tambien_se_recortan_los_adjuntos(ajustes) -> None:
    """Un adjunto enorme se recorta también, y el aviso lo menciona."""
    ajuste = ajustes(TURN_TOKEN_BUDGET="7000")
    ajustado = fit_turn_context(
        settings=ajuste,
        transcript="proyecto",
        history_text="",
        attachments_block="adjunto " * 5000,
        tools_block="dato " * 100,
    )
    assert "recortado por el presupuesto" in ajustado.attachments_block
    assert ajustado.tools_block == ""
    assert any("los adjuntos" in w for w in ajustado.warnings)
