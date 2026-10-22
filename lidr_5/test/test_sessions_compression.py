"""Tests de punta a punta del Paso 9 en los endpoints de sesiones.

QUÉ SE COMPRUEBA (con la aplicación real; solo se reemplazan los proveedores de LLM)

- Compresión: con una ventana de 1 turno, el turno con un compromiso pasa a las anclas y
  el corriente al resumen; el tercer turno viaja al modelo con resumen, anclas y ventana,
  en ese orden. Vale para `/estimate` y para `/estimate-acb`.
- Si el resumidor falla, la estimación se entrega igual con un aviso y el estado de la
  sesión no se corrompe.
- Tier dinámico: una conversación con confidencialidad llega al prompt con el bloque
  `<audience>` y el estado de la sesión dice qué regla decidió; sin reglas, no hay bloque.
- `GET /sessions/{id}` informa anclas, resumen y audiencia.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from test_acb_critic_boss import indep
from test_estimate_structured_endpoint import _WrapperFalso
from test_sessions_acb import _CriticoFalso
from test_sessions_memory_endpoint import FORM, _ExtractorFalso

from app.config import get_settings
from app.dependencies import (
    get_compression_tools,
    get_critic,
    get_llm_wrapper,
    get_metadata_extractor,
)
from app.main import create_app
from app.sessions.compression import AnchorDetector, CompressionTools
from app.sessions.models import SUMMARY_PREFIX

ANCLA = "Cliente: el alcance está congelado, no se añade nada más al proyecto."
CORRIENTE = "Cliente: ahora queremos un informe mensual de reservas por sala."
TERCERO = "Cliente: ¿cuánto sube la estimación si sumamos notificaciones por correo?"
CUARTO = "Cliente: y si además sumamos un panel de métricas para la dirección."


class _ResumidorFalso:
    """Resumidor de prueba: devuelve un texto fijo, o falla, y anota lo que recibió."""

    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.llamadas: list[list[str]] = []

    async def summarize(self, *, previous_summary, evicted):
        """Anota los mensajes que iban al resumen y devuelve un texto reconocible."""
        self.llamadas.append([m.content for m in evicted])
        if self.error:
            raise self.error
        return "RESUMEN-DE-PRUEBA"


@pytest.fixture
def montar(monkeypatch):
    """Fábrica: app con wrapper, extractor y resumidor falsos, ventana de 1 turno y sesión."""

    def _montar(resumidor, *, critico=None):
        """Devuelve `(cliente, wrapper, session_id)`."""
        monkeypatch.setenv("REDIS_URL", "")
        monkeypatch.setenv("MAX_TURNS", "1")
        get_settings.cache_clear()
        cliente = TestClient(create_app())
        wrapper = _WrapperFalso()
        herramientas = CompressionTools(detector=AnchorDetector(), summarizer=resumidor)
        cliente.app.dependency_overrides[get_llm_wrapper] = lambda: wrapper
        cliente.app.dependency_overrides[get_metadata_extractor] = lambda: _ExtractorFalso()
        cliente.app.dependency_overrides[get_compression_tools] = lambda: herramientas
        if critico is not None:
            cliente.app.dependency_overrides[get_critic] = lambda: critico
        return cliente, wrapper, cliente.post("/sessions").json()["session_id"]

    yield _montar
    get_settings.cache_clear()


def estimar(cliente, sid, transcript, ruta="estimate"):
    """Envía un turno como formulario multipart a `/estimate` o `/estimate-acb`."""
    return cliente.post(f"/sessions/{sid}/{ruta}", data={"transcript": transcript, **FORM})


@pytest.mark.parametrize("ruta", ["estimate", "estimate-acb"])
def test_el_cuarto_turno_viaja_con_resumen_anclas_y_ventana(montar, ruta) -> None:
    resumidor = _ResumidorFalso()
    cliente, wrapper, sid = montar(resumidor, critico=_CriticoFalso(indep("accept")))

    for texto in (ANCLA, CORRIENTE, TERCERO, CUARTO):
        assert estimar(cliente, sid, texto, ruta).status_code == 200

    # El turno 1 sale de la ventana al guardar el 2 (ancla); el 2 sale al guardar el 3 (resumen).
    historial = wrapper.llamadas[3]["history"]
    assert [m["role"] for m in historial] == ["user", "assistant"] * 3
    # 1) resumen, 2) ancla textual (el turno del compromiso), 3) ventana (el turno 3).
    assert historial[0]["content"] == SUMMARY_PREFIX + "RESUMEN-DE-PRUEBA"
    assert historial[2]["content"] == ANCLA
    assert historial[4]["content"] == TERCERO
    # Al resumidor fueron los turnos corrientes (2 y 3); el del compromiso se conservó entero.
    assert [c[0] for c in resumidor.llamadas] == [CORRIENTE, TERCERO]

    estado = cliente.get(f"/sessions/{sid}").json()
    assert estado["anchors"] == 1
    assert estado["has_summary"] is True
    assert estado["turns"] == 1


@pytest.mark.parametrize("ruta", ["estimate", "estimate-acb"])
def test_un_resumidor_roto_no_pierde_la_estimacion_y_avisa(montar, ruta) -> None:
    cliente, _wrapper, sid = montar(
        _ResumidorFalso(error=RuntimeError("proveedor caído")),
        critico=_CriticoFalso(indep("accept")),
    )
    estimar(cliente, sid, CORRIENTE, ruta)

    respuesta = estimar(cliente, sid, TERCERO, ruta)

    assert respuesta.status_code == 200
    assert any("resumen de la conversación" in aviso for aviso in respuesta.json()["warnings"])
    assert cliente.get(f"/sessions/{sid}").json()["has_summary"] is False


@pytest.mark.parametrize("ruta", ["estimate", "estimate-acb"])
def test_la_confidencialidad_activa_la_audiencia_ejecutiva(montar, ruta) -> None:
    cliente, wrapper, sid = montar(_ResumidorFalso(), critico=_CriticoFalso(indep("accept")))

    estimar(cliente, sid, "Cliente: el proyecto está bajo NDA, una web de reservas de salas.", ruta)

    assert "<audience>" in wrapper.llamadas[0]["system_prompt"]
    estado = cliente.get(f"/sessions/{sid}").json()
    assert (estado["tier"], estado["tier_rule"]) == ("executive", "nda_detected")


@pytest.mark.parametrize("ruta", ["estimate", "estimate-acb"])
def test_sin_reglas_no_hay_bloque_de_audiencia(montar, ruta) -> None:
    cliente, wrapper, sid = montar(_ResumidorFalso(), critico=_CriticoFalso(indep("accept")))

    estimar(cliente, sid, CORRIENTE, ruta)

    assert "<audience>" not in wrapper.llamadas[0]["system_prompt"]
    estado = cliente.get(f"/sessions/{sid}").json()
    assert (estado["tier"], estado["tier_rule"]) == ("default", "default")


def test_el_estado_de_una_sesion_nueva_no_trae_anclas_ni_resumen(montar) -> None:
    cliente, _wrapper, sid = montar(_ResumidorFalso())
    estado = cliente.get(f"/sessions/{sid}").json()
    assert estado["anchors"] == 0
    assert estado["has_summary"] is False
    assert estado["tier"] is None and estado["tier_rule"] is None
