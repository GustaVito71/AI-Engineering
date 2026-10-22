"""Tests de integración de sesiones (Paso 7): tres casos de punta a punta con `httpx.AsyncClient`.

Se levanta la aplicación real (rutas, router, almacén de sesiones, servicio de estimación y
construcción de prompts) y solo se reemplazan los dos puntos que llaman a un proveedor: el
wrapper del LLM y el extractor de metadata. Sin red ni Redis.

1. Una sesión enlaza dos peticiones: la `project_metadata` se actualiza y el segundo turno la
   recibe en el system prompt.
2. Una petición con un PDF adjunto: el texto del PDF llega al prompt y, sin adjunto, no
   aparece. (Con un modelo falso solo se puede comprobar que el texto llega; que la
   estimación del modelo real cambie es una prueba cualitativa que se hace a mano.)
3. Ocho turnos en la misma sesión: el historial enviado al LLM nunca supera `MAX_TURNS`.
"""

from __future__ import annotations

import httpx
import pytest
from test_attachments import hacer_pdf
from test_estimate_structured_endpoint import _WrapperFalso
from test_sessions_memory_endpoint import MEMORIA_1, MEMORIA_2, TURNO_1, TURNO_2, _ExtractorFalso

from app.config import get_settings
from app.dependencies import get_llm_wrapper, get_metadata_extractor
from app.main import create_app

FORM = {"project_type": "web_saas", "detail_level": "detailed", "output_format": "phases_table"}
MAX_TURNS = 3


@pytest.fixture
async def montar(monkeypatch):
    """Fábrica: arma la app con wrapper y extractor falsos y devuelve `(cliente, wrapper, id)`."""
    clientes: list[httpx.AsyncClient] = []

    async def _montar(extractor):
        """Crea la app de prueba y una sesión nueva, hablando con ella por ASGI (sin red)."""
        monkeypatch.setenv("REDIS_URL", "")
        monkeypatch.setenv("MAX_TURNS", str(MAX_TURNS))
        get_settings.cache_clear()
        app = create_app()
        wrapper = _WrapperFalso()
        app.dependency_overrides[get_llm_wrapper] = lambda: wrapper
        app.dependency_overrides[get_metadata_extractor] = lambda: extractor
        cliente = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")
        clientes.append(cliente)
        session_id = (await cliente.post("/sessions")).json()["session_id"]
        return cliente, wrapper, session_id

    yield _montar
    for cliente in clientes:
        await cliente.aclose()
    get_settings.cache_clear()


async def estimar(cliente, session_id, transcript, **extra):
    """Envía un turno como formulario multipart."""
    return await cliente.post(
        f"/sessions/{session_id}/estimate", data={"transcript": transcript, **FORM}, **extra
    )


async def test_una_sesion_enlaza_dos_peticiones_y_actualiza_la_metadata(montar) -> None:
    cliente, wrapper, sid = await montar(_ExtractorFalso([MEMORIA_1, MEMORIA_2]))

    assert (await estimar(cliente, sid, TURNO_1)).status_code == 200
    estado = (await cliente.get(f"/sessions/{sid}")).json()
    assert estado["project_metadata"]["project_name"] == "Bookflow"
    assert estado["project_metadata"]["mentioned_technologies"] == ["Rails"]

    assert (await estimar(cliente, sid, TURNO_2)).status_code == 200
    # El segundo turno viajó con la memoria del primero en el system prompt...
    assert "Nombre del proyecto: Bookflow" in wrapper.llamadas[1]["system_prompt"]
    assert "Rails" in wrapper.llamadas[1]["system_prompt"]
    # ...y la revisión explícita quedó anotada: Rails pasó a descartadas.
    metadata = (await cliente.get(f"/sessions/{sid}")).json()["project_metadata"]
    assert metadata["mentioned_technologies"] == ["Node"]
    assert metadata["rejected_options"] == ["Rails"]


async def test_el_texto_del_pdf_adjunto_llega_al_prompt(montar) -> None:
    cliente, wrapper, sid = await montar(_ExtractorFalso())
    pdf = hacer_pdf(["El sistema debe integrarse con el ERP LEGACYSOFT.", "Segunda pagina."])

    sin_adjunto = await estimar(cliente, sid, TURNO_1)
    con_adjunto = await estimar(
        cliente, sid, TURNO_2, files={"attachments": ("requisitos.pdf", pdf, "application/pdf")}
    )

    assert sin_adjunto.status_code == con_adjunto.status_code == 200
    assert "LEGACYSOFT" not in wrapper.llamadas[0]["user_prompt"]
    prompt = wrapper.llamadas[1]["user_prompt"]
    assert "<attachments>" in prompt
    assert "requisitos.pdf" in prompt
    assert "LEGACYSOFT" in prompt


async def test_ocho_turnos_nunca_envian_mas_de_max_turns_pares(montar) -> None:
    cliente, wrapper, sid = await montar(_ExtractorFalso())

    for numero in range(1, 9):
        respuesta = await estimar(cliente, sid, f"Cliente: mensaje número {numero} del proyecto.")
        assert respuesta.status_code == 200

    assert len(wrapper.llamadas) == 8
    for llamada in wrapper.llamadas:
        assert len(llamada["history"]) <= MAX_TURNS * 2
    # El historial se llenó y desde el cuarto turno la ventana descarta los más viejos.
    assert [len(c["history"]) for c in wrapper.llamadas] == [0, 2, 4, 6, 6, 6, 6, 6]
    assert (await cliente.get(f"/sessions/{sid}")).json()["turns"] == MAX_TURNS
