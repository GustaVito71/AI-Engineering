"""Tests del Paso 5: ventana deslizante en el endpoint, caducidad y tope de sesiones.

QUÉ SE COMPRUEBA

- Cada estimación exitosa anota un par (usuario + asistente) en el historial de la sesión.
- El primer turno viaja sin historial; el siguiente trae el turno anterior.
- Ningún envío al modelo supera `MAX_TURNS` pares de historial, por largo que sea
  el diálogo (criterio «hecho cuando» del plan).
- Una estimación fallida no deja rastro en el historial.
- El mensaje del usuario guardado se recorta, y los adjuntos no entran al historial.
- Caducidad: una sesión sin uso más allá del plazo da 404 (en el almacén y en el
  endpoint); usarla renueva el plazo.
- Tope: con el almacén lleno se desaloja la sesión menos usada.
- La configuración (`SESSION_TTL_SECONDS`, `MAX_SESSIONS`) rechaza valores menores a 1 y
  toma el valor por defecto si queda vacía.

Sin red ni Redis: wrapper y extractor son falsos; el reloj del almacén se controla a mano.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient
from test_estimate_structured_endpoint import _WrapperFalso

from app.config import Settings, get_settings
from app.dependencies import get_llm_wrapper, get_metadata_extractor
from app.main import create_app
from app.routers.sessions import MAX_HISTORY_USER_CHARS
from app.services.llm_wrapper import StructuredOutputError
from app.sessions import SessionNotFoundError, SessionStore

FORM = {"project_type": "web_saas", "detail_level": "detailed", "output_format": "phases_table"}


def _turno(n: int) -> str:
    """Un mensaje de usuario distinguible por número (supera el mínimo de 20 caracteres)."""
    return f"Cliente: mensaje número {n} sobre el proyecto de reservas."


@pytest.fixture
def montar(monkeypatch):
    """Arma la app con un wrapper falso, sin extractor, y devuelve `(cliente, wrapper, id)`."""

    def _montar(max_turns: str = "6", wrapper=None):
        """Crea la app de prueba con la ventana indicada."""
        monkeypatch.setenv("REDIS_URL", "")
        monkeypatch.setenv("MAX_TURNS", max_turns)
        get_settings.cache_clear()
        wrapper = wrapper or _WrapperFalso()
        cliente = TestClient(create_app())
        cliente.app.dependency_overrides[get_llm_wrapper] = lambda: wrapper
        cliente.app.dependency_overrides[get_metadata_extractor] = lambda: None
        session_id = cliente.post("/sessions").json()["session_id"]
        return cliente, wrapper, session_id

    yield _montar
    get_settings.cache_clear()


def estimar(cliente, session_id, transcript, **extra):
    """Envía el formulario multipart de un turno."""
    return cliente.post(
        f"/sessions/{session_id}/estimate", data={"transcript": transcript, **FORM}, **extra
    )


def test_el_primer_turno_viaja_sin_historial_y_el_segundo_trae_el_anterior(montar) -> None:
    cliente, wrapper, sid = montar()
    assert estimar(cliente, sid, _turno(1)).status_code == 200
    assert estimar(cliente, sid, _turno(2)).status_code == 200

    assert wrapper.llamadas[0]["history"] == []
    historial = wrapper.llamadas[1]["history"]
    assert [m["role"] for m in historial] == ["user", "assistant"]
    assert historial[0]["content"] == _turno(1)


def test_cada_turno_exitoso_queda_anotado_en_la_sesion(montar) -> None:
    cliente, _, sid = montar()
    estimar(cliente, sid, _turno(1))
    sesion = cliente.app.state.session_store.get_or_raise(sid)
    assert [m.role for m in sesion.history.messages] == ["user", "assistant"]
    # El asistente se guarda como el resumen de la estimación (equipo y totales).
    assert "Totales:" in sesion.history.messages[1].content


def test_ningun_envio_supera_max_turns_pares_de_historial(montar) -> None:
    cliente, wrapper, sid = montar(max_turns="3")
    for n in range(1, 9):
        assert estimar(cliente, sid, _turno(n)).status_code == 200

    for llamada in wrapper.llamadas:
        assert len(llamada["history"]) <= 3 * 2
    # En el último envío quedan los 3 turnos previos más recientes (5, 6 y 7).
    ultimo = wrapper.llamadas[-1]["history"]
    assert [m["content"] for m in ultimo if m["role"] == "user"] == [
        _turno(5),
        _turno(6),
        _turno(7),
    ]


def test_una_estimacion_fallida_no_deja_rastro_en_el_historial(montar) -> None:
    error = StructuredOutputError(attempts=2, cost_usd=0.0, last_error="inválido")
    cliente, _, sid = montar(wrapper=_WrapperFalso(error=error))
    assert estimar(cliente, sid, _turno(1)).status_code != 200
    assert cliente.app.state.session_store.get_or_raise(sid).history.messages == []


def test_el_mensaje_del_usuario_guardado_se_recorta(montar) -> None:
    cliente, _, sid = montar()
    estimar(cliente, sid, "x" * (MAX_HISTORY_USER_CHARS + 500))
    guardado = cliente.app.state.session_store.get_or_raise(sid).history.messages[0].content
    assert len(guardado) == MAX_HISTORY_USER_CHARS


def test_los_adjuntos_no_entran_al_historial(montar) -> None:
    cliente, _, sid = montar()
    archivo = ("notas.txt", b"SECRETO-DEL-ADJUNTO texto del documento", "text/plain")
    respuesta = estimar(cliente, sid, _turno(1), files={"attachments": archivo})
    assert respuesta.status_code == 200
    mensajes = cliente.app.state.session_store.get_or_raise(sid).history.messages
    assert all("SECRETO-DEL-ADJUNTO" not in m.content for m in mensajes)


# --- Caducidad y tope del almacén -----------------------------------------------


class _Reloj:
    """Reloj manual: la hora solo cambia cuando el test la adelanta."""

    def __init__(self) -> None:
        self.ahora = datetime(2026, 10, 19, 12, 0, tzinfo=UTC)

    def __call__(self) -> datetime:
        """La hora actual del reloj de prueba."""
        return self.ahora

    def avanzar(self, **kwargs) -> None:
        """Adelanta el reloj (por ejemplo, `avanzar(hours=25)`)."""
        self.ahora += timedelta(**kwargs)


def test_una_sesion_sin_uso_caduca_y_se_borra() -> None:
    reloj = _Reloj()
    store = SessionStore(ttl_seconds=3600, clock=reloj)
    sid = store.create().session_id
    reloj.avanzar(hours=2)
    assert store.get(sid) is None
    assert len(store) == 0
    with pytest.raises(SessionNotFoundError):
        store.get_or_raise(sid)


def test_usar_la_sesion_renueva_el_plazo() -> None:
    reloj = _Reloj()
    store = SessionStore(ttl_seconds=3600, clock=reloj)
    sid = store.create().session_id
    for _ in range(3):
        reloj.avanzar(minutes=50)  # cada uso llega antes de la hora, así que nunca caduca
        assert store.get(sid) is not None


def test_crear_una_sesion_limpia_las_caducadas() -> None:
    reloj = _Reloj()
    store = SessionStore(ttl_seconds=3600, clock=reloj)
    for _ in range(3):
        store.create()
    reloj.avanzar(hours=2)
    store.create()
    assert len(store) == 1


def test_con_el_almacen_lleno_se_desaloja_la_sesion_menos_usada() -> None:
    reloj = _Reloj()
    store = SessionStore(max_sessions=2, clock=reloj)
    vieja = store.create().session_id
    reloj.avanzar(minutes=1)
    reciente = store.create().session_id
    reloj.avanzar(minutes=1)
    store.get(vieja)  # usarla la vuelve la más reciente; ahora la menos usada es la otra
    reloj.avanzar(minutes=1)
    nueva = store.create().session_id
    assert len(store) == 2
    assert store.get(reciente) is None
    assert store.get(vieja) is not None
    assert store.get(nueva) is not None


def test_el_endpoint_responde_404_si_la_sesion_caduco(montar) -> None:
    cliente, _, sid = montar()
    # Se reemplaza el reloj del almacén vivo y se adelanta más allá de las 24 h.
    reloj = _Reloj()
    store = cliente.app.state.session_store
    store._clock = reloj
    store.get_or_raise(sid).touch(reloj())
    reloj.avanzar(hours=25)
    assert estimar(cliente, sid, _turno(1)).status_code == 404


# --- Configuración --------------------------------------------------------------


def test_la_configuracion_tiene_valores_por_defecto_y_acepta_vacios(monkeypatch) -> None:
    monkeypatch.setenv("SESSION_TTL_SECONDS", "")
    monkeypatch.setenv("MAX_SESSIONS", "")
    ajustes = Settings(_env_file=None)
    assert ajustes.session_ttl_seconds == 86_400
    assert ajustes.max_sessions == 1_000


@pytest.mark.parametrize("variable", ["SESSION_TTL_SECONDS", "MAX_SESSIONS"])
def test_la_configuracion_rechaza_valores_menores_a_uno(monkeypatch, variable) -> None:
    monkeypatch.setenv(variable, "0")
    with pytest.raises(ValueError, match=variable):
        Settings(_env_file=None)


# --- GET /sessions/{id}: estado visible -----------------------------------------


def test_get_session_devuelve_turnos_ventana_y_metadata(montar) -> None:
    cliente, _, sid = montar(max_turns="3")
    vacio = cliente.get(f"/sessions/{sid}").json()
    assert vacio["turns"] == 0
    assert vacio["max_turns"] == 3
    assert vacio["project_metadata"]["project_name"] is None

    estimar(cliente, sid, _turno(1))
    estimar(cliente, sid, _turno(2))
    assert cliente.get(f"/sessions/{sid}").json()["turns"] == 2


def test_get_session_no_expone_el_texto_de_los_mensajes(montar) -> None:
    cliente, _, sid = montar()
    estimar(cliente, sid, _turno(1))
    assert _turno(1) not in cliente.get(f"/sessions/{sid}").text


def test_get_session_responde_404_si_no_existe(montar) -> None:
    cliente, _, _ = montar()
    assert cliente.get("/sessions/no-existe").status_code == 404
