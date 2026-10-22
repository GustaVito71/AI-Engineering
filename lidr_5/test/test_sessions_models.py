"""Tests unitarios del módulo de sesiones (`app/sessions/`) y de `MAX_TURNS`.

QUÉ SE COMPRUEBA, EN PALABRAS SIMPLES

- Historial: guarda solo los últimos N turnos (ventana deslizante) y el system prompt
  nunca se pierde.
- Metadata del proyecto: empieza vacía, respeta sus topes y se puede combinar con
  datos nuevos sin perder los anteriores.
- Sesión y almacén: cada sesión es independiente y tiene un identificador único.
- Configuración: `MAX_TURNS` fija cuántos turnos se recuerdan (6 por defecto, mínimo 1).

Son tests sin red, sin Redis y sin claves: solo lógica en memoria.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.sessions import (
    ConversationHistory,
    ProjectMetadata,
    Session,
    SessionNotFoundError,
    SessionStore,
)

# --- Historial y ventana -------------------------------------------------------


def _llenar(history: ConversationHistory, turnos: int) -> None:
    """Agrega `turnos` turnos de prueba al historial: (u1, a1), (u2, a2), etc."""
    for n in range(1, turnos + 1):
        history.append(user=f"u{n}", assistant=f"a{n}")


def test_history_keeps_only_the_last_max_turns_pairs() -> None:
    """Con ventana de 2 turnos y 3 enviados, se olvida el primero y quedan los dos últimos."""
    history = ConversationHistory(max_turns=2)
    _llenar(history, 3)
    assert [m.content for m in history.messages] == ["u2", "a2", "u3", "a3"]


def test_history_below_the_limit_is_kept_whole() -> None:
    """Si no se supera el límite, no se olvida nada."""
    history = ConversationHistory(max_turns=6)
    _llenar(history, 3)
    assert len(history.messages) == 6


def test_window_never_exceeds_max_turns_and_keeps_roles_alternating() -> None:
    """Aunque entren 10 turnos, nunca se guardan más de 3 y los mensajes siguen alternando usuario y asistente."""
    history = ConversationHistory(max_turns=3)
    for n in range(1, 11):
        history.append(user=f"u{n}", assistant=f"a{n}")
        assert len(history.messages) <= 3 * 2
        assert [m.role for m in history.messages][::2] == ["user"] * (len(history.messages) // 2)
        assert [m.role for m in history.messages][1::2] == ["assistant"] * (
            len(history.messages) // 2
        )


def test_max_turns_must_be_at_least_one() -> None:
    """Una ventana de 0 turnos no tiene sentido: se rechaza al crearla."""
    with pytest.raises(ValidationError):
        ConversationHistory(max_turns=0)


def test_to_messages_list_puts_the_system_prompt_first() -> None:
    """La lista que se envía al modelo empieza por el system prompt y sigue con el historial."""
    history = ConversationHistory(max_turns=2)
    _llenar(history, 1)
    assert history.to_messages_list("SISTEMA") == [
        {"role": "system", "content": "SISTEMA"},
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
    ]


def test_system_prompt_survives_any_trimming() -> None:
    """Por más que la ventana recorte mensajes, el system prompt no se pierde: no es parte del historial."""
    history = ConversationHistory(max_turns=1)
    _llenar(history, 5)
    mensajes = history.to_messages_list("SISTEMA")
    assert mensajes[0] == {"role": "system", "content": "SISTEMA"}
    assert [m["content"] for m in mensajes[1:]] == ["u5", "a5"]


def test_to_messages_list_with_empty_history_is_only_the_system_prompt() -> None:
    """Sin historial, la lista contiene solo el system prompt."""
    assert ConversationHistory().to_messages_list("S") == [{"role": "system", "content": "S"}]


# --- ProjectMetadata ------------------------------------------------------------


def test_metadata_is_empty_by_default() -> None:
    """Una metadata recién creada no conoce ningún hecho del proyecto."""
    metadata = ProjectMetadata()
    assert metadata.is_empty()
    assert metadata.mentioned_technologies == []


@pytest.mark.parametrize(
    "campo",
    [
        {"project_name": "X"},
        {"assumed_team_size": 3},
        {"mentioned_technologies": ["React"]},
        {"agreed_scope": "Fase 1"},
    ],
)
def test_any_field_makes_the_metadata_non_empty(campo) -> None:
    """Con que un solo campo tenga valor, la metadata deja de estar vacía."""
    assert not ProjectMetadata(**campo).is_empty()


def test_merge_overwrites_scalars_unions_technologies_and_keeps_missing() -> None:
    """Al combinar: nombre y equipo nuevos reemplazan a los viejos, las tecnologías se suman sin repetir y lo que la actualización no trae se conserva."""
    base = ProjectMetadata(
        project_name="Nimbus CRM",
        assumed_team_size=3,
        mentioned_technologies=["React", "Postgres"],
        agreed_scope="Solo fase 1",
    )
    update = ProjectMetadata(
        project_name="Nimbus CRM v2",
        mentioned_technologies=["postgres", "Redis"],
    )
    merged = base.merge_with(update)
    assert merged.project_name == "Nimbus CRM v2"  # lo nuevo pisa
    assert merged.assumed_team_size == 3  # la actualización no lo trae: se conserva
    assert merged.agreed_scope == "Solo fase 1"
    assert merged.mentioned_technologies == ["React", "Postgres", "Redis"]


def test_merge_does_not_modify_its_operands() -> None:
    """Combinar devuelve un objeto nuevo: las dos metadatas originales quedan intactas."""
    base = ProjectMetadata(mentioned_technologies=["React"])
    update = ProjectMetadata(mentioned_technologies=["Redis"])
    base.merge_with(update)
    assert base.mentioned_technologies == ["React"]
    assert update.mentioned_technologies == ["Redis"]


def test_merge_into_empty_metadata_returns_the_update() -> None:
    """Combinar con una metadata vacía da como resultado la actualización."""
    update = ProjectMetadata(project_name="X", assumed_team_size=2)
    assert ProjectMetadata().merge_with(update) == update


@pytest.mark.parametrize(
    "campo",
    [
        {"assumed_team_size": 0},
        {"assumed_team_size": 51},
        {"project_name": "x" * 121},
        {"agreed_scope": "x" * 2001},
    ],
)
def test_metadata_limits_are_enforced(campo) -> None:
    """Se rechazan un equipo fuera de 1 a 50, un nombre de más de 120 caracteres y un alcance de más de 2.000."""
    with pytest.raises(ValidationError):
        ProjectMetadata(**campo)


def test_scope_at_the_limit_is_accepted() -> None:
    """Un alcance de exactamente 2.000 caracteres (el máximo) es válido."""
    assert ProjectMetadata(agreed_scope="x" * 2000).agreed_scope == "x" * 2000


# --- Session y SessionStore -------------------------------------------------------


def test_session_starts_empty_with_a_uuid4_id() -> None:
    """Una sesión nueva está vacía y su identificador es un UUID versión 4."""
    from uuid import UUID

    session = Session()
    assert UUID(session.session_id).version == 4
    assert session.history.messages == []
    assert session.metadata.is_empty()


def test_store_creates_sessions_with_unique_ids_and_registers_them() -> None:
    """Cada sesión creada tiene un identificador distinto y queda registrada en el almacén."""
    store = SessionStore()
    s1, s2 = store.create(), store.create()
    assert s1.session_id != s2.session_id
    assert len(store) == 2
    assert store.get(s1.session_id) is s1


def test_store_applies_the_configured_window_to_new_sessions() -> None:
    """Las sesiones nuevas heredan la ventana de turnos con la que se creó el almacén."""
    session = SessionStore(max_turns=2).create()
    assert session.history.max_turns == 2


def test_store_get_returns_none_and_get_or_raise_raises_for_unknown_ids() -> None:
    """Con un identificador desconocido, `get` devuelve None y `get_or_raise` lanza el error."""
    store = SessionStore()
    assert store.get("no-existe") is None
    with pytest.raises(SessionNotFoundError):
        store.get_or_raise("no-existe")


def test_sessions_do_not_share_state() -> None:
    """Lo que se escribe en una sesión no aparece en otra."""
    store = SessionStore()
    a, b = store.create(), store.create()
    a.history.append(user="hola", assistant="mundo")
    a.metadata = ProjectMetadata(project_name="A")
    assert b.history.messages == []
    assert b.metadata.is_empty()


# --- Configuración (MAX_TURNS) -------------------------------------------------------


def test_max_turns_defaults_to_six(settings_predeterminada) -> None:
    """Sin configurar nada, la ventana es de 6 turnos."""
    assert settings_predeterminada.max_turns == 6


def test_max_turns_is_read_from_the_environment(monkeypatch) -> None:
    """La variable de entorno MAX_TURNS fija la ventana."""
    from app.config import Settings

    monkeypatch.setenv("MAX_TURNS", "4")
    assert Settings().max_turns == 4


def test_empty_max_turns_falls_back_to_the_default(monkeypatch) -> None:
    """MAX_TURNS vacío (como queda en un .env sin completar) equivale a 6."""
    from app.config import Settings

    monkeypatch.setenv("MAX_TURNS", "")
    assert Settings().max_turns == 6


@pytest.mark.parametrize("valor", [0, -1])
def test_max_turns_below_one_is_rejected(construir_settings, valor) -> None:
    """Con MAX_TURNS menor que 1 (0 o -1) el arranque falla con un mensaje que nombra la variable."""
    with pytest.raises(ValidationError, match="MAX_TURNS"):
        construir_settings(max_turns=valor)
