"""Tests del tier dinámico (Paso 9): `resolve_tier` y su cadena de reglas.

QUÉ SE COMPRUEBA

- El override de quien llama gana siempre, aunque una regla también se cumpliera.
- Cada regla se activa con palabras en español y en inglés, y devuelve su nombre.
- El orden de prioridad: confidencialidad y normas (ejecutivo) pasan por delante de lo técnico.
- Una sola palabra técnica no basta; hacen falta dos distintas.
- Un equipo de 2 personas o menos da audiencia de jefe de proyecto.
- Sin ninguna regla se usa el tier por defecto, y una regla que falla se salta.
"""

from __future__ import annotations

import pytest

from app.sessions import tier_resolver
from app.sessions.models import ProjectMetadata
from app.sessions.tier_resolver import Tier, resolve_tier

NEUTRO = "Queremos una web para gestionar reservas de salas."


def _resolver(transcript: str = NEUTRO, metadata: ProjectMetadata | None = None, **extra):
    """Atajo: resuelve el tier con metadata vacía si no se da otra."""
    return resolve_tier(transcript=transcript, metadata=metadata or ProjectMetadata(), **extra)


def test_el_override_gana_aunque_una_regla_se_cumpla() -> None:
    tier, regla = _resolver("Hay un NDA firmado con el cliente.", override=Tier.DEVELOPER)
    assert (tier, regla) == (Tier.DEVELOPER, "override")


@pytest.mark.parametrize(
    "texto",
    [
        "El proyecto está bajo NDA.",
        "Es información confidencial del cliente.",
        "Firmamos un acuerdo de confidencialidad.",
        "The work is under embargo until launch.",
    ],
)
def test_la_confidencialidad_da_audiencia_ejecutiva(texto) -> None:
    assert _resolver(texto) == (Tier.EXECUTIVE, "nda_detected")


def test_la_confidencialidad_en_el_alcance_acordado_tambien_cuenta() -> None:
    memoria = ProjectMetadata(agreed_scope="Todo bajo acuerdo de confidencialidad.")
    assert _resolver(NEUTRO, memoria) == (Tier.EXECUTIVE, "nda_detected")


@pytest.mark.parametrize(
    "texto",
    [
        "Debe cumplir RGPD.",
        "Hay que cumplir GDPR y HIPAA.",
        "Aplica la ley 25.326.",
        "Con PCI-DSS.",
    ],
)
def test_una_norma_da_audiencia_ejecutiva(texto) -> None:
    assert _resolver(texto) == (Tier.EXECUTIVE, "regulatory_context")


def test_una_norma_en_las_tecnologias_vigentes_tambien_cuenta() -> None:
    memoria = ProjectMetadata(mentioned_technologies=["Cifrado HIPAA"])
    assert _resolver(NEUTRO, memoria) == (Tier.EXECUTIVE, "regulatory_context")


def test_dos_palabras_tecnicas_distintas_dan_audiencia_de_desarrollo() -> None:
    texto = "Desplegamos con Docker sobre Kubernetes."
    assert _resolver(texto) == (Tier.DEVELOPER, "technical_audience")


@pytest.mark.parametrize(
    "texto",
    ["Corremos todo en docker.", "docker, docker y más docker.", "Usamos Kafka."],
)
def test_una_sola_palabra_tecnica_no_basta(texto) -> None:
    assert _resolver(texto) == (Tier.DEFAULT, "default")


@pytest.mark.parametrize(("equipo", "esperado"), [(1, Tier.PM), (2, Tier.PM), (3, Tier.DEFAULT)])
def test_un_equipo_de_dos_o_menos_da_audiencia_de_jefe_de_proyecto(equipo, esperado) -> None:
    tier, _regla = _resolver(NEUTRO, ProjectMetadata(assumed_team_size=equipo))
    assert tier == esperado


def test_la_confidencialidad_pasa_por_delante_de_lo_tecnico_y_del_equipo() -> None:
    texto = "Bajo NDA: Docker y Kubernetes."
    tier, regla = _resolver(texto, ProjectMetadata(assumed_team_size=1))
    assert (tier, regla) == (Tier.EXECUTIVE, "nda_detected")


def test_sin_ninguna_regla_se_usa_el_tier_por_defecto_indicado() -> None:
    assert _resolver() == (Tier.DEFAULT, "default")
    assert _resolver(default=Tier.PM) == (Tier.PM, "default")


def test_una_regla_que_falla_se_salta(monkeypatch) -> None:
    def rota(_ctx):
        raise RuntimeError("fallo inesperado")

    reglas = (
        tier_resolver.TierRule("rota", Tier.EXECUTIVE, rota),
        tier_resolver.TierRule("equipo", Tier.PM, lambda _ctx: True),
    )
    monkeypatch.setattr(tier_resolver, "_RULES", reglas)
    assert _resolver() == (Tier.PM, "equipo")
