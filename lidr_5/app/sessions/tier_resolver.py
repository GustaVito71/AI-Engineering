"""Audiencia (tier) de la respuesta: se deduce de la conversación (Paso 9).

QUÉ HACE ESTE ARCHIVO

El tier decide a quién va dirigida la estimación (ejecutivo, jefe de proyecto,
desarrollador o neutro) y el prompt v5 lo usa en su bloque `<audience>`. Aquí se decide
cuál corresponde en cada turno.

CÓMO SE DECIDE (tomado de session_05_live, con reglas también en español)

1. Si quien llama pide un tier (`override`), gana siempre.
2. Si no, se recorre una cadena de reglas en orden de prioridad: la primera que se
   cumple decide. Cada regla es una función sencilla que responde sí o no.
3. Si ninguna se cumple, se usa el tier por defecto (`default`, sin bloque `<audience>`).

Junto con el tier se devuelve el NOMBRE de la regla que decidió («executive
(nda_detected)»): así se puede explicar en pantalla por qué se eligió esa audiencia.

REGLAS (de mayor a menor prioridad)

- `nda_detected` → executive: se habla de confidencialidad, NDA o embargo.
- `regulatory_context` → executive: aparece una norma (GDPR/RGPD, HIPAA, PCI-DSS…).
- `technical_audience` → developer: al menos dos palabras técnicas distintas (docker,
  kubernetes, kafka…). Con una sola hay demasiado ruido («corremos en docker»).
- `low_budget_pm` → pm: el equipo asumido es de 2 personas o menos.

DIFERENCIA CON LA REFERENCIA

La referencia solo reconoce palabras en inglés. Como los usuarios escriben en español,
cada lista suma sus equivalentes en español; sin ellas, una conversación en español no
activaría casi ninguna regla. Las reglas usan expresiones regulares, así que no cuestan
ninguna llamada a un modelo.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

from app.sessions.models import ProjectMetadata
from app.tracing import emit


class Tier(str, Enum):
    """Audiencias posibles; los valores coinciden con `config.TIER_VALUES`."""

    EXECUTIVE = "executive"
    PM = "pm"
    DEVELOPER = "developer"
    DEFAULT = "default"


# Tier que se usa cuando nadie pidió otro y ninguna regla se cumple (el prompt no agrega
# bloque `<audience>`).
DEFAULT_TIER = Tier.DEFAULT

# Con este equipo asumido (o menos) se considera un proyecto de presupuesto ajustado.
SMALL_TEAM_MAX = 2

# Cuántas palabras técnicas DISTINTAS hacen falta para pasar a audiencia técnica.
MIN_TECH_KEYWORDS = 2


@dataclass
class ResolutionContext:
    """Lo que ven las reglas: el texto del turno y la memoria del proyecto."""

    transcript: str
    metadata: ProjectMetadata


@dataclass
class TierRule:
    """Una regla de la cadena: su nombre, el tier que da y la condición que la activa."""

    name: str
    tier: Tier
    predicate: Callable[[ResolutionContext], bool]


# --- Listas de palabras (inglés y español) -----------------------------------------------

_NDA_PATTERN = re.compile(
    r"\b(nda|non[- ]?disclosure|confidential|under embargo|legal hold|"
    r"acuerdo de confidencialidad|confidencial(?:es|idad)?|bajo embargo|"
    r"retenci[óo]n legal)\b",
    re.IGNORECASE,
)
_REGULATORY_PATTERN = re.compile(
    r"\b(hipaa|gdpr|rgpd|lopd|sox|pci[- ]?dss|fda|iso[- ]?27001|ccpa)\b|"
    r"\bley\s+25\.?326\b",
    re.IGNORECASE,
)
_DEV_KEYWORDS_PATTERN = re.compile(
    r"\b(docker|kubernetes|k8s|microservices?|microservicios?|terraform|iac|"
    r"helm|grpc|graphql|kafka|airflow|spark|rabbitmq)\b",
    re.IGNORECASE,
)


# --- Condiciones de las reglas -----------------------------------------------------------


def _has_nda(ctx: ResolutionContext) -> bool:
    """Confidencialidad en el mensaje o en el alcance acordado."""
    if _NDA_PATTERN.search(ctx.transcript):
        return True
    return bool(ctx.metadata.agreed_scope and _NDA_PATTERN.search(ctx.metadata.agreed_scope))


def _has_regulatory_context(ctx: ResolutionContext) -> bool:
    """Una norma en el mensaje, en el alcance acordado o en las tecnologías vigentes."""
    if _REGULATORY_PATTERN.search(ctx.transcript):
        return True
    if ctx.metadata.agreed_scope and _REGULATORY_PATTERN.search(ctx.metadata.agreed_scope):
        return True
    return any(_REGULATORY_PATTERN.search(tech) for tech in ctx.metadata.mentioned_technologies)


def _technical_audience(ctx: ResolutionContext) -> bool:
    """Al menos `MIN_TECH_KEYWORDS` palabras técnicas distintas en el mensaje."""
    hits = {hit.lower() for hit in _DEV_KEYWORDS_PATTERN.findall(ctx.transcript)}
    return len(hits) >= MIN_TECH_KEYWORDS


def _is_small_team(ctx: ResolutionContext) -> bool:
    """Equipo asumido de `SMALL_TEAM_MAX` personas o menos."""
    size = ctx.metadata.assumed_team_size
    return size is not None and size <= SMALL_TEAM_MAX


# La cadena, en orden de prioridad: la primera regla que se cumple decide.
_RULES: tuple[TierRule, ...] = (
    TierRule("nda_detected", Tier.EXECUTIVE, _has_nda),
    TierRule("regulatory_context", Tier.EXECUTIVE, _has_regulatory_context),
    TierRule("technical_audience", Tier.DEVELOPER, _technical_audience),
    TierRule("low_budget_pm", Tier.PM, _is_small_team),
)


def resolve_tier(
    *,
    transcript: str,
    metadata: ProjectMetadata,
    override: Tier | None = None,
    default: Tier = DEFAULT_TIER,
) -> tuple[Tier, str]:
    """Devuelve `(tier, regla)`: el tier elegido y el nombre de la regla que lo decidió.

    Orden: 1) el `override` de quien llama (regla `override`); 2) la primera regla de la
    cadena que se cumple; 3) `default` (regla `default`). Una regla que falla por un error
    inesperado se salta y se registra: elegir la audiencia nunca debe tumbar una estimación.
    """
    if override is not None:
        return override, "override"

    ctx = ResolutionContext(transcript=transcript, metadata=metadata)
    for rule in _RULES:
        try:
            if rule.predicate(ctx):
                emit(__name__, "tier_resuelto", tier=rule.tier.value, regla=rule.name)
                return rule.tier, rule.name
        except Exception as exc:  # noqa: BLE001 - una regla rota no puede romper el turno
            emit(
                __name__,
                "tier_regla_fallo",
                level="warning",
                regla=rule.name,
                error=type(exc).__name__,
            )
    return default, "default"
