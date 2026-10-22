"""Loader Jinja2 de plantillas de prompt versionadas.

La estructura en disco es ``app/prompts/<use_case>/<version>/<role>.j2``. El
versionado es obligatorio desde el primer día: cambiar de prompt es cambiar un
string en el punto de llamada (``version="v2"``), no refactorizar código.

Cada versión tiene su propio ``examples.yaml`` con sus roles (tarifa y etiquetas
de presentación), reglas de redondeo y datos de ejemplo. Los prompts publicados
son inmutables: cambiar el comportamiento implica un directorio de versión nuevo.

Reparto de responsabilidades:
- ``examples.yaml``: los datos (roles, tarifas, etiquetas, redondeo, ejemplos) y,
  opcionalmente, el tipo de salida de la versión (``output``, ver
  ``output_of_version``).
- ``loader.py``: solo aritmética genérica (horas, costes, totales, dotación).
  Nunca decide cómo se muestra nada, así que una versión nueva puede cambiar
  la presentación sin tocar Python.
- ``system.j2``: las instrucciones al modelo (rol, reglas, formatos, niveles de
  detalle). Incluye ``examples.j2`` de su propia versión.
- ``examples.j2``: la presentación de los ejemplos (etiquetas, plurales, resumen
  de equipo, maquetación).

Además de las plantillas de estimación, ``app/prompts/metadata_extraction/<version>/``
guarda las del extractor de ``project_metadata`` (Paso 4), que se renderizan con
``render_metadata_extraction_prompt``. Los textos que vienen del usuario o de un LLM y
que se encierran entre etiquetas pasan por los filtros ``prompt_data`` y ``escape_tags``.
"""

from __future__ import annotations

import json
import re
from copy import deepcopy
from functools import lru_cache
from math import ceil
from pathlib import Path

import yaml
from jinja2 import Environment, FileSystemLoader, StrictUndefined, TemplateNotFound

from app.schemas.critic import CriticFeedback
from app.schemas.estimation import EstimationRequest
from app.schemas.structured_estimation import (
    MIN_CONFIDENCE_PCT,
    OUT_OF_SCOPE_PREFIX,
    TOTALS_MAX_DEVIATION_PCT,
    UNESTIMATED_PHASE,
    StructuredResult,
    format_number,
)
from app.sessions.models import ExtractedMetadata, Message, ProjectMetadata

_BASE_DIR = Path(__file__).resolve().parent

# Campos que todo rol de `rates` tiene que declarar. `label`/`plural` los usa la
# plantilla, y `eur_per_hour` la aritmética de abajo. Se validan al cargar el
# YAML, para que uno mal formado falle en la primera request y no a mitad de un
# render.
_ROLE_FIELDS = ("label", "plural", "eur_per_hour")


# --- Structured response ----------------------------------------------------
# Tipos de salida de una versión de prompt (clave `output` de su examples.yaml).
# Cada endpoint acepta solo las versiones de su tipo: `/estimate` las de texto
# libre y `/estimate/structured` las estructuradas.
OUTPUT_TEXT = "text"
OUTPUT_STRUCTURED = "structured"
# --- Rendered response ------------------------------------------------------
# `v4`: salida estructurada con la presentación (`rendered`) escrita por el modelo.
# La acepta solo `/estimate/rendered`.
OUTPUT_RENDERED = "rendered"
_OUTPUT_KINDS = (OUTPUT_TEXT, OUTPUT_STRUCTURED, OUTPUT_RENDERED)


def output_of_version(version: str) -> str:
    """Tipo de salida que declara una versión en la clave ``output`` de su examples.yaml.

    Sin la clave, o sin examples.yaml, la versión es de texto libre: así `v1` y
    `v2`, ya publicadas, no se tocan. Un valor que no es ninguno de los tipos
    conocidos es un examples.yaml mal escrito y falla con el nombre del archivo.
    """
    return _output_at(str(_BASE_DIR / "estimation" / version / "examples.yaml"))


@lru_cache(maxsize=32)
def _output_at(path: str) -> str:
    """Lee la clave ``output`` una vez por archivo. La caché va por ruta y no por
    versión, para que un test que cambia ``_BASE_DIR`` no reciba un valor viejo."""
    try:
        with open(path, encoding="utf-8") as f:
            output = (yaml.safe_load(f) or {}).get("output", OUTPUT_TEXT)
    except FileNotFoundError:
        return OUTPUT_TEXT
    if output not in _OUTPUT_KINDS:
        raise ValueError(
            f"{path}: output '{output}' no es válido. Valores posibles: {', '.join(_OUTPUT_KINDS)}."
        )
    return output


def _to_json(value: object) -> str:
    """Filtro ``json`` de las plantillas: JSON indentado, con tildes legibles.

    El ``tojson`` de Jinja2 escapa los caracteres no ASCII (``Dise\\u00f1o``) y
    los ejemplos de `v3` tienen que mostrar el JSON tal como lo debe escribir el
    modelo. Es formato genérico, no decide qué se muestra.
    """
    return json.dumps(value, ensure_ascii=False, indent=2)


def available_versions(output: str = OUTPUT_TEXT) -> list[str]:
    """Versiones de prompt publicadas de un tipo de salida.

    Una versión publicada es un directorio ``vN/`` con un ``system.j2``. Su tipo
    lo declara la clave ``output`` de su ``examples.yaml`` (ver
    ``output_of_version``); por defecto se listan las de texto libre, que son
    las que acepta ``/estimate``.

    Ordenadas por número (v2 antes que v10). Los endpoints las usan para validar
    ``?prompt_version=`` antes de armar una ruta con lo que mandó el cliente.
    """
    root_dir = _BASE_DIR / "estimation"
    versions = [
        d.name
        for d in root_dir.iterdir()
        if d.is_dir()
        and re.fullmatch(r"v\d+", d.name)
        and (d / "system.j2").is_file()
        and output_of_version(d.name) == output
    ]
    return sorted(versions, key=lambda v: int(v[1:]))


def _round_up(n: float, base: int) -> int:
    """Redondea hacia arriba al múltiplo de `base` más cercano. Acepta floats."""
    return int(ceil(n / base) * base)


# Caché del YAML sin procesar, por versión.
@lru_cache(maxsize=8)
def _load_raw_yaml(version: str) -> dict:
    """Lee el archivo `examples.yaml` (ejemplos de la estimación) de una versión del
    prompt.

    Lanza `TemplateNotFound` si esa versión no tiene el archivo.
    """
    yaml_path = _BASE_DIR / "estimation" / version / "examples.yaml"
    try:
        with yaml_path.open(encoding="utf-8") as f:
            return yaml.safe_load(f)
    except FileNotFoundError as e:
        raise TemplateNotFound(f"No existe examples.yaml para la versión {version}") from e


def _validate(data: dict, version: str) -> None:
    """Falla rápido ante un examples.yaml mal formado, nombrando lo que falta."""
    rates = data["rates"]
    for role, spec in rates.items():
        missing = [f for f in _ROLE_FIELDS if f not in (spec or {})]
        if missing:
            raise ValueError(
                f"Al rol '{role}' de estimation/{version}/examples.yaml le falta "
                f"{', '.join(missing)}. Todo rol necesita: {', '.join(_ROLE_FIELDS)}."
            )
    for ex in data["examples"]:
        for p in ex["phases"]:
            for role in p.get("team", {}):
                if role not in rates:
                    raise ValueError(
                        f"El rol '{role}' del ejemplo '{ex['project_description'][:40]}...' "
                        f"no está definido en rates. Roles disponibles: {list(rates)}"
                    )


@lru_cache(maxsize=8)
def _compute_version_data(version: str) -> dict:
    """Lee examples.yaml una vez por versión y hace toda la aritmética.

    Cacheada: el YAML se lee y los números se calculan una sola vez por proceso.
    Quien la necesite pasa por `_load_version_data`, que entrega una copia.
    """
    data = _load_raw_yaml(version)
    _validate(data, version)

    rates = data["rates"]
    productive_hours = data["productive_hours_per_week"]
    hours_base = data["rounding"]["hours_base"]
    cost_base = data["rounding"]["cost_base"]

    computed_examples = []
    for ex in data["examples"]:
        computed_phases = []
        for p in ex["phases"]:
            breakdown = []
            for role, headcount in p.get("team", {}).items():
                rate = rates[role]["eur_per_hour"]
                hours = _round_up(p["duration_weeks"] * productive_hours * headcount, hours_base)
                cost = _round_up(hours * rate, cost_base)
                breakdown.append(
                    {
                        "role": role,
                        "headcount": headcount,
                        "hours": hours,
                        "cost_eur": cost,
                        "rate_eur_per_hour": rate,
                    }
                )
            computed_phases.append(
                {
                    **p,
                    "hours": sum(b["hours"] for b in breakdown),
                    "cost_eur": sum(b["cost_eur"] for b in breakdown),
                    "team_breakdown": breakdown,
                }
            )

        # Dotación máxima de cada rol entre todas las fases, en el orden en que
        # se declaran los roles en `rates`. Es solo agregación: cómo se muestra
        # (etiquetas, plurales, separadores) lo decide la plantilla.
        team_headcount = {
            role: max(p.get("team", {}).get(role, 0) for p in ex["phases"]) for role in rates
        }
        team_headcount = {role: hc for role, hc in team_headcount.items() if hc > 0}

        computed_examples.append(
            {
                # --- Structured response ----------------------------------------------------
                # Campos propios de la versión (por ejemplo, `summary` y
                # `confidence_pct` en v3) pasan tal cual; lo calculado se pisa abajo.
                **ex,
                "project_type": ex["project_type"],
                "project_description": ex["project_description"],
                "phases": computed_phases,
                "totals": {
                    "total_hours": sum(p["hours"] for p in computed_phases),
                    "total_cost_eur": sum(p["cost_eur"] for p in computed_phases),
                    "total_duration_weeks": sum(p["duration_weeks"] for p in computed_phases),
                },
                "team_headcount": team_headcount,
            }
        )

    return {
        "rates": rates,
        "rounding": data["rounding"],
        "productive_hours_per_week": productive_hours,
        "examples": computed_examples,
    }


def _load_version_data(version: str) -> dict:
    """Datos calculados de una versión, como copia para que nadie modifique la caché.

    Devuelve un dict con las claves:
    - rates: {role: {label, plural, eur_per_hour}}
    - rounding: {hours_base, cost_base}
    - productive_hours_per_week: int
    - examples: lista de ejemplos calculados (fases con hours, cost_eur y
      team_breakdown; totals; team_headcount)
    """
    return deepcopy(_compute_version_data(version))


def _escape_tags(value: object) -> str:
    """Cambia `<` y `>` por `&lt;` y `&gt;` para que un texto no pueda abrir ni cerrar etiquetas.

    Se usa con texto que viene del usuario o de un LLM y que el prompt encierra entre
    etiquetas (`<user_turn>`, `<project_metadata>`): sin esto, un texto con
    `</project_metadata>` cerraría el bloque y lo que siga parecería una instrucción.
    Conserva los saltos de línea.
    """
    return str(value).replace("<", "&lt;").replace(">", "&gt;")


def _prompt_data(value: object) -> str:
    """Un dato de la metadata en una sola línea y sin etiquetas (ver `_escape_tags`)."""
    return _escape_tags(" ".join(str(value).split()))


_env = Environment(
    loader=FileSystemLoader(_BASE_DIR),
    undefined=StrictUndefined,
    trim_blocks=True,
    lstrip_blocks=True,
    autoescape=False,
    keep_trailing_newline=True,
)

# --- Structured response ----------------------------------------------------
_env.filters["json"] = _to_json

# --- Rendered response ------------------------------------------------------
# Cifras en formato castellano (29.850) en los `rendered` de los ejemplos de v4:
# la misma función que usa el validador, así ejemplo y validación no discrepan.
_env.filters["number"] = format_number
_env.filters["prompt_data"] = _prompt_data
_env.filters["escape_tags"] = _escape_tags


def render_estimation_prompt(
    request: EstimationRequest,
    version: str = "v1",
    *,
    project_metadata: ProjectMetadata | None = None,
    tier: str = "default",
    critic_feedback: CriticFeedback | None = None,
) -> tuple[str, str]:
    """Renderiza los prompts de sistema y de usuario para el caso de estimación.

    `project_metadata` y `tier` son la memoria de una sesión conversacional (Paso 4) y
    solo los usa la versión v5; las demás versiones los ignoran. Una metadata vacía
    equivale a no pasar ninguna: el bloque `<project_metadata>` no se renderiza.

    `critic_feedback` es la revisión del crítico del ACB (Paso 8) sobre el intento
    anterior; solo la v5 la usa y agrega un bloque `<critic_feedback>` al mensaje de
    usuario. Sin feedback (o con otra versión) el prompt no cambia.

    Devuelve:
        Una tupla ``(system_prompt, user_prompt)`` lista para enviar al LLM
        como mensajes separados ``role: "system"`` y ``role: "user"``.
    """
    version_data = _load_version_data(version)

    context = {
        "description": request.description,
        "project_type": request.project_type.value,
        "detail_level": request.detail_level.value,
        "output_format": request.output_format.value,
        "rates": version_data["rates"],
        "productive_hours_per_week": version_data["productive_hours_per_week"],
        "hours_rounding_base": version_data["rounding"]["hours_base"],
        "cost_rounding_base": version_data["rounding"]["cost_base"],
        "examples": version_data["examples"],
        # system.j2 incluye el examples.j2 de su propia versión con esta variable,
        # así la ruta del include no lleva una versión escrita a mano.
        "version": version,
        # --- Structured response ----------------------------------------------------
        # Las reglas de rechazo de v3 salen de las constantes del schema que las
        # valida, así el prompt y el validador no pueden discrepar.
        "min_confidence_pct": MIN_CONFIDENCE_PCT,
        "out_of_scope_prefix": OUT_OF_SCOPE_PREFIX,
        "unestimated_phase": UNESTIMATED_PHASE,
        # --- Sesiones conversacionales (v5) -----------------------------------------
        # None cuando no hay hechos conocidos: un objeto Pydantic siempre es "verdadero"
        # para Jinja, así que la plantilla no podría distinguir "vacía" de "con datos".
        "project_metadata": (
            project_metadata
            if project_metadata is not None and not project_metadata.is_empty()
            else None
        ),
        "tier": tier,
        # --- ACB (Paso 8): None en la primera vuelta y en las versiones sin ACB.
        "critic_feedback": critic_feedback if critic_feedback and critic_feedback.issues else None,
    }
    system = _env.get_template(f"estimation/{version}/system.j2").render(**context)
    user = _env.get_template(f"estimation/{version}/user.j2").render(**context)
    return system, user


# --- Extractor de project_metadata (Paso 4) -----------------------------------------


def render_metadata_extraction_prompt(
    current: ProjectMetadata,
    *,
    user_turn: str,
    assistant_turn: str,
    version: str = "v1",
) -> tuple[str, str]:
    """Prompts (sistema y usuario) del extractor de `project_metadata`.

    Entrada: la metadata actual y el último turno (lo que dijo el usuario y lo que
    respondió el asistente). La plantilla vive en
    ``app/prompts/metadata_extraction/<version>/`` y es independiente de las de
    estimación. Devuelve ``(system_prompt, user_prompt)``.
    """
    context = {
        "current_metadata": ExtractedMetadata.from_project_metadata(current).model_dump(),
        "user_turn": user_turn,
        "assistant_turn": assistant_turn,
    }
    system = _env.get_template(f"metadata_extraction/{version}/system.j2").render(**context)
    user = _env.get_template(f"metadata_extraction/{version}/user.j2").render(**context)
    return system, user


# --- Resumidor de la conversación (Paso 9) ---------------------------------------------

# Versión por defecto de la plantilla del resumidor (app/prompts/conversation_summary/<versión>/).
SUMMARY_PROMPT_VERSION = "v1"


def render_conversation_summary_prompt(
    *,
    previous_summary: str | None,
    evicted: list[Message],
    version: str = SUMMARY_PROMPT_VERSION,
) -> tuple[str, str]:
    """Prompts (sistema y usuario) del resumidor acumulativo del historial.

    Entrada: el resumen que ya existía (o `None`) y los mensajes que acaban de salir de
    la ventana. Devuelve ``(system_prompt, user_prompt)``. El texto de los mensajes pasa
    por `escape_tags` en la plantilla, así que no puede cerrar los bloques del prompt.
    """
    context = {"previous_summary": previous_summary, "evicted": evicted}
    system = _env.get_template(f"conversation_summary/{version}/system.j2").render(**context)
    user = _env.get_template(f"conversation_summary/{version}/user.j2").render(**context)
    return system, user


# --- Bucle agéntico (Paso 10) ---------------------------------------------------------

# Versión por defecto de la plantilla del bucle agéntico (app/prompts/agent/<versión>/).
AGENT_PROMPT_VERSION = "v1"


def render_agent_prompt(
    *,
    transcript: str,
    attachments_block: str = "",
    version: str = AGENT_PROMPT_VERSION,
) -> tuple[str, str]:
    """Prompts (sistema y usuario) de la fase de recogida de datos con herramientas.

    El sistema no tiene variables; el usuario lleva la transcripción (escapada con
    `escape_tags`) y el bloque de adjuntos ya armado. Devuelve ``(system_prompt, user_prompt)``.
    """
    context = {"transcript": transcript, "attachments_block": attachments_block}
    system = _env.get_template(f"agent/{version}/system.j2").render(**context)
    user = _env.get_template(f"agent/{version}/user.j2").render(**context)
    return system, user


# --- Crítico del ACB (Paso 8) -------------------------------------------------------

# Versión por defecto de la plantilla del crítico (app/prompts/critic/<versión>/).
CRITIC_PROMPT_VERSION = "v1"


def render_critic_prompt(
    *,
    transcript: str,
    attachments_block: str,
    project_metadata: ProjectMetadata,
    tier: str,
    detail_level: str,
    result: StructuredResult,
    actor_version: str,
    version: str = CRITIC_PROMPT_VERSION,
) -> tuple[str, str]:
    """Prompts (sistema y usuario) del crítico que audita una estimación.

    El crítico recibe lo mismo que vio el actor (transcripción, adjuntos, memoria del
    proyecto y audiencia) más la estimación a revisar, serializada como JSON. Las
    tarifas y las horas productivas salen de la versión de prompt del actor
    (`actor_version`), de modo que el crítico juzga con las mismas reglas con las que se
    estimó. La estimación y la transcripción pasan por `escape_tags`: un texto con
    ``</estimation_under_review>`` no puede cerrar el bloque. Devuelve
    ``(system_prompt, user_prompt)``.
    """
    version_data = _load_version_data(actor_version)
    context = {
        "rates": version_data["rates"],
        "productive_hours_per_week": version_data["productive_hours_per_week"],
        "out_of_scope_prefix": OUT_OF_SCOPE_PREFIX,
        "min_confidence_pct": MIN_CONFIDENCE_PCT,
        "totals_max_deviation_pct": TOTALS_MAX_DEVIATION_PCT,
        "transcript": transcript,
        "attachments_block": attachments_block,
        "project_metadata": None if project_metadata.is_empty() else project_metadata,
        "tier": tier,
        "detail_level": detail_level,
        "result_json": _to_json(result.model_dump(mode="json")),
    }
    system = _env.get_template(f"critic/{version}/system.j2").render(**context)
    user = _env.get_template(f"critic/{version}/user.j2").render(**context)
    return system, user
