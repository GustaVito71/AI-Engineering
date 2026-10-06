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

from app.schemas.estimation import EstimationRequest
from app.schemas.structured_estimation import (
    MIN_CONFIDENCE_PCT,
    OUT_OF_SCOPE_PREFIX,
    UNESTIMATED_PHASE,
)

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
_OUTPUT_KINDS = (OUTPUT_TEXT, OUTPUT_STRUCTURED)


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


def render_estimation_prompt(
    request: EstimationRequest,
    version: str = "v1",
) -> tuple[str, str]:
    """Renderiza los prompts de sistema y de usuario para el caso de estimación.

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
    }
    system = _env.get_template(f"estimation/{version}/system.j2").render(**context)
    user = _env.get_template(f"estimation/{version}/user.j2").render(**context)
    return system, user
