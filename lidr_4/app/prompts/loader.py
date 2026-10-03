"""Jinja2 loader for versioned prompt templates.

The on-disk layout is ``app/prompts/<use_case>/<version>/<role>.j2``. Versioning
is required from day one: switching prompts becomes a string change at the
call site (``version="v2"``), not a code refactor.

Each version has an ``examples.yaml`` with its own roles (rate + display
labels), rounding rules and example data. Published prompts are immutable:
changing behavior means a new version dir.

Division of responsibilities:
- ``examples.yaml``: the data (roles, rates, labels, rounding, examples).
- ``loader.py``: generic arithmetic only (hours, costs, totals, headcount).
  It never decides how anything is displayed, so a new version can change
  the presentation without touching Python.
- ``system.j2``: all presentation (labels, plurals, team summary, layout).
"""

from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
from math import ceil
from pathlib import Path

import yaml
from jinja2 import Environment, FileSystemLoader, StrictUndefined, TemplateNotFound

from app.schemas.estimation import EstimationRequest

_BASE_DIR = Path(__file__).resolve().parent

# Fields every role in `rates` must declare. `label`/`plural` are consumed by
# the template, `eur_per_hour` by the arithmetic below. Validated at load time
# so a malformed YAML fails on the first request, not halfway through a render.
_ROLE_FIELDS = ("label", "plural", "eur_per_hour")


def _round_up(n: float, base: int) -> int:
    """Round up to the nearest multiple of base. Works with floats."""
    return int(ceil(n / base) * base)


# Cache raw YAML data per version
@lru_cache(maxsize=8)
def _load_raw_yaml(version: str) -> dict:
    yaml_path = _BASE_DIR / "estimation" / version / "examples.yaml"
    try:
        with yaml_path.open(encoding="utf-8") as f:
            return yaml.safe_load(f)
    except FileNotFoundError as e:
        raise TemplateNotFound(f"examples.yaml for version {version}") from e


def _validate(data: dict, version: str) -> None:
    """Fail fast on a malformed examples.yaml, naming what is missing."""
    rates = data["rates"]
    for role, spec in rates.items():
        missing = [f for f in _ROLE_FIELDS if f not in (spec or {})]
        if missing:
            raise ValueError(
                f"Role '{role}' in estimation/{version}/examples.yaml is missing "
                f"{', '.join(missing)}. Every role needs: {', '.join(_ROLE_FIELDS)}."
            )
    for ex in data["examples"]:
        for p in ex["phases"]:
            for role in p.get("team", {}):
                if role not in rates:
                    raise ValueError(
                        f"Role '{role}' in example '{ex['project_description'][:40]}...' "
                        f"not defined in rates. Available: {list(rates)}"
                    )


@lru_cache(maxsize=8)
def _compute_version_data(version: str) -> dict:
    """Read examples.yaml once per version and do all the arithmetic.

    Cached: the YAML is read and the numbers computed once per process.
    Callers go through `_load_version_data`, which hands out a copy.
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

        # Peak headcount per role across phases, in the order roles are
        # declared in `rates`. Pure aggregation: the template decides how to
        # display it (labels, plurals, separators).
        team_headcount = {
            role: max(p.get("team", {}).get(role, 0) for p in ex["phases"]) for role in rates
        }
        team_headcount = {role: hc for role, hc in team_headcount.items() if hc > 0}

        computed_examples.append(
            {
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
    """Computed data for a version, as a copy so callers can't mutate the cache.

    Returns a dict with keys:
    - rates: {role: {label, plural, eur_per_hour}}
    - rounding: {hours_base, cost_base}
    - productive_hours_per_week: int
    - examples: list of computed examples (phases with hours, cost_eur,
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


def render_estimation_prompt(
    request: EstimationRequest,
    version: str = "v1",
) -> tuple[str, str]:
    """Render the system and user prompts for the estimation use case.

    Returns:
        A tuple ``(system_prompt, user_prompt)`` ready to be sent to the LLM
        as separate ``role: "system"`` and ``role: "user"`` messages.
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
    }
    system = _env.get_template(f"estimation/{version}/system.j2").render(**context)
    user = _env.get_template(f"estimation/{version}/user.j2").render(**context)
    return system, user
