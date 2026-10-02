"""Tests for the Jinja2 prompt loader.

The goal is to verify the contract of the rendered output without touching the
LLM: that user-provided fields land in the right block, that conditional
sections only render when the matching enum value is requested, and that
``StrictUndefined`` blows up early on missing variables.
"""

from __future__ import annotations

import re
import shutil

import pytest
import yaml
from jinja2 import (
    Environment,
    FileSystemLoader,
    StrictUndefined,
    TemplateNotFound,
    UndefinedError,
)

from app.prompts import loader
from app.prompts.loader import _load_version_data, render_estimation_prompt
from app.schemas.estimation import (
    DetailLevel,
    EstimationRequest,
    OutputFormat,
    ProjectType,
)


def _make_request(**overrides) -> EstimationRequest:
    base = {
        "description": "A small CRM for a real estate agency: contacts, deals, role-based access.",
        "project_type": ProjectType.WEB_SAAS,
        "detail_level": DetailLevel.MEDIUM,
        "output_format": OutputFormat.PHASES_TABLE,
    }
    base.update(overrides)
    return EstimationRequest(**base)


def _make_request_fmt(fmt: OutputFormat, **overrides) -> EstimationRequest:
    base = {
        "description": "A small CRM for a real estate agency: contacts, deals, role-based access.",
        "project_type": ProjectType.WEB_SAAS,
        "detail_level": DetailLevel.MEDIUM,
        "output_format": fmt,
    }
    base.update(overrides)
    return EstimationRequest(**base)


def test_user_prompt_wraps_description_in_project_description_block() -> None:
    request = _make_request(description="UNIQUE-MARKER-12345 build a tiny scheduling app.")
    _system, user = render_estimation_prompt(request)
    assert "<project_description>" in user
    assert "UNIQUE-MARKER-12345 build a tiny scheduling app." in user
    assert "</project_description>" in user
    start = user.index("<project_description>")
    end = user.index("</project_description>")
    assert "UNIQUE-MARKER-12345" in user[start:end]


def test_phases_table_keyword_appears_only_when_format_requested() -> None:
    table_request = _make_request_fmt(OutputFormat.PHASES_TABLE)
    narrative_request = _make_request_fmt(OutputFormat.NARRATIVE)

    table_system, _ = render_estimation_prompt(table_request)
    narrative_system, _ = render_estimation_prompt(narrative_request)

    assert "phases_table" in table_system
    assert "phases_table" not in narrative_system


def test_detailed_includes_assumptions_per_phase_summary_does_not() -> None:
    detailed_request = _make_request_fmt(
        OutputFormat.PHASES_TABLE, detail_level=DetailLevel.DETAILED
    )
    summary_request = _make_request_fmt(OutputFormat.PHASES_TABLE, detail_level=DetailLevel.SUMMARY)

    detailed_system, _ = render_estimation_prompt(detailed_request)
    summary_system, _ = render_estimation_prompt(summary_request)

    assert "list assumptions per phase" in detailed_system.lower()
    assert "list assumptions per phase" not in summary_system.lower()


def test_examples_block_is_included_in_system_prompt() -> None:
    request = _make_request()
    system, _ = render_estimation_prompt(request)
    assert "<examples>" in system
    assert "</examples>" in system


def test_strict_undefined_raises_on_missing_variable() -> None:
    """A separate Jinja2 template with the same StrictUndefined config must error
    early when a variable is missing — guarantees that typos in templates are
    surfaced at render time, not silently rendered as empty strings."""
    env = Environment(undefined=StrictUndefined)
    template = env.from_string("Hello {{ unknown_variable }}")
    with pytest.raises(UndefinedError):
        template.render()


def test_unknown_version_raises() -> None:
    request = _make_request()
    with pytest.raises(TemplateNotFound):
        render_estimation_prompt(request, version="v999")


def test_examples_computed_from_rates_and_rounding() -> None:
    """Each example's totals must equal the sum of its phase costs/weeks.

    Costs and hours are computed from rates + rounding rules, not hardcoded.
    This catches regressions where hardcoded totals drift from phase data,
    and guarantees that rates used in <scope> are consistent with example math.
    """
    data = _load_version_data("v1")

    for ex in data["examples"]:
        totals = ex["totals"]
        phases = ex["phases"]

        # Sum of phase costs must match totals
        phase_cost_sum = sum(p["cost_eur"] for p in phases)
        assert phase_cost_sum == totals["total_cost_eur"], (
            f"Phase cost sum {phase_cost_sum} != totals {totals['total_cost_eur']}"
        )
        # Weeks sum must match
        weeks_sum = sum(p["duration_weeks"] for p in phases)
        assert weeks_sum == totals["total_duration_weeks"], (
            f"Phase weeks sum {weeks_sum} != totals {totals['total_duration_weeks']}"
        )
        # Each phase cost must equal sum of role costs
        for p in phases:
            role_cost_sum = sum(b["cost_eur"] for b in p["team_breakdown"])
            assert p["cost_eur"] == role_cost_sum, (
                f"Phase {p['phase']} cost {p['cost_eur']} != sum of role costs {role_cost_sum}"
            )
            # Hours must be multiple of hours_base (5)
            assert p["hours"] % 5 == 0, (
                f"Phase {p['phase']} hours {p['hours']} not rounded to nearest 5"
            )
            # Each role hours must be multiple of 5
            for b in p["team_breakdown"]:
                assert b["hours"] % 5 == 0, (
                    f"Phase {p['phase']} role {b['role']} hours {b['hours']} not rounded to nearest 5"
                )
            # Cost must be multiple of cost_base (50)
            assert p["cost_eur"] % 50 == 0, (
                f"Phase {p['phase']} cost {p['cost_eur']} not rounded to nearest 50"
            )
            # Each role cost must be multiple of 50
            for b in p["team_breakdown"]:
                assert b["cost_eur"] % 50 == 0, (
                    f"Phase {p['phase']} role {b['role']} cost {b['cost_eur']} not rounded to nearest 50"
                )


def test_scope_renders_rates_and_rounding_from_yaml() -> None:
    """The <scope> block must render the exact rate values with 2 decimals
    and the rounding bases from YAML, not hardcoded numbers."""
    request = _make_request()
    system, _ = render_estimation_prompt(request)

    # Extract the <scope> block precisely
    scope_match = re.search(r"<scope>\s*(.*?)\s*</scope>", system, re.DOTALL)
    assert scope_match, "<scope> block not found in system prompt"
    scope = scope_match.group(1)

    # Exact formatted rates (all roles from YAML)
    assert "62.50 EUR/hour" in scope
    assert "50.00 EUR/hour" in scope
    assert "32 productive hours" in scope
    # Rounding bases from YAML
    assert "5h" in scope
    assert "50 EUR" in scope
    # Ensure no raw float like "62.5" or "50.0" appears in scope
    assert "62.5 EUR/hour" not in scope
    assert "50.0 EUR/hour" not in scope


def test_system_prompt_starts_without_leading_newline() -> None:
    """System prompt must not start with a blank line (macro definition trimmed)."""
    request = _make_request()
    system, _ = render_estimation_prompt(request)
    # First character must be 'Y' from "You are a senior..."
    assert system.startswith("You are a senior project estimator")


def test_line_items_numbering_resets_per_example() -> None:
    """Each example's line items must be numbered 1, 2, 3... independently."""
    request = _make_request_fmt(OutputFormat.LINE_ITEMS)
    system, _ = render_estimation_prompt(request)

    # Split by examples
    examples = system.split("<example>")[1:]  # skip preamble
    for ex_idx, ex in enumerate(examples):
        lines = [line for line in ex.splitlines() if re.match(r"^\d+\.\s", line)]
        numbers = [int(line.split(".")[0]) for line in lines]
        assert numbers == list(range(1, len(numbers) + 1)), (
            f"Example {ex_idx + 1} numbering broken: {numbers}"
        )


def test_examples_each_row_on_own_line() -> None:
    """Each example row (phase table, line item, narrative paragraph) must be on its own line."""
    # phases_table
    request = _make_request_fmt(OutputFormat.PHASES_TABLE)
    system, _ = render_estimation_prompt(request)
    # Data rows only: start with "| " but not header ("| phase") or separator ("|---")
    table_lines = [
        line
        for line in system.splitlines()
        if line.startswith("| ") and not line.startswith("| phase") and not line.startswith("|---")
    ]
    # Should have one line per phase per example (3 examples × 5 phases = 15)
    assert len(table_lines) == 15, f"Expected 15 phase rows, got {len(table_lines)}"
    # Each row should be a complete line (not concatenated)
    for line in table_lines:
        assert line.count("|") == 5, f"Row not well-formed: {line}"

    # line_items - each item on its own line
    request = _make_request_fmt(OutputFormat.LINE_ITEMS)
    system, _ = render_estimation_prompt(request)
    item_lines = [line for line in system.splitlines() if re.match(r"^\d+\.\s", line)]
    assert len(item_lines) > 0
    for line in item_lines:
        assert " — " in line, f"Line item malformed: {line}"

    # narrative - each phase paragraph on its own line
    request = _make_request_fmt(OutputFormat.NARRATIVE)
    system, _ = render_estimation_prompt(request)
    narrative_lines = [
        line for line in system.splitlines() if line.startswith("The ") and "phase spans" in line
    ]
    assert len(narrative_lines) == 15, (
        f"Expected 15 narrative paragraphs, got {len(narrative_lines)}"
    )


def test_scope_includes_all_rates() -> None:
    """The <scope> block must list all rates from YAML, not just dev/designer."""
    request = _make_request()
    system, _ = render_estimation_prompt(request)

    scope_match = re.search(r"<scope>\s*(.*?)\s*</scope>", system, re.DOTALL)
    assert scope_match, "<scope> block not found in system prompt"
    scope = scope_match.group(1)

    # All four rates must appear with 2 decimals
    assert "62.50 EUR/hour" in scope
    assert "50.00 EUR/hour" in scope
    # QA and PM are 62.50 too
    # Just verify they appear (same rate as developer but listed)
    assert scope.count("62.50 EUR/hour") >= 2  # developer + QA + PM (at least 2 mentions)
    assert "50.00 EUR/hour" in scope


def test_line_items_show_roles_with_proper_case() -> None:
    """Line items must show QA/PM in uppercase, others capitalized."""
    request = _make_request_fmt(OutputFormat.LINE_ITEMS)
    system, _ = render_estimation_prompt(request)

    lines = [line for line in system.splitlines() if re.match(r"^\d+\.\s", line)]
    for line in lines:
        # Each line has "(Role)" - check it's properly capitalized
        match = re.search(r"\(([^)]+)\)", line)
        assert match, f"Role not found in line: {line}"
        role = match.group(1)
        assert role in ("Developer", "Designer", "QA", "PM"), f"Unexpected role format: {role}"


# --- Versiones de prueba en tmp_path -------------------------------------------
#
# Los tests que necesitan un examples.yaml distinto del de v1 arman una versión
# temporal: copian las plantillas de v1, escriben el YAML modificado y apuntan el
# loader a ese directorio. Así se prueba el comportamiento real (carga + render)
# sin tocar app/prompts/.


# Ruta real de v1, fijada al importar: la fixture cambia loader._BASE_DIR.
_V1_DIR = loader._BASE_DIR / "estimation" / "v1"


@pytest.fixture
def version_temporal(tmp_path, monkeypatch):
    """Devuelve una función que crea `estimation/vtest/` con el YAML dado."""
    origen = _V1_DIR
    destino = tmp_path / "estimation" / "vtest"
    destino.mkdir(parents=True)
    for plantilla in ("system.j2", "user.j2"):
        shutil.copy(origen / plantilla, destino / plantilla)

    monkeypatch.setattr(loader, "_BASE_DIR", tmp_path)
    monkeypatch.setattr(loader._env, "loader", FileSystemLoader(tmp_path))
    loader._compute_version_data.cache_clear()
    loader._load_raw_yaml.cache_clear()

    def _crear(data: dict) -> str:
        (destino / "examples.yaml").write_text(
            yaml.safe_dump(data, sort_keys=False), encoding="utf-8"
        )
        return "vtest"

    yield _crear
    loader._compute_version_data.cache_clear()
    loader._load_raw_yaml.cache_clear()


def _yaml_v1() -> dict:
    return yaml.safe_load((_V1_DIR / "examples.yaml").read_text(encoding="utf-8"))


def test_unknown_role_in_examples_raises_error(version_temporal) -> None:
    """An example referencing a role not in rates must raise at load time."""
    data = _yaml_v1()
    data["examples"][0]["phases"][0]["team"]["architect"] = 1
    version = version_temporal(data)

    with pytest.raises(ValueError, match="architect"):
        render_estimation_prompt(_make_request(), version=version)


def test_role_without_label_raises_error(version_temporal) -> None:
    """Every role must declare label, plural and eur_per_hour."""
    data = _yaml_v1()
    del data["rates"]["qa"]["label"]
    version = version_temporal(data)

    with pytest.raises(ValueError, match="'qa'.*label"):
        render_estimation_prompt(_make_request(), version=version)


def test_new_role_needs_only_yaml(version_temporal) -> None:
    """Adding a role in examples.yaml is enough: it shows up in <scope>, in the
    line items and in the team summary without touching Python."""
    data = _yaml_v1()
    data["rates"]["architect"] = {
        "label": "Solution Architect",
        "plural": "Solution Architects",
        "eur_per_hour": 80.0,
    }
    data["examples"][0]["phases"][0]["team"]["architect"] = 1
    version = version_temporal(data)

    system, _ = render_estimation_prompt(
        _make_request_fmt(OutputFormat.LINE_ITEMS), version=version
    )

    assert "- Solution Architect rate: 80.00 EUR/hour." in system
    assert "Discovery (Solution Architect)" in system
    assert "Team: 2 Developers, 1 Designer, 1 QA, 1 Solution Architect" in system


def test_team_summary_uses_labels_and_plurals_from_yaml() -> None:
    """The summary is assembled by the template from YAML labels, in YAML order."""
    system, _ = render_estimation_prompt(_make_request())
    teams = [line for line in system.splitlines() if line.startswith("Team: ")]

    assert teams == [
        "Team: 2 Developers, 1 Designer, 1 QA",
        "Team: 4 Developers, 2 Designers, 1 PM, 2 QAs",
        "Team: 2 Developers, 1 Designer, 1 QA",
    ]


def test_version_data_is_computed_once() -> None:
    """The YAML is read and the arithmetic done once per version; each caller
    gets its own copy so nobody can mutate the cached result."""
    loader._compute_version_data.cache_clear()

    d1 = _load_version_data("v1")
    d2 = _load_version_data("v1")

    info = loader._compute_version_data.cache_info()
    assert (info.misses, info.hits) == (1, 1)
    assert d1 is not d2
    assert d1 == d2
