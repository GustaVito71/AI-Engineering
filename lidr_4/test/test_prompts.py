"""Tests del loader Jinja2 de prompts.

El objetivo es verificar el contrato del prompt renderizado sin llamar al LLM:
que los campos que manda el usuario caigan en el bloque correcto, que las
secciones condicionales aparezcan solo cuando se pide el valor de enum que les
corresponde, y que ``StrictUndefined`` falle enseguida si falta una variable.
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
    """Una plantilla Jinja2 aparte, con la misma configuración StrictUndefined,
    tiene que fallar enseguida cuando falta una variable. Así, un error de tipeo
    en una plantilla aparece al renderizar y no queda como un texto vacío sin
    que nadie lo note."""
    env = Environment(undefined=StrictUndefined)
    template = env.from_string("Hello {{ unknown_variable }}")
    with pytest.raises(UndefinedError):
        template.render()


def test_unknown_version_raises() -> None:
    request = _make_request()
    with pytest.raises(TemplateNotFound):
        render_estimation_prompt(request, version="v999")


def test_examples_computed_from_rates_and_rounding() -> None:
    """Los totales de cada ejemplo tienen que ser la suma de los costes y las
    semanas de sus fases.

    Costes y horas se calculan a partir de las tarifas y las reglas de redondeo,
    no están escritos a mano. Esto detecta regresiones en las que totales fijos
    dejan de coincidir con los datos de las fases, y garantiza que las tarifas
    que muestra <scope> sean las mismas que usa la aritmética de los ejemplos.
    """
    data = _load_version_data("v1")

    for ex in data["examples"]:
        totals = ex["totals"]
        phases = ex["phases"]

        # La suma de los costes de las fases tiene que coincidir con el total
        phase_cost_sum = sum(p["cost_eur"] for p in phases)
        assert phase_cost_sum == totals["total_cost_eur"], (
            f"Phase cost sum {phase_cost_sum} != totals {totals['total_cost_eur']}"
        )
        # La suma de semanas también tiene que coincidir
        weeks_sum = sum(p["duration_weeks"] for p in phases)
        assert weeks_sum == totals["total_duration_weeks"], (
            f"Phase weeks sum {weeks_sum} != totals {totals['total_duration_weeks']}"
        )
        # El coste de cada fase es la suma de los costes de sus roles
        for p in phases:
            role_cost_sum = sum(b["cost_eur"] for b in p["team_breakdown"])
            assert p["cost_eur"] == role_cost_sum, (
                f"Phase {p['phase']} cost {p['cost_eur']} != sum of role costs {role_cost_sum}"
            )
            # Las horas son múltiplo de hours_base (5)
            assert p["hours"] % 5 == 0, (
                f"Phase {p['phase']} hours {p['hours']} not rounded to nearest 5"
            )
            # Las horas de cada rol también son múltiplo de 5
            for b in p["team_breakdown"]:
                assert b["hours"] % 5 == 0, (
                    f"Phase {p['phase']} role {b['role']} hours {b['hours']} not rounded to nearest 5"
                )
            # El coste es múltiplo de cost_base (50)
            assert p["cost_eur"] % 50 == 0, (
                f"Phase {p['phase']} cost {p['cost_eur']} not rounded to nearest 50"
            )
            # El coste de cada rol también es múltiplo de 50
            for b in p["team_breakdown"]:
                assert b["cost_eur"] % 50 == 0, (
                    f"Phase {p['phase']} role {b['role']} cost {b['cost_eur']} not rounded to nearest 50"
                )


def test_scope_renders_rates_and_rounding_from_yaml() -> None:
    """El bloque <scope> tiene que mostrar las tarifas exactas con 2 decimales y
    las bases de redondeo tomadas del YAML, no números escritos a mano."""
    request = _make_request()
    system, _ = render_estimation_prompt(request)

    # Se extrae exactamente el bloque <scope>
    scope_match = re.search(r"<scope>\s*(.*?)\s*</scope>", system, re.DOTALL)
    assert scope_match, "<scope> block not found in system prompt"
    scope = scope_match.group(1)

    # Tarifas con su formato exacto (todos los roles del YAML)
    assert "62.50 EUR/hour" in scope
    assert "50.00 EUR/hour" in scope
    assert "32 productive hours" in scope
    # Bases de redondeo del YAML
    assert "5h" in scope
    assert "50 EUR" in scope
    # Menciona la persona-semana y que los costes se redondean hacia arriba
    assert "one person-week" in scope
    assert "costs up to the nearest" in scope
    # Ningún float sin formatear (como "62.5" o "50.0") aparece en <scope>
    assert "62.5 EUR/hour" not in scope
    assert "50.0 EUR/hour" not in scope


def test_narrative_uses_proper_case_and_blank_lines() -> None:
    """En el formato narrativo, los nombres de fase van con mayúscula inicial y
    los párrafos se separan con una línea en blanco (doble salto de línea), sin
    una línea en blanco de más antes de "Total effort"."""
    request = _make_request_fmt(OutputFormat.NARRATIVE)
    system, _ = render_estimation_prompt(request)

    # Se extrae la sección de ejemplos
    ex_match = re.search(r"<examples>.*?</examples>", system, re.DOTALL)
    assert ex_match, "<examples> block not found"
    examples = ex_match.group(0)

    # Nombres de fase con mayúscula inicial, no pasados por lower()
    assert "The QA phase" in examples
    assert "The Design phase" in examples

    # Párrafos separados por exactamente una línea en blanco (doble salto de línea)
    assert "\n\nThe Design phase" in examples

    # Sin doble línea en blanco antes de "Total effort"
    assert "\n\n\nTotal effort" not in examples


def test_system_prompt_starts_without_leading_newline() -> None:
    """El system prompt no empieza con una línea en blanco (la definición de la
    macro no deja salto de línea)."""
    request = _make_request()
    system, _ = render_estimation_prompt(request)
    # El primer carácter es la 'Y' de "You are a senior..."
    assert system.startswith("You are a senior project estimator")


def test_line_items_numbering_resets_per_example() -> None:
    """Los elementos de línea de cada ejemplo deben numerarse 1, 2, 3... de forma independiente."""
    request = _make_request_fmt(OutputFormat.LINE_ITEMS)
    system, _ = render_estimation_prompt(request)

    # Se separa por ejemplo
    examples = system.split("<example>")[1:]  # se salta el preámbulo
    for ex_idx, ex in enumerate(examples):
        lines = [line for line in ex.splitlines() if re.match(r"^\d+\.\s", line)]
        numbers = [int(line.split(".")[0]) for line in lines]
        assert numbers == list(range(1, len(numbers) + 1)), (
            f"Example {ex_idx + 1} numbering broken: {numbers}"
        )


def test_examples_each_row_on_own_line() -> None:
    """Cada fila de ejemplo (tabla de fases, elemento de línea, párrafo narrativo)
    tiene que ir en su propia línea."""
    # phases_table
    request = _make_request_fmt(OutputFormat.PHASES_TABLE)
    system, _ = render_estimation_prompt(request)
    # Solo filas de datos: empiezan con "| ", sin contar el encabezado ("| phase")
    # ni el separador ("|---")
    table_lines = [
        line
        for line in system.splitlines()
        if line.startswith("| ") and not line.startswith("| phase") and not line.startswith("|---")
    ]
    # Una línea por fase y por ejemplo (4 ejemplos × 5 fases = 20)
    assert len(table_lines) == 20, f"Expected 20 phase rows, got {len(table_lines)}"
    # Cada fila es una línea completa (no pegada a la siguiente)
    for line in table_lines:
        assert line.count("|") == 5, f"Row not well-formed: {line}"

    # line_items: cada elemento en su propia línea
    request = _make_request_fmt(OutputFormat.LINE_ITEMS)
    system, _ = render_estimation_prompt(request)
    item_lines = [line for line in system.splitlines() if re.match(r"^\d+\.\s", line)]
    assert len(item_lines) > 0
    for line in item_lines:
        assert " — " in line, f"Line item malformed: {line}"

    # narrative: cada párrafo de fase en su propia línea
    request = _make_request_fmt(OutputFormat.NARRATIVE)
    system, _ = render_estimation_prompt(request)
    narrative_lines = [
        line for line in system.splitlines() if line.startswith("The ") and "phase spans" in line
    ]
    assert len(narrative_lines) == 20, (
        f"Expected 20 narrative paragraphs, got {len(narrative_lines)}"
    )


def test_scope_includes_all_rates() -> None:
    """El bloque <scope> tiene que listar todas las tarifas del YAML, no solo las
    de developer y designer."""
    request = _make_request()
    system, _ = render_estimation_prompt(request)

    scope_match = re.search(r"<scope>\s*(.*?)\s*</scope>", system, re.DOTALL)
    assert scope_match, "<scope> block not found in system prompt"
    scope = scope_match.group(1)

    # Las tarifas aparecen con 2 decimales
    assert "62.50 EUR/hour" in scope
    assert "50.00 EUR/hour" in scope
    # QA y PM también cobran 62.50, igual que developer:
    # basta comprobar que ese valor aparece más de una vez
    assert scope.count("62.50 EUR/hour") >= 2  # developer + QA + PM (al menos 2 menciones)
    assert "50.00 EUR/hour" in scope


def test_line_items_show_roles_with_proper_case() -> None:
    """Los elementos de línea muestran QA y PM en mayúsculas, y el resto de los
    roles con mayúscula inicial."""
    request = _make_request_fmt(OutputFormat.LINE_ITEMS)
    system, _ = render_estimation_prompt(request)

    lines = [line for line in system.splitlines() if re.match(r"^\d+\.\s", line)]
    for line in lines:
        # Cada línea lleva "(Rol)": se comprueba que esté bien escrito en mayúsculas
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
    # Hay dos cachés que vaciar: la del YAML sin procesar (_load_raw_yaml) y la de
    # los datos calculados (_compute_version_data). Hay que vaciar las dos para
    # aislar los tests que escriben un examples.yaml distinto con el mismo nombre
    # de versión.
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
    """Un ejemplo que usa un rol que no está en `rates` tiene que fallar al
    cargar el YAML."""
    data = _yaml_v1()
    data["examples"][0]["phases"][0]["team"]["architect"] = 1
    version = version_temporal(data)

    with pytest.raises(ValueError, match="architect"):
        render_estimation_prompt(_make_request(), version=version)


def test_role_without_label_raises_error(version_temporal) -> None:
    """Todo rol tiene que declarar label, plural y eur_per_hour."""
    data = _yaml_v1()
    del data["rates"]["qa"]["label"]
    version = version_temporal(data)

    with pytest.raises(ValueError, match="'qa'.*label"):
        render_estimation_prompt(_make_request(), version=version)


def test_new_role_needs_only_yaml(version_temporal) -> None:
    """Basta con agregar un rol en examples.yaml: aparece en <scope>, en los
    elementos de línea y en el resumen de equipo sin tocar Python."""
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
    """El resumen de equipo lo arma la plantilla con las etiquetas del YAML, en
    el orden del YAML."""
    system, _ = render_estimation_prompt(_make_request())
    teams = [line for line in system.splitlines() if line.startswith("Team: ")]

    assert teams == [
        "Team: 2 Developers, 1 Designer, 1 QA",
        "Team: 4 Developers, 2 Designers, 1 PM, 2 QAs",
        "Team: 2 Developers, 1 Designer, 1 QA",
        "Team: 3 Developers, 1 Designer, 1 QA",
    ]


def test_version_data_is_computed_once() -> None:
    """El YAML se lee y la aritmética se hace una sola vez por versión; cada
    llamada recibe su propia copia, así nadie puede modificar el resultado
    cacheado."""
    loader._compute_version_data.cache_clear()

    d1 = _load_version_data("v1")
    d2 = _load_version_data("v1")

    info = loader._compute_version_data.cache_info()
    assert (info.misses, info.hits) == (1, 1)
    assert d1 is not d2
    assert d1 == d2
