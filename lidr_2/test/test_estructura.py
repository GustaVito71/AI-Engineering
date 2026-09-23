"""Valida la estructura de carpetas que exige el ejercicio.

Es la "manera de validar que la estructura de carpetas es correcta": corre en
la suite (pytest) y el pipeline (CI) la ejecuta junto al resto. Si alguien
renombra `test/` a `tests/`, mueve `datos/`, o borra un archivo que el ejercicio
pide, este test falla y el pipeline rojo lo dice.
"""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

SE_ESPERA = [
    # Paquetes de la arquitectura (capas del flujo E2E)
    "app",
    "app/main.py",
    "app/config.py",
    "app/context",
    "app/context/examples.py",
    "app/services",
    "app/services/llm_service.py",
    "app/services/pricing.py",
    "app/providers",
    "app/routers",
    "app/routers/estimations.py",
    # Tests y configuración del ejercicio
    "test",
    "test/conftest.py",
    "test/test_estimations.py",
    "README.md",
    ".env.example",
    "pyproject.toml",
    # La transcripción canónica (parámetro del ejercicio)
    "datos",
    "datos/transcripcion_reunion.md",
]


def test_el_pipeline_vive_en_la_raiz_del_monorepo():
    # GitHub Actions solo descubre workflows en el .github/workflows de la
    # raíz del repo. Un .github anidado en lidr_2/ es invisible para el CI
    # (y de hecho el pipeline no corrió hasta moverlo a la raíz).
    pipeline = REPO_ROOT.parent / ".github" / "workflows" / "ci.yml"
    assert pipeline.is_file()
    assert "working-directory: lidr_2" in pipeline.read_text(encoding="utf-8")
    assert "pytest -q" in pipeline.read_text(encoding="utf-8")
    assert "ruff check ." in pipeline.read_text(encoding="utf-8")


def test_la_estructura_de_carpetas_es_la_del_ejercicio():
    faltantes = [p for p in SE_ESPERA if not (REPO_ROOT / p).exists()]
    assert not faltantes, (
        "Faltan archivos/carpetas que el ejercicio exige:\n  "
        + "\n  ".join(faltantes)
    )


def test_los_secretos_no_viven_en_el_repo():
    # Los archivos susceptibles de llevar secretos no se versionan:
    # .env queda fuera (gitignore) y .env.example existe como plantilla.
    assert (REPO_ROOT / ".env.example").exists()
    assert not (REPO_ROOT / "datos").joinpath(".env").exists()


def test_la_transcripcion_canonica_sirve_al_endpoint():
    """La transcripción del ejercicio es un parámetro válido: el slice entre
    marcadores pasa la validación de entrada del router
    (≥ min chars y ≤ max chars) y contiene diálogo real, no un placeholder."""
    from conftest import make_settings

    settings = make_settings()
    md = (REPO_ROOT / "datos" / "transcripcion_reunion.md").read_text(
        encoding="utf-8"
    )
    inicio = md.index("<!-- transcripcion -->") + len("<!-- transcripcion -->")
    final = md.index("<!-- /transcripcion -->")
    texto = md[inicio:final].strip()
    assert settings.estimation_min_chars <= len(texto) <= settings.estimation_max_chars
    # Dialogo real: hablan al menos dos interlocutores distintos.
    # Formato de cada turno: **Nombre:** texto.
    hablantes = {
        line[2 : line.index(":**")]
        for line in texto.splitlines()
        if line.startswith("**") and ":**" in line
    }
    assert len(hablantes) >= 2