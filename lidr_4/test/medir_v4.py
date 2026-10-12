"""Medición de `v4` con el modelo real: intentos, coste, tarifas, redondeo y variación.

NO es un test: llama al proveedor (cuesta dinero), no es determinista y no tiene
aprobado ni fallo, solo números. Por eso no empieza con `test_` y pytest no lo
recoge. Usa el LLMWrapper real, sin caché (una repetición no puede salir de la
caché), con las claves y el modelo que lea Settings del `.env`.

Corre los 9 casos (3 niveles de detalle × 3 formatos de salida) con la misma
descripción, `--repetitions` veces cada uno, y por cada ejecución anota:

- intentos: respuestas que pidió Instructor (1 = la primera sirvió), y el motivo
  de cada reintento. Responde al riesgo «el validador de `rendered` rechaza una
  respuesta correcta», que PLAN §9 deja sin medir;
- coste en USD y tokens;
- avisos de totales (diferencias de hasta TOTALS_MAX_DEVIATION_PCT);
- fases cuyo coste por hora queda fuera de las tarifas del prompt;
- cifras que no son múltiplo del redondeo del prompt (horas y coste);
- variación entre repeticiones del mismo caso (horas y coste totales).

Salida: una tabla por pantalla y dos CSV, uno con una fila por ejecución
(`--csv`) y otro con una fila por fase (`--phases-csv`), para ver cuál fase queda
fuera de tarifa o sin redondear.

Uso, desde la raíz de lidr_4 (Settings lee `.env` del directorio actual):

    uv run python test/medir_v4.py                    # 9 casos × 3 repeticiones
    uv run python test/medir_v4.py -n 2 --csv med.csv

Hace 9 × N llamadas reales, una tras otra.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import itertools
import sys
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import instructor

# El script vive en test/: se agrega la raíz del proyecto para importar `app`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import LLMConfigurationError, get_settings
from app.main import configure_logging
from app.prompts.loader import _load_version_data, render_estimation_prompt
from app.schemas.estimation import (
    DetailLevel,
    EstimationRequest,
    OutputFormat,
    ProjectType,
)
from app.schemas.structured_estimation import RenderedResult, format_number
from app.services.cache import create_estimation_cache
from app.services.llm_wrapper import LLMWrapper, StructuredOutputError

DESCRIPTION = (
    "Plataforma web SaaS B2B para que empresas medianas gestionen el préstamo de "
    "equipos a sus empleados: inventario, solicitudes con aprobación, avisos por "
    "correo, informes mensuales e integración con el directorio corporativo."
)
# Margen para comparar números decimales sin falsos desvíos.
EPSILON = 1e-9
# Largo máximo del motivo de un reintento o de un error en el CSV y en pantalla.
REASON_CHARS = 300


@dataclass
class Row:
    """Una ejecución. Los campos de la estimación quedan en 0 si no hubo respuesta válida."""

    detail: str
    output_format: str
    repetition: int
    ok: bool = False
    model: str = ""
    attempts: int = 0
    retry_reasons: str = ""
    cost_usd: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    seconds: float = 0.0
    phases: int = 0
    total_hours: int = 0
    total_cost_eur: float = 0.0
    warnings: int = 0
    off_rate_phases: int = 0
    not_rounded: int = 0
    values: int = 0
    out_of_scope: bool = False
    error: str = ""


@dataclass
class PhaseRow:
    """Una fase de una ejecución válida."""

    detail: str
    output_format: str
    repetition: int
    name: str
    hours: int
    cost_eur: float
    eur_per_hour: float
    off_rate: bool
    hours_rounded: bool
    cost_rounded: bool


# Motivos de reintento de la ejecución en curso. Instructor avisa de cada respuesta
# que no cumplió el schema por el hook `parse:error`; el wrapper no los expone,
# así que el script engancha el hook en cada cliente que el wrapper crea.
_parse_errors: list[str] = []


def _short(text: object) -> str:
    """Una línea, sin el volcado del input que agrega pydantic tras el mensaje."""
    return " ".join(str(text).split()).split(" [type=")[0][:REASON_CHARS]


def _track_parse_errors() -> None:
    original = instructor.from_litellm

    def from_litellm(*args, **kwargs):
        client = original(*args, **kwargs)
        client.on("parse:error", lambda error: _parse_errors.append(_short(error)))
        return client

    instructor.from_litellm = from_litellm


def _is_multiple(value: float, base: int) -> bool:
    return abs(value / base - round(value / base)) < EPSILON


def _is_off_rate(
    hours: int, cost_eur: float, min_rate: float, max_rate: float, hours_base: int, cost_base: int
):
    """El coste no cabe entre `horas × tarifa mínima` y `horas × máxima`.

    Una fase mezcla roles, así que su coste por hora cae entre la tarifa más baja
    y la más alta. Se deja un margen por el redondeo del prompt: el modelo calcula
    el coste con horas sin redondear y las muestra redondeadas (hasta `hours_base`
    horas de diferencia), y el coste se redondea hacia arriba a `cost_base`.
    """
    return hours > 0 and not (
        max(hours - hours_base, 0) * min_rate - EPSILON
        <= cost_eur
        <= (hours + hours_base) * max_rate + cost_base
    )


def _not_rounded(estimation, hours_base: int, cost_base: int) -> tuple[int, int]:
    """(cifras que no son múltiplo del redondeo, cifras revisadas)."""
    hours = [f.hours for f in estimation.phases] + [estimation.totals.hours]
    costs = [f.cost_eur for f in estimation.phases] + [estimation.totals.cost_eur]
    bad = sum(not _is_multiple(h, hours_base) for h in hours)
    bad += sum(not _is_multiple(c, cost_base) for c in costs)
    return bad, len(hours) + len(costs)


async def _run_case(
    wrapper, version, request, repetition, rates, rounding
) -> tuple[Row, list[PhaseRow]]:
    row = Row(request.detail_level.value, request.output_format.value, repetition)
    phase_rows: list[PhaseRow] = []
    system_prompt, user_prompt = render_estimation_prompt(request, version)
    _parse_errors.clear()
    start = time.perf_counter()
    try:
        result = await wrapper.estimate_structured(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            prompt_version=version,
            response_model=RenderedResult,
            context={"output_format": request.output_format.value},
        )
    except StructuredOutputError as exc:
        row.attempts, row.cost_usd = exc.attempts, exc.cost_usd
        row.error = _short(f"intentos agotados: {exc.last_error}")
    except Exception as exc:  # noqa: BLE001 - se anota y se sigue con el resto
        row.error = _short(f"{type(exc).__name__}: {exc}")
    else:
        estimation = result.estimation
        row.ok = True
        row.model = result.model
        row.attempts = result.attempts
        row.cost_usd = result.cost_usd
        row.prompt_tokens = result.prompt_tokens
        row.completion_tokens = result.completion_tokens
        row.phases = len(estimation.phases)
        row.total_hours = estimation.totals.hours
        row.total_cost_eur = estimation.totals.cost_eur
        row.warnings = len(estimation.total_discrepancies())
        row.out_of_scope = estimation.out_of_scope
        rate_values = [r["eur_per_hour"] for r in rates.values()]
        min_rate, max_rate = min(rate_values), max(rate_values)
        for phase in estimation.phases:
            phase_rows.append(
                PhaseRow(
                    detail=row.detail,
                    output_format=row.output_format,
                    repetition=repetition,
                    name=phase.name,
                    hours=phase.hours,
                    cost_eur=phase.cost_eur,
                    eur_per_hour=round(phase.cost_eur / phase.hours, 2) if phase.hours else 0.0,
                    off_rate=_is_off_rate(
                        phase.hours,
                        phase.cost_eur,
                        min_rate,
                        max_rate,
                        rounding["hours_base"],
                        rounding["cost_base"],
                    ),
                    hours_rounded=_is_multiple(phase.hours, rounding["hours_base"]),
                    cost_rounded=_is_multiple(phase.cost_eur, rounding["cost_base"]),
                )
            )
        row.off_rate_phases = sum(p.off_rate for p in phase_rows)
        row.not_rounded, row.values = _not_rounded(
            estimation, rounding["hours_base"], rounding["cost_base"]
        )
    row.retry_reasons = " | ".join(_parse_errors)
    row.seconds = round(time.perf_counter() - start, 1)
    return row, phase_rows


def _lines(row: Row, phase_rows: list[PhaseRow]) -> list[str]:
    base = f"{row.output_format:<13} {row.detail:<9} #{row.repetition}"
    if not row.ok:
        lines = [f"{base}  FALLÓ  intentos={row.attempts}  {row.cost_usd:.4f} USD  {row.error}"]
    else:
        lines = [
            f"{base}  ok  intentos={row.attempts}  {row.cost_usd:.4f} USD  "
            f"{row.phases} fases  {format_number(row.total_hours)} h  "
            f"{format_number(row.total_cost_eur)} EUR  avisos={row.warnings}  "
            f"fuera_de_tarifa={row.off_rate_phases}  "
            f"sin_redondeo={row.not_rounded}/{row.values}  {row.seconds} s"
            + ("  FUERA DE ALCANCE" if row.out_of_scope else "")
        ]
    lines += [
        f"    fuera de tarifa: {p.name} — {format_number(p.hours)} h, "
        f"{format_number(p.cost_eur)} EUR = {format_number(p.eur_per_hour)} EUR/h"
        for p in phase_rows
        if p.off_rate
    ]
    if row.retry_reasons:
        lines.append(f"    motivo del reintento: {row.retry_reasons}")
    return lines


def _spread_pct(values: list[float]) -> str:
    """Rango relativo (máx − mín) / media, en %. «—» si no hay dos ejecuciones válidas."""
    if len(values) < 2 or sum(values) == 0:
        return "—"
    return f"{(max(values) - min(values)) / (sum(values) / len(values)) * 100:.1f} %"


def _print_summary(rows: list[Row]) -> None:
    print("\n--- Variación entre repeticiones (rango relativo de los totales) ---")
    print(f"{'formato':<13} {'detalle':<9} {'horas':>8} {'coste':>8}  fases")
    for (detail, output_format), group in itertools.groupby(
        sorted(rows, key=lambda r: (r.output_format, r.detail)),
        key=lambda r: (r.detail, r.output_format),
    ):
        ok = [r for r in group if r.ok]
        print(
            f"{output_format:<13} {detail:<9} "
            f"{_spread_pct([r.total_hours for r in ok]):>8} "
            f"{_spread_pct([r.total_cost_eur for r in ok]):>8}  "
            f"{sorted({r.phases for r in ok}) or '—'}"
        )

    ok = [r for r in rows if r.ok]
    values = sum(r.values for r in ok)
    print("\n--- Resumen ---")
    print(
        f"Ejecuciones:            {len(rows)} ({len(ok)} válidas, {len(rows) - len(ok)} fallidas)"
    )
    print(f"Con reintento:          {sum(r.attempts > 1 for r in rows)} (intentos > 1)")
    print(f"Intentos por ejecución: {sum(r.attempts for r in rows) / len(rows):.2f} de media")
    print(f"Con aviso de totales:   {sum(r.warnings > 0 for r in ok)}")
    print(f"Fases fuera de tarifa:  {sum(r.off_rate_phases for r in ok)}")
    print(f"Cifras sin redondeo:    {sum(r.not_rounded for r in ok)} de {values}")
    print(f"Fuera de alcance:       {sum(r.out_of_scope for r in ok)}")
    print(f"Coste total:            {sum(r.cost_usd for r in rows):.4f} USD")


async def _measure(args: argparse.Namespace) -> tuple[list[Row], list[PhaseRow]]:
    settings = get_settings()
    configure_logging("WARNING")
    _track_parse_errors()
    try:
        wrapper = LLMWrapper(settings=settings, cache=create_estimation_cache(None, 0))
    except LLMConfigurationError as exc:
        sys.exit(f"Configuración incompleta: {exc}")

    data = _load_version_data(args.version)
    rows: list[Row] = []
    phase_rows: list[PhaseRow] = []
    print(
        f"Modelo {settings.primary_model}, prompt {args.version}, {args.repetitions} repeticiones"
    )
    for detail, output_format in itertools.product(DetailLevel, OutputFormat):
        request = EstimationRequest(
            description=args.description,
            project_type=ProjectType(args.project_type),
            detail_level=detail,
            output_format=output_format,
        )
        for repetition in range(1, args.repetitions + 1):
            row, phases = await _run_case(
                wrapper, args.version, request, repetition, data["rates"], data["rounding"]
            )
            rows.append(row)
            phase_rows.extend(phases)
            print("\n".join(_lines(row, phases)), flush=True)
    return rows, phase_rows


def _write_csv(path: str, dataclass_type, items) -> None:
    with Path(path).open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=[campo.name for campo in fields(dataclass_type)])
        writer.writeheader()
        writer.writerows(asdict(item) for item in items)


def main() -> None:
    parser = argparse.ArgumentParser(description="Mide el prompt renderizado con el modelo real.")
    parser.add_argument("-n", "--repetitions", type=int, default=3, help="repeticiones por caso")
    parser.add_argument(
        "--version", default=None, help="prompt renderizado (por defecto, el del servicio)"
    )
    parser.add_argument(
        "--project-type", default="web_saas", choices=[p.value for p in ProjectType]
    )
    parser.add_argument("--description", default=DESCRIPTION, help="descripción del proyecto")
    parser.add_argument(
        "--csv", default="medir_v4.csv", help="una fila por ejecución (archivo de salida)"
    )
    parser.add_argument(
        "--phases-csv", default="medir_v4_fases.csv", help="una fila por fase (archivo de salida)"
    )
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error("--repetitions debe ser al menos 1")
    args.version = args.version or get_settings().rendered_prompt_version

    rows, phase_rows = asyncio.run(_measure(args))
    _print_summary(rows)

    _write_csv(args.csv, Row, rows)
    _write_csv(args.phases_csv, PhaseRow, phase_rows)
    print(f"\nDetalle por ejecución: {args.csv}\nDetalle por fase:      {args.phases_csv}")


if __name__ == "__main__":
    main()
