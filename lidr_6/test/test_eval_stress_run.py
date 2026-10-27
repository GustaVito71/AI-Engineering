"""Tests de `evals/stress/run.py` (Bloque 5, LIDR_6): la parte de lógica pura (CLI,
CSV, tablas del reporte) — nada que llame al servicio ni al LLM.

El runner real (`correr_barrido_memoria`, `correr_barrido_adjuntos`,
`correr_barrido_cache`, y `main()`) no se prueba acá por la misma razón que
`evals/run.py` y el runner HTTP de `evals/stress/attachments.py`: llama a proveedores
reales, cuesta dinero y no es determinista. Lo que SÍ es determinista y gratis, y por
eso sí se prueba, es: resolver los argumentos de la CLI, armar y leer el CSV, y las
cuentas del reporte (percentiles, promedios, las tablas) a partir de filas ya armadas
a mano — exactamente lo que `leer_csv` devolvería después de una corrida real.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evals.stress.run import (
    COLUMNAS_CSV,
    _fila_base,
    _percentil,
    _resolver_enteros,
    _resolver_perfiles,
    _tabla_md,
    escribir_csv,
    generar_reporte,
    leer_csv,
)

# --- Resolución de argumentos de la CLI -------------------------------------------------------


def test_resolver_perfiles_vacio_da_los_tres() -> None:
    assert _resolver_perfiles("") == ["growing", "pivoting", "contradicting"]


def test_resolver_perfiles_acepta_los_alias_del_enunciado() -> None:
    assert _resolver_perfiles("growing,pivot,contradiction") == ["growing", "pivoting", "contradicting"]


def test_resolver_perfiles_acepta_los_nombres_reales() -> None:
    assert _resolver_perfiles("pivoting") == ["pivoting"]


def test_resolver_perfiles_desconocido_termina_el_programa() -> None:
    with pytest.raises(SystemExit):
        _resolver_perfiles("no_existe")


def test_resolver_enteros_separa_por_coma() -> None:
    assert _resolver_enteros("1,3,6") == [1, 3, 6]


def test_resolver_enteros_vacio_usa_los_validos() -> None:
    assert _resolver_enteros("", validos=(1, 3, 6)) == [1, 3, 6]


def test_resolver_enteros_fuera_de_los_validos_termina_el_programa() -> None:
    with pytest.raises(SystemExit):
        _resolver_enteros("7", validos=(1, 3, 6))


# --- CSV: escribir y releer ------------------------------------------------------------------


def test_escribir_y_leer_csv_hace_ida_y_vuelta(tmp_path) -> None:
    fila = _fila_base(
        kind="memory",
        mode="in_process",
        profile="growing",
        n_turns=3,
        repeat_index=1,
        turn_index=1,
        status_code=200,
        cost_usd=0.001234,
        latency_ms=1500.5,
        recall_frac=0.75,
        metric_latency_budget=1,
        metric_cost_budget=1,
        metric_memory_recall_full=0,
    )
    ruta = tmp_path / "results.csv"
    escribir_csv([fila], ruta)
    releida = leer_csv(ruta)

    assert len(releida) == 1
    r = releida[0]
    assert r["profile"] == "growing"
    assert r["n_turns"] == 3
    assert r["cost_usd"] == pytest.approx(0.001234)
    assert r["latency_ms"] == pytest.approx(1500.5)
    assert r["recall_frac"] == pytest.approx(0.75)
    assert r["metric_memory_recall_full"] == 0
    # Columnas que esta fila no llenó: vacías en el CSV, `None` al releer, no "" ni 0.
    assert r["attachment_kb"] is None
    assert r["cached"] is None


def test_fila_base_tiene_todas_las_columnas_del_csv() -> None:
    fila = _fila_base(kind="memory")
    assert set(fila.keys()) == set(COLUMNAS_CSV)
    assert fila["kind"] == "memory"
    assert fila["profile"] == ""  # todo lo demás, sin overrides, queda en ""


def test_leer_csv_convierte_cached_a_booleano(tmp_path) -> None:
    fila = _fila_base(kind="cache_probe", cached=True)
    ruta = tmp_path / "results.csv"
    escribir_csv([fila], ruta)
    releida = leer_csv(ruta)
    assert releida[0]["cached"] is True


# --- Percentiles y tabla Markdown -------------------------------------------------------------


def test_percentil_50_es_la_mediana() -> None:
    assert _percentil([10.0, 20.0, 30.0], 50) == 20.0


def test_percentil_de_lista_vacia_es_none() -> None:
    assert _percentil([], 50) is None


def test_percentil_de_un_solo_valor() -> None:
    assert _percentil([42.0], 95) == 42.0


def test_percentil_95_interpola_entre_los_dos_mas_altos() -> None:
    valores = list(range(1, 101))  # 1..100
    p95 = _percentil([float(v) for v in valores], 95)
    assert p95 is not None
    assert 94.0 <= p95 <= 96.0  # interpolación lineal cerca de 95.05


def test_tabla_md_tiene_encabezado_y_separador() -> None:
    tabla = _tabla_md(["a", "b"], [["1", "2"]])
    lineas = tabla.splitlines()
    assert lineas[0] == "| a | b |"
    assert lineas[1] == "| --- | --- |"
    assert lineas[2] == "| 1 | 2 |"


# --- El reporte completo, con filas sintéticas ------------------------------------------------


def _fila_memoria(*, profile, n_turns, turn_index, repeat_index=1, cost_usd=0.01, latency_ms=1000.0, recall_frac=1.0):
    return _fila_base(
        kind="memory",
        mode="in_process",
        profile=profile,
        n_turns=n_turns,
        turn_index=turn_index,
        repeat_index=repeat_index,
        status_code=200,
        cost_usd=cost_usd,
        latency_ms=latency_ms,
        tokens_in=500,
        tokens_out=200,
        recall_frac=recall_frac,
        metric_memory_recall_full=int(recall_frac >= 1.0),
    )


def test_generar_reporte_no_rompe_con_filas_sinteticas_y_marca_la_fuga_de_recall() -> None:
    filas_dict = [
        _fila_memoria(profile="growing", n_turns=1, turn_index=1, recall_frac=1.0),
        _fila_memoria(profile="growing", n_turns=3, turn_index=1, recall_frac=1.0),
        _fila_memoria(profile="growing", n_turns=3, turn_index=2, recall_frac=1.0),
        _fila_memoria(profile="growing", n_turns=3, turn_index=3, recall_frac=0.5),  # fuga al final
        _fila_base(
            kind="cache_probe",
            mode="in_process",
            profile="growing",
            repeat_index=1,
            turn_index=1,
            status_code=200,
            cost_usd=0.01,
            latency_ms=800.0,
            cached=False,
            cache_hit_kind="miss",
        ),
        _fila_base(
            kind="cache_probe",
            mode="in_process",
            profile="growing",
            repeat_index=2,
            turn_index=1,
            status_code=200,
            cost_usd=0.0,
            latency_ms=50.0,
            cached=True,
            cache_hit_kind="exacta",
        ),
    ]
    # Pasa por el mismo camino que una corrida real: CSV -> disco -> `leer_csv`.
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        ruta = Path(tmp) / "r.csv"
        escribir_csv(filas_dict, ruta)
        filas = leer_csv(ruta)

    reporte = generar_reporte(filas)

    assert "# Bloque 5" in reporte
    assert "N=3" in reporte  # la fuga de recall detectada en n_turns=3
    assert "exacta" in reporte.lower()


def test_generar_reporte_sin_ninguna_fila_no_rompe() -> None:
    reporte = generar_reporte([])
    assert "# Bloque 5" in reporte
    assert "s/d" in reporte or "Sin datos" in reporte
