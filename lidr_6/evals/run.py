"""Runner CLI del framework de evals (Bloque 1, LIDR_6).

Corre los casos de `golden_dataset.json` contra el servicio REAL, turno a turno, y los
juzga con las tres métricas de `evals/metrics.py`. No es un test de pytest (pytest no
debe recogerlo: no empieza con `test_`) porque llama a los proveedores de verdad y
cuesta dinero real; se corre a mano, cuando se quiere medir el servicio, no en cada
commit.

CÓMO CORRE

En proceso, sin servidor aparte: `TestClient(create_app())`, igual que los tests, pero
SIN reemplazar el LLMWrapper por uno falso. Cada turno es una llamada real al
proveedor que el `.env` de la raíz tenga configurado; por eso el coste y la latencia
que mide son los de verdad. Un caso con varios turnos crea una sesión y manda sus
turnos en orden, en esa misma sesión (la memoria conversacional es justamente lo que
varios casos quieren poner a prueba). Si el caso declara `expected_facts`, al final
consulta `GET /sessions/{id}` para ver qué quedó en la memoria persistida.

El coste de cada turno no viaja en el cuerpo de la respuesta HTTP: se junta de los
eventos de trazabilidad que el servicio emite en cada llamada al proveedor
(`coste_usd` en `estimacion_completada`, `critico_completado`...), capturados con
`structlog.testing.capture_logs` alrededor de cada POST.

MODOS

- `actor` (por defecto): cada turno va a `POST /sessions/{id}/estimate`.
- `acb`: cada turno va a `POST /sessions/{id}/estimate-acb` (actor + crítico + jefe;
  más caro y más lento, pero es el camino que de verdad usa un cliente que pide
  revisión).

USO (desde la raíz de lidr_5 o lidr_6)

    python evals/run.py
    python evals/run.py --mode acb
    python evals/run.py --casos golden-01,golden-09
    python evals/run.py --dataset evals/golden_dataset.json --salida evals/resultados.csv

Las claves de los proveedores se leen del `.env` de la raíz, como en el resto del
servicio: este script no las pide ni las imprime.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
# Para poder hacer `from app... import ...` corriendo este archivo suelto (`python
# evals/run.py`), sin depender de que quien lo invoca ya tenga la raíz en PYTHONPATH.
sys.path.insert(0, str(RAIZ))

from app.main import create_app
from fastapi.testclient import TestClient
from structlog.testing import capture_logs

from evals.metrics import METRICS, CaseResult, TurnResult

DATASET_POR_DEFECTO = Path(__file__).resolve().parent / "golden_dataset.json"
SALIDA_POR_DEFECTO = Path(__file__).resolve().parent / "resultados.csv"
MODOS = ("actor", "acb")


def cargar_casos(ruta: Path, *, filtro: set[str] | None) -> list[dict]:
    """Los casos golden del JSON (ignora `_schema`: es documentación, no un caso).

    `filtro`, si se da, deja solo los casos cuyo `id` está en el conjunto (para
    `--casos`, que permite correr un subconjunto sin editar el dataset).
    """
    data = json.loads(ruta.read_text(encoding="utf-8"))
    casos = list(data["casos"])
    if filtro:
        casos = [c for c in casos if c["id"] in filtro]
        encontrados = {c["id"] for c in casos}
        faltantes = filtro - encontrados
        if faltantes:
            sys.exit(f"No existen en el dataset: {', '.join(sorted(faltantes))}")
    return casos


def _coste_de_los_logs(logs: list[dict]) -> float:
    """Suma `coste_usd` de todos los eventos capturados durante una llamada HTTP.

    Cubre el camino actor (un evento `estimacion_completada`) y el acb (varias vueltas
    de actor y crítico, cada una con el suyo): no hace falta saber de antemano cuántas
    llamadas al proveedor hizo una vuelta, solo sumar lo que de verdad se gastó. Un
    acierto de caché ya deja `coste_usd=0.0` en el evento (ver `emit_completion`), así
    que no hay riesgo de contar dos veces lo que no se pagó.
    """
    return sum(evento.get("coste_usd", 0.0) or 0.0 for evento in logs if "coste_usd" in evento)


def correr_turno(cliente: TestClient, ruta: str, turno: dict) -> TurnResult:
    """Manda un turno (un POST multipart) y mide su estado, coste y latencia."""
    with capture_logs() as logs:
        inicio = time.perf_counter()
        respuesta = cliente.post(
            ruta,
            data={
                "transcript": turno["transcript"],
                "project_type": turno["project_type"],
                "detail_level": turno["detail_level"],
                "output_format": turno["output_format"],
            },
        )
        latencia = time.perf_counter() - inicio
    cuerpo = respuesta.json() if respuesta.status_code == 200 else None
    coste = _coste_de_los_logs(logs)
    return TurnResult(status_code=respuesta.status_code, body=cuerpo, cost_usd=coste, latency_s=latencia)


def correr_caso(cliente: TestClient, caso: dict, *, modo: str) -> CaseResult:
    """Crea una sesión y manda los turnos del caso, en orden, a `/estimate` o
    `/estimate-acb`. Un turno que no responde 200 corta el caso ahí: no tiene sentido
    seguir una conversación que ya se rompió, y las métricas lo marcan como fallo.
    """
    session_id = cliente.post("/sessions").json()["session_id"]
    resultado = CaseResult(case_id=caso["id"], session_id=session_id)
    sufijo = "/estimate-acb" if modo == "acb" else "/estimate"
    ruta = f"/sessions/{session_id}{sufijo}"

    for turno in caso["turns"]:
        turn_result = correr_turno(cliente, ruta, turno)
        resultado.turns.append(turn_result)
        if turn_result.status_code != 200:
            break

    if resultado.all_turns_ok and caso.get("expected_facts"):
        estado = cliente.get(f"/sessions/{session_id}")
        if estado.status_code == 200:
            resultado.session_metadata = estado.json().get("project_metadata")
    return resultado


def evaluar_caso(caso: dict, resultado: CaseResult) -> list[dict]:
    """Las filas de resultado (una por métrica) de UN caso ya corrido."""
    filas = []
    for metrica in METRICS:
        aprobado, motivo = metrica.evaluate(caso, resultado)
        filas.append(
            {
                "caso": caso["id"],
                "metrica": metrica.name,
                "aprobado": int(aprobado),
                "motivo": motivo,
                "turnos": len(resultado.turns),
                "coste_usd_total": round(resultado.total_cost_usd, 6),
                "latencia_s_total": round(sum(t.latency_s for t in resultado.turns), 2),
            }
        )
    return filas


def imprimir_resumen(filas: list[dict]) -> None:
    """Tabla por métrica (cuántos casos aprobó) y el detalle de cada reprobado."""
    nombres_metrica = {f["metrica"] for f in filas}
    print("\nResumen por métrica:")
    print(f"{'métrica':<18} {'aprobados':>10} {'de':>4}")
    for nombre in sorted(nombres_metrica):
        grupo = [f for f in filas if f["metrica"] == nombre]
        aprobados = sum(f["aprobado"] for f in grupo)
        print(f"{nombre:<18} {aprobados:>10} {len(grupo):>4}")

    reprobados = [f for f in filas if not f["aprobado"]]
    if reprobados:
        print(f"\nCasos reprobados ({len(reprobados)}):")
        for f in reprobados:
            print(f"  {f['caso']:<12} {f['metrica']:<18} {f['motivo']}")
    else:
        print("\nTodos los casos aprobaron las tres métricas.")

    coste_total = sum(f["coste_usd_total"] for f in filas if f["metrica"] == filas[0]["metrica"])
    print(f"\nCoste total de la corrida: ${coste_total:.5f}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--dataset", default=str(DATASET_POR_DEFECTO), help="Ruta al JSON de casos golden")
    ap.add_argument("--mode", choices=MODOS, default="actor", help="actor (/estimate) o acb (/estimate-acb)")
    ap.add_argument("--casos", default="", help="IDs separados por coma; vacío corre todo el dataset")
    ap.add_argument("--salida", default=str(SALIDA_POR_DEFECTO), help="CSV con el detalle por caso y métrica")
    args = ap.parse_args()

    filtro = {c.strip() for c in args.casos.split(",") if c.strip()} or None
    casos = cargar_casos(Path(args.dataset), filtro=filtro)
    if not casos:
        sys.exit(f"El dataset está vacío: {args.dataset}")

    print(f"Modo: {args.mode}. Casos: {len(casos)}.")
    cliente = TestClient(create_app())
    filas: list[dict] = []
    for caso in casos:
        resultado = correr_caso(cliente, caso, modo=args.mode)
        filas_caso = evaluar_caso(caso, resultado)
        filas.extend(filas_caso)
        estado = "OK" if all(f["aprobado"] for f in filas_caso) else "FALLÓ"
        print(
            f"  {caso['id']:<12} {estado:<5} "
            f"(${resultado.total_cost_usd:.5f}, {len(resultado.turns)} turno(s))"
        )

    with open(args.salida, "w", newline="", encoding="utf-8") as f:
        escritor = csv.DictWriter(f, fieldnames=list(filas[0].keys()))
        escritor.writeheader()
        escritor.writerows(filas)

    imprimir_resumen(filas)
    print(f"\nCSV con detalle por caso y métrica: {args.salida}")


if __name__ == "__main__":
    main()
