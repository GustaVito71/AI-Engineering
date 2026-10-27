"""Bloque 5 de LIDR_6: el runner que junta los Bloques 2, 3 y 4 en una sola corrida, y
el reporte que se lee de su CSV.

QUÉ CORRE (tres barridos, en una sola invocación)

1. MEMORIA (Bloque 2 + Bloque 4): por cada perfil de `--scenarios` y cada largo de
   conversación de `--turn-counts`, corre la conversación turno a turno contra
   `POST /sessions/{id}/estimate` (repitiéndola `--repeats` veces, con sesiones
   nuevas, para no medir una sola muestra de un LLM que no es determinista). Por cada
   turno mide todo lo que ya definió el Bloque 1 (`turn_observed`, ver más abajo) y,
   con `MemoryDriftMetric` del Bloque 4, si los hechos que deberían seguir vigentes
   después de ese turno (`Scenario.facts_expected_after`) de verdad sobreviven.
2. ADJUNTOS (Bloque 3): por cada tamaño de `--attachment-sizes`, un turno suelto con
   el PDF sintético calibrado de `evals/stress/attachments.py`, repetido `--repeats`
   veces. Mide lo mismo que el punto 1, más si los dos requisitos funcionales del
   adjunto sobreviven (igual que `correr_punto` del Bloque 3).
3. CACHÉ (necesario para el reporte, no estaba en los otros bloques): por cada perfil
   de `--scenarios`, el texto del turno 1 se manda `--repeats` veces a
   `POST /api/v1/estimate/structured` (el único camino con caché: ver "POR QUÉ UN
   TERCER BARRIDO" más abajo). Mide si la respuesta salió de caché (`cached` en el
   cuerpo) y, en proceso, si fue la exacta o la semántica.

Una fila del CSV es UN turno de UNO de los tres barridos: todas comparten las columnas
de `turn_observed` (vacías lo que ese barrido no mide) más una columna por métrica
binaria (`metric_latency_budget`, `metric_cost_budget`, `metric_memory_recall_full`).

POR QUÉ UN TERCER BARRIDO (CACHÉ) QUE NINGÚN BLOQUE ANTERIOR PEDÍA

Es, en sí mismo, el primer hallazgo del "dónde se rompe": `turn_observed` fija
`cache_hit_kind="none"` SIEMPRE para el camino conversacional (`estimate_conversation`
no consulta ninguna caché; ver `app/routers/sessions.py:_emit_turn_observed`). Las dos
cachés (exacta y semántica) solo existen en el camino SIN sesión
(`EstimationService.estimate_structured`, montado en `/api/v1/estimate/structured`),
que tampoco acepta adjuntos. Es decir: memoria conversacional, adjuntos y caché son
TRES caminos que no se solapan en este servicio. Pedir "hit rate de ambas cachés" sin
tocar ese tercer camino daría siempre 0%, no porque la caché no funcione, sino porque
los barridos 1 y 2 nunca pasan por ella. Este barrido existe para medir la caché donde
de verdad se puede medir, y la tensión en sí misma es parte de la lectura del Bloque 5
(ver el reporte, sección 3).

MODOS: `--http` vs. en proceso

- En proceso (por defecto, sin `--http`): `TestClient(create_app())`, igual que
  `evals/run.py`. Cada turno es una llamada real al proveedor configurado en el
  `.env`. El coste y los campos de `turn_observed` se leen de los eventos de log
  capturados con `structlog.testing.capture_logs` (igual que el resto del framework:
  no viajan en el cuerpo HTTP). Además, SOLO en este modo, el runner puede sacar del
  almacén en memoria (`app.sessions.store.SessionStore`, vía `app.state` del cliente
  de test) una copia de la `Session` después de cada turno: es lo que necesita
  `MemoryDriftMetric` para mirar `summary`/`anchors`/`metadata` de verdad, porque
  `GET /sessions/{id}` NO expone ese texto (solo cuenta anclas, no las muestra; ver
  `app/schemas/session.py:SessionStateResponse`).
- `--http <url>`: `httpx.Client(base_url=url)` contra un servicio YA corriendo en otro
  proceso (por ejemplo `uvicorn app.main:create_app --factory`). Es el modo real para
  medir latencia de punta a punta a través de la red. Pero NINGUNO de los dos trucos
  de arriba funciona: no hay logs que capturar (corren en el otro proceso) ni almacén
  al que mirar. Por eso en este modo:
    - `cost_usd` y casi todas las columnas de `turn_observed` quedan vacías (no hay
      forma de medirlas desde afuera del proceso sin que el servicio las exponga en
      la respuesta, y hoy no lo hace: otro hallazgo del "dónde se rompe").
    - `MemoryDriftMetric` no se puede evaluar (no hay `session_snapshot`). Se cae a un
      sustituto más débil, ya usado en el Bloque 3: buscar el hecho como subcadena en
      el texto de la ESTIMACIÓN de la última respuesta (lo que el modelo escribió),
      no en la memoria interna de la sesión. Queda marcado en el CSV
      (`memory_check_kind="text_fallback"` contra `"internal_snapshot"`) para que el
      reporte no mezcle las dos cosas como si fueran la misma medición.
    - `cached` SÍ se puede leer (viaja en el cuerpo de `/estimate/structured`), pero
      no se puede distinguir exacta de semántica sin los eventos de log: queda como
      un solo `cache_hit_kind="any"` en vez de `"exacta"`/`"semantica"`.

USO

    # Corrida chica, para probar que todo funciona sin gastar mucho:
    uv run python -m evals.stress.run --scenarios growing --turn-counts 1,3 \\
        --attachment-sizes 0,20 --repeats 2

    # La corrida completa que describe el enunciado:
    uv run python -m evals.stress.run --http http://localhost:8000 \\
        --scenarios growing,pivot,contradiction --attachment-sizes 0,5,20,50,100 \\
        --repeats 3 --output evals/stress/results.csv

    # Rehacer solo el reporte a partir de un CSV ya generado (no llama al servicio):
    uv run python -m evals.stress.run --solo-reporte --output evals/stress/results.csv

`--scenarios` acepta los tres nombres de `evals/stress/scenarios.py:PROFILES`
("growing", "pivoting", "contradicting") y también los alias cortos del enunciado del
bloque ("pivot", "contradiction"). `--turn-counts` filtra los N de
`scenarios.TURN_COUNTS`; por defecto corre (1, 3, 6), no los cinco (hasta 20 turnos):
con `--repeats 3` y los tres perfiles, los cinco N completos son 3 × 3 × 40 = 360
turnos reales al proveedor (horas, no minutos). El set de cinco completo
(`--turn-counts 1,3,6,10,20`) es el que corresponde para el reporte final que se lleva
al directo; el default más chico es para poder probar el runner sin quemar presupuesto.
"""

from __future__ import annotations

import argparse
import copy
import csv
import statistics
import sys
import time
from pathlib import Path
from typing import Any

RAIZ = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(RAIZ))

from evals.stress.attachments import (
    RECALL_KEYWORD_END,
    RECALL_KEYWORD_START,
    SIZE_POINTS_KB,
    build_attachment,
)
from evals.stress.metrics import CostBudgetMetric, LatencyBudgetMetric, MemoryDriftMetric
from evals.stress.scenarios import PROFILES, TURN_COUNTS, Scenario, build_scenario

SALIDA_CSV_POR_DEFECTO = Path(__file__).resolve().parent / "results.csv"
SALIDA_REPORTE_POR_DEFECTO = Path(__file__).resolve().parent / "REPORT.md"

# Alias cortos, tal como los escribe el enunciado del bloque, hacia los nombres reales
# de `scenarios.PROFILES`. Los nombres reales siguen aceptándose tal cual.
ALIAS_PERFIL = {"pivot": "pivoting", "contradiction": "contradicting", "growing": "growing"}

# Presupuestos por defecto de las dos métricas de presupuesto (Bloque 4). Generosos a
# propósito, como ya explica `CostBoundsMetric` en `evals/metrics.py`: son topes de
# regresión (detectar un desbarranco real), no un benchmark de qué tan rápido/barato
# "debería" ser el servicio. $0.05/turno es el mismo tope que ya usa
# `evals/golden_dataset.json`; 60s es holgado frente a lo medido en el Bloque 3
# (~40-50s por turno con el proveedor real).
LATENCY_BUDGET_MS_POR_DEFECTO = 60_000
COST_BUDGET_USD_POR_DEFECTO = 0.05

# Las columnas del CSV, en orden fijo. `restval=""` en el DictWriter rellena lo que un
# barrido no mide (por ejemplo, un turno de adjuntos no tiene `profile` ni `n_turns`).
COLUMNAS_CSV = [
    "kind",  # "memory" | "attachment" | "cache_probe"
    "mode",  # "in_process" | "http"
    "profile",
    "n_turns",
    "attachment_kb",
    "repeat_index",
    "turn_index",
    "session_id",
    "status_code",
    # --- las columnas de `turn_observed` (Bloque 1), tal cual las emite el servicio ---
    "enriched_transcript_chars",
    "attachments_total_chars",
    "messages_in_window",
    "anchors_count",
    "summary_chars",
    "tokens_in",
    "tokens_out",
    "cost_usd",
    "latency_ms",
    "cache_hit_kind",
    "last_resolved_tier",
    # --- métricas binarias (Bloque 4) ---
    "metric_latency_budget",
    "metric_cost_budget",
    "metric_memory_recall_full",
    # --- extras que no son `turn_observed` pero hacen falta para el reporte ---
    "recall_frac",  # fracción de hechos vigentes recuperados (continuo, no binario)
    "memory_check_kind",  # "internal_snapshot" | "text_fallback" | "" (no aplica)
    "cached",  # bool, solo en filas "cache_probe"
]


# --- El cliente: TestClient o httpx.Client, con la misma interfaz ---------------------------


class Cliente:
    """Envoltorio fino sobre `TestClient` (en proceso) o `httpx.Client` (`--http`).

    Las dos clases ya comparten la firma de `.post`/`.get` (`TestClient` de Starlette
    hereda de `httpx.Client`), así que no hace falta adaptar nada ahí: lo único que
    cambia entre modos es si se puede capturar el log del proceso y mirar el almacén de
    sesiones, y eso es justo lo que expone `en_proceso`.
    """

    def __init__(self, *, http_url: str | None) -> None:
        self.en_proceso = http_url is None
        if self.en_proceso:
            from app.main import create_app
            from fastapi.testclient import TestClient

            self._cliente = TestClient(create_app())
            # CLAVE: sin entrar al context manager, el lifespan de la app
            # (`app/main.py:lifespan`) nunca corre, así que `app.state.estimation_cache`
            # y `app.state.semantic_cache` no existen y CADA request cae al fallback sin
            # caché de `get_estimation_cache`/`get_semantic_cache` (siempre None/NoOp),
            # aunque Redis esté arriba. `evals/run.py` y `evals/stress/attachments.py` no
            # lo necesitan (no miden caché), por eso no lo hacían; el barrido 3 de este
            # módulo sí, así que acá hace falta de verdad.
            self._cliente.__enter__()
        else:
            import httpx

            self._cliente = httpx.Client(base_url=http_url, timeout=120.0)

    def post(self, ruta: str, **kwargs: Any) -> Any:
        return self._cliente.post(ruta, **kwargs)

    def get(self, ruta: str, **kwargs: Any) -> Any:
        return self._cliente.get(ruta, **kwargs)

    def close(self) -> None:
        """Cierra el cliente (y, en proceso, corre el `shutdown` del lifespan: cierra
        la conexión a Redis en vez de dejarla colgada hasta que el proceso termina)."""
        if self.en_proceso:
            self._cliente.__exit__(None, None, None)
        else:
            self._cliente.close()

    def session_snapshot(self, session_id: str) -> Any | None:
        """Copia de la `Session` interna, o `None` en modo `--http` (no hay a qué
        almacén mirar desde afuera del proceso: ver el docstring del módulo)."""
        if not self.en_proceso:
            return None
        store = getattr(self._cliente.app.state, "session_store", None)
        if store is None:
            return None
        session = store.get(session_id)
        return copy.deepcopy(session) if session is not None else None


def _coste_de_los_logs(logs: list[dict]) -> float:
    """Igual que en `evals/run.py` y `evals/stress/attachments.py`: suma `coste_usd`
    de todos los eventos capturados durante la llamada."""
    return sum(evento.get("coste_usd", 0.0) or 0.0 for evento in logs if "coste_usd" in evento)


def _turn_observed_de_los_logs(logs: list[dict]) -> dict[str, Any] | None:
    """El evento `turn_observed` de esta llamada, si el servicio lo emitió (solo en el
    camino conversacional y solo si el turno llegó a responder 200)."""
    for evento in logs:
        if evento.get("event") == "turn_observed":
            return evento
    return None


# --- Fila base: lo que comparten los tres barridos -------------------------------------------


def _fila_base(**overrides: Any) -> dict[str, Any]:
    """Una fila con todas las columnas del CSV vacías, para que cada barrido solo
    tenga que llenar las que de verdad mide."""
    fila = dict.fromkeys(COLUMNAS_CSV, "")
    fila.update(overrides)
    return fila


def _aplicar_metricas_de_presupuesto(
    fila: dict[str, Any], *, latency_budget_ms: int, cost_budget_usd: float
) -> None:
    """Completa `metric_latency_budget`/`metric_cost_budget` IN PLACE, si la fila tiene
    con qué evaluarlas (un turno que no respondió 200 no tiene latencia ni coste
    fiables, así que queda sin estas dos columnas, no en 0 a la fuerza)."""
    if fila["latency_ms"] != "":
        resultado = LatencyBudgetMetric(latency_budget_ms).evaluate({"latency_ms": fila["latency_ms"]})
        fila["metric_latency_budget"] = int(resultado.passed)
    if fila["cost_usd"] != "":
        resultado = CostBudgetMetric(cost_budget_usd).evaluate({"cost_usd": fila["cost_usd"]})
        fila["metric_cost_budget"] = int(resultado.passed)


# --- Barrido 1: memoria (Bloque 2 + Bloque 4) -------------------------------------------------


def _recall_fraccion(
    *,
    hechos_vigentes: frozenset[str],
    snapshot: Any | None,
    texto_respuesta: str,
) -> tuple[float, str]:
    """La fracción de `hechos_vigentes` que sobrevive, y CÓMO se comprobó.

    Con `snapshot` (modo en proceso): `MemoryDriftMetric` contra la memoria interna
    de verdad (resumen, anclas, metadata). Sin él (`--http`): sustituto más débil,
    buscar cada hecho en el texto de la última respuesta (igual que el Bloque 3 busca
    sus marcadores en la respuesta del modelo). Una conversación sin hechos vigentes
    todavía (el primer turno de un perfil que recién arranca) da recall perfecto por
    definición: no hay nada que se pueda haber perdido.
    """
    if not hechos_vigentes:
        return 1.0, "internal_snapshot" if snapshot is not None else "text_fallback"
    if snapshot is not None:
        aciertos = sum(MemoryDriftMetric(hecho).evaluate(snapshot).passed for hecho in hechos_vigentes)
        return aciertos / len(hechos_vigentes), "internal_snapshot"
    texto = texto_respuesta.lower()
    aciertos = sum(1 for hecho in hechos_vigentes if hecho.lower() in texto)
    return aciertos / len(hechos_vigentes), "text_fallback"


def _correr_turno_conversacional(
    cliente: Cliente,
    ruta: str,
    *,
    transcript: str,
    project_type: str,
    detail_level: str,
    output_format: str,
    files: list[tuple[str, tuple[str, bytes, str]]] | None = None,
) -> tuple[dict[str, Any], str]:
    """Un POST a `/sessions/{id}/estimate...`: la fila (sin las métricas todavía) y el
    texto de la estimación de la respuesta (para el sustituto de recall en `--http`)."""
    datos = {
        "transcript": transcript,
        "project_type": project_type,
        "detail_level": detail_level,
        "output_format": output_format,
    }
    if cliente.en_proceso:
        from structlog.testing import capture_logs

        with capture_logs() as logs:
            inicio = time.perf_counter()
            respuesta = cliente.post(ruta, data=datos, files=files)
            latencia_cliente_ms = (time.perf_counter() - inicio) * 1000
        observado = _turn_observed_de_los_logs(logs)
        fila = _fila_base(
            mode="in_process",
            status_code=respuesta.status_code,
            cost_usd=round(_coste_de_los_logs(logs), 6),
        )
        if observado:
            for columna in (
                "enriched_transcript_chars",
                "attachments_total_chars",
                "messages_in_window",
                "anchors_count",
                "summary_chars",
                "tokens_in",
                "tokens_out",
                "latency_ms",
                "cache_hit_kind",
                "last_resolved_tier",
            ):
                if columna in observado:
                    fila[columna] = observado[columna]
        if fila["latency_ms"] == "":
            fila["latency_ms"] = round(latencia_cliente_ms, 1)
    else:
        inicio = time.perf_counter()
        respuesta = cliente.post(ruta, data=datos, files=files)
        latencia_cliente_ms = (time.perf_counter() - inicio) * 1000
        # En `--http` no hay logs que mirar: la única columna de `turn_observed` que
        # se puede rellenar desde afuera del proceso es la latencia, medida del lado
        # del cliente (incluye red; `turn_observed.latency_ms` mide solo el servicio).
        fila = _fila_base(mode="http", status_code=respuesta.status_code, latency_ms=round(latencia_cliente_ms, 1))

    texto_estimacion = ""
    if respuesta.status_code == 200:
        try:
            texto_estimacion = str(respuesta.json().get("estimation", ""))
        except ValueError:
            pass
    return fila, texto_estimacion


def correr_barrido_memoria(
    cliente: Cliente,
    *,
    perfiles: list[str],
    turn_counts: list[int],
    repeticiones: int,
    latency_budget_ms: int,
    cost_budget_usd: float,
) -> list[dict[str, Any]]:
    """El barrido 1: cada (perfil, N) de `evals/stress/scenarios.py`, `repeticiones`
    veces, turno a turno. Devuelve las filas ya con sus métricas."""
    filas: list[dict[str, Any]] = []
    for perfil in perfiles:
        for n_turns in turn_counts:
            escenario: Scenario = build_scenario(perfil, n_turns)
            for repeticion in range(1, repeticiones + 1):
                session_id = cliente.post("/sessions").json()["session_id"]
                ruta = f"/sessions/{session_id}/estimate"
                for turno in escenario.turns:
                    fila, texto_estimacion = _correr_turno_conversacional(
                        cliente,
                        ruta,
                        transcript=turno.transcript,
                        project_type=escenario.project_type,
                        detail_level=escenario.detail_level,
                        output_format=escenario.output_format,
                    )
                    fila.update(
                        kind="memory",
                        profile=perfil,
                        n_turns=n_turns,
                        repeat_index=repeticion,
                        turn_index=turno.index,
                        session_id=session_id,
                    )
                    if fila["status_code"] == 200:
                        hechos_vigentes = escenario.facts_expected_after(turno.index)
                        snapshot = cliente.session_snapshot(session_id)
                        recall, tipo_chequeo = _recall_fraccion(
                            hechos_vigentes=hechos_vigentes, snapshot=snapshot, texto_respuesta=texto_estimacion
                        )
                        fila["recall_frac"] = round(recall, 4)
                        fila["memory_check_kind"] = tipo_chequeo
                        fila["metric_memory_recall_full"] = int(recall >= 1.0)
                    _aplicar_metricas_de_presupuesto(
                        fila, latency_budget_ms=latency_budget_ms, cost_budget_usd=cost_budget_usd
                    )
                    filas.append(fila)
                    if fila["status_code"] != 200:
                        break  # como en evals/run.py: una conversación rota no sigue
    return filas


# --- Barrido 2: adjuntos (Bloque 3) -----------------------------------------------------------


def correr_barrido_adjuntos(
    cliente: Cliente,
    *,
    tamanos_kb: list[int],
    repeticiones: int,
    latency_budget_ms: int,
    cost_budget_usd: float,
) -> list[dict[str, Any]]:
    """El barrido 2: cada tamaño de `evals/stress/attachments.py`, `repeticiones`
    veces, un turno suelto cada vez (mismo transcript/project_type que el Bloque 3)."""
    from evals.stress.attachments import DETAIL_LEVEL, OUTPUT_FORMAT, PROJECT_TYPE, TRANSCRIPT

    filas: list[dict[str, Any]] = []
    for kb in tamanos_kb:
        files = None
        if kb > 0:
            adjunto = build_attachment(kb)
            files = [("attachments", (adjunto.filename, adjunto.pdf_bytes, "application/pdf"))]
        for repeticion in range(1, repeticiones + 1):
            session_id = cliente.post("/sessions").json()["session_id"]
            ruta = f"/sessions/{session_id}/estimate"
            fila, texto_estimacion = _correr_turno_conversacional(
                cliente,
                ruta,
                transcript=TRANSCRIPT,
                project_type=PROJECT_TYPE,
                detail_level=DETAIL_LEVEL,
                output_format=OUTPUT_FORMAT,
                files=files,
            )
            fila.update(
                kind="attachment",
                attachment_kb=kb,
                repeat_index=repeticion,
                turn_index=1,
                session_id=session_id,
            )
            if fila["status_code"] == 200 and kb > 0:
                texto = texto_estimacion.lower()
                hechos = {RECALL_KEYWORD_START, RECALL_KEYWORD_END}
                aciertos = sum(1 for h in hechos if h in texto)
                fila["recall_frac"] = round(aciertos / len(hechos), 4)
                fila["memory_check_kind"] = "text_fallback"  # el recall del Bloque 3 siempre es sobre texto
                fila["metric_memory_recall_full"] = int(aciertos == len(hechos))
            _aplicar_metricas_de_presupuesto(
                fila, latency_budget_ms=latency_budget_ms, cost_budget_usd=cost_budget_usd
            )
            filas.append(fila)
    return filas


# --- Barrido 3: caché (necesario para el reporte; ver el docstring del módulo) ---------------


def correr_barrido_cache(
    cliente: Cliente,
    *,
    perfiles: list[str],
    repeticiones: int,
    latency_budget_ms: int,
    cost_budget_usd: float,
) -> list[dict[str, Any]]:
    """El barrido 3: el transcript del turno 1 de cada perfil, mandado `repeticiones`
    veces IDÉNTICO a `/api/v1/estimate/structured` (sin sesión: ese endpoint no tiene).

    Con repeticiones idénticas la caché EXACTA debería acertar desde la segunda
    llamada en adelante (misma clave); la SEMÁNTICA, en este experimento concreto,
    rara vez llega a intervenir: solo se consulta cuando la exacta ya falló, y con
    texto idéntico la exacta no vuelve a fallar después de la primera vez (ver el
    docstring de `EstimationService.estimate_structured`). El reporte explica esta
    lectura; este runner solo mide lo que de verdad pasó.
    """
    filas: list[dict[str, Any]] = []
    for perfil in perfiles:
        escenario = build_scenario(perfil, 1)
        descripcion = escenario.turns[0].transcript
        for repeticion in range(1, repeticiones + 1):
            datos = {
                "description": descripcion,
                "project_type": escenario.project_type,
                "detail_level": escenario.detail_level,
                "output_format": escenario.output_format,
            }
            if cliente.en_proceso:
                from structlog.testing import capture_logs

                with capture_logs() as logs:
                    inicio = time.perf_counter()
                    respuesta = cliente.post("/api/v1/estimate/structured", json=datos)
                    latencia_ms = (time.perf_counter() - inicio) * 1000
                acierto_semantico = any(e.get("event") == "cache_semantica_acierto" for e in logs)
                coste = round(_coste_de_los_logs(logs), 6)
            else:
                inicio = time.perf_counter()
                respuesta = cliente.post("/api/v1/estimate/structured", json=datos)
                latencia_ms = (time.perf_counter() - inicio) * 1000
                acierto_semantico = None  # no se puede saber sin los logs del proceso
                coste = ""

            cached = None
            if respuesta.status_code == 200:
                cached = bool(respuesta.json().get("cached"))

            if cached is None:
                cache_hit_kind = ""
            elif not cached:
                cache_hit_kind = "miss"
            elif acierto_semantico is None:
                cache_hit_kind = "any"  # --http: se sabe que acertó, no cuál de las dos
            else:
                cache_hit_kind = "semantica" if acierto_semantico else "exacta"

            fila = _fila_base(
                kind="cache_probe",
                mode="in_process" if cliente.en_proceso else "http",
                profile=perfil,
                repeat_index=repeticion,
                turn_index=1,
                status_code=respuesta.status_code,
                latency_ms=round(latencia_ms, 1),
                cost_usd=coste,
                cache_hit_kind=cache_hit_kind,
                cached=cached if cached is not None else "",
            )
            _aplicar_metricas_de_presupuesto(
                fila, latency_budget_ms=latency_budget_ms, cost_budget_usd=cost_budget_usd
            )
            filas.append(fila)
    return filas


# --- CSV ---------------------------------------------------------------------------------------


def escribir_csv(filas: list[dict[str, Any]], ruta: Path) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with open(ruta, "w", newline="", encoding="utf-8") as f:
        escritor = csv.DictWriter(f, fieldnames=COLUMNAS_CSV, restval="")
        escritor.writeheader()
        escritor.writerows(filas)


def leer_csv(ruta: Path) -> list[dict[str, Any]]:
    """El CSV de vuelta a dicts, con los campos numéricos convertidos (todo llega como
    string de un CSV) y los campos vacíos como `None`, para que el reporte no tenga
    que manejar `""` por todos lados."""
    _NUMERICOS_INT = {
        "n_turns",
        "attachment_kb",
        "repeat_index",
        "turn_index",
        "status_code",
        "enriched_transcript_chars",
        "attachments_total_chars",
        "messages_in_window",
        "anchors_count",
        "summary_chars",
        "tokens_in",
        "tokens_out",
        "metric_latency_budget",
        "metric_cost_budget",
        "metric_memory_recall_full",
    }
    _NUMERICOS_FLOAT = {"cost_usd", "latency_ms", "recall_frac"}
    filas: list[dict[str, Any]] = []
    with open(ruta, encoding="utf-8") as f:
        for fila in csv.DictReader(f):
            limpia: dict[str, Any] = {}
            for clave, valor in fila.items():
                if valor == "":
                    limpia[clave] = None
                elif clave in _NUMERICOS_INT:
                    limpia[clave] = int(valor)
                elif clave in _NUMERICOS_FLOAT:
                    limpia[clave] = float(valor)
                elif clave == "cached":
                    limpia[clave] = valor == "True"
                else:
                    limpia[clave] = valor
            filas.append(limpia)
    return filas


# --- Reporte -------------------------------------------------------------------------------


def _percentil(valores: list[float], p: float) -> float | None:
    """El percentil `p` (0-100) de `valores`, por interpolación lineal entre los dos
    puntos más cercanos (el método "de libro", sin depender de una librería de
    estadística aparte para algo tan chico). `None` si la lista está vacía: un
    percentil sin datos no es cero, es "no hay nada que reportar"."""
    if not valores:
        return None
    ordenados = sorted(valores)
    if len(ordenados) == 1:
        return ordenados[0]
    posicion = (p / 100) * (len(ordenados) - 1)
    inferior = int(posicion)
    superior = min(inferior + 1, len(ordenados) - 1)
    fraccion = posicion - inferior
    return ordenados[inferior] + (ordenados[superior] - ordenados[inferior]) * fraccion


def _tabla_md(encabezados: list[str], filas: list[list[str]]) -> str:
    """Una tabla Markdown mínima: sin alinear columnas a mano (no hace falta para que
    se vea bien en cualquier visor de Markdown, que es lo único que importa acá)."""
    out = ["| " + " | ".join(encabezados) + " |", "|" + "|".join([" --- "] * len(encabezados)) + "|"]
    for fila in filas:
        out.append("| " + " | ".join(fila) + " |")
    return "\n".join(out)


def _tabla_resumen(filas: list[dict[str, Any]]) -> str:
    """Sección 1 del reporte: P50/P95 de latencia, coste acumulado por escenario, hit
    rate de ambas cachés, recall medio del fact-tracker."""
    con_latencia = [f["latency_ms"] for f in filas if f["latency_ms"] is not None]
    p50 = _percentil(con_latencia, 50)
    p95 = _percentil(con_latencia, 95)

    costo_por_escenario: dict[str, float] = {}
    for f in filas:
        if f["kind"] == "memory" and f["cost_usd"] is not None:
            clave = f"{f['profile']}-{f['n_turns']}"
            costo_por_escenario[clave] = costo_por_escenario.get(clave, 0.0) + f["cost_usd"]
    # Promedio entre repeticiones: se suma por (escenario, repeat_index) y se promedia.
    costo_por_escenario_repeticion: dict[tuple[str, int], float] = {}
    for f in filas:
        if f["kind"] == "memory" and f["cost_usd"] is not None:
            clave = (f"{f['profile']}-{f['n_turns']}", f["repeat_index"])
            costo_por_escenario_repeticion[clave] = costo_por_escenario_repeticion.get(clave, 0.0) + f["cost_usd"]
    promedio_por_escenario: dict[str, list[float]] = {}
    for (escenario, _rep), costo in costo_por_escenario_repeticion.items():
        promedio_por_escenario.setdefault(escenario, []).append(costo)

    probes_cache = [f for f in filas if f["kind"] == "cache_probe" and f["cached"] is not None]
    hit_rate_cualquiera = (
        sum(1 for f in probes_cache if f["cached"]) / len(probes_cache) if probes_cache else None
    )
    exactas = [f for f in probes_cache if f["cache_hit_kind"] == "exacta"]
    semanticas = [f for f in probes_cache if f["cache_hit_kind"] == "semantica"]
    hit_rate_exacta = len(exactas) / len(probes_cache) if probes_cache else None
    hit_rate_semantica = len(semanticas) / len(probes_cache) if probes_cache else None

    con_recall = [f["recall_frac"] for f in filas if f["recall_frac"] is not None]
    recall_medio = statistics.mean(con_recall) if con_recall else None

    def _fmt(valor: float | None, sufijo: str = "") -> str:
        return f"{valor:.1f}{sufijo}" if valor is not None else "s/d"

    filas_tabla = [
        ["Latencia P50 (ms)", _fmt(p50)],
        ["Latencia P95 (ms)", _fmt(p95)],
        [
            "Coste acumulado por escenario (USD, promedio entre repeticiones)",
            ", ".join(f"{k}: ${statistics.mean(v):.5f}" for k, v in sorted(promedio_por_escenario.items())) or "s/d",
        ],
        ["Hit rate caché exacta", _fmt(hit_rate_exacta * 100 if hit_rate_exacta is not None else None, "%")],
        [
            "Hit rate caché semántica",
            _fmt(hit_rate_semantica * 100 if hit_rate_semantica is not None else None, "%"),
        ],
        [
            "Hit rate de cualquier caché (fallback en --http)",
            _fmt(hit_rate_cualquiera * 100 if hit_rate_cualquiera is not None else None, "%"),
        ],
        ["Recall medio del fact-tracker (todas las filas con hechos)", _fmt((recall_medio or 0) * 100, "%")],
    ]
    return _tabla_md(["Métrica", "Valor"], filas_tabla)


def _curva_latencia_vs_tokens(filas: list[dict[str, Any]]) -> str:
    """Sección 2a: latencia vs. tokens totales (in+out), en baldes de 500 tokens.
    Baldes en vez de un punto por fila porque dos llamadas con casi los mismos tokens
    no tienen por qué dar la misma latencia (el proveedor varía turno a turno)."""
    puntos = [
        (f["tokens_in"] + f["tokens_out"], f["latency_ms"])
        for f in filas
        if f["tokens_in"] is not None and f["tokens_out"] is not None and f["latency_ms"] is not None
    ]
    if not puntos:
        return "_Sin datos de tokens (ningún turno en proceso con `turn_observed` completo)._"
    baldes: dict[int, list[float]] = {}
    for tokens, latencia in puntos:
        balde = (tokens // 500) * 500
        baldes.setdefault(balde, []).append(latencia)
    filas_tabla = [
        [f"{b}-{b + 499}", str(len(lat)), f"{statistics.mean(lat):.0f}"]
        for b, lat in sorted(baldes.items())
    ]
    return _tabla_md(["Tokens (in+out)", "n", "Latencia media (ms)"], filas_tabla)


def _curva_costo_acumulado_vs_turno(filas: list[dict[str, Any]]) -> str:
    """Sección 2b: coste acumulado turno a turno, por perfil (promedio entre N y
    repeticiones en ese turno): la curva que muestra si el coste por turno crece,
    se achata, o escala con el contexto que arrastra la sesión."""
    por_perfil_turno: dict[tuple[str, int], list[float]] = {}
    acumulado: dict[tuple[str, int, int], float] = {}  # (perfil, n_turns, repeat) -> acumulado
    for f in sorted(
        (f for f in filas if f["kind"] == "memory" and f["cost_usd"] is not None),
        key=lambda f: f["turn_index"],
    ):
        clave_run = (f["profile"], f["n_turns"], f["repeat_index"])
        acumulado[clave_run] = acumulado.get(clave_run, 0.0) + f["cost_usd"]
        por_perfil_turno.setdefault((f["profile"], f["turn_index"]), []).append(acumulado[clave_run])
    if not por_perfil_turno:
        return "_Sin datos del barrido de memoria._"
    filas_tabla = [
        [perfil, str(turno), f"${statistics.mean(valores):.5f}"]
        for (perfil, turno), valores in sorted(por_perfil_turno.items())
    ]
    return _tabla_md(["Perfil", "Turno", "Coste acumulado medio (USD)"], filas_tabla)


def _curva_recall_vs_n(filas: list[dict[str, Any]]) -> str:
    """Sección 2c: recall al ÚLTIMO turno de cada (perfil, N) — la pregunta de fondo
    del Bloque 2: ¿sobrevive lo de antes cuando la conversación se alarga?"""
    por_perfil_n: dict[tuple[str, int], list[float]] = {}
    for f in filas:
        if f["kind"] != "memory" or f["recall_frac"] is None:
            continue
        if f["turn_index"] != f["n_turns"]:
            continue  # solo el ÚLTIMO turno de la conversación cuenta para esta curva
        por_perfil_n.setdefault((f["profile"], f["n_turns"]), []).append(f["recall_frac"])
    if not por_perfil_n:
        return "_Sin datos del barrido de memoria._"
    filas_tabla = [
        [perfil, str(n), f"{statistics.mean(valores) * 100:.0f}%"]
        for (perfil, n), valores in sorted(por_perfil_n.items())
    ]
    return _tabla_md(["Perfil", "N (turnos)", "Recall al último turno"], filas_tabla)


def _lectura(filas: list[dict[str, Any]]) -> str:
    """Sección 3: los dos párrafos de lectura, armados a partir de lo que esta
    corrida en particular midió (no un texto genérico pegado siempre igual)."""
    recall_por_n: dict[int, list[float]] = {}
    for f in filas:
        if f["kind"] == "memory" and f["recall_frac"] is not None and f["turn_index"] == f["n_turns"]:
            recall_por_n.setdefault(f["n_turns"], []).append(f["recall_frac"])
    ns_con_fuga = sorted(n for n, vs in recall_por_n.items() if statistics.mean(vs) < 1.0)

    adjuntos_100 = [f for f in filas if f["kind"] == "attachment" and f["attachment_kb"] == 100]
    hubo_truncamiento_100kb = any(f["metric_memory_recall_full"] == 0 for f in adjuntos_100 if f["metric_memory_recall_full"] is not None)

    probes_cache = [f for f in filas if f["kind"] == "cache_probe" and f["cached"] is not None]
    exacta_ok = any(f["cache_hit_kind"] == "exacta" for f in probes_cache)
    semantica_ok = any(f["cache_hit_kind"] == "semantica" for f in probes_cache)

    parrafo_1 = (
        "**Memoria conversacional.** "
        + (
            f"El recall al último turno cae por debajo del 100% a partir de N={ns_con_fuga[0]} "
            f"(de los N medidos en esta corrida: {sorted(recall_por_n)}); con menos turnos no se "
            "llega a perder nada."
            if ns_con_fuga
            else f"En los N medidos en esta corrida ({sorted(recall_por_n) or 'ninguno'}) el recall "
            "al último turno se mantuvo en 100%: no alcanzó a verse una fuga con este largo de "
            "conversación."
        )
        + " La ventana deslizante es la sospechosa natural (los hechos viejos dependen de haber "
        "sido promovidos a ancla o resumen antes de salir de la ventana; ver "
        "`app/sessions/compression.py`), pero esta corrida mide el síntoma (qué se recuerda), no "
        "la causa exacta turno por turno."
    )

    parrafo_2 = (
        "**Caché y adjuntos.** "
        + (
            "El adjunto de 100 KB vuelve a perder el requisito del final por el recorte de "
            "`MAX_TEXT_CHARS` (igual que en el Bloque 3): el límite de entrada de texto rompe antes "
            "que cualquier otra cosa medida acá. "
            if hubo_truncamiento_100kb
            else ""
        )
        + (
            f"La caché {'exacta' if exacta_ok else 'exacta (sin aciertos en esta corrida)'}"
            f"{' y la semántica' if semantica_ok else ', pero no la semántica,'} "
            if exacta_ok or semantica_ok
            else "Ninguna de las dos cachés acertó en esta corrida "
        )
        + (
            "acertó en el barrido de caché, como se esperaba con texto idéntico repetido "
            "(la semántica solo se consulta cuando la exacta falla, así que con repeticiones "
            "literales rara vez llega a intervenir). "
            if (exacta_ok or semantica_ok)
            else "(revisar que Redis esté arriba y la clave del proveedor de embeddings esté "
            "configurada antes de sacar conclusiones). "
        )
        + "Más de fondo: memoria conversacional, adjuntos y caché son tres caminos que en este "
        "servicio NO se solapan (`estimate_conversation` nunca consulta ninguna caché, y "
        "`estimate_structured`, el único con caché, no acepta adjuntos ni sesión). Es ahí donde mi "
        "CAG empieza a romperse: cualquier escenario real que combine las tres cosas a la vez "
        "(una conversación larga, con adjuntos, que además debería cachear lo repetido) hoy no "
        "tiene un camino único que lo cubra."
    )
    return parrafo_1 + "\n\n" + parrafo_2


def generar_reporte(filas: list[dict[str, Any]]) -> str:
    return (
        "# Bloque 5 — Reporte de estrés (LIDR_6)\n\n"
        "Generado por `evals/stress/run.py` a partir de `results.csv`. Ver el docstring del "
        "módulo para qué mide cada barrido y las limitaciones de `--http`.\n\n"
        "## 1. Resumen\n\n" + _tabla_resumen(filas) + "\n\n"
        "## 2. Curvas\n\n"
        "### 2a. Latencia vs. tokens\n\n" + _curva_latencia_vs_tokens(filas) + "\n\n"
        "### 2b. Coste acumulado vs. turno\n\n" + _curva_costo_acumulado_vs_turno(filas) + "\n\n"
        "### 2c. Recall vs. N\n\n" + _curva_recall_vs_n(filas) + "\n\n"
        "## 3. Dónde empieza a romperse mi CAG, y por qué\n\n" + _lectura(filas) + "\n"
    )


# --- CLI -----------------------------------------------------------------------------------


def _resolver_perfiles(crudo: str) -> list[str]:
    """`--scenarios`: nombres reales de `PROFILES` o los alias cortos del enunciado,
    separados por coma. Vacío = los tres."""
    if not crudo.strip():
        return list(PROFILES)
    resueltos = []
    for nombre in (n.strip() for n in crudo.split(",") if n.strip()):
        resuelto = ALIAS_PERFIL.get(nombre, nombre)
        if resuelto not in PROFILES:
            sys.exit(f"Perfil desconocido: {nombre!r} (válidos: {', '.join(PROFILES)}, o sus alias)")
        resueltos.append(resuelto)
    return resueltos


def _resolver_enteros(crudo: str, *, validos: tuple[int, ...] | None = None, nombre_campo: str = "valor") -> list[int]:
    if not crudo.strip():
        return list(validos) if validos else []
    enteros = [int(v.strip()) for v in crudo.split(",") if v.strip()]
    if validos is not None:
        desconocidos = [v for v in enteros if v not in validos]
        if desconocidos:
            sys.exit(f"{nombre_campo} inválido(s): {desconocidos} (válidos: {list(validos)})")
    return enteros


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--http", default=None, help="URL de un servicio ya corriendo; sin esto, en proceso")
    ap.add_argument("--scenarios", default="", help="Perfiles separados por coma (vacío = los tres)")
    ap.add_argument(
        "--turn-counts",
        default="1,3,6",
        help=f"N de turnos separados por coma, de {list(TURN_COUNTS)} (default: 1,3,6, no los 5 completos)",
    )
    ap.add_argument("--attachment-sizes", default="0,5,20,50,100", help="KB separados por coma")
    ap.add_argument("--repeats", type=int, default=3, help="Repeticiones por punto (sesiones nuevas)")
    ap.add_argument("--latency-budget-ms", type=int, default=LATENCY_BUDGET_MS_POR_DEFECTO)
    ap.add_argument("--cost-budget-usd", type=float, default=COST_BUDGET_USD_POR_DEFECTO)
    ap.add_argument("--output", default=str(SALIDA_CSV_POR_DEFECTO), help="CSV de salida")
    ap.add_argument("--report", default=str(SALIDA_REPORTE_POR_DEFECTO), help="Ruta del REPORT.md")
    ap.add_argument(
        "--solo-reporte",
        action="store_true",
        help="No llama al servicio: regenera --report a partir de un --output ya existente",
    )
    args = ap.parse_args()

    ruta_csv = Path(args.output)
    ruta_reporte = Path(args.report)

    if args.solo_reporte:
        if not ruta_csv.exists():
            sys.exit(f"No existe {ruta_csv}: no hay nada de donde regenerar el reporte")
        filas = leer_csv(ruta_csv)
    else:
        perfiles = _resolver_perfiles(args.scenarios)
        turn_counts = _resolver_enteros(args.turn_counts, validos=TURN_COUNTS, nombre_campo="--turn-counts")
        tamanos_kb = _resolver_enteros(args.attachment_sizes, validos=SIZE_POINTS_KB, nombre_campo="--attachment-sizes")
        if args.repeats < 1:
            sys.exit("--repeats tiene que ser al menos 1")

        modo = f"--http {args.http}" if args.http else "en proceso"
        print(f"Modo: {modo}. Perfiles: {perfiles}. N: {turn_counts}. Adjuntos (KB): {tamanos_kb}. Repeticiones: {args.repeats}.")

        cliente = Cliente(http_url=args.http)
        filas = []
        if perfiles and turn_counts:
            print("Barrido 1/3: memoria...")
            filas += correr_barrido_memoria(
                cliente,
                perfiles=perfiles,
                turn_counts=turn_counts,
                repeticiones=args.repeats,
                latency_budget_ms=args.latency_budget_ms,
                cost_budget_usd=args.cost_budget_usd,
            )
        if tamanos_kb:
            print("Barrido 2/3: adjuntos...")
            filas += correr_barrido_adjuntos(
                cliente,
                tamanos_kb=tamanos_kb,
                repeticiones=args.repeats,
                latency_budget_ms=args.latency_budget_ms,
                cost_budget_usd=args.cost_budget_usd,
            )
        if perfiles:
            print("Barrido 3/3: caché...")
            filas += correr_barrido_cache(
                cliente,
                perfiles=perfiles,
                repeticiones=args.repeats,
                latency_budget_ms=args.latency_budget_ms,
                cost_budget_usd=args.cost_budget_usd,
            )
        cliente.close()

        escribir_csv(filas, ruta_csv)
        print(f"\nCSV con {len(filas)} filas: {ruta_csv}")
        filas = leer_csv(ruta_csv)  # releer: el reporte trabaja con los tipos ya convertidos

    ruta_reporte.parent.mkdir(parents=True, exist_ok=True)
    ruta_reporte.write_text(generar_reporte(filas), encoding="utf-8")
    print(f"Reporte: {ruta_reporte}")


if __name__ == "__main__":
    main()
