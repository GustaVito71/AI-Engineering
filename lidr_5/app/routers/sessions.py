"""Router de los endpoints de sesiones conversacionales.

QUÉ HACE ESTE ARCHIVO

Define las direcciones HTTP que tienen que ver con sesiones:

- `POST /sessions`: crea una sesión vacía y devuelve su `session_id`.
- `GET /sessions/{session_id}`: estado visible de la sesión (turnos recordados y
  metadata del proyecto), para que un cliente muestre qué recuerda el servicio.
- `POST /sessions/{session_id}/estimate`: estima a partir de una transcripción y,
  opcionalmente, de documentos adjuntos (PDF, Word, .txt o .md). Llega por formulario
  multipart (el formato que usan los navegadores para subir archivos).

Memoria del proyecto (Paso 4): el endpoint de estimar inyecta en el prompt la metadata de
la sesión (`<project_metadata>`, versión v5) y, cuando la estimación está lista, le pide a
un extractor LLM que la actualice con lo que aportó el turno. Si el extractor falla, la
estimación se entrega igual y la respuesta lo avisa en `warnings`.

Historial (Paso 5): el endpoint le pasa al modelo los últimos `MAX_TURNS` pares de la
sesión (ventana deslizante) y, tras una estimación exitosa, anota el turno nuevo. Los
adjuntos no se guardan en el historial: solo viajan en el turno que los subió.

QUÉ ES UNA SESIÓN

Una sesión es la "libreta" de una conversación: guarda los últimos mensajes y los
hechos del proyecto en curso (nombre, equipo, tecnologías, alcance). Al crearla está
vacía. El cliente conserva el `session_id` y lo manda en cada petición para que el
servicio recupere esa libreta.

RESPONSABILIDAD DEL ROUTER

El router solo traduce HTTP: recibe la petición, le pide al almacén (`SessionStore`)
que cree la sesión y devuelve el resultado con el código correcto. No decide cómo se
guarda una sesión ni cuántos turnos recuerda; eso lo hace `app/sessions/`.

LÍMITE CONOCIDO

Las sesiones viven en la memoria del proceso: se pierden al reiniciar el servicio. Para
que no crezcan sin tope caducan tras `SESSION_TTL_SECONDS` sin uso (se responde 404) y
hay un máximo de `MAX_SESSIONS` vivas (se descarta la menos usada). Ver `app/sessions/`.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status

from app.agent.context import fit_turn_context, gather_tool_context
from app.agent.tools import ToolRegistry
from app.config import Settings, get_settings
from app.dependencies import (
    get_compression_tools,
    get_critic,
    get_estimation_service,
    get_metadata_extractor,
    get_session_store,
    get_tool_registry,
)
from app.prompts.loader import OUTPUT_STRUCTURED
from app.routers.estimations import _resolve_version, translate_estimation_errors
from app.schemas.acb import ACBResponse
from app.schemas.critic import CriticFeedback
from app.schemas.estimation import DetailLevel, EstimationRequest, OutputFormat, ProjectType
from app.schemas.session import CreateSessionResponse, SessionStateResponse
from app.schemas.structured_estimation import StructuredResult
from app.schemas.tier_views import TieredEstimationResponse
from app.services.attachments import (
    MAX_FILE_BYTES,
    MAX_FILES,
    AttachmentError,
    ExtractedAttachment,
    TooManyAttachmentsError,
    apply_total_limit,
    build_attachments_block,
    check_attachment_name,
    collect_warnings,
    extract_attachment,
)
from app.services.boss import Boss, Draft
from app.services.critic import Critic, CriticReview
from app.services.estimation import EstimationService
from app.services.llm_wrapper import StructuredCallResult
from app.services.metadata_extractor import MetadataExtractor
from app.services.tier_projection import project_to_tier
from app.sessions import ProjectMetadata, Session, SessionNotFoundError, SessionStore
from app.sessions.compression import CompressionTools, apply_compression
from app.sessions.tier_resolver import Tier, resolve_tier
from app.tracing import emit

# Largo permitido de la transcripción (en caracteres, ya sin espacios en los extremos).
MIN_TRANSCRIPT_CHARS = 20
MAX_TRANSCRIPT_CHARS = 80_000
# Cuánto del mensaje del usuario se guarda en el historial para los turnos siguientes.
MAX_HISTORY_USER_CHARS = 4_000

# `prefix` antepone "/sessions" a todas las rutas de este archivo, y `tags` las agrupa
# bajo el mismo título en la documentación automática (`/docs`).
router = APIRouter(prefix="/sessions", tags=["sessions"])


@router.post(
    "",
    response_model=CreateSessionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Crea una sesión conversacional vacía",
)
async def create_session(
    # `Depends` le pide a FastAPI que entregue el almacén compartido de sesiones (el
    # mismo para todas las peticiones; ver `get_session_store` en app/dependencies.py).
    store: SessionStore = Depends(get_session_store),
) -> CreateSessionResponse:
    """Crea una sesión sin mensajes ni metadata y devuelve su `session_id`.

    Responde 201 ("creado"), el código estándar cuando una petición crea algo nuevo.
    No recibe datos: la sesión nace vacía y se va llenando con las estimaciones.
    """
    session = store.create()
    # Evento de trazabilidad: queda en el log con el identificador y cuántas sesiones
    # hay vivas. No guarda ningún contenido de la conversación (todavía no hay).
    emit(
        __name__,
        "sesion_creada",
        session_id=session.session_id,
        sesiones_activas=len(store),
    )
    return CreateSessionResponse(session_id=session.session_id)


@router.get(
    "/{session_id}",
    response_model=SessionStateResponse,
    summary="Estado visible de una sesión: turnos recordados y metadata del proyecto",
)
async def get_session_state(
    session_id: str,
    store: SessionStore = Depends(get_session_store),
) -> SessionStateResponse:
    """Devuelve cuántos turnos recuerda la sesión y los hechos del proyecto anotados.

    Responde 404 si la sesión no existe o caducó. Consultarla cuenta como uso: renueva
    su plazo de caducidad, igual que estimar.
    """
    try:
        session = store.get_or_raise(session_id)
    except SessionNotFoundError:
        raise HTTPException(status_code=404, detail="La sesión no existe.") from None
    return SessionStateResponse(
        session_id=session.session_id,
        turns=len(session.history.messages) // 2,
        max_turns=session.history.max_turns,
        project_metadata=session.metadata,
        anchors=len(session.history.anchors) // 2,
        has_summary=bool(session.history.summary),
        tier=session.last_tier,
        tier_rule=session.last_tier_rule,
    )


@router.post(
    "/{session_id}/estimate",
    response_model=TieredEstimationResponse,
    summary="Estima a partir de una transcripción y documentos adjuntos",
    description=(
        "Recibe un formulario multipart con la `transcript` (20 a 80.000 caracteres) y, "
        "opcionalmente, hasta 5 `attachments` (.pdf, .docx, .txt o .md, de hasta 10 MB "
        "cada uno). El texto de los adjuntos se extrae en el servicio y se agrega al "
        "prompt. Un adjunto sin texto o ilegible no hace fallar la petición: queda en "
        "`warnings`. Con `?prompt_version=v3` se elige la versión del prompt. La "
        "respuesta trae, además de `estimation`, la vista propia del tier resuelto "
        "para este turno en `tier_view` (null con tier `default`)."
    ),
)
async def estimate_session(
    session_id: str,
    transcript: Annotated[str, Form(description="Conversación o descripción del proyecto.")],
    project_type: Annotated[ProjectType, Form()],
    detail_level: Annotated[DetailLevel, Form()],
    output_format: Annotated[OutputFormat, Form()],
    attachments: Annotated[list[UploadFile] | None, File()] = None,
    prompt_version: str | None = Query(
        default=None,
        description="Versión del prompt de salida estructurada (por ejemplo `v3`).",
        examples=["v3"],
    ),
    store: SessionStore = Depends(get_session_store),
    service: EstimationService = Depends(get_estimation_service),
    extractor: MetadataExtractor | None = Depends(get_metadata_extractor),
    compression: CompressionTools = Depends(get_compression_tools),
    registry: ToolRegistry = Depends(get_tool_registry),
) -> TieredEstimationResponse:
    """Estima un proyecto a partir de la transcripción y de los adjuntos.

    Orden de las comprobaciones, de la más barata a la más cara: sesión existente (404),
    versión de prompt (422), largo de la transcripción (422), cantidad y formato de los
    archivos (422 / 415), tamaño al leerlos (413) y, recién entonces, extracción de texto
    y llamada al modelo.
    """
    turn = await _prepare_turn(
        store=store,
        session_id=session_id,
        transcript=transcript,
        project_type=project_type,
        detail_level=detail_level,
        output_format=output_format,
        attachments=attachments,
        prompt_version=prompt_version,
    )
    session, settings = turn.session, turn.settings
    resolved_tier, tier_rule = resolve_tier(
        transcript=turn.transcript,
        metadata=session.metadata,
        default=Tier(settings.default_tier),
    )

    # Resumen, anclas y la ventana deslizante: como mucho `MAX_TURNS` pares anteriores.
    history = session.history.to_history_messages()
    with translate_estimation_errors("conversacional"):
        attachments_block, extra_warnings = await _turn_context(
            turn, settings, history, registry, service
        )
        start = time.perf_counter()
        estimation = await service.estimate_conversation(
            turn.request,
            turn.version,
            attachments_block=attachments_block,
            extra_warnings=extra_warnings,
            project_metadata=session.metadata,
            tier=resolved_tier.value,
            history=history,
        )
    latency_ms = round((time.perf_counter() - start) * 1000, 1)

    memory_warnings = await _record_turn(
        session,
        extractor,
        compression,
        turn.transcript,
        estimation.call.estimation,
        tier=resolved_tier,
        tier_rule=tier_rule,
    )
    _emit_turn("estimacion_conversacional", session_id, turn, settings_version=turn.version)
    _emit_turn_observed(
        session,
        turn,
        tier=resolved_tier,
        call=estimation.call,
        latency_ms=latency_ms,
    )
    return TieredEstimationResponse(
        estimation=estimation.call.estimation,
        prompt_version=estimation.call.prompt_version,
        cached=False,
        warnings=[*estimation.warnings, *memory_warnings],
        tier_view=project_to_tier(estimation.call.estimation, resolved_tier.value),
    )


@router.post(
    "/{session_id}/estimate-acb",
    response_model=ACBResponse,
    summary="Estima y revisa la estimación con un crítico (Actor-Crítico-Jefe)",
    description=(
        "Mismo formulario que `/estimate` más un `tier` opcional (`executive`, `pm`, "
        "`developer` o `default`). El actor estima, un crítico de otro proveedor revisa "
        "y el jefe acepta, vuelve a pedir una estimación con el feedback o entrega la "
        "última con salvedades. Hace como máximo `BOSS_MAX_ITERATIONS` vueltas de dos "
        "llamadas cada una. La respuesta trae la estimación, en `acb` la traza, y en "
        "`tier_view` la vista propia del tier resuelto para este turno (null con tier "
        "`default`)."
    ),
)
async def estimate_session_acb(
    session_id: str,
    transcript: Annotated[str, Form(description="Conversación o descripción del proyecto.")],
    project_type: Annotated[ProjectType, Form()],
    detail_level: Annotated[DetailLevel, Form()],
    output_format: Annotated[OutputFormat, Form()],
    tier: Annotated[
        Tier | None, Form(description="Audiencia de la respuesta. Sin valor, la configurada.")
    ] = None,
    attachments: Annotated[list[UploadFile] | None, File()] = None,
    prompt_version: str | None = Query(
        default=None,
        description="Versión del prompt de salida estructurada del actor (por ejemplo `v5`).",
        examples=["v5"],
    ),
    store: SessionStore = Depends(get_session_store),
    service: EstimationService = Depends(get_estimation_service),
    extractor: MetadataExtractor | None = Depends(get_metadata_extractor),
    critic: Critic = Depends(get_critic),
    compression: CompressionTools = Depends(get_compression_tools),
    registry: ToolRegistry = Depends(get_tool_registry),
) -> ACBResponse:
    """Estima con el patrón Actor-Crítico-Jefe y devuelve la estimación con su traza.

    Hace las mismas comprobaciones de entrada que `/estimate` (ver `_prepare_turn`). El
    actor es `EstimationService.estimate_conversation`, con la misma ventana deslizante y
    la misma memoria; el crítico es `Critic.review` y el jefe `Boss.run`. El turno se
    anota en el historial con la estimación FINAL (la que el usuario ve).
    """
    turn = await _prepare_turn(
        store=store,
        session_id=session_id,
        transcript=transcript,
        project_type=project_type,
        detail_level=detail_level,
        output_format=output_format,
        attachments=attachments,
        prompt_version=prompt_version,
    )
    session, settings = turn.session, turn.settings
    resolved_tier, tier_rule = resolve_tier(
        transcript=turn.transcript,
        metadata=session.metadata,
        override=tier,
        default=Tier(settings.default_tier),
    )
    # La ventana se fija antes del bucle: todas las vueltas ven el mismo historial.
    history = session.history.to_history_messages()
    # Adjuntos y herramientas se resuelven UNA vez, antes de la primera vuelta: las vueltas
    # siguientes del actor reutilizan lo recogido (el bucle de herramientas no se multiplica).
    with translate_estimation_errors("conversacional"):
        attachments_block, extra_warnings = await _turn_context(
            turn, settings, history, registry, service
        )

    async def actor(feedback: CriticFeedback | None) -> Draft:
        """Una llamada del actor; con feedback, el prompt incluye los problemas señalados."""
        estimation = await service.estimate_conversation(
            turn.request,
            turn.version,
            attachments_block=attachments_block,
            extra_warnings=extra_warnings,
            project_metadata=session.metadata,
            tier=resolved_tier.value,
            history=history,
            critic_feedback=feedback,
        )
        return Draft(
            result=estimation.call.estimation,
            provider=estimation.call.provider,
            payload=estimation,
        )

    async def review(draft: Draft) -> CriticReview:
        """Una llamada del crítico sobre un borrador del actor."""
        return await critic.review(
            transcript=turn.transcript,
            attachments_block=attachments_block,
            project_metadata=session.metadata,
            tier=resolved_tier.value,
            detail_level=detail_level.value,
            result=draft.result,
            actor_version=turn.version,
            actor_provider=draft.provider,
        )

    with translate_estimation_errors("conversacional"):
        outcome = await Boss(max_iterations=settings.boss_max_iterations).run(
            actor=actor, critic=review
        )

    memory_warnings = await _record_turn(
        session,
        extractor,
        compression,
        turn.transcript,
        outcome.result,
        tier=resolved_tier,
        tier_rule=tier_rule,
    )
    emit(
        __name__,
        "estimacion_acb",
        session_id=session_id,
        prompt_version=turn.version,
        tier=resolved_tier.value,
        regla_tier=tier_rule,
        vueltas=outcome.trace.iterations_run,
        decision_final=outcome.trace.final_decision,
        estados_revision=[it.review_status for it in outcome.trace.iterations],
        transcript_chars=len(turn.transcript),
        attachments_received=len(turn.extracted),
    )
    return ACBResponse(
        estimation=outcome.result,
        prompt_version=outcome.draft.payload.call.prompt_version,
        cached=False,
        warnings=[*outcome.draft.payload.warnings, *memory_warnings],
        acb=outcome.trace,
        tier_view=project_to_tier(outcome.result, resolved_tier.value),
    )


@dataclass
class _Turn:
    """Un turno ya validado: la sesión, la configuración y la entrada lista para estimar."""

    session: Session
    settings: Settings
    version: str
    transcript: str
    extracted: list[ExtractedAttachment]
    request: EstimationRequest


async def _prepare_turn(
    *,
    store: SessionStore,
    session_id: str,
    transcript: str,
    project_type: ProjectType,
    detail_level: DetailLevel,
    output_format: OutputFormat,
    attachments: list[UploadFile] | None,
    prompt_version: str | None,
) -> _Turn:
    """Las comprobaciones de entrada que comparten `/estimate` y `/estimate-acb`.

    Orden, de la más barata a la más cara: sesión existente (404), versión de prompt (422),
    largo de la transcripción (422), cantidad y formato de los archivos (422 / 415), tamaño
    al leerlos (413) y, recién entonces, extracción de texto. Devuelve el turno listo.
    """
    try:
        session = store.get_or_raise(session_id)
    except SessionNotFoundError:
        raise HTTPException(status_code=404, detail="La sesión no existe.") from None

    settings = get_settings()
    # Sin `?prompt_version=` se usa la versión de las sesiones (v5 por defecto), que es la
    # que inyecta la memoria del proyecto; las anteriores la ignoran.
    version = _resolve_version(
        prompt_version or settings.conversation_prompt_version, OUTPUT_STRUCTURED
    )

    transcript = transcript.strip()
    if not MIN_TRANSCRIPT_CHARS <= len(transcript) <= MAX_TRANSCRIPT_CHARS:
        raise HTTPException(
            status_code=422,
            detail=(
                f"La transcripción debe tener entre {MIN_TRANSCRIPT_CHARS} y "
                f"{MAX_TRANSCRIPT_CHARS:,} caracteres.".replace(",", ".")
            ),
        )

    try:
        extracted = await _read_attachments(attachments)
    except AttachmentError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from None

    # La transcripción puede superar los 2.000 caracteres de una descripción de lidr_4,
    # así que se arma la petición sin pasar por esa validación (el largo ya se comprobó
    # arriba). Los guardrails de entrada sí revisan la transcripción en el servicio.
    request = EstimationRequest.model_construct(
        description=transcript,
        project_type=project_type,
        detail_level=detail_level,
        output_format=output_format,
    )
    return _Turn(session, settings, version, transcript, extracted, request)


async def _turn_context(
    turn: _Turn,
    settings: Settings,
    history: list[dict[str, str]],
    registry: ToolRegistry,
    service: EstimationService,
) -> tuple[str, list[str]]:
    """El contexto extra del turno: bloque de adjuntos (+ resultados de herramientas) y avisos.

    Pasos: (1) arma el bloque de adjuntos y sus avisos; (2) si el bucle agéntico está
    activado y hay herramientas, lo corre y obtiene sus resultados como datos; (3) ajusta
    todo al presupuesto de tokens del turno, recortando primero los resultados de
    herramientas y luego los adjuntos. Con el bucle apagado o sin herramientas, solo se aplica
    el presupuesto a los adjuntos y no hay ninguna llamada extra al modelo.
    """
    attachments_block = build_attachments_block(turn.extracted)
    warnings = collect_warnings(turn.extracted)
    use_tools = settings.agent_loop_enabled and len(registry) > 0
    tools_block, _ = await gather_tool_context(
        settings=settings,
        registry=registry,
        llm=service.llm_wrapper if use_tools else None,
        transcript=turn.transcript,
        attachments_block=attachments_block,
    )
    fitted = fit_turn_context(
        settings=settings,
        transcript=turn.transcript,
        history_text="\n".join(m["content"] for m in history),
        attachments_block=attachments_block,
        tools_block=tools_block,
    )
    block = "\n\n".join(b for b in (fitted.attachments_block, fitted.tools_block) if b)
    return block, [*warnings, *fitted.warnings]


async def _record_turn(
    session: Session,
    extractor: MetadataExtractor | None,
    compression: CompressionTools,
    transcript: str,
    estimation: StructuredResult,
    *,
    tier: Tier,
    tier_rule: str,
) -> list[str]:
    """Anota el turno en el historial, lo comprime y actualiza la memoria; devuelve los avisos.

    Solo se llama con una estimación que salió bien: un turno fallido no se recuerda. El
    texto del usuario se recorta para que un mensaje enorme no encarezca los turnos
    siguientes; los adjuntos no se guardan (solo viajan en el turno que los subió). Luego se
    actualiza la memoria con lo que aportó el turno; si eso falla, la estimación no se
    pierde (ver `_update_memory`).
    """
    evicted = session.history.append(
        user=transcript[:MAX_HISTORY_USER_CHARS],
        assistant=_describe_estimation(estimation),
    )
    session.turns_total += 1
    session.last_tier, session.last_tier_rule = tier.value, tier_rule
    # Lo que salió de la ventana no se olvida: pasa a anclas (si es un compromiso) o al
    # resumen acumulado. Un fallo del resumidor solo agrega un aviso.
    compression_warnings = await apply_compression(
        session.history,
        evicted,
        detector=compression.detector,
        summarizer=compression.summarizer,
    )
    memory_warnings = await _update_memory(session, extractor, transcript, estimation)
    return [*compression_warnings, *memory_warnings]


def _emit_turn(event: str, session_id: str, turn: _Turn, *, settings_version: str) -> None:
    """Evento de trazabilidad del turno: solo cantidades, nunca el contenido."""
    emit(
        __name__,
        event,
        session_id=session_id,
        prompt_version=settings_version,
        transcript_chars=len(turn.transcript),
        attachments_received=len(turn.extracted),
        attachments_with_text=sum(1 for a in turn.extracted if a.text),
    )


def _emit_turn_observed(
    session: Session,
    turn: _Turn,
    *,
    tier: Tier,
    call: StructuredCallResult,
    latency_ms: float,
) -> None:
    """Evento `turn_observed`: todo lo que describe un turno, en un único registro del log.

    Lo emite `estimate_session` una vez por turno exitoso, después de comprimir la memoria,
    así que la ventana y las anclas que registra son las de después del turno. Reúne en un
    solo evento datos que antes salían sueltos, para poder extraer una tabla en una pasada.
    Solo lleva cantidades, nunca el contenido de la conversación.

    `cache_hit_kind` es siempre "none": el camino conversacional no usa caché (ver
    `EstimationService.estimate_conversation`). `tokens_in`, `tokens_out` y `cost_usd` son los
    de la llamada del actor, sumando todos sus intentos; no incluyen el extractor de memoria
    ni el resumidor, que emiten sus propios eventos. `latency_ms` mide solo la llamada al
    servicio de estimación.
    """
    attachments_total_chars = sum(len(a.text or "") for a in turn.extracted)
    history = session.history
    emit(
        __name__,
        "turn_observed",
        turn_index=session.turns_total,
        session_id=session.session_id,
        enriched_transcript_chars=len(turn.transcript) + attachments_total_chars,
        attachments_total_chars=attachments_total_chars,
        messages_in_window=len(history.messages),
        anchors_count=len(history.anchors),
        summary_chars=len(history.summary or ""),
        tokens_in=call.prompt_tokens,
        tokens_out=call.completion_tokens,
        cost_usd=call.cost_usd,
        latency_ms=latency_ms,
        cache_hit_kind="none",
        last_resolved_tier=tier.value,
    )


async def _read_attachments(files: list[UploadFile] | None) -> list[ExtractedAttachment]:
    """Lee los archivos subidos y extrae su texto, respetando los límites de entrada.

    Ignora las partes sin nombre (un formulario sin archivo manda una parte vacía).
    Comprueba cantidad y formato ANTES de leer, y lee como máximo un byte más que el
    tope para rechazar un archivo enorme sin cargarlo entero en memoria. Al final aplica el
    tope de texto entre todos los archivos. La extracción
    (PDF y Word pueden tardar) corre en un hilo aparte para no frenar al servidor.
    """
    files = [f for f in files or [] if f.filename]
    if len(files) > MAX_FILES:
        raise TooManyAttachmentsError(f"Se admiten como máximo {MAX_FILES} adjuntos por petición.")
    for upload in files:
        check_attachment_name(upload.filename or "")

    extracted = []
    for upload in files:
        data = await upload.read(MAX_FILE_BYTES + 1)
        extracted.append(await asyncio.to_thread(extract_attachment, upload.filename or "", data))
    # Tope de texto entre todos los archivos: lo que sobra se recorta o se descarta con aviso.
    return apply_total_limit(extracted)


async def _update_memory(
    session: Session,
    extractor: MetadataExtractor | None,
    transcript: str,
    estimation: StructuredResult,
) -> list[str]:
    """Actualiza `session.metadata` con el turno que acaba de pasar; devuelve los avisos.

    La memoria es un apoyo, no el producto: si no hay extractor o este falla (proveedor
    caído, respuesta inválida), la metadata anterior se conserva y el usuario recibe un
    aviso en `warnings`, pero la estimación se entrega igual. Al log va solo el tipo de
    error y cantidades, nunca el contenido de la conversación.
    """
    if extractor is None:
        return ["La memoria del proyecto no se actualizó: el extractor no está configurado."]
    try:
        extraction = await extractor.extract(
            session.metadata,
            user_turn=transcript,
            assistant_turn=_describe_estimation(estimation),
        )
    except Exception as exc:  # noqa: BLE001 - cualquier fallo del extractor es no fatal
        emit(
            __name__,
            "extractor_fallo",
            level="warning",
            session_id=session.session_id,
            error=type(exc).__name__,
        )
        return ["No se pudo actualizar la memoria del proyecto; la estimación no se vio afectada."]

    previous = session.metadata
    session.metadata = extraction.metadata
    emit(
        __name__,
        "metadata_actualizada",
        session_id=session.session_id,
        modelo=extraction.model,
        coste_usd=extraction.cost_usd,
        hechos_antes=_count_facts(previous),
        hechos_despues=_count_facts(extraction.metadata),
    )
    return []


def _describe_estimation(estimation: StructuredResult) -> str:
    """La estimación como texto breve para el extractor: resumen, equipo y totales."""
    team = ", ".join(f"{member.headcount} {member.role}" for member in estimation.team)
    totals = estimation.totals
    return (
        f"{estimation.summary}\n"
        f"Equipo: {team or 'sin definir'}.\n"
        f"Totales: {totals.hours} h, {totals.cost_eur} EUR, {totals.duration_weeks} semanas."
    )


def _count_facts(metadata: ProjectMetadata) -> int:
    """Cuántos hechos guarda la metadata (para el log, sin volcar su contenido)."""
    scalars = sum(
        value is not None
        for value in (metadata.project_name, metadata.assumed_team_size, metadata.agreed_scope)
    )
    return (
        scalars
        + len(metadata.mentioned_technologies)
        + len(metadata.explicit_constraints)
        + len(metadata.rejected_options)
        + len(metadata.other_facts)
    )
