# odd/tasks/async-concurrencia.md

> Feature: conversión del Estimador CAG a concurrencia async completa (punta a punta).
> Proyecto: lidr_3 (entrega 3 del curso AI Engineering 2026-09, base copiada de lidr_2).
> Fecha: 2026-09-24 · Estado: en curso.

## Objetivo

Convertir la cadena completa «HTTP handler → servicio → adaptador LLM» de síncrona
(bloqueante sobre threadpool) a `async/await` real, usando los clientes async nativos
de los SDK (AsyncOpenAI / AsyncAnthropic), para que una llamada lenta al LLM no
congele el event loop ni bloquee /health ni las requests concurrentes.

## Decisión técnica

Async de punta a punta (interfaz + adaptadores + servicio + router), NO parches:

- `BaseProvider.chat` → `async def chat` (contrato).
- `OpenAIProvider` → `AsyncOpenAI`, `await self.client.responses.create(...)`.
- `AnthropicProvider` → `AsyncAnthropic`, `await self.client.messages.create(...)`.
- `generate_estimation` → `async def` con `await provider.chat(...)`.
- `create_estimation` (router) → `async def` con `await generate_estimation(...)`.
- Lo que NO cambia: `health()` (sin I/O, no gana nada con async), `factory.create_provider`
  (pura construcción, sin I/O), `pricing.estimate_cost` y `context.examples`
  (funciones puras síncronas).

Decisión posterior (2026-09-25): se eliminó el comentario del router que advertía contra
volver a un cliente síncrono del SDK. Se descartó como meramente hipotético, no un riesgo
presente ni futuro: la cadena es async de punta a punta y el caso que describía (SDK
síncrono sin `await` dentro de un `async def`) no existe en este código.

## Checklist (stable IDs)

- [x] T1 — BaseProvider.chat → async (base.py)
- [x] T2 — OpenAIProvider: cliente AsyncOpenAI + chat async (openai_provider.py)
- [x] T3 — AnthropicProvider: cliente AsyncAnthropic + chat async (anthropic_provider.py)
- [x] T4 — llm_service.generate_estimation async con await (llm_service.py)
- [x] T5 — router create_estimation async (routers/estimations.py)
- [x] T6 — tests: fakes con `async def chat`, tests directos async, test de concurrencia
        que demuestre solapamiento real en el event loop (test/test_estimations.py)
- [x] T7 — Verificación: `uv run pytest -q` → 24 passed; `uv run ruff check .` → 0 errores.
- [x] T8 — Contrato Swagger completo: `_completar_schema_openapi()` en main.py deriva
      `minLength`/`maxLength` del request desde Settings activos (una sola fuente de
      verdad; el `model_validator` que lee Settings no es expresable como JSON Schema
      estático). Tests: `test_openapi_documenta_los_limites_del_request` y
      `test_openapi_sigue_los_limites_si_settings_cambian` (24 passed en total).
      (Además: refinado el segundo test para forzar env vars — no depender de un
      `.env` local del dev.)
- [x] T9 — Capa de schemas siguiendo la convención del curso (referencia
      `LIDR-academy/ai-engineering` rama `session_2`, `estimator/app/schemas/`):
      `EstimationRequest`/`EstimationResponse` extraídos del router a
      `app/schemas/estimation.py`. Se conservó TODO el comportamiento propio:
      validación con Settings configurables (no `Field(min_length=50)` hardcodeado),
      límites en Swagger derivados, respuesta con `truncated`/`cost_usd`/`cost_note`.
      El router importa los modelos desde schemas. 24 passed, ruff limpio.
        Demo en vivo: pendiente (requiere key real; no bloquea).

## Rutas y autorización

- Ruta elegida: **directa inline** (delegación imposible en este runtime: el mecanismo
  de subagentes respondió "OpenCode's free tier can only be used from within OpenCode",
  fallo no transitorio; se documenta como trigger de delegación incumplido por entorno).
- Trigger de writer (6 archivos no triviales) detectado y documentado.
- Autorización: el usuario pidió explícitamente "concurrencia real con async completo".
- Git: sin commits/push (lo maneja el humano; regla del entorno del autor).
- RDD: on (global) — aplicar ciclo de review nativo al finalizar.

## Estado del ciclo de review nativo (RDD on)

- `gentle-ai review assess --cwd <repo> --json` → riesgo `high`/`unassessable` (candidate en
  archivos untracked en el worktree, sin commits: el Go exige declaración explícita).
- Declaración de intended-untracked (7 archivos) → STATUS válido → START con consent relayed.
- Consent del humano: **otorgado** ("Revisar este cambio") → START creó lineage
  `review-522c2485a82b19a5` (riesgo medium, 8 archivos / 858 líneas, lente única
  `review-reliability`, corrección 200).
- STATUS → `collect` de `reviewer_result` (`review.capture-result`). El provider_task requiere
  un Task de subagente `review-reliability`; el mecanismo de subagentes del runtime falló con
  "OpenCode's free tier can only be used from within OpenCode" (no transitorio; 2 intentos,
  mismo error). El STATUS reofrece el mismo slot → **verificador nativo no disponible por runtime**.
- Resultado: no se inventa PASS; la captura del lente queda pendiente hasta que el mecanismo de
  subagentes esté disponible en un runtime que lo soporte. No corresponde handoff de defecto de
  Gentle AI: la falla la produce el runtime del cliente (free tier), no una invocación de
  gentle-ai.
- Decisión del humano (2026-09-24): **pausar**. No entregar todavía; el lineage queda en estado
  `reviewing` y se retoma en otra sesión.
- Retomada: ejecutar el STATUS exacto del binding (cwd = raíz del monorepo):
  `gentle-ai review status --contract=gentle-ai.review-integration/v2 --next-transition=true --lineage=review-522c2485a82b19a5 --repository-context=rctx2_df251f9e775139615aabc208fa8c51d4f476508c74964e38dd1aeb91c34ca06e --agent=opencode --projection=workspace`
  → debería reofrecer `collect` de `reviewer_result` (`review.capture-result`) para el lente
  `review-reliability`; lanzar el Task del `provider_task` solo cuando el mecanismo de subagentes
  funcione en el runtime.
- Retomada intentada (2026-09-24, segunda sesión): STATUS reofreció el mismo slot, el Task del
  `provider_task` se relanzó UNA vez y falló con el mismo `OpenCode's free tier can only be used
  from within OpenCode` (no transitorio). No se reintenta a ciegas ni se inventa PASS: el lente
  sigue sin capturar y la review queda en pausa hasta un runtime con subagentes disponibles.
- **Cierre (decisión del humano, 2026-09-24)**: cerrar con la verificación funcional. Se aceptan
  los 22 passed + ruff limpio como verificación del cambio. La review nativa NO se captura ni
  produce PASS: el lineage `review-522c2485a82b19a5` queda documentado como **no disponible por
  runtime** (subagentes caídos, fallo no transitorio). La entrega sigue la política ordinaria del
  repo; el lineage permanece en estado `reviewing` sin autoridad quemada, y puede retomarse desde
  un runtime con subagentes ejecutando el STATUS exacto del binding (comando en la sección
  «Retomada») si algún día hace falta.

## Verificación de evidencia

- `uv run pytest -q` → suite completa verde (incluye el nuevo test de concurrencia).
- `uv run ruff check .` → 0 errores.
- Verificación del SDK no asumida: `AsyncOpenAI`/`AsyncAnthropic` confirmados presentes
  en el venv instalado antes de escribir (lección del `temperature` de Anthropic).