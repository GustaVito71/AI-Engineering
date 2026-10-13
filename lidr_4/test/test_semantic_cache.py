"""Tests de la caché semántica (WU10): app/services/semantic_cache.py y su uso en el servicio.

Sin red ni claves: Redis es fakeredis y los embeddings, vectores a mano. Lo que se fija:

- un acierto exige el mismo espacio de búsqueda, la similitud por encima del umbral y las
  mismas cifras en la descripción;
- solo se guardan estimaciones sanas (guarda anti-envenenamiento);
- todo fallo (embedding, Redis, entradas corruptas) es un miss, nunca un error;
- el modo `log_only` busca y registra pero no sirve;
- el servicio busca solo si la exacta falló, copia el acierto a la exacta y avisa al usuario;
- la configuración y el adaptador de embeddings.
"""

from __future__ import annotations

import asyncio
import copy
import itertools
import json
from types import SimpleNamespace

import httpx
import pytest
from fakeredis import FakeAsyncRedis
from openai import AsyncOpenAI
from pydantic import ValidationError
from redis.exceptions import RedisError
from structlog.testing import capture_logs

from app.schemas.estimation import DetailLevel, EstimationRequest, OutputFormat, ProjectType
from app.schemas.structured_estimation import RenderedResult, StructuredResult
from app.services import semantic_cache as modulo
from app.services.cache import EstimationCache
from app.services.estimation import EstimationService
from app.services.llm_wrapper import StructuredCallResult
from app.services.semantic_cache import (
    LiteLLMEmbedder,
    SemanticLookup,
    SemanticRedisCache,
    create_semantic_cache,
    extract_numbers,
)

NS = '["openai/gpt-4o-mini", "v3", "web_saas", "medium", ""]'
DESC = "Un SaaS B2B pequeño para gestionar el préstamo de equipos a empleados."

ESTIMACION = {
    "phases": [
        {
            "name": "Implementación",
            "summary": "Trabajo de la fase de Implementación.",
            "duration_weeks": 5,
            "hours": 320,
            "cost_eur": 20000,
            "confidence_pct": 80,
            "assumptions": [],
            "risks": [],
        }
    ],
    "team": [{"role": "Desarrollador", "headcount": 2}],
    "totals": {"hours": 320, "cost_eur": 20000, "duration_weeks": 5},
    "summary": "Estimación de prueba con las fases y el equipo indicados.",
    "confidence_pct": 90,
}
RECHAZO = {
    "phases": [
        {
            "name": "Sin estimar",
            "summary": "No se puede dimensionar sin más información.",
            "duration_weeks": 0,
            "hours": 0,
            "cost_eur": 0,
            "confidence_pct": 0,
            "assumptions": [],
            "risks": [],
        }
    ],
    "team": [],
    "totals": {"hours": 0, "cost_eur": 0, "duration_weeks": 0},
    "summary": "Fuera de alcance: falta el alcance.",
    "confidence_pct": 10,
}


def _estimacion(**cambios) -> StructuredResult:
    return StructuredResult.model_validate({**copy.deepcopy(ESTIMACION), **cambios})


def _entrada(**cambios) -> dict:
    estimacion = _estimacion(**cambios)
    return {
        "content": estimacion.model_dump_json(),
        "model": "openai/gpt-4o-mini",
        "provider": "openai",
        "prompt_tokens": 10,
        "completion_tokens": 20,
        "cost_usd": 0.001,
    }


class _Embedder:
    """Vectores a mano por texto; lo que no está, falla."""

    def __init__(self, vectores: dict[str, list[float]], error: Exception | None = None) -> None:
        self.vectores = vectores
        self.error = error
        self.llamadas: list[str] = []

    async def embed(self, text: str) -> list[float]:
        self.llamadas.append(text)
        if self.error is not None:
            raise self.error
        return self.vectores[text]


def _cache(embedder, redis=None, **kwargs) -> SemanticRedisCache:
    valores = {
        "embedding_model": "m",
        "threshold": 0.9,
        "ttl": 60,
        "min_confidence_pct": 70,
        "max_entries": 10,
    }
    return SemanticRedisCache(
        redis or FakeAsyncRedis(decode_responses=True), embedder, **{**valores, **kwargs}
    )


def _vectores(**por_texto: float) -> dict[str, list[float]]:
    """Textos a vectores unitarios con el ángulo dado: coseno = cos(ángulo entre ellos)."""
    import math

    return {texto: [math.cos(a), math.sin(a)] for texto, a in por_texto.items()}


# --- Cifras -------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("texto", "esperado"),
    [
        ("Sin cifras", ()),
        ("20.000 usuarios", (20000.0,)),
        ("20,000 usuarios", (20000.0,)),
        ("20000 usuarios", (20000.0,)),
        ("1,5 millones y 2.5 años", (1.5, 2.5)),
        ("150.000.000 de pesos", (150000000.0,)),
        ("12 meses, 12 equipos y 3 sprints", (3.0, 12.0)),
        ("versión 1.2.3", (123.0,)),
        ("Un SaaS B2B con i18n, S3 y Web3", ()),
        ("20k usuarios y 5M de filas", (5.0, 20.0)),
    ],
)
def test_extract_numbers(texto, esperado) -> None:
    assert extract_numbers(texto) == esperado


# --- Búsqueda y guarda ----------------------------------------------------------------


async def _guardar(cache, embedder, descripcion=DESC, namespace=NS, estimacion=None) -> bool:
    busqueda = await cache.lookup(namespace, descripcion)
    return await cache.store(
        namespace, descripcion, busqueda, _entrada(), estimacion or _estimacion()
    )


async def test_una_descripcion_parecida_acierta_y_trae_la_entrada() -> None:
    embedder = _Embedder(_vectores(**{DESC: 0.0, "otra": 0.1}))  # coseno 0,995
    cache = _cache(embedder)
    assert await _guardar(cache, embedder)

    with capture_logs() as logs:
        busqueda = await cache.lookup(NS, "otra")

    assert busqueda.entry == _entrada()
    assert busqueda.similarity == pytest.approx(0.995, abs=1e-3)
    assert [e["event"] for e in logs] == ["cache_semantica_acierto"]


async def test_por_debajo_del_umbral_es_un_miss_con_la_mejor_similitud() -> None:
    embedder = _Embedder(_vectores(**{DESC: 0.0, "lejana": 0.8}))  # coseno ~0,70
    cache = _cache(embedder)
    await _guardar(cache, embedder)

    with capture_logs() as logs:
        busqueda = await cache.lookup(NS, "lejana")

    assert busqueda.entry is None
    assert busqueda.similarity == pytest.approx(0.697, abs=1e-2)
    assert busqueda.embedding is not None
    [evento] = logs
    assert evento["event"] == "cache_semantica_fallo" and evento["entradas"] == 1


async def test_el_umbral_es_inclusivo() -> None:
    embedder = _Embedder({DESC: [1.0, 0.0], "borde": [1.0, 0.0]})
    cache = _cache(embedder, threshold=1.0)
    await _guardar(cache, embedder)
    assert (await cache.lookup(NS, "borde")).entry is not None


async def test_otro_espacio_de_busqueda_nunca_acierta() -> None:
    embedder = _Embedder({DESC: [1.0, 0.0]})
    cache = _cache(embedder)
    await _guardar(cache, embedder)

    otro = await cache.lookup(NS.replace("v3", "v4"), DESC)

    assert otro.entry is None and otro.similarity is None


async def test_otro_modelo_de_embeddings_no_comparte_vectores() -> None:
    redis = FakeAsyncRedis(decode_responses=True)
    embedder = _Embedder({DESC: [1.0, 0.0]})
    await _guardar(_cache(embedder, redis, embedding_model="a"), embedder)

    assert (await _cache(embedder, redis, embedding_model="b").lookup(NS, DESC)).entry is None


async def test_cifras_distintas_no_aciertan_aunque_la_similitud_sea_alta() -> None:
    a, b = "App con 20.000 usuarios", "App con 200.000 usuarios"
    embedder = _Embedder({a: [1.0, 0.0], b: [1.0, 0.0]})  # idénticas para el embedding
    cache = _cache(embedder)
    await _guardar(cache, embedder, descripcion=a)

    with capture_logs() as logs:
        busqueda = await cache.lookup(NS, b)

    assert busqueda.entry is None
    assert [e["event"] for e in logs] == [
        "cache_semantica_cifras_distintas",
        "cache_semantica_fallo",
    ]


async def test_las_mismas_cifras_con_otro_formato_aciertan() -> None:
    a, b = "App con 20.000 usuarios", "Una app para 20000 usuarios"
    embedder = _Embedder({a: [1.0, 0.0], b: [1.0, 0.0]})
    cache = _cache(embedder)
    await _guardar(cache, embedder, descripcion=a)
    assert (await cache.lookup(NS, b)).entry is not None


async def test_si_la_mejor_tiene_otras_cifras_se_prueba_la_siguiente() -> None:
    a = "App con 20.000 usuarios"
    b = "App con 200.000 usuarios"
    c = "App para 20.000 usuarios y pagos"
    embedder = _Embedder(
        {a: [1.0, 0.0], b: [1.0, 0.0], c: [0.99, 0.141], "q": [1.0, 0.0]},
    )
    cache = _cache(embedder, threshold=0.9)
    await _guardar(cache, embedder, descripcion=b)
    await _guardar(cache, embedder, descripcion=c)

    # «q» no trae cifras: ninguna coincide; con 20.000 acierta la tercera aunque la segunda sea mejor.
    embedder.vectores["App con 20.000 usuarios y mas"] = [1.0, 0.0]
    busqueda = await cache.lookup(NS, "App con 20.000 usuarios y mas")
    assert busqueda.entry is not None


async def test_log_only_busca_y_registra_pero_no_sirve() -> None:
    embedder = _Embedder({DESC: [1.0, 0.0], "igual": [1.0, 0.0]})
    cache = _cache(embedder, log_only=True)
    await _guardar(cache, embedder)

    with capture_logs() as logs:
        busqueda = await cache.lookup(NS, "igual")

    assert busqueda.entry is None
    assert busqueda.similarity == pytest.approx(1.0)
    assert busqueda.embedding is not None
    assert [e["event"] for e in logs] == ["cache_semantica_solo_log"]


async def test_ningun_evento_lleva_la_descripcion() -> None:
    secreto = "descripcion-secreta-con-12345 usuarios"
    embedder = _Embedder({secreto: [1.0, 0.0]})
    cache = _cache(embedder)
    with capture_logs() as logs:
        await _guardar(cache, embedder, descripcion=secreto)
        await cache.lookup(NS, secreto)
    assert logs and "secreta" not in str(logs)


# --- Qué se guarda -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("estimacion", "se_guarda"),
    [
        (_estimacion(), True),
        (_estimacion(confidence_pct=70), True),
        (_estimacion(confidence_pct=69), False),
        (StructuredResult.model_validate(RECHAZO), False),
        (_estimacion(totals={"hours": 330, "cost_eur": 20000, "duration_weeks": 5}), False),
    ],
)
async def test_solo_se_guardan_estimaciones_sanas(estimacion, se_guarda) -> None:
    embedder = _Embedder({DESC: [1.0, 0.0]})
    cache = _cache(embedder)

    with capture_logs() as logs:
        guardada = await _guardar(cache, embedder, estimacion=estimacion)

    assert guardada is se_guarda
    assert (await cache.lookup(NS, DESC)).entry is not None if se_guarda else True
    assert any(e["event"] == "cache_semantica_no_guardada" for e in logs) is (not se_guarda)
    if not se_guarda:
        assert (await cache.lookup(NS, DESC)).entry is None


async def test_sin_embedding_no_se_guarda() -> None:
    cache = _cache(_Embedder({}))
    assert await cache.store(NS, DESC, SemanticLookup(), _entrada(), _estimacion()) is False


async def test_la_entrada_tiene_ttl_y_el_hash_tambien() -> None:
    redis = FakeAsyncRedis(decode_responses=True)
    embedder = _Embedder({DESC: [1.0, 0.0]})
    cache = _cache(embedder, redis, ttl=60)
    await _guardar(cache, embedder)

    claves = [k async for k in redis.scan_iter("estimator:sem:*")]
    assert len(claves) == 2
    for clave in claves:
        assert 0 < await redis.ttl(clave) <= 60


async def test_la_misma_descripcion_sobrescribe_su_entrada() -> None:
    redis = FakeAsyncRedis(decode_responses=True)
    embedder = _Embedder({DESC: [1.0, 0.0]})
    cache = _cache(embedder, redis)
    await _guardar(cache, embedder)
    await _guardar(cache, embedder)
    [vectores] = [k async for k in redis.scan_iter("estimator:sem:v:*")]
    assert await redis.hlen(vectores) == 1


async def test_el_tope_desaloja_las_mas_antiguas(monkeypatch) -> None:
    redis = FakeAsyncRedis(decode_responses=True)
    textos = {f"descripcion {i}": [1.0, 0.0] for i in range(4)}
    embedder = _Embedder(textos)
    cache = _cache(embedder, redis, max_entries=2)
    reloj = itertools.count(100)
    monkeypatch.setattr(modulo, "time", SimpleNamespace(time=lambda: next(reloj)))

    for texto in textos:
        await _guardar(cache, embedder, descripcion=texto)

    [vectores] = [k async for k in redis.scan_iter("estimator:sem:v:*")]
    assert await redis.hlen(vectores) == 2
    assert len([k async for k in redis.scan_iter("estimator:sem:e:*")]) == 2
    assert (await cache.lookup(NS, "descripcion 0")).entry is None  # desalojada
    assert (await cache.lookup(NS, "descripcion 3")).entry is not None
    ids_vivos = set(await redis.hkeys(vectores))
    assert modulo.SemanticRedisCache._entry_id(NS, "descripcion 3") in ids_vivos
    assert modulo.SemanticRedisCache._entry_id(NS, "descripcion 0") not in ids_vivos


# --- Fail soft ------------------------------------------------------------------------


async def test_un_fallo_del_embedding_es_un_miss_sin_vector() -> None:
    cache = _cache(_Embedder({}, error=TimeoutError("lento")))
    with capture_logs() as logs:
        busqueda = await cache.lookup(NS, DESC)

    assert busqueda == SemanticLookup()
    [evento] = logs
    assert (evento["event"], evento["operacion"], evento["tipo_error"]) == (
        "cache_semantica_no_disponible",
        "embedding",
        "TimeoutError",
    )


class _RedisCaido:
    """Todo falla con RedisError."""

    def __getattr__(self, _nombre):
        async def fallar(*_a, **_k):
            raise RedisError("caído")

        return fallar

    def pipeline(self, **_kwargs):
        raise RedisError("caído")


async def test_con_redis_caido_buscar_y_guardar_no_fallan() -> None:
    embedder = _Embedder({DESC: [1.0, 0.0]})
    cache = _cache(embedder, _RedisCaido())

    with capture_logs() as logs:
        busqueda = await cache.lookup(NS, DESC)
        guardada = await cache.store(NS, DESC, busqueda, _entrada(), _estimacion())

    assert busqueda.entry is None and busqueda.embedding is not None
    assert guardada is False
    assert {e["operacion"] for e in logs if e["event"] == "cache_semantica_no_disponible"} == {
        "lectura",
        "escritura",
    }


async def test_una_entrada_corrupta_o_de_otra_dimension_se_ignora() -> None:
    redis = FakeAsyncRedis(decode_responses=True)
    embedder = _Embedder({DESC: [1.0, 0.0]})
    cache = _cache(embedder, redis)
    await _guardar(cache, embedder)
    [vectores] = [k async for k in redis.scan_iter("estimator:sem:v:*")]
    await redis.hset(vectores, "basura", "no es json")
    await redis.hset(vectores, "sin_campos", json.dumps({"x": 1}))
    await redis.hset(
        vectores, "otra_dim", json.dumps({"v": modulo._pack([1.0, 0.0, 0.0]), "n": []})
    )

    assert (await cache.lookup(NS, DESC)).entry is not None


async def test_un_vector_de_otra_dimension_no_acierta_aunque_su_entrada_exista() -> None:
    redis = FakeAsyncRedis(decode_responses=True)
    embedder = _Embedder({DESC: [1.0, 0.0]})
    cache = _cache(embedder, redis)
    await _guardar(cache, embedder)
    [vectores] = [k async for k in redis.scan_iter("estimator:sem:v:*")]
    [entry_id] = await redis.hkeys(vectores)
    await redis.hset(vectores, entry_id, json.dumps({"v": modulo._pack([1.0, 0.0, 0.0]), "n": []}))

    assert (await cache.lookup(NS, DESC)).entry is None


async def test_un_rechazo_no_se_guarda_ni_con_el_umbral_de_confianza_en_cero() -> None:
    embedder = _Embedder({DESC: [1.0, 0.0]})
    cache = _cache(embedder, min_confidence_pct=0)

    assert not await _guardar(cache, embedder, estimacion=StructuredResult.model_validate(RECHAZO))


async def test_de_varias_candidatas_acierta_la_mas_parecida() -> None:
    a, b, q = "Plataforma A", "Plataforma B", "Plataforma Q"
    embedder = _Embedder({a: [1.0, 0.0], b: [0.96, 0.28], q: [1.0, 0.0]})
    cache = _cache(embedder, threshold=0.9)
    for texto, nombre in ((a, "A"), (b, "B")):
        busqueda = await cache.lookup(NS, texto)
        await cache.store(NS, texto, busqueda, {**_entrada(), "model": nombre}, _estimacion())

    assert (await cache.lookup(NS, q)).entry["model"] == "A"


async def test_una_entrada_vencida_se_descarta_y_limpia_su_vector() -> None:
    redis = FakeAsyncRedis(decode_responses=True)
    embedder = _Embedder({DESC: [1.0, 0.0]})
    cache = _cache(embedder, redis)
    await _guardar(cache, embedder)
    [entrada] = [k async for k in redis.scan_iter("estimator:sem:e:*")]
    await redis.delete(entrada)

    assert (await cache.lookup(NS, DESC)).entry is None
    [vectores] = [k async for k in redis.scan_iter("estimator:sem:v:*")] or [None]
    assert vectores is None or await redis.hlen(vectores) == 0


async def test_una_entrada_que_no_es_un_objeto_se_descarta() -> None:
    redis = FakeAsyncRedis(decode_responses=True)
    embedder = _Embedder({DESC: [1.0, 0.0]})
    cache = _cache(embedder, redis)
    await _guardar(cache, embedder)
    [entrada] = [k async for k in redis.scan_iter("estimator:sem:e:*")]
    await redis.set(entrada, "[1, 2]")
    assert (await cache.lookup(NS, DESC)).entry is None


# --- Construcción desde la configuración -----------------------------------------------


def test_por_defecto_esta_encendida(settings_predeterminada, construir_settings) -> None:
    assert settings_predeterminada.semantic_cache_enabled is True
    apagada = construir_settings(semantic_cache_enabled=False)
    with capture_logs() as logs:
        assert create_semantic_cache(apagada, FakeAsyncRedis()) is None
    assert logs == []


def test_encendida_sin_redis_queda_apagada_con_un_aviso(construir_settings) -> None:
    ajustes = construir_settings(semantic_cache_enabled=True, openai_api_key="k")
    with capture_logs() as logs:
        assert create_semantic_cache(ajustes, None) is None
    [evento] = logs
    assert evento["event"] == "cache_semantica_desactivada"
    assert evento["motivo"] == "REDIS_URL vacío"


def test_encendida_sin_la_clave_de_embeddings_queda_apagada_con_un_aviso(
    construir_settings,
) -> None:
    ajustes = construir_settings(semantic_cache_enabled=True)
    with capture_logs() as logs:
        assert create_semantic_cache(ajustes, FakeAsyncRedis()) is None
    [evento] = logs
    assert evento["motivo"] == "falta la clave de openai"


def test_encendida_con_todo_se_arma_con_la_configuracion(construir_settings) -> None:
    ajustes = construir_settings(
        semantic_cache_enabled=True,
        openai_api_key="k",
        cache_semantic_threshold=0.8,
        semantic_cache_log_only=True,
        semantic_cache_max_entries=5,
        low_confidence_warn_pct=60,
        cache_ttl=99,
    )
    with capture_logs() as logs:
        cache = create_semantic_cache(ajustes, FakeAsyncRedis())

    assert isinstance(cache, SemanticRedisCache)
    assert (cache._threshold, cache._log_only, cache._max_entries) == (0.8, True, 5)
    assert (cache._min_confidence_pct, cache._ttl) == (60, 99)
    [evento] = logs
    assert evento["event"] == "cache_semantica_activada"
    assert (evento["modelo"], evento["umbral"], evento["solo_log"]) == (
        "text-embedding-3-small",
        0.8,
        True,
    )


def test_los_valores_por_defecto_y_vacios(settings_predeterminada, construir_settings) -> None:
    assert settings_predeterminada.cache_semantic_threshold == 0.95
    assert settings_predeterminada.embedding_model == "text-embedding-3-small"
    assert settings_predeterminada.semantic_cache_max_entries == 200
    vacios = construir_settings(
        semantic_cache_enabled="",
        cache_semantic_threshold="",
        semantic_cache_log_only="",
        semantic_cache_max_entries="",
        embedding_model="",
    )
    assert vacios.semantic_cache_enabled is True and vacios.semantic_cache_log_only is False
    assert vacios.cache_semantic_threshold == 0.95
    assert vacios.semantic_cache_max_entries == 200
    assert vacios.embedding_model == "text-embedding-3-small"


@pytest.mark.parametrize(
    ("campo", "valor", "texto"),
    [
        ("cache_semantic_threshold", 0, "CACHE_SEMANTIC_THRESHOLD"),
        ("cache_semantic_threshold", 1.01, "CACHE_SEMANTIC_THRESHOLD"),
        ("embedding_timeout", 0, "EMBEDDING_TIMEOUT"),
        ("semantic_cache_max_entries", 0, "SEMANTIC_CACHE_MAX_ENTRIES"),
    ],
)
def test_una_configuracion_invalida_rompe_el_arranque(
    construir_settings, campo, valor, texto
) -> None:
    with pytest.raises(ValidationError, match=texto):
        construir_settings(**{campo: valor})


def test_el_umbral_uno_es_valido(construir_settings) -> None:
    assert construir_settings(cache_semantic_threshold=1).cache_semantic_threshold == 1


# --- Adaptador de embeddings --------------------------------------------------------------


def _cliente_openai(pedidos: list[httpx.Request]) -> AsyncOpenAI:
    def manejar(pedido: httpx.Request) -> httpx.Response:
        pedidos.append(pedido)
        cuerpo = {
            "object": "list",
            "model": "text-embedding-3-small",
            "data": [{"object": "embedding", "index": 0, "embedding": [0.25, -0.5, 1.0]}],
            "usage": {"prompt_tokens": 3, "total_tokens": 3},
        }
        return httpx.Response(200, json=cuerpo)

    http = httpx.AsyncClient(transport=httpx.MockTransport(manejar))
    return AsyncOpenAI(api_key="sk-de-prueba", http_client=http, max_retries=0)


async def test_el_adaptador_pide_el_modelo_con_la_clave_y_el_texto() -> None:
    pedidos: list[httpx.Request] = []
    embedder = LiteLLMEmbedder(
        api_key="sk-de-prueba",
        model="text-embedding-3-small",
        timeout=5,
        client=_cliente_openai(pedidos),
    )

    vector = await embedder.embed("hola")

    assert vector == [0.25, -0.5, 1.0]
    [pedido] = pedidos
    assert pedido.url.path == "/v1/embeddings"
    assert pedido.headers["authorization"] == "Bearer sk-de-prueba"
    cuerpo = json.loads(pedido.content)
    assert cuerpo["model"] == "text-embedding-3-small" and cuerpo["input"] == ["hola"]


async def test_el_adaptador_corta_la_llamada_en_el_timeout(monkeypatch) -> None:
    async def lenta(**_kwargs):
        await asyncio.sleep(5)

    monkeypatch.setattr(modulo.litellm, "aembedding", lenta)
    with pytest.raises(asyncio.TimeoutError):
        await LiteLLMEmbedder(api_key="k", model="m", timeout=0.05).embed("hola")


# --- En el servicio -------------------------------------------------------------------------

SOLICITUD = EstimationRequest(
    description=DESC,
    project_type=ProjectType.WEB_SAAS,
    detail_level=DetailLevel.MEDIUM,
    output_format=OutputFormat.PHASES_TABLE,
)
PARECIDA = SOLICITUD.model_copy(
    update={"description": "Un SaaS B2B pequeño para gestionar el préstamo de portátiles."}
)


class _Wrapper:
    primary_model = "openai/gpt-4o-mini"
    warnings: tuple[str, ...] = ()

    def __init__(self, estimacion: dict = ESTIMACION) -> None:
        self.llamadas = 0
        self._estimacion = estimacion

    async def estimate_structured(self, *, prompt_version, response_model, context, **_kwargs):
        self.llamadas += 1
        estimacion = response_model.model_validate(self._estimacion, context=context)
        return StructuredCallResult(
            content=estimacion.model_dump_json(),
            model="openai/gpt-4o-mini",
            provider="openai",
            prompt_tokens=10,
            completion_tokens=20,
            cost_usd=0.001,
            prompt_version=prompt_version,
            estimation=estimacion,
        )

    async def estimate(self, **_kwargs):  # pragma: no cover - el texto no usa la semántica
        raise AssertionError("no se esperaba una llamada de texto")


def _servicio(wrapper=None, semantic=None, redis=None) -> EstimationService:
    cache = EstimationCache(redis or FakeAsyncRedis(decode_responses=True), ttl=60)
    return EstimationService(wrapper or _Wrapper(), cache=cache, semantic_cache=semantic)


def _semantica(redis, **kwargs) -> SemanticRedisCache:
    embedder = _Embedder({DESC: [1.0, 0.0], PARECIDA.description: [0.999, 0.045]})
    return _cache(embedder, redis, **kwargs)


async def test_la_parecida_se_sirve_sin_llamar_al_modelo_y_avisa() -> None:
    redis = FakeAsyncRedis(decode_responses=True)
    wrapper = _Wrapper()
    servicio = _servicio(wrapper, _semantica(redis), redis)

    primera = await servicio.estimate_structured(SOLICITUD, "v3", StructuredResult)
    with capture_logs() as logs:
        segunda = await servicio.estimate_structured(PARECIDA, "v3", StructuredResult)

    assert wrapper.llamadas == 1
    assert primera.call.cached is False and segunda.call.cached is True
    assert segunda.call.estimation == primera.call.estimation
    assert len(segunda.warnings) == 1 and "similitud 100 %" in segunda.warnings[0]
    completada = [e for e in logs if e["event"] == "estimacion_completada"]
    assert completada[0]["desde_cache"] is True and completada[0]["tipo_cache"] == "semantica"


async def test_el_acierto_semantico_se_copia_a_la_cache_exacta() -> None:
    redis = FakeAsyncRedis(decode_responses=True)
    servicio = _servicio(_Wrapper(), _semantica(redis), redis)
    await servicio.estimate_structured(SOLICITUD, "v3", StructuredResult)
    await servicio.estimate_structured(PARECIDA, "v3", StructuredResult)

    embedder = _Embedder({})  # fallaría si se buscara por la semántica
    apagada = _servicio(_Wrapper(), _cache(embedder, redis), redis)
    tercera = await apagada.estimate_structured(PARECIDA, "v3", StructuredResult)

    assert tercera.call.cached is True and embedder.llamadas == []
    assert tercera.warnings == []  # un acierto exacto no lleva el aviso de similitud


async def test_un_acierto_de_la_exacta_no_busca_ni_paga_embedding() -> None:
    redis = FakeAsyncRedis(decode_responses=True)
    embedder = _Embedder({DESC: [1.0, 0.0]})
    servicio = _servicio(_Wrapper(), _cache(embedder, redis), redis)
    await servicio.estimate_structured(SOLICITUD, "v3", StructuredResult)
    llamadas = len(embedder.llamadas)

    await servicio.estimate_structured(SOLICITUD, "v3", StructuredResult)

    assert len(embedder.llamadas) == llamadas  # solo el de la primera (búsqueda y escritura juntas)


async def test_el_embedding_se_calcula_una_vez_por_estimacion_nueva() -> None:
    redis = FakeAsyncRedis(decode_responses=True)
    embedder = _Embedder({DESC: [1.0, 0.0]})
    servicio = _servicio(_Wrapper(), _cache(embedder, redis), redis)

    await servicio.estimate_structured(SOLICITUD, "v3", StructuredResult)

    assert embedder.llamadas == [DESC]


async def test_el_espacio_cambia_con_la_version_el_tipo_y_el_formato_de_la_renderizada() -> None:
    servicio = _servicio()
    v3 = servicio._namespace(SOLICITUD, "v3", StructuredResult)
    assert servicio._namespace(SOLICITUD, "v4", StructuredResult) != v3
    otro_tipo = SOLICITUD.model_copy(update={"project_type": ProjectType.MOBILE_APP})
    assert servicio._namespace(otro_tipo, "v3", StructuredResult) != v3
    renderizada = servicio._namespace(SOLICITUD, "v4", RenderedResult)
    narrativa = SOLICITUD.model_copy(update={"output_format": OutputFormat.NARRATIVE})
    assert servicio._namespace(narrativa, "v4", RenderedResult) != renderizada
    assert servicio._namespace(narrativa, "v3", StructuredResult) == v3


async def test_otro_tipo_de_proyecto_no_acierta() -> None:
    redis = FakeAsyncRedis(decode_responses=True)
    wrapper = _Wrapper()
    servicio = _servicio(wrapper, _semantica(redis), redis)
    await servicio.estimate_structured(SOLICITUD, "v3", StructuredResult)

    otro_tipo = PARECIDA.model_copy(update={"project_type": ProjectType.MOBILE_APP})
    await servicio.estimate_structured(otro_tipo, "v3", StructuredResult)

    assert wrapper.llamadas == 2


async def test_en_v3_el_formato_de_salida_no_separa_el_espacio() -> None:
    redis = FakeAsyncRedis(decode_responses=True)
    wrapper = _Wrapper()
    servicio = _servicio(wrapper, _semantica(redis), redis)
    await servicio.estimate_structured(SOLICITUD, "v3", StructuredResult)

    otro_formato = PARECIDA.model_copy(update={"output_format": OutputFormat.NARRATIVE})
    resultado = await servicio.estimate_structured(otro_formato, "v3", StructuredResult)

    assert wrapper.llamadas == 1 and resultado.call.cached is True


async def test_una_estimacion_dudosa_no_se_reparte_a_las_parecidas() -> None:
    redis = FakeAsyncRedis(decode_responses=True)
    wrapper = _Wrapper({**ESTIMACION, "confidence_pct": 40})
    servicio = _servicio(wrapper, _semantica(redis), redis)

    await servicio.estimate_structured(SOLICITUD, "v3", StructuredResult)
    await servicio.estimate_structured(PARECIDA, "v3", StructuredResult)

    assert wrapper.llamadas == 2


async def test_log_only_no_sirve_y_la_llamada_sigue() -> None:
    redis = FakeAsyncRedis(decode_responses=True)
    wrapper = _Wrapper()
    servicio = _servicio(wrapper, _semantica(redis, log_only=True), redis)
    await servicio.estimate_structured(SOLICITUD, "v3", StructuredResult)

    segunda = await servicio.estimate_structured(PARECIDA, "v3", StructuredResult)

    assert wrapper.llamadas == 2 and segunda.call.cached is False


async def test_una_entrada_semantica_que_ya_no_cumple_el_schema_se_trata_como_miss() -> None:
    redis = FakeAsyncRedis(decode_responses=True)
    wrapper = _Wrapper()
    servicio = _servicio(wrapper, _semantica(redis), redis)
    await servicio.estimate_structured(SOLICITUD, "v3", StructuredResult)
    for clave in [k async for k in redis.scan_iter("estimator:sem:e:*")]:
        await redis.set(clave, json.dumps({**_entrada(), "content": '{"phases": []}'}))

    resultado = await servicio.estimate_structured(PARECIDA, "v3", StructuredResult)

    assert wrapper.llamadas == 2 and resultado.call.cached is False


async def test_sin_semantica_el_servicio_se_comporta_como_antes() -> None:
    redis = FakeAsyncRedis(decode_responses=True)
    wrapper = _Wrapper()
    servicio = _servicio(wrapper, None, redis)

    await servicio.estimate_structured(SOLICITUD, "v3", StructuredResult)
    await servicio.estimate_structured(PARECIDA, "v3", StructuredResult)

    assert wrapper.llamadas == 2


async def test_un_embedding_caido_no_rompe_la_estimacion() -> None:
    redis = FakeAsyncRedis(decode_responses=True)
    wrapper = _Wrapper()
    semantica = _cache(_Embedder({}, error=TimeoutError("x")), redis)
    servicio = _servicio(wrapper, semantica, redis)

    resultado = await servicio.estimate_structured(SOLICITUD, "v3", StructuredResult)

    assert wrapper.llamadas == 1 and resultado.warnings == []


# --- Cableado --------------------------------------------------------------------------------


def _app(monkeypatch, **entorno):
    from app.config import get_settings
    from app.main import create_app

    monkeypatch.setenv("REDIS_URL", "redis://127.0.0.1:1/0")  # nadie escucha: fail soft
    for variable, valor in entorno.items():
        monkeypatch.setenv(variable, valor)
    get_settings.cache_clear()
    return create_app()


def test_el_lifespan_no_arma_la_semantica_por_defecto(monkeypatch) -> None:
    from fastapi.testclient import TestClient

    app = _app(monkeypatch)
    with TestClient(app):
        assert app.state.semantic_cache is None


def test_el_lifespan_arma_la_semantica_si_se_pide_y_hay_clave(monkeypatch) -> None:
    from fastapi.testclient import TestClient

    app = _app(monkeypatch, SEMANTIC_CACHE_ENABLED="true", OPENAI_API_KEY="k")
    with TestClient(app):
        assert isinstance(app.state.semantic_cache, SemanticRedisCache)


def test_el_endpoint_sirve_la_parecida_con_su_aviso(monkeypatch) -> None:
    from fastapi.testclient import TestClient

    from app.dependencies import get_estimation_cache, get_llm_wrapper, get_semantic_cache

    redis = FakeAsyncRedis(decode_responses=True)
    wrapper = _Wrapper()
    semantica = _semantica(redis)
    app = _app(monkeypatch)
    app.dependency_overrides[get_llm_wrapper] = lambda: wrapper
    app.dependency_overrides[get_estimation_cache] = lambda: EstimationCache(redis, ttl=60)
    app.dependency_overrides[get_semantic_cache] = lambda: semantica
    cuerpo = {"project_type": "web_saas", "detail_level": "medium", "output_format": "narrative"}

    with TestClient(app) as cliente:
        ruta = "/api/v1/estimate/structured"
        primera = cliente.post(ruta, json={**cuerpo, "description": DESC})
        segunda = cliente.post(ruta, json={**cuerpo, "description": PARECIDA.description})

    assert primera.status_code == segunda.status_code == 200
    assert primera.json()["warnings"] == []
    assert any("similitud" in aviso for aviso in segunda.json()["warnings"])
    assert wrapper.llamadas == 1
