"""Tests del extractor de `project_metadata`, de `ProjectMetadata.normalized` y de su configuración.

QUÉ SE COMPRUEBA

- `normalized` aplica la política de olvido nº 1 por código: una opción descartada sale de
  las tecnologías vigentes (sin distinguir mayúsculas), y las listas quedan limpias.
- `merge_with` acumula también las listas nuevas.
- `MetadataExtractor` arma el prompt con la metadata actual y el turno, recorta los
  turnos largos, pide el schema correcto y devuelve la metadata ya normalizada.
- La configuración del extractor: valores por defecto, proveedores válidos y distintos,
  claves de Gemini y DeepSeek, y la copia de `Settings` que usa el wrapper.
- El wrapper etiqueta la llamada del extractor como `metadata` y usa un evento propio.

Sin red: el wrapper se reemplaza por uno falso.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from pydantic import ValidationError
from structlog.testing import capture_logs

from app.services.llm_wrapper import emit_completion, output_label
from app.services.metadata_extractor import (
    MAX_ASSISTANT_TURN_CHARS,
    MAX_USER_TURN_CHARS,
    MetadataExtractor,
)
from app.sessions import ExtractedMetadata, ProjectMetadata
from app.sessions.models import MAX_ITEM_CHARS, MAX_LIST_ITEMS

# --- ProjectMetadata ----------------------------------------------------------------


def test_los_campos_nuevos_cuentan_para_is_empty() -> None:
    """Una restricción o una opción descartada ya son hechos conocidos."""
    assert not ProjectMetadata(explicit_constraints=["Plazo 3 meses"]).is_empty()
    assert not ProjectMetadata(rejected_options=["Rails"]).is_empty()


def test_normalized_quita_de_las_vigentes_lo_descartado() -> None:
    """Rails no puede estar a la vez entre las tecnologías vigentes y las descartadas."""
    metadata = ProjectMetadata(
        mentioned_technologies=["Rails", "Node", "PostgreSQL"], rejected_options=["rails"]
    ).normalized()
    assert metadata.mentioned_technologies == ["Node", "PostgreSQL"]
    assert metadata.rejected_options == ["rails"]


def test_normalized_limpia_las_listas() -> None:
    """Sin vacíos ni repetidos (sin distinguir mayúsculas), con espacios colapsados y acotadas."""
    metadata = ProjectMetadata(
        mentioned_technologies=["  Python ", "python", "", "Cython"],
        explicit_constraints=["A" * (MAX_ITEM_CHARS + 50)],
        rejected_options=[f"opción {i}" for i in range(MAX_LIST_ITEMS + 10)],
        project_name="   ",
        agreed_scope="",
    ).normalized()
    assert metadata.mentioned_technologies == ["Python", "Cython"]
    assert len(metadata.explicit_constraints[0]) == MAX_ITEM_CHARS
    assert len(metadata.rejected_options) == MAX_LIST_ITEMS
    assert metadata.project_name is None and metadata.agreed_scope is None


def test_merge_with_acumula_restricciones_y_descartadas() -> None:
    """La unión agrega las listas nuevas sin repetir y sin perder las anteriores."""
    base = ProjectMetadata(explicit_constraints=["Plazo 3 meses"], rejected_options=["Rails"])
    update = ProjectMetadata(
        explicit_constraints=["plazo 3 meses", "Solo cloud público"], rejected_options=["Java"]
    )
    merged = base.merge_with(update)
    assert merged.explicit_constraints == ["Plazo 3 meses", "Solo cloud público"]
    assert merged.rejected_options == ["Rails", "Java"]


# --- MetadataExtractor --------------------------------------------------------------


class _WrapperFalso:
    """Reemplaza al LLMWrapper: guarda lo que recibe y devuelve la metadata preparada."""

    def __init__(self, devuelve: ExtractedMetadata) -> None:
        self._devuelve = devuelve
        self.llamadas: list[dict] = []

    async def estimate_structured(self, **kwargs):
        """Guarda los argumentos y devuelve un resultado con la forma del real."""
        self.llamadas.append(kwargs)
        return SimpleNamespace(
            estimation=self._devuelve, model="gemini/gemini-3.1-flash-lite", cost_usd=0.002
        )


def _plano(**campos) -> ExtractedMetadata:
    """Una respuesta del modelo en la forma plana, con «no se conoce» por defecto."""
    base = {
        "project_name": "",
        "assumed_team_size": 0,
        "mentioned_technologies": [],
        "agreed_scope": "",
        "explicit_constraints": [],
        "rejected_options": [],
        "other_facts": [],
    }
    return ExtractedMetadata(**{**base, **campos})


def _extraer(wrapper: _WrapperFalso, current: ProjectMetadata, user: str, assistant: str):
    """Ejecuta `extract` de forma síncrona para los tests."""
    return asyncio.run(
        MetadataExtractor(wrapper).extract(current, user_turn=user, assistant_turn=assistant)
    )


def test_el_prompt_trata_el_numero_de_un_rol_como_dato_del_rol() -> None:
    """«3 desarrolladores» va a other_facts; el tamaño asumido es solo el del equipo entero."""
    from app.prompts.loader import render_metadata_extraction_prompt

    sistema, _ = render_metadata_extraction_prompt(
        ProjectMetadata(), user_turn="x", assistant_turn="y"
    )
    assert "3 desarrolladores" in sistema
    assert "other_facts" in sistema
    assert "un equipo de 3 personas" in sistema


def test_el_extractor_pide_el_schema_y_arma_el_prompt() -> None:
    """Pide `ProjectMetadata`, versiona el prompt y le pasa la metadata actual y el turno."""
    wrapper = _WrapperFalso(_plano(project_name="Bookflow"))
    actual = ProjectMetadata(project_name="Bookflow", mentioned_technologies=["Rails"])
    resultado = _extraer(wrapper, actual, "Ya no Rails, vamos con Node", "Estimación")

    llamada = wrapper.llamadas[0]
    assert llamada["response_model"] is ExtractedMetadata
    assert llamada["prompt_version"] == "metadata_extraction/v1"
    assert '"mentioned_technologies": [\n    "Rails"\n  ]' in llamada["user_prompt"]
    # La memoria actual se muestra en la forma plana: «no se conoce» es 0 o cadena vacía.
    assert '"assumed_team_size": 0' in llamada["user_prompt"]
    assert "Ya no Rails, vamos con Node" in llamada["user_prompt"]
    assert resultado.model == "gemini/gemini-3.1-flash-lite"
    assert resultado.provider == "gemini"
    assert resultado.cost_usd == 0.002


def test_el_resultado_sale_normalizado() -> None:
    """Aunque el modelo deje Rails en las dos listas, la metadata guardada es coherente."""
    devuelto = _plano(mentioned_technologies=["Rails", "Node"], rejected_options=["Rails"])
    resultado = _extraer(_WrapperFalso(devuelto), ProjectMetadata(), "Ya no Rails", "x")
    assert resultado.metadata.mentioned_technologies == ["Node"]
    assert resultado.metadata.rejected_options == ["Rails"]


def test_los_turnos_largos_se_recortan() -> None:
    """Del usuario se conserva el final y de la estimación el principio, para acotar el coste."""
    usuario = "INICIO" + "u" * (MAX_USER_TURN_CHARS + 100) + "FINAL"
    asistente = "COMIENZO" + "a" * (MAX_ASSISTANT_TURN_CHARS + 100) + "COLA"
    wrapper = _WrapperFalso(_plano())
    _extraer(wrapper, ProjectMetadata(), usuario, asistente)
    prompt = wrapper.llamadas[0]["user_prompt"]
    assert "FINAL" in prompt and "INICIO" not in prompt
    assert "COMIENZO" in prompt and "COLA" not in prompt


# --- ExtractedMetadata: la forma que se le pide al modelo -----------------------------


def test_el_esquema_para_el_proveedor_es_plano() -> None:
    """Sin campos opcionales, límites ni `null`, y con los seis campos obligatorios.

    Con el esquema de `ProjectMetadata` (opcionales con `anyOf`/`null` y límites) Gemini
    volcó los nombres de campo dentro de `rejected_options`; este test impide que el
    esquema vuelva a complicarse sin darse cuenta.
    """
    import json

    esquema = ExtractedMetadata.model_json_schema()
    texto = json.dumps(esquema)
    for prohibido in ("anyOf", '"null"', "minimum", "maximum", "maxLength"):
        assert prohibido not in texto
    assert (
        set(esquema["required"]) == set(esquema["properties"]) == set(ProjectMetadata.model_fields)
    )


def test_ida_y_vuelta_entre_la_forma_plana_y_la_interna() -> None:
    """Convertir a la forma plana y volver no cambia la metadata."""
    original = ProjectMetadata(
        project_name="Bookflow",
        assumed_team_size=3,
        mentioned_technologies=["Node"],
        agreed_scope="MVP",
        explicit_constraints=["Plazo 3 meses"],
        rejected_options=["Rails"],
    )
    assert ExtractedMetadata.from_project_metadata(original).to_project_metadata() == original
    assert (
        ExtractedMetadata.from_project_metadata(ProjectMetadata()).to_project_metadata().is_empty()
    )


def test_no_se_conoce_se_convierte_en_none() -> None:
    """Cadena vacía, 0 o un equipo fuera de 1 a 50 quedan como «sin dato», sin rechazar todo."""
    for equipo in (0, -2, 51, 500):
        metadata = _plano(
            project_name="  ", assumed_team_size=equipo, agreed_scope=""
        ).to_project_metadata()
        assert metadata.project_name is None
        assert metadata.assumed_team_size is None
        assert metadata.agreed_scope is None
    assert _plano(assumed_team_size=50).to_project_metadata().assumed_team_size == 50


def test_los_textos_largos_se_recortan_en_vez_de_fallar() -> None:
    """Un nombre o alcance demasiado largo se recorta a su tope y la respuesta se acepta."""
    metadata = _plano(project_name="N" * 500, agreed_scope="A" * 5000).to_project_metadata()
    assert len(metadata.project_name) == 120
    assert len(metadata.agreed_scope) == 2000


# --- Wrapper: etiqueta y evento del extractor --------------------------------------


def test_output_label_distingue_la_metadata() -> None:
    """La llamada del extractor se etiqueta `metadata`, no `estructurada`."""
    assert output_label(ExtractedMetadata) == "metadata"


def test_emit_completion_acepta_un_evento_propio() -> None:
    """El evento del extractor no se mezcla con `estimacion_completada` (coste separado)."""
    resultado = SimpleNamespace(
        model="gemini/x", provider="gemini", cached=False, prompt_tokens=1,
        completion_tokens=1, cost_usd=0.001, prompt_version="metadata_extraction/v1",
    )  # fmt: skip
    with capture_logs() as logs:
        emit_completion(resultado, 0.0, primary_model="gemini/x", event="metadata_extraida")
    assert [e["event"] for e in logs] == ["metadata_extraida"]


# --- Configuración del extractor ----------------------------------------------------


def test_defaults_del_extractor(settings_predeterminada) -> None:
    """Gemini primario y DeepSeek de respaldo, con sus modelos y la versión v5 de las sesiones."""
    s = settings_predeterminada
    assert (s.extractor_default, s.extractor_fallback) == ("gemini", "deepseek")
    assert s.gemini_model == "gemini-3.1-flash-lite" and s.deepseek_model == "deepseek-flash"
    assert s.conversation_prompt_version == "v5" and s.default_tier == "default"


def test_for_extractor_cambia_primario_y_respaldo(construir_settings) -> None:
    """La copia para el wrapper usa los proveedores del extractor, sin tocar el original."""
    s = construir_settings(gemini_api_key="g-key", deepseek_api_key="d-key")
    e = s.for_extractor()
    assert e.primary_model == "gemini/gemini-3.1-flash-lite"
    assert e.fallback_model == "deepseek/deepseek-flash"
    assert e.active_api_key("gemini") == "g-key" and e.active_api_key("deepseek") == "d-key"
    assert s.primary_model == "openai/gpt-4o-mini"  # el servicio no cambió


def test_sin_clave_active_api_key_es_none(settings_predeterminada) -> None:
    """Sin GEMINI_API_KEY no hay clave (y el wrapper no se podrá armar)."""
    assert settings_predeterminada.active_api_key("gemini") is None


def test_extractor_con_proveedores_iguales_se_rechaza(construir_settings) -> None:
    """Primario y respaldo del extractor tienen que ser distintos."""
    with pytest.raises(ValidationError, match="EXTRACTOR_DEFAULT y EXTRACTOR_FALLBACK son iguales"):
        construir_settings(extractor_default="gemini", extractor_fallback="gemini")


@pytest.mark.parametrize("campo", ["extractor_default", "extractor_fallback"])
def test_extractor_con_proveedor_desconocido_se_rechaza(construir_settings, campo) -> None:
    """Un proveedor fuera de la lista (openai, anthropic, gemini, deepseek) no se acepta."""
    with pytest.raises(ValidationError, match="no es un proveedor soportado"):
        construir_settings(**{campo: "mistral"})


def test_extractor_acepta_los_proveedores_del_servicio(construir_settings) -> None:
    """El extractor puede usar OpenAI o Anthropic (el servicio, en cambio, no puede usar Gemini)."""
    s = construir_settings(extractor_default="anthropic", extractor_fallback="openai")
    assert s.for_extractor().primary_model == "anthropic/claude-haiku-5-5"


def test_default_tier_invalido_se_rechaza(construir_settings) -> None:
    """El tier solo admite executive, pm, developer o default."""
    with pytest.raises(ValidationError, match="DEFAULT_TIER"):
        construir_settings(default_tier="boss")


def test_valores_vacios_vuelven_al_default(construir_settings) -> None:
    """Un `.env` copiado de `.env.example` con variables vacías no rompe el arranque."""
    s = construir_settings(
        extractor_default="", extractor_fallback="", gemini_model="", conversation_prompt_version="",
        default_tier="",
    )  # fmt: skip
    assert (s.extractor_default, s.extractor_fallback) == ("gemini", "deepseek")
    assert s.gemini_model == "gemini-3.1-flash-lite"
    assert s.conversation_prompt_version == "v5" and s.default_tier == "default"


def test_deepseek_usa_json_simple_y_el_resto_json_schema() -> None:
    """DeepSeek rechaza `response_format` con schema: se le pide JSON simple."""
    import instructor

    from app.services.llm_wrapper import instructor_mode

    assert instructor_mode("deepseek/deepseek-flash") is instructor.Mode.JSON
    assert instructor_mode("gemini/gemini-3.1-flash-lite") is instructor.Mode.JSON_SCHEMA
    assert instructor_mode("openai/gpt-4o-mini") is instructor.Mode.JSON_SCHEMA


def test_other_facts_hace_el_viaje_completo_y_se_limpia() -> None:
    """El campo libre pasa por la forma plana, se normaliza y no se pierde al fusionar."""
    plano = _plano(
        other_facts=["Cliente: cadena de gimnasios", "  cliente: CADENA de gimnasios ", ""]
    )
    metadata = plano.to_project_metadata()
    assert metadata.other_facts == ["Cliente: cadena de gimnasios"]
    assert not metadata.is_empty()
    assert ExtractedMetadata.from_project_metadata(metadata).other_facts == metadata.other_facts

    fusionada = metadata.merge_with(ProjectMetadata(other_facts=["Idioma: español"]))
    assert fusionada.other_facts == ["Cliente: cadena de gimnasios", "Idioma: español"]


def test_other_facts_tiene_topes() -> None:
    """Como las otras listas: como mucho 30 elementos de 200 caracteres."""
    metadata = _plano(
        other_facts=[f"hecho {n} " + "x" * 300 for n in range(50)]
    ).to_project_metadata()
    assert len(metadata.other_facts) == 30
    assert all(len(f) <= 200 for f in metadata.other_facts)
