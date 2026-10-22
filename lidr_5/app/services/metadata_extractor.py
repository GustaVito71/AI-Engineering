"""Extractor de `project_metadata`: mantiene al día la memoria del proyecto.

QUÉ HACE ESTE ARCHIVO

Después de cada estimación de una sesión, hace una segunda llamada a un LLM (más
barato y rápido que el de la estimación) para actualizar los hechos del proyecto:
nombre, equipo, tecnologías, alcance, restricciones y opciones descartadas. Esa
memoria se inyecta en el system prompt del turno siguiente (prompt v5), de modo que
el modelo sigue sabiendo lo acordado aunque el mensaje original ya haya salido de la
ventana deslizante.

POR QUÉ UN LLM Y NO REGLAS FIJAS

La conversación es libre y puede mezclar idiomas, así que una lista de palabras
clave perdería hechos. El benchmark del proyecto (ver CONTEXTO.md) comparó proveedores
y dejó a Gemini como primario por su latencia, con DeepSeek de respaldo.

CÓMO SE ACTUALIZA (política de olvido nº 1)

El extractor recibe la metadata actual y el último turno, y devuelve la metadata
COMPLETA ya revisada, no un parche. Así puede quitar o mover hechos cuando el usuario
cambia de opinión («ya no Rails, vamos con Node»). Después, `ProjectMetadata.normalized`
garantiza por código lo que el modelo podría olvidar: una tecnología descartada no puede
seguir entre las vigentes, y las listas quedan limpias y acotadas.

QUÉ PASA SI FALLA

La memoria es un apoyo, no el producto: si el extractor falla (sin clave, proveedor
caído, respuesta inválida), la estimación del usuario NO se pierde. El endpoint conserva
la metadata anterior y agrega un aviso en `warnings`. Por eso `extract` lanza la
excepción y es el llamador quien decide qué hacer.

CÓMO LLAMA AL PROVEEDOR

Reutiliza `LLMWrapper` (Router de LiteLLM, reintentos, respaldo y coste) con la
configuración del extractor (`Settings.for_extractor`), así que no hay un segundo
cliente que mantener. Lo que se registra en el log es solo cantidades y costes, nunca
el contenido de la conversación.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.config import Settings
from app.prompts.loader import render_metadata_extraction_prompt
from app.services.llm_wrapper import LLMWrapper
from app.sessions.models import ExtractedMetadata, ProjectMetadata

# Versión de la plantilla del extractor (app/prompts/metadata_extraction/<versión>/).
EXTRACTION_PROMPT_VERSION = "v1"

# Cuánto del turno se le pasa al extractor. La transcripción puede tener 80.000
# caracteres, pero los hechos acordados recientes están al final y el extractor solo
# necesita lo último para detectar cambios. El tope acota el coste de la segunda llamada.
MAX_USER_TURN_CHARS = 20_000
MAX_ASSISTANT_TURN_CHARS = 4_000

# Tope de tokens de la respuesta: la metadata es un objeto chico.
EXTRACTION_MAX_TOKENS = 1_000


@dataclass(frozen=True)
class MetadataExtraction:
    """Resultado de una extracción: la metadata nueva y lo que costó obtenerla."""

    metadata: ProjectMetadata
    model: str
    cost_usd: float

    @property
    def provider(self) -> str:
        """El proveedor que respondió (la parte antes de la `/` del nombre del modelo)."""
        return self.model.split("/", 1)[0]


class MetadataExtractor:
    """Actualiza la `ProjectMetadata` de una sesión a partir del último turno."""

    def __init__(self, wrapper: LLMWrapper) -> None:
        # El wrapper ya viene armado con el primario y el respaldo del extractor.
        self._wrapper = wrapper

    @classmethod
    def from_settings(cls, settings: Settings) -> MetadataExtractor:
        """Arma el extractor con EXTRACTOR_DEFAULT / EXTRACTOR_FALLBACK.

        Lanza `LLMConfigurationError` si falta la clave del proveedor primario del
        extractor (por ejemplo `GEMINI_API_KEY`); sin la del respaldo funciona igual.
        """
        return cls(LLMWrapper(settings=settings.for_extractor()))

    async def extract(
        self,
        current: ProjectMetadata,
        *,
        user_turn: str,
        assistant_turn: str,
    ) -> MetadataExtraction:
        """Devuelve la metadata actualizada con lo que aportó este turno.

        Al modelo se le pide la forma plana `ExtractedMetadata` (más simple para la
        generación estructurada de los proveedores) y aquí se convierte a la interna.

        Lanza las mismas excepciones que `LLMWrapper.estimate_structured`
        (`StructuredOutputError`, `IncompleteJSONError`, errores del proveedor).
        """
        system_prompt, user_prompt = render_metadata_extraction_prompt(
            current,
            # Del turno del usuario se conserva el final; del asistente, el principio.
            user_turn=user_turn[-MAX_USER_TURN_CHARS:],
            assistant_turn=assistant_turn[:MAX_ASSISTANT_TURN_CHARS],
            version=EXTRACTION_PROMPT_VERSION,
        )
        result = await self._wrapper.estimate_structured(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            prompt_version=f"metadata_extraction/{EXTRACTION_PROMPT_VERSION}",
            response_model=ExtractedMetadata,
            max_tokens=EXTRACTION_MAX_TOKENS,
        )
        return MetadataExtraction(
            metadata=result.estimation.to_project_metadata(),
            model=result.model,
            cost_usd=result.cost_usd,
        )
