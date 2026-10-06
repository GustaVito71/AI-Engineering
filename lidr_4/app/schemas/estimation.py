"""Schemas del contrato HTTP de estimaciones.

Son dos capas de límites sobre `description`, y no es redundancia:

- El `Field(20, 2000)` es el CONTRATO con el cliente. Lo que Swagger muestra y
  lo que Pydantic valida salen de acá, y no se puede reconfigurar: es la
  promesa de la API.
- `Settings.description_min_chars/max_chars` es el TECHO DEL OPERADOR. Existe
  porque el tamaño de la entrada es la factura, y un techo de coste que exige
  redeploy es un techo blando. Solo puede restringir el contrato, nunca
  ampliarlo (ver validate_description_ceiling en app/config.py).

Pydantic no puede volcar a un JSON Schema estático un `model_validator` que lee
configuración, así que app.main._complete_openapi_schema deriva esos dos
números a mano para que Swagger no prometa más de lo que el servicio tolera.
"""

from enum import Enum

from pydantic import BaseModel, Field, model_validator

from ..config import get_settings


class ProjectType(str, Enum):
    MOBILE_APP = "mobile_app"
    WEB_SAAS = "web_saas"
    INTERNAL_TOOL = "internal_tool"
    DATA_PIPELINE = "data_pipeline"


class DetailLevel(str, Enum):
    SUMMARY = "summary"
    MEDIUM = "medium"
    DETAILED = "detailed"


class OutputFormat(str, Enum):
    PHASES_TABLE = "phases_table"
    LINE_ITEMS = "line_items"
    NARRATIVE = "narrative"


class EstimationRequest(BaseModel):
    description: str = Field(
        min_length=20,
        max_length=2000,
        description="Descripción en texto libre o transcripción del proyecto para estimar.",
    )
    project_type: ProjectType
    detail_level: DetailLevel
    output_format: OutputFormat

    @model_validator(mode="after")
    def _validate_operator_ceiling(self) -> "EstimationRequest":
        """Aplica el techo configurable de Settings, ya validado contra el
        contrato al arrancar.

        El `Field` de arriba corre antes que este validador, así que acá solo
        puede llegar texto que ya cumple 20/2000. Solo resta el caso de un
        operador que bajó el techo a mano: se rechaza con un mensaje que
        nombra el número, para que el 422 sea accionable sin leer el traceback.

        Se valida en el borde del sistema: una entrada inválida se rechaza con
        422 sin tocar el proveedor, es decir, sin gastar un token. Poner solo
        min_length es validar lo que enseñan los tutoriales; el `max` es el
        que protege el coste."""
        cfg = get_settings()
        length = len(self.description)
        if length < cfg.description_min_chars:
            raise ValueError(
                f"La descripción debe tener al menos {cfg.description_min_chars} "
                f"caracteres (recibidos {length})."
            )
        if length > cfg.description_max_chars:
            raise ValueError(
                f"La descripción no puede superar {cfg.description_max_chars} "
                f"caracteres (recibidos {length})."
            )
        return self


class EstimationResponse(BaseModel):
    text: str = Field(description="Estimación provista por el LLM como texto libre.")
    prompt_version: str = Field(description="Identificador de la plantilla de prompt utilizada.")
    # Avisos para el usuario sobre cómo se generó la estimación (por ejemplo,
    # que el modelo de respaldo no está disponible). Lista vacía si no hay
    # ninguno: los clientes que no conocen el campo pueden ignorarlo.
    warnings: list[str] = Field(
        default_factory=list,
        description="Avisos para el usuario. Vacío si no hay ninguno.",
    )
