"""Configuración de la aplicación (pydantic-settings, leída de .env/entorno).

Heredada de lidr_3, con los cambios que impone LiteLLM. Los tres principios
del original se conservan intactos:

1. Las API keys son opcionales en la construcción (`SecretStr | None`) y se
   exigen en el punto de uso: al construir el wrapper LLM, en la primera
   request que lo necesita (ver app/dependencies.py). Si la validación ocurre al
   importar el módulo, el proceso muere ANTES de que exista la aplicación:
   no hay /health, no hay /docs, no hay nada que le diga al orquestador qué
   falta. Un health check tiene que sobrevivir a la avería que diagnostica.

2. `aplicar_defaults` normaliza lo que pydantic-settings no puede: un `X=`
   vacío se parsea como `""`, no como `None`, así que el "no configurado" se
   detecta con falsy y no con `is None`. Un `.env` copiado de `.env.example`
   no debe romper el arranque. Un valor presente pero inválido sí lo rompe:
   fail fast.

3. El render del logging no vive acá, vive en main.configure_logging.

QUÉ CAMBIÓ RESPECTO DE LIDR_3 Y POR QUÉ

- Desaparecen `LLM_PROVIDER` y `LLM_FALLBACK`. Ya no hay un switch de
  proveedor: el `Router` de LiteLLM recibe un deployment por modelo (el
  primario y, si tiene clave, el de respaldo) y decide el failover. Escribir
  el retry a mano sería tener dos dueños de la misma decisión.

- El provider se deriva del prefijo del MODELO ("openai/gpt-4o-mini" ->
  "openai") en vez de declararse aparte. Ojo con la diferencia respecto de la
  referencia session_3, que adivina el provider del NOMBRE del modelo con
  `startswith("claude")`. Acá no se adivina nada: se parsea configuración que
  el operador escribió explícitamente. Qué deployment *respondió* no se
  infiere del nombre: el wrapper lo lee de
  `response._hidden_params["model_id"]`.

- `PROMPT_VERSION` es obligatoria y no decorativa: elige la plantilla
  (prompts/estimation/vN/) y viaja en cada respuesta y en cada evento de
  trazabilidad. Hoy NO invalida la caché, ni hace falta: la caché es
  exact-match y su clave es un hash del prompt completo ya renderizado, así
  que cualquier cambio de plantilla produce otra clave. Con el caché semántico
  de WU10 la clave embeberá solo la consulta del usuario, y entonces esta
  variable pasará a ser el mecanismo de invalidación: sin ella, cambiar la
  plantilla seguiría sirviendo respuestas viejas en silencio.
"""

from functools import lru_cache
from typing import Annotated, Literal

from pydantic import BeforeValidator, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

VARIABLE_DE_API_KEY = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}

# `log_level` es el único campo con dominio cerrado que además decide algo:
# main.configure_logging lo pasa a logging.setLevel, así que un valor inválido
# muere al arrancar. `Literal` no cambia ese comportamiento —la stdlib ya
# rechazaba con `ValueError: Unknown level`— pero mueve el error al lado de
# Pydantic, que nombra el campo y lista los valores válidos en vez de dejar
# `Unknown level: 'info'` a secas.
#
# El `BeforeValidator` no es decorativo: sin él, un `LOG_LEVEL=` vacío rompería
# el arranque, porque `Literal` rechaza `""` antes de que `aplicar_defaults`
# pueda convertirlo en el default. Eso rompería la regla heredada de `lidr_3`
# —cadena vacía = "no configurado"— que el punto 2 del docstring declara como
# principio. Se normaliza antes de validar el dominio: vacío y sin espacios van
# al default, y cualquier otra cosa se pasa a `Literal` sin tocar, para que un
# `info` en minúscula siga siendo un error en vez de un(DEFAULT silencioso.
#
# `app_env` sigue siendo `str` a propósito: hoy no ramifica en ningún lado, se
# solo muestra en /health. Tiparlo sería validar un valor del que nadie depende.
NIVELES_DE_LOG = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


def _normalizar_log_level(valor: str) -> str:
    """Vacío = "no configurado" = default, igual que el resto de Settings.

    Solo contempla el caso vacío. No hace `.upper()` a propósito: un `LOG_LEVEL=info`
    es un `.env` mal puesto, y silenciarlo a `INFO` esconde el error detrás de un
    default. Que muera al arrancar es el comportamiento que se busca."""
    return valor or "INFO"


# El contrato de `description` vive en el Field de
# app.schemas.estimation.EstimationRequest. Se replica acá porqueSettings no
# puede importar al schema (sería un ciclo: el schema importa get_settings) y
# el validador de abajo necesita compararse contra estos números. Si se cambia
# uno hay que cambiar el otro: el test de WU2 falla si se desincronizan.
CONTRATO_MIN_CHARS = 20
CONTRATO_MAX_CHARS = 2000


class LLMConfigurationError(Exception):
    """Falta configuración para usar el LLM (no es un fallo del proveedor).

    Diferente de un fallo de llamada: aquí el problema es local y se sabe
    exactamente qué falta. Nombra la variable en el mensaje."""


def provider_de_modelo(model: str) -> str:
    """'openai/gpt-4o-mini' -> 'openai'.

    Un modelo sin prefijo asume 'openai', que es el default de LiteLLM.
    Solo se usa para elegir contra qué API key construir el deployment, en
    tiempo de configuración. Nunca para inferir qué deployment respondió.
    """
    return model.split("/", 1)[0] if "/" in model else "openai"


class Settings(BaseSettings):
    """Configuración de la aplicación cargada desde variables de entorno
    y el archivo .env.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- API keys. Opcionales acá, exigidas en el punto de uso.
    openai_api_key: SecretStr | None = None
    anthropic_api_key: SecretStr | None = None

    # --- LiteLLM. Un deployment por modelo; el Router decide el failover.
    # Ver el docstring del módulo.
    primary_model: str = "openai/gpt-4o-mini"
    fallback_model: str = "anthropic/claude-haiku-4-5"

    llm_timeout: float = 30.0
    llm_max_retries: int = 2
    # 4000 y no los 2000 de lidr_3: la salida estructurada (resumen + tareas
    # + equipo + supuestos + riesgos) es bastante más larga que el markdown
    # equivalente, y un max_tokens corto produce JSON truncado.
    llm_max_tokens: int = 4000

    # --- Capa 2. Qué plantilla se usa. Ver el docstring sobre su relación
    # con la caché.
    prompt_version: str = "v1"

    # --- Caché. REDIS_URL vacío = caché desactivado, sin tocar código.
    redis_url: str = "redis://localhost:6379/0"
    cache_ttl: int = 86400

    # --- Techo del operador sobre `description`. El contrato con el cliente
    # (20/2000) vive en el Field de app.schemas.estimation.EstimationRequest y
    # no se toca acá: estos dos valores solo pueden RESTRINGIR ese rango, nunca
    # ampliarlo, porque el `max` es lo que protege el coste (el tamaño de la
    # entrada es la factura) y un techo que necesita redeploy es un techo blando.
    #
    # El default es el contrato mismo (20/2000), no un valor más permisivo: si
    # fuera 50/5000, el `min_length=20` del schema nunca se cumpliría y el
    # operador ni se enteraría. Configurar esto más estricto es deliberado y
    # `validar_techo_descripcion` lo acepta; configurarlo más laxo es un error
    # de configuración y muere al arrancar.
    descripcion_min_chars: int = 20
    descripcion_max_chars: int = 2000

    # --- Frontend. URL base del API para Streamlit (cliente HTTP separado).
    estimator_api_base_url: str = "http://localhost:8001"

    # --- Servicio.
    app_env: str = "local"
    log_level: Annotated[NIVELES_DE_LOG, BeforeValidator(_normalizar_log_level)] = "INFO"
    app_port: int = 8001

    @model_validator(mode="after")
    def aplicar_defaults(self) -> "Settings":
        """Normaliza lo opcional y falla rápido ante lo inválido.

        Se conserva la regla de lidr_3: cadena vacía = "no configurado" =
        se usa el default. Un valor presente pero inválido es un error de
        configuración y muere acá, no a mitad de una request.
        """
        self.app_env = self.app_env or "local"
        self.primary_model = self.primary_model or "openai/gpt-4o-mini"
        self.fallback_model = self.fallback_model or "anthropic/claude-haiku-4-5"
        # pydantic-settings parsea un `PROMPT_VERSION=` vacío como "", y "" no
        # es un nombre de archivo válido: un .env a medio completar rompería el
        # arranque en el peor momento. Acá se valida el dominio a mano porque
        # `log_level` sí es un Literal y no necesita esto: su campo normaliza
        # el vacío en un BeforeValidator, que corre antes de validar el dominio.
        if not self.prompt_version:
            raise ValueError("PROMPT_VERSION no puede estar vacío")
        return self

    @model_validator(mode="after")
    def validar_modelos(self) -> "Settings":
        """Los dos deployments del Router tienen que ser distinguibles.

        Si PRIMARY_MODEL == FALLBACK_MODEL, el "fallback" es un no-op que
        igual paga una llamada extra y una latencia. Es un error de
        configuración, no un caso degenerado aceptable.
        """
        if self.primary_model == self.fallback_model:
            raise ValueError(
                f"PRIMARY_MODEL y FALLBACK_MODEL son iguales ('{self.primary_model}'). "
                "El fallback no podría nunca atender una request."
            )
        return self

    @model_validator(mode="after")
    def validar_techo_descripcion(self) -> "Settings":
        """Settings solo puede hacer el contrato MÁS ESTRICTO, nunca más laxo.

        La intersección de los dos límites es la que se aplica, y el `Field`
        (20/2000) siempre está: no se puede relajar. Entonces hay dos
        direcciones, y solo una rompe la documentación:

        - Settings más estricto (min >= 20, max <= 2000): manda Settings, y el
          parche escribe ese mismo número. Swagger coincide con el servicio.
          Acá es donde vive la reducción de coste sin redeploy.
        - Settings más laxo (min < 20, max > 2000): el `Field` sigue cortando,
          pero el parche escribe el número laxo, así que Swagger promete un
          rango que el servicio no acepta. Es la misma mentira que tenía WU1,
          solo que con un default que la dispara.

        Los dos casos laxos son un `.env` mal puesto y mueren acá, al arrancar.
        Si se dejaran pasar, el síntoma aparecería como un 422 inexplicable
        en producción, que es el peor lugar y el peor momento."""
        if self.descripcion_min_chars < CONTRATO_MIN_CHARS:
            raise ValueError(
                f"DESCRIPCION_MIN_CHARS={self.descripcion_min_chars} es menor que el "
                f"mínimo del contrato ({CONTRATO_MIN_CHARS}). Este valor solo puede "
                "estrechar el rango: el `min_length` del schema seguiría rechazando "
                "texto más corto y Swagger documentaría un mínimo que el servicio "
                "no acepta. Para subir el mínimo no hay que tocar esto."
            )
        if self.descripcion_max_chars > CONTRATO_MAX_CHARS:
            raise ValueError(
                f"DESCRIPCION_MAX_CHARS={self.descripcion_max_chars} es mayor que el "
                f"máximo del contrato ({CONTRATO_MAX_CHARS}). Este valor solo puede "
                "estrechar el rango: el `max_length` del schema seguiría rechazando "
                "texto más largo y Swagger documentaría un máximo que el servicio "
                f"no acepta. Para BAJAR el techo de coste, ponelo en {CONTRATO_MAX_CHARS} o menos."
            )
        if self.descripcion_min_chars > self.descripcion_max_chars:
            raise ValueError(
                f"DESCRIPCION_MIN_CHARS={self.descripcion_min_chars} es mayor que "
                f"DESCRIPCION_MAX_CHARS={self.descripcion_max_chars}: ningún texto "
                "sería aceptado."
            )
        return self

    @property
    def primary_provider(self) -> str:
        """Provider del deployment primario, parseado de PRIMARY_MODEL."""
        return provider_de_modelo(self.primary_model)

    @property
    def fallback_provider(self) -> str:
        """Provider del deployment de respaldo, parseado de FALLBACK_MODEL."""
        return provider_de_modelo(self.fallback_model)

    @property
    def is_configured(self) -> bool:
        """True si hay API key para el deployment primario.

        Reporta el primario y no ambos a propósito: /health debe describir
        si el camino de una request normal funciona, no si están todos los
        cables conectados. El fallback se valida en su propio momento, cuando
        se usa.
        """
        return self.active_api_key(self.primary_provider) is not None

    def active_api_key(self, provider: str | None = None) -> str | None:
        """La key del provider indicado (SecretStr -> str), sin lanzar.

        provider=None usa el primario. Devuelve None si falta la key.
        """
        provider = provider or self.primary_provider
        if provider not in VARIABLE_DE_API_KEY:
            return None
        raw = getattr(self, f"{provider}_api_key", None)
        return raw.get_secret_value() if raw else None


@lru_cache
def get_settings() -> Settings:
    return Settings()
