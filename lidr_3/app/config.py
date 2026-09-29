"""Configuración de la aplicación (pydantic-settings, leída de .env/entorno).

Dos decisiones del diseño de este ejercicio, y por qué:

1. Las API keys son opcionales en la construcción (`SecretStr | None`) y se
   exigen en el punto de uso (`active_api_key()`). Si la validación ocurre al
   importar el módulo, el proceso muere ANTES de que exista la aplicación:
   no hay /health, no hay /docs, no hay nada que le diga al orquestador
   qué falta. Un health check tiene que sobrevivir a la avería que diagnostica.

2. El modelo se RESUELVE desde el proveedor, no se declara como campo
   independiente fijo. Si LLM_MODEL no viene, se usa el del proveedor activo
   (OPENAI_MODEL o ANTHROPIC_MODEL). La incoherencia "openai + claude-haiku-4-5"
   solo se evita si un campo sabe del otro. `Literal` además hace que un typo
   en .env falle al arrancar.
3. APP_ENV y LOG_LEVEL son opcionales: un "vacío" se normaliza a su default
   en `aplicar_defaults`, para que un .env copiado de .env.example no rompa.
   Un LOG_LEVEL inválido (no vacío) sí hace fallar el arranque: fail fast.
"""

from functools import lru_cache
from typing import Literal

from pydantic import SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMConfigurationError(Exception):
    """Falta configuración para usar el LLM (no es un fallo del proveedor).

    Diferente de un fallo de llamada: aquí el problema es local y se sabe
    exactamente qué falta. Nombra la variable en el mensaje."""


# Piso de `max_output_tokens` en la API de OpenAI (Responses). Anthropic
# acepta 1, así que este número no restringe al otro proveedor.
_MIN_MAX_TOKENS = 16


class Settings(BaseSettings):
    """Configuración de la aplicación cargada desde variables de entorno
    y el archivo .env.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- LIDR 2
    openai_api_key: SecretStr | None = None
    anthropic_api_key: SecretStr | None = None

    llm_provider: Literal["openai", "anthropic"] = "openai"
    llm_model: str | None = None

    openai_model: str = "gpt-4o-mini"
    anthropic_model: str = "claude-haiku-4-5"

    app_env: str = "local"
    log_level: str = "INFO"
    app_port: int = 8001

    # --- LIDR 3
    llm_timeout: float = 30.0
    llm_max_retries: int = 2
    llm_max_tokens: int = 2000
    # Proveedor alterno si el primario falla (rate limit, timeout, 5xx).
    # None = fallback desactivado. str (no Literal) para que un valor vacío
    # llegue al validator y se normalice a None (ver llm_model).
    llm_fallback: str | None = None

    # Cache Redis: URL de conexión y TTL en segundos (24h por defecto).
    redis_url: str = "redis://localhost:6379/0"
    cache_ttl: int = 86400

    estimation_min_chars: int = 50
    estimation_max_chars: int = 50_000

    @model_validator(mode="after")
    def aplicar_defaults(self) -> "Settings":
        """Resuelve el modelo y normaliza los campos opcionales.

        - Modelo: si LLM_MODEL no viene, el default es el del proveedor activo
          (OPENAI_MODEL o ANTHROPIC_MODEL). Así es imposible configurar
          "openai + un modelo de Anthropic" sin elegirlo expresamente.
        - APP_ENV / LOG_LEVEL / modelos vacíos significan "no configurados":
          se usa el default. Un `.env` copiado de `.env.example` (que trae
          APP_ENV= y LOG_LEVEL= vacíos) no debe romper nada.
        Un LOG_LEVEL inválido no vacío falla en logging.setLevel al arrancar.
        """
        self.app_env = self.app_env or "local"
        self.log_level = self.log_level or "INFO"
        self.openai_model = self.openai_model or "gpt-4o-mini"
        self.anthropic_model = self.anthropic_model or "claude-haiku-4-5"
        # La API de OpenAI rechaza `max_output_tokens` por debajo de 16 con un
        # 400 críptico ("integer below minimum value") que no dice qué variable
        # lo usó mal. Se valida al arrancar, como el resto del dominio, para
        # que un `.env` mal puesto falle con un mensaje que sí se entiende.
        # 16 es el piso de OpenAI; Anthropic acepta 1, así que el mismo número
        # no le hace mal a nadie.
        if self.llm_max_tokens < _MIN_MAX_TOKENS:
            raise ValueError(
                f"LLM_MAX_TOKENS={self.llm_max_tokens} está por debajo del mínimo "
                f"de {_MIN_MAX_TOKENS} que exige la API de OpenAI "
                "(max_output_tokens)"
            )
        # pydantic-settings parsea un `LLM_MODEL=` vacío como "" y no como
        # None: el "no configurado" se detecta con falsy, no con `is None`.
        if not self.llm_model:
            self.llm_model = (
                self.openai_model if self.llm_provider == "openai" else self.anthropic_model
            )
        # Un LLM_FALLBACK= vacío se normaliza a None (fallback desactivado).
        if not self.llm_fallback:
            self.llm_fallback = None
            return self
        # Como el campo es str (no Literal), validamos el dominio acá:
        # un typo se detecta al arrancar, igual que LLM_PROVIDER.
        if self.llm_fallback not in ("openai", "anthropic"):
            raise ValueError(
                f"LLM_FALLBACK ('{self.llm_fallback}') debe ser 'openai' o 'anthropic'"
            )
        # Fallback al mismo proveedor es un no-op con el costo de una llamada
        # extra: falla al arrancar (fail fast), como un LOG_LEVEL inválido.
        if self.llm_fallback == self.llm_provider:
            raise ValueError(
                f"LLM_FALLBACK ('{self.llm_fallback}') no puede ser igual a "
                f"LLM_PROVIDER ('{self.llm_provider}')"
            )
        return self

    @property
    def is_configured(self) -> bool:
        """True si el proveedor activo tiene su API key en el entorno."""
        return self.active_api_key() is not None

    def active_api_key(self, provider: str | None = None) -> str | None:
        """La key del proveedor indicado (SecretStr -> str), sin lanzar.

        provider=None usa el proveedor activo (llm_provider), lo que conserva
        el comportamiento de /health. Devuelve None si falta la key."""
        provider = provider or self.llm_provider
        raw = self.openai_api_key if provider == "openai" else self.anthropic_api_key
        return raw.get_secret_value() if raw else None

    def require_api_key(self, provider: str | None = None) -> str:
        """La key del proveedor indicado, exigida en el punto de uso.

        provider=None usa el proveedor activo. Se llama justo antes de construir
        el cliente LLM, nunca al importar (filosofía del módulo: el proceso
        sobrevive al problema que diagnostica)."""
        provider = provider or self.llm_provider
        key_var = "OPENAI_API_KEY" if provider == "openai" else "ANTHROPIC_API_KEY"
        key = self.active_api_key(provider)
        if not key:
            raise LLMConfigurationError(f"Falta {key_var} para el proveedor '{provider}'")
        return key

    def resolve_model(self, provider: str | None = None) -> str:
        """El modelo para un proveedor: LLM_MODEL (override) solo aplica al
        activo; un proveedor alterno usa su default (OPENAI_MODEL/ANTHROPIC_MODEL)."""
        provider = provider or self.llm_provider
        if provider == self.llm_provider and self.llm_model:
            return self.llm_model
        return self.openai_model if provider == "openai" else self.anthropic_model


@lru_cache
def get_settings() -> Settings:
    return Settings()
