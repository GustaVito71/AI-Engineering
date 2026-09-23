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


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openai_api_key: SecretStr | None = None
    anthropic_api_key: SecretStr | None = None

    llm_provider: Literal["openai", "anthropic"] = "openai"
    llm_model: str | None = None
    openai_model: str = "gpt-4o-mini"
    anthropic_model: str = "claude-haiku-4-5"

    llm_timeout: float = 30.0
    llm_max_retries: int = 2
    llm_max_tokens: int = 2000

    estimation_min_chars: int = 50
    estimation_max_chars: int = 50_000

    app_env: str = "local"
    log_level: str = "INFO"
    app_port: int = 8001

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
        # pydantic-settings parsea un `LLM_MODEL=` vacío como "" y no como
        # None: el "no configurado" se detecta con falsy, no con `is None`.
        if not self.llm_model:
            self.llm_model = (
                self.openai_model
                if self.llm_provider == "openai"
                else self.anthropic_model
            )
        return self

    @property
    def is_configured(self) -> bool:
        """True si el proveedor activo tiene su API key en el entorno."""
        return self.active_api_key() is not None

    def active_api_key(self) -> str | None:
        """La key del proveedor activo (SecretStr -> str), sin lanzar.

        Devuelve None si falta: eso es lo que /health necesita poder contar.
        """
        raw = (
            self.openai_api_key
            if self.llm_provider == "openai"
            else self.anthropic_api_key
        )
        return raw.get_secret_value() if raw else None

    def require_api_key(self) -> str:
        """La key del proveedor activo, exigida en el punto de uso.

        Se llama justo antes de construir el cliente LLM, nunca al importar.
        """
        key_var = "OPENAI_API_KEY" if self.llm_provider == "openai" else "ANTHROPIC_API_KEY"
        key = self.active_api_key()
        if not key:
            raise LLMConfigurationError(f"Falta {key_var} para el proveedor '{self.llm_provider}'")
        return key


@lru_cache
def get_settings() -> Settings:
    return Settings()