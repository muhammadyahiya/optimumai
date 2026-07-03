"""Runtime configuration via environment variables (``OPTIMUMAI_*``).

Follows the workspace convention of ``pydantic-settings`` + ``.env``. Nothing
here is required for the core math — every field has a sensible default and the
LLM fields are only consulted by the (optional) tutor layer.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict

from optimumai.core.explain import ExplainLevel


class Settings(BaseSettings):
    """Central settings object, populated from the environment or a ``.env`` file."""

    model_config = SettingsConfigDict(
        env_prefix="OPTIMUMAI_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Presentation defaults
    explain_level: ExplainLevel = ExplainLevel.INTERMEDIATE
    color: bool = True

    # LLM tutor (optional; used only when the `[llm]` extra is installed)
    llm_model: str = "gpt-4o-mini"
    api_key: str | None = None
    api_base: str | None = None
    temperature: float = 0.3


@lru_cache
def get_settings() -> Settings:
    """Return a cached :class:`Settings` instance."""
    return Settings()
