"""Application settings for the Geo Strategist scaffold."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

from dotenv import dotenv_values
from pydantic import BaseModel, ConfigDict, Field, field_validator


ENV_KEYS = {
    "GEMINI_API_KEY",
    "GEMINI_API_KEY2",
    "GEMINI_API_KEY3",
    "GOOGLE_API_KEY",
    "OPENCODE_API_KEY",
    "YAHOO_CLIENT_ID",
    "ESTAT_APP_ID",
    "REINFOLIB_API_KEY",
    "OPENROUTER_API_KEY",
    "OPENROUTER_BASE_URL",
    "GEO_STRATEGIST_CACHE_DIR",
    "GEO_STRATEGIST_DATA_DIR",
    "GEO_STRATEGIST_RUNS_DIR",
    "GEO_STRATEGIST_LOG_LEVEL",
    "DEFAULT_JUDGE_MODEL",
    "DEFAULT_OPENROUTER_MODEL",
}

LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}


class AppSettings(BaseModel):
    """Typed settings loaded from explicit dotenv files and process env."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    gemini_api_key: str = Field(default="", alias="GEMINI_API_KEY")
    gemini_api_key2: str = Field(default="", alias="GEMINI_API_KEY2")
    gemini_api_key3: str = Field(default="", alias="GEMINI_API_KEY3")
    google_api_key: str = Field(default="", alias="GOOGLE_API_KEY")
    opencode_api_key: str = Field(default="", alias="OPENCODE_API_KEY")
    yahoo_client_id: str = Field(default="", alias="YAHOO_CLIENT_ID")
    estat_app_id: str = Field(default="", alias="ESTAT_APP_ID")
    reinfolib_api_key: str = Field(default="", alias="REINFOLIB_API_KEY")
    openrouter_api_key: str = Field(default="", alias="OPENROUTER_API_KEY")
    openrouter_base_url: str = Field(
        default="https://openrouter.ai/api/v1",
        alias="OPENROUTER_BASE_URL",
    )
    cache_dir: Path = Field(default=Path(".cache"), alias="GEO_STRATEGIST_CACHE_DIR")
    data_dir: Path = Field(default=Path(".data"), alias="GEO_STRATEGIST_DATA_DIR")
    runs_dir: Path = Field(default=Path(".runs"), alias="GEO_STRATEGIST_RUNS_DIR")
    log_level: str = Field(default="INFO", alias="GEO_STRATEGIST_LOG_LEVEL")
    default_judge_model: str = Field(default="gemini", alias="DEFAULT_JUDGE_MODEL")
    default_openrouter_model: str = Field(default="", alias="DEFAULT_OPENROUTER_MODEL")

    @field_validator("cache_dir", "data_dir", "runs_dir", mode="before")
    @classmethod
    def _coerce_path(cls, value: object) -> Path:
        return value if isinstance(value, Path) else Path(str(value))

    @field_validator("log_level")
    @classmethod
    def _normalise_log_level(cls, value: str) -> str:
        level = value.upper()
        if level not in LOG_LEVELS:
            msg = f"log level must be one of {sorted(LOG_LEVELS)}"
            raise ValueError(msg)
        return level


def _clean_env_mapping(values: Mapping[str, str | None]) -> dict[str, str]:
    return {key: value for key, value in values.items() if key in ENV_KEYS and value is not None}


def load_settings(
    env_file: str | Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> AppSettings:
    """Load settings without implicitly reading a local `.env` file.

    Pass `env_file=Path(".env")` from an application entry point when reading a
    real local environment file is intentional. Process environment values take
    precedence over dotenv values.
    """

    raw_values: dict[str, str] = {}

    if env_file is not None:
        path = Path(env_file)
        if path.exists():
            raw_values.update(_clean_env_mapping(dotenv_values(path)))

    env_values = os.environ if environ is None else environ
    raw_values.update(_clean_env_mapping(env_values))

    return AppSettings.model_validate(raw_values)
