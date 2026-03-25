"""Project Aegis — Configuration system.

Loads settings from config.yaml with environment variable overrides.
Uses Pydantic Settings for strict validation.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


# --- Nested config models ---


class AppConfig(BaseModel):
    name: str = "Project Aegis"
    version: str = "0.1.0"
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 8000


class DatabaseConfig(BaseModel):
    url: str = "sqlite:///data/aegis.db"
    echo: bool = False


class OllamaConfig(BaseModel):
    base_url: str = "http://localhost:11434"
    orchestrator_model: str = "qwen3:1.7b"
    embedding_model: str = "nomic-embed-text"
    timeout: int = 120


class QdrantConfig(BaseModel):
    host: str = "localhost"
    port: int = 6333
    grpc_port: int = 6334


class NIMProviderConfig(BaseModel):
    api_keys: list[str] = Field(default_factory=list)
    base_url: str = "https://integrate.api.nvidia.com/v1"
    default_model: str = "meta/llama-3.1-70b-instruct"
    timeout: int = 60


class DeepSeekProviderConfig(BaseModel):
    api_keys: list[str] = Field(default_factory=list)
    base_url: str = "https://api.deepseek.com"
    reasoning_model: str = "deepseek-reasoner"
    general_model: str = "deepseek-chat"
    timeout: int = 90


class SimpleProviderConfig(BaseModel):
    api_key: str = ""
    default_model: str = ""
    timeout: int = 30


class ProvidersConfig(BaseModel):
    nim: NIMProviderConfig = Field(default_factory=NIMProviderConfig)
    deepseek: DeepSeekProviderConfig = Field(default_factory=DeepSeekProviderConfig)
    mistral: SimpleProviderConfig = Field(
        default_factory=lambda: SimpleProviderConfig(default_model="mistral-small-latest")
    )
    groq: SimpleProviderConfig = Field(
        default_factory=lambda: SimpleProviderConfig(default_model="llama-3.1-70b-versatile")
    )
    gemini: SimpleProviderConfig = Field(
        default_factory=lambda: SimpleProviderConfig(default_model="gemini-2.0-flash")
    )


class SearchSerperConfig(BaseModel):
    api_key: str = ""
    max_results: int = 5


class SearchConfig(BaseModel):
    serper: SearchSerperConfig = Field(default_factory=SearchSerperConfig)


class RateLimitTier(BaseModel):
    messages_per_day: int = 0
    messages_per_hour: int = 0
    messages_burst_5min: int = 0
    model_calls_per_day: int = 0
    model_calls_per_hour: int = 0
    messages_per_day_per_ip: int = 0
    messages_per_session: int = 0
    model_calls_per_day_per_ip: int = 0
    model_calls_per_session: int = 0


class RateLimitsConfig(BaseModel):
    owner: RateLimitTier = Field(default_factory=RateLimitTier)
    named_guest: RateLimitTier = Field(default_factory=RateLimitTier)
    anonymous_guest: RateLimitTier = Field(default_factory=RateLimitTier)


class LoggingConfig(BaseModel):
    level: str = "INFO"
    log_dir: str = "data/logs"
    json_format: bool = True
    correlation_ids: bool = True


class BackupConfig(BaseModel):
    enabled: bool = True
    interval_minutes: int = 30
    qdrant_snapshot_timeout: int = 30
    write_queue_max: int = 100


# --- Main Settings ---


class Settings(BaseSettings):
    """Main application settings. Loads from config.yaml + environment variables."""

    app: AppConfig = Field(default_factory=AppConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    qdrant: QdrantConfig = Field(default_factory=QdrantConfig)
    providers: ProvidersConfig = Field(default_factory=ProvidersConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    rate_limits: RateLimitsConfig = Field(default_factory=RateLimitsConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    backup: BackupConfig = Field(default_factory=BackupConfig)

    model_config = {"env_prefix": "AEGIS_", "env_nested_delimiter": "__"}


def _resolve_env_vars(data: Any) -> Any:
    """Recursively resolve ${VAR_NAME} patterns in config values."""
    if isinstance(data, str) and "${" in data:
        import re

        def _replace(match: re.Match[str]) -> str:
            var_name = match.group(1)
            return os.environ.get(var_name, "")

        return re.sub(r"\$\{(\w+)\}", _replace, data)
    if isinstance(data, dict):
        return {k: _resolve_env_vars(v) for k, v in data.items()}
    if isinstance(data, list):
        return [_resolve_env_vars(item) for item in data]
    return data


def _collect_provider_keys() -> dict[str, Any]:
    """Collect multi-account API keys from environment variables.

    Only includes keys that are actually set in the environment.
    NIM_API_KEY_1..4 → providers.nim.api_keys
    DEEPSEEK_API_KEY_1..4 → providers.deepseek.api_keys
    """
    overrides: dict[str, Any] = {}

    nim_keys = [
        os.environ[f"NIM_API_KEY_{i}"]
        for i in range(1, 5)
        if f"NIM_API_KEY_{i}" in os.environ
    ]
    if nim_keys:
        overrides.setdefault("providers", {}).setdefault("nim", {})["api_keys"] = nim_keys

    deepseek_keys = [
        os.environ[f"DEEPSEEK_API_KEY_{i}"]
        for i in range(1, 5)
        if f"DEEPSEEK_API_KEY_{i}" in os.environ
    ]
    if deepseek_keys:
        overrides.setdefault("providers", {}).setdefault("deepseek", {})["api_keys"] = deepseek_keys

    # Single-key providers — only override if env var is set and non-empty
    for var, path in [
        ("MISTRAL_API_KEY", ("providers", "mistral", "api_key")),
        ("GROQ_API_KEY", ("providers", "groq", "api_key")),
        ("GEMINI_API_KEY", ("providers", "gemini", "api_key")),
        ("SERPER_API_KEY", ("search", "serper", "api_key")),
    ]:
        val = os.environ.get(var, "")
        if val:
            d = overrides
            for key in path[:-1]:
                d = d.setdefault(key, {})
            d[path[-1]] = val

    return overrides


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep merge override dict into base dict.

    Explicit falsey values (False, 0, empty string) ARE applied as overrides.
    Only missing keys (not present in override) are left untouched.
    """
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_settings(config_path: str | Path | None = None) -> Settings:
    """Load settings from config.yaml, env vars, and provider keys.

    Priority (highest to lowest):
    1. Environment variables (AEGIS__* prefix)
    2. Provider-specific env vars (NIM_API_KEY_1, MISTRAL_API_KEY, etc.)
    3. config.yaml values
    4. Pydantic defaults
    """
    # Find config.yaml
    if config_path is None:
        # Look relative to this file's location
        config_path = Path(__file__).parent / "config.yaml"

    yaml_data: dict[str, Any] = {}
    config_file = Path(config_path)
    if config_file.exists():
        with open(config_file) as f:
            raw = yaml.safe_load(f)
            if raw:
                yaml_data = _resolve_env_vars(raw)

    # Collect provider keys from environment
    provider_overrides = _collect_provider_keys()

    # Merge: yaml base + provider key overrides
    merged = _deep_merge(yaml_data, provider_overrides)

    return Settings(**merged)


# Singleton — import this in other modules
settings = load_settings()
