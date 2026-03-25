"""Tests for the configuration system.

Sprint 1.1 Task 2: Verify config loading, validation, and env var interpolation.
"""

from __future__ import annotations

import os
from pathlib import Path

from aegis.config import Settings, load_settings, _resolve_env_vars, _deep_merge


class TestConfigDefaults:
    """Test that default configuration values are sensible."""

    def test_default_app_name(self) -> None:
        s = Settings()
        assert s.app.name == "Project Aegis"

    def test_default_app_port(self) -> None:
        s = Settings()
        assert s.app.port == 8000

    def test_default_ollama_model(self) -> None:
        s = Settings()
        assert s.ollama.orchestrator_model == "qwen3:1.7b"

    def test_default_qdrant_port(self) -> None:
        s = Settings()
        assert s.qdrant.port == 6333

    def test_default_rate_limits_owner(self) -> None:
        s = Settings()
        assert s.rate_limits.owner.messages_per_day == 0  # Defaults to 0 without yaml


class TestEnvVarResolution:
    """Test ${VAR} interpolation in config values."""

    def test_resolve_single_var(self) -> None:
        os.environ["TEST_VAR_123"] = "hello"
        result = _resolve_env_vars("prefix_${TEST_VAR_123}_suffix")
        assert result == "prefix_hello_suffix"
        del os.environ["TEST_VAR_123"]

    def test_resolve_missing_var(self) -> None:
        result = _resolve_env_vars("${NONEXISTENT_VAR_XYZ}")
        assert result == ""

    def test_resolve_nested_dict(self) -> None:
        os.environ["TEST_NESTED"] = "value"
        data = {"key": "${TEST_NESTED}", "nested": {"inner": "${TEST_NESTED}"}}
        result = _resolve_env_vars(data)
        assert result == {"key": "value", "nested": {"inner": "value"}}
        del os.environ["TEST_NESTED"]

    def test_resolve_list(self) -> None:
        os.environ["TEST_LIST"] = "item"
        data = ["${TEST_LIST}", "static"]
        result = _resolve_env_vars(data)
        assert result == ["item", "static"]
        del os.environ["TEST_LIST"]

    def test_no_interpolation_passthrough(self) -> None:
        assert _resolve_env_vars("plain string") == "plain string"
        assert _resolve_env_vars(42) == 42
        assert _resolve_env_vars(None) is None


class TestDeepMerge:
    """Test deep merge utility."""

    def test_simple_merge(self) -> None:
        base = {"a": 1, "b": 2}
        override = {"b": 3, "c": 4}
        result = _deep_merge(base, override)
        assert result == {"a": 1, "b": 3, "c": 4}

    def test_nested_merge(self) -> None:
        base = {"a": {"x": 1, "y": 2}}
        override = {"a": {"y": 3, "z": 4}}
        result = _deep_merge(base, override)
        assert result == {"a": {"x": 1, "y": 3, "z": 4}}

    def test_empty_override(self) -> None:
        base = {"a": 1}
        result = _deep_merge(base, {})
        assert result == {"a": 1}

    def test_falsey_override_applied(self) -> None:
        """Codex finding #1: explicit False/0/empty-string must override base values."""
        base = {"debug": True, "count": 5, "name": "default"}
        override = {"debug": False, "count": 0, "name": ""}
        result = _deep_merge(base, override)
        assert result["debug"] is False
        assert result["count"] == 0
        assert result["name"] == ""


class TestLoadSettings:
    """Test loading settings from config.yaml file."""

    def test_load_from_yaml(self) -> None:
        config_path = Path(__file__).parent.parent / "aegis" / "config.yaml"
        if config_path.exists():
            s = load_settings(config_path)
            assert s.app.name == "Project Aegis"
            assert s.ollama.orchestrator_model == "qwen3:1.7b"
            assert s.rate_limits.owner.messages_per_day == 600

    def test_load_nonexistent_yaml(self, tmp_path: Path) -> None:
        """Should fall back to defaults if yaml doesn't exist."""
        s = load_settings(tmp_path / "nonexistent.yaml")
        assert s.app.name == "Project Aegis"


class TestProviderKeyCollection:
    """Test that provider API keys are collected from environment."""

    def test_nim_keys_from_env(self) -> None:
        os.environ["NIM_API_KEY_1"] = "key1"
        os.environ["NIM_API_KEY_2"] = "key2"
        try:
            s = load_settings()
            assert "key1" in s.providers.nim.api_keys
            assert "key2" in s.providers.nim.api_keys
        finally:
            del os.environ["NIM_API_KEY_1"]
            del os.environ["NIM_API_KEY_2"]

    def test_single_provider_key(self) -> None:
        os.environ["MISTRAL_API_KEY"] = "mist_key"
        try:
            s = load_settings()
            assert s.providers.mistral.api_key == "mist_key"
        finally:
            del os.environ["MISTRAL_API_KEY"]
