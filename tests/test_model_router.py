"""Tests for the model router and account pool.

Sprint 1.2: Validates provider routing, fallback chains, circuit breaker,
and multi-account pool management.
"""

from __future__ import annotations

import pytest

from unittest.mock import patch

from aegis.core.intent import ComplexityTier
from aegis.power.model_router import ModelRouter, ProviderInfo, ProviderStatus
from aegis.power.account_pool import AccountPool, AccountState, _pools, get_pool


class TestProviderInfo:
    """Test individual provider state management."""

    def test_available_with_key(self) -> None:
        p = ProviderInfo(name="mistral", model="mistral-small", api_key="test-key")
        assert p.is_available()

    def test_unavailable_without_key(self) -> None:
        p = ProviderInfo(name="mistral", model="mistral-small", api_key="")
        assert not p.is_available()

    def test_ollama_available_without_key(self) -> None:
        p = ProviderInfo(name="ollama", model="qwen3:1.7b")
        assert p.is_available()

    def test_mark_error_circuit_breaker(self) -> None:
        p = ProviderInfo(name="test", model="test", api_key="key")
        p.mark_error("timeout")
        p.mark_error("timeout")
        assert p.status == ProviderStatus.HEALTHY
        p.mark_error("timeout")  # 3rd error → unavailable
        assert p.status == ProviderStatus.UNAVAILABLE
        assert not p.is_available()

    def test_mark_success_resets_errors(self) -> None:
        p = ProviderInfo(name="test", model="test", api_key="key")
        p.mark_error("timeout")
        p.mark_error("timeout")
        p.mark_success()  # Resets consecutive errors
        assert p.error_count == 0
        assert p.status == ProviderStatus.HEALTHY

    def test_reset(self) -> None:
        p = ProviderInfo(name="test", model="test", api_key="key")
        p.mark_error("timeout")
        p.mark_error("timeout")
        p.mark_error("timeout")
        assert p.status == ProviderStatus.UNAVAILABLE
        p.reset()
        assert p.status == ProviderStatus.HEALTHY
        assert p.is_available()


class TestModelRouter:
    """Test complexity-based routing with fallback chains."""

    def test_simple_routes_to_first_available(self) -> None:
        router = ModelRouter()
        result = router.route(ComplexityTier.SIMPLE)
        # Without API keys, should fall back through chain to ollama
        assert result.provider.name in ("mistral", "groq", "gemini", "ollama")

    def test_medium_routing(self) -> None:
        router = ModelRouter()
        result = router.route(ComplexityTier.MEDIUM)
        assert result.provider.name in ("nim", "groq", "gemini", "ollama")

    def test_complex_routing(self) -> None:
        router = ModelRouter()
        result = router.route(ComplexityTier.COMPLEX)
        assert result.provider.name in ("deepseek_r1", "nim", "gemini", "ollama")

    def test_heavy_routing(self) -> None:
        router = ModelRouter()
        result = router.route(ComplexityTier.HEAVY)
        assert result.provider.name in ("deepseek_v3", "deepseek_r1", "gemini", "ollama")

    def test_ollama_always_available(self) -> None:
        """Ollama should always be the final fallback."""
        router = ModelRouter()
        # Ollama doesn't need an API key
        provider = router.get_provider("ollama")
        assert provider is not None
        assert provider.is_available()

    def test_fallback_detection(self) -> None:
        """When primary is unavailable, result should indicate fallback."""
        router = ModelRouter()
        result = router.route(ComplexityTier.SIMPLE)
        # Without any API keys configured, should fall back
        if result.provider.name != "mistral":
            assert result.is_fallback

    def test_get_status(self) -> None:
        router = ModelRouter()
        status = router.get_status()
        assert "ollama" in status
        assert "mistral" in status
        assert status["ollama"]["has_key"] is True  # Ollama always True


class TestAccountPool:
    """Test multi-account round-robin pool."""

    def test_creation(self) -> None:
        pool = AccountPool("nim", ["key1", "key2", "key3"])
        assert pool.size == 3
        assert pool.available_count == 3
        assert not pool.is_exhausted

    def test_empty_keys_filtered(self) -> None:
        pool = AccountPool("nim", ["key1", "", "key3", ""])
        assert pool.size == 2

    def test_round_robin(self) -> None:
        pool = AccountPool("nim", ["key1", "key2", "key3"])
        accounts = [pool.next_account() for _ in range(6)]
        keys = [a.key for a in accounts if a]
        # Should cycle: key1, key2, key3, key1, key2, key3
        assert keys == ["key1", "key2", "key3", "key1", "key2", "key3"]

    def test_skip_exhausted(self) -> None:
        pool = AccountPool("nim", ["key1", "key2", "key3"])
        # Exhaust key2
        pool._accounts[1].mark_exhausted()
        accounts = [pool.next_account() for _ in range(4)]
        keys = [a.key for a in accounts if a]
        assert "key2" not in keys

    def test_all_exhausted(self) -> None:
        pool = AccountPool("nim", ["key1", "key2"])
        pool._accounts[0].mark_exhausted()
        pool._accounts[1].mark_exhausted()
        assert pool.is_exhausted
        assert pool.next_account() is None

    def test_record_usage(self) -> None:
        pool = AccountPool("nim", ["key1"])
        pool.record_usage(0, tokens=100)
        assert pool._accounts[0].request_count == 1
        assert pool._accounts[0].token_count == 100

    def test_record_error_exhausts(self) -> None:
        pool = AccountPool("nim", ["key1"])
        pool.record_error(0, "quota exceeded")
        pool.record_error(0, "quota exceeded")
        pool.record_error(0, "quota exceeded")
        assert pool._accounts[0].is_exhausted

    def test_reset_all(self) -> None:
        pool = AccountPool("nim", ["key1", "key2"])
        pool._accounts[0].mark_exhausted()
        pool._accounts[1].mark_exhausted()
        assert pool.is_exhausted
        pool.reset_all()
        assert not pool.is_exhausted
        assert pool.available_count == 2

    def test_get_status(self) -> None:
        pool = AccountPool("nim", ["key1", "key2"])
        status = pool.get_status()
        assert status["provider"] == "nim"
        assert status["total_accounts"] == 2
        assert len(status["accounts"]) == 2

    def test_masked_key(self) -> None:
        state = AccountState(key="sk-1234567890abcdef", index=0)
        assert "1234567890" not in state.masked_key
        assert state.masked_key.startswith("sk-1")
        assert state.masked_key.endswith("cdef")

    def test_short_key_masked(self) -> None:
        state = AccountState(key="short", index=0)
        assert state.masked_key == "***"


class TestAccountState:
    """Test individual account state tracking."""

    def test_record_usage_resets_errors(self) -> None:
        state = AccountState(key="key1", index=0)
        state.record_error("timeout")
        state.record_error("timeout")
        assert state.error_count == 2
        state.record_usage(tokens=50)
        assert state.error_count == 0

    def test_exhaustion_after_3_errors(self) -> None:
        state = AccountState(key="key1", index=0)
        state.record_error("e1")
        state.record_error("e2")
        assert not state.is_exhausted
        state.record_error("e3")
        assert state.is_exhausted

    def test_reset(self) -> None:
        state = AccountState(key="key1", index=0)
        state.record_error("e1")
        state.record_error("e2")
        state.record_error("e3")
        assert state.is_exhausted
        state.reset()
        assert not state.is_exhausted
        assert state.error_count == 0


class TestPoolIntegrationWithRouter:
    """Test that model router uses account pools for NIM/DeepSeek (Codex fix #3)."""

    def _make_router_with_nim_keys(self) -> ModelRouter:
        """Create a router with NIM keys injected via pool."""
        router = ModelRouter()
        # Manually inject a pool and mark provider as pool-managed
        pool = AccountPool("nim", ["nim-key-1", "nim-key-2", "nim-key-3"])
        _pools["nim"] = pool
        router._providers["nim"].api_key = "pool-managed"
        return router

    def teardown_method(self) -> None:
        """Clean up global pool registry."""
        _pools.clear()

    def test_nim_route_selects_from_pool(self) -> None:
        router = self._make_router_with_nim_keys()
        result = router.route(ComplexityTier.MEDIUM)
        # NIM is first in medium chain and pool has keys
        assert result.provider.name == "nim"
        assert result.account_index is not None
        assert result.provider.api_key in ("nim-key-1", "nim-key-2", "nim-key-3")

    def test_nim_round_robin_across_calls(self) -> None:
        router = self._make_router_with_nim_keys()
        indices = []
        for _ in range(6):
            result = router.route(ComplexityTier.MEDIUM)
            assert result.provider.name == "nim"
            indices.append(result.account_index)
        # Should cycle through 0,1,2,0,1,2
        assert indices == [0, 1, 2, 0, 1, 2]

    def test_exhausted_pool_falls_back(self) -> None:
        router = self._make_router_with_nim_keys()
        pool = _pools["nim"]
        # Exhaust all accounts
        for acc in pool._accounts:
            acc.mark_exhausted()
        result = router.route(ComplexityTier.MEDIUM)
        # Should fall back past NIM to next in chain
        assert result.provider.name != "nim"
        assert result.is_fallback

    def test_deepseek_route_uses_pool(self) -> None:
        router = ModelRouter()
        pool = AccountPool("deepseek", ["ds-key-1", "ds-key-2"])
        _pools["deepseek"] = pool
        router._providers["deepseek_r1"].api_key = "pool-managed"
        result = router.route(ComplexityTier.COMPLEX)
        # DeepSeek-R1 is first in complex chain
        assert result.provider.name == "deepseek_r1"
        assert result.account_index is not None
        assert result.provider.api_key in ("ds-key-1", "ds-key-2")
