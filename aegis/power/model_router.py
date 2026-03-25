"""Project Aegis — Model router.

Sprint 1.2: Routes requests to the appropriate provider tier based on
complexity classification (DR-010) with fallback chain support.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, UTC
from enum import Enum

import structlog

from aegis.config import settings
from aegis.core.intent import ComplexityTier
from aegis.power.account_pool import AccountPool, get_pool

logger = structlog.get_logger(__name__)


class ProviderStatus(str, Enum):
    """Provider health status."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"


@dataclass
class ProviderInfo:
    """Runtime info for a single provider."""

    name: str
    model: str
    api_key: str = ""
    base_url: str = ""
    timeout: int = 30
    status: ProviderStatus = ProviderStatus.HEALTHY
    last_error: str | None = None
    last_error_time: datetime | None = None
    request_count: int = 0
    error_count: int = 0

    def is_available(self) -> bool:
        """Check if provider is available for requests."""
        if self.status == ProviderStatus.UNAVAILABLE:
            return False
        if not self.api_key and self.name != "ollama":
            return False
        return True

    def mark_error(self, error: str) -> None:
        """Record a provider error."""
        self.error_count += 1
        self.last_error = error
        self.last_error_time = datetime.now(UTC)
        # Mark unavailable after 3 consecutive errors
        if self.error_count >= 3:
            self.status = ProviderStatus.UNAVAILABLE
            logger.warning(
                "provider.unavailable",
                provider=self.name,
                errors=self.error_count,
            )

    def mark_success(self) -> None:
        """Record a successful request."""
        self.request_count += 1
        self.error_count = 0
        self.status = ProviderStatus.HEALTHY

    def reset(self) -> None:
        """Reset error state (e.g., after health check passes)."""
        self.error_count = 0
        self.status = ProviderStatus.HEALTHY
        self.last_error = None


@dataclass
class RoutingResult:
    """Result of model routing decision."""

    provider: ProviderInfo
    complexity: ComplexityTier
    is_fallback: bool = False
    fallback_reason: str | None = None
    account_index: int | None = None  # Set when provider uses account pool


# --- Tier → Provider fallback chains (from orchestrator_spec.md §4.2) ---

TIER_FALLBACK_CHAINS: dict[ComplexityTier, list[str]] = {
    ComplexityTier.SIMPLE: ["mistral", "groq", "gemini", "ollama"],
    ComplexityTier.MEDIUM: ["nim", "groq", "gemini", "ollama"],
    ComplexityTier.COMPLEX: ["deepseek_r1", "nim", "gemini", "ollama"],
    ComplexityTier.HEAVY: ["deepseek_v3", "deepseek_r1", "gemini", "ollama"],
}


class ModelRouter:
    """Routes requests to providers based on complexity tier.

    Maintains provider state and handles fallback chains when providers
    are unavailable or have exhausted quotas.
    """

    def __init__(self) -> None:
        self._providers: dict[str, ProviderInfo] = {}
        self._initialize_providers()

    def _initialize_providers(self) -> None:
        """Build provider registry from config."""
        cfg = settings.providers

        # Mistral
        self._providers["mistral"] = ProviderInfo(
            name="mistral",
            model=cfg.mistral.default_model,
            api_key=cfg.mistral.api_key,
            timeout=cfg.mistral.timeout,
        )

        # Groq
        self._providers["groq"] = ProviderInfo(
            name="groq",
            model=cfg.groq.default_model,
            api_key=cfg.groq.api_key,
            timeout=cfg.groq.timeout,
        )

        # Gemini (emergency fallback)
        self._providers["gemini"] = ProviderInfo(
            name="gemini",
            model=cfg.gemini.default_model,
            api_key=cfg.gemini.api_key,
            timeout=cfg.gemini.timeout,
        )

        # NIM — pool-managed, availability based on pool state
        nim_has_keys = bool(cfg.nim.api_keys)
        self._providers["nim"] = ProviderInfo(
            name="nim",
            model=cfg.nim.default_model,
            api_key="pool-managed" if nim_has_keys else "",
            base_url=cfg.nim.base_url,
            timeout=cfg.nim.timeout,
        )

        # DeepSeek-R1 (reasoning) — pool-managed
        ds_has_keys = bool(cfg.deepseek.api_keys)
        self._providers["deepseek_r1"] = ProviderInfo(
            name="deepseek_r1",
            model=cfg.deepseek.reasoning_model,
            api_key="pool-managed" if ds_has_keys else "",
            base_url=cfg.deepseek.base_url,
            timeout=cfg.deepseek.timeout,
        )

        # DeepSeek-V3 (general heavy) — pool-managed
        self._providers["deepseek_v3"] = ProviderInfo(
            name="deepseek_v3",
            model=cfg.deepseek.general_model,
            api_key="pool-managed" if ds_has_keys else "",
            base_url=cfg.deepseek.base_url,
            timeout=cfg.deepseek.timeout,
        )

        # Ollama (local, always available)
        self._providers["ollama"] = ProviderInfo(
            name="ollama",
            model=settings.ollama.orchestrator_model,
            base_url=settings.ollama.base_url,
            timeout=settings.ollama.timeout,
        )

    def get_provider(self, name: str) -> ProviderInfo | None:
        """Get a provider by name."""
        return self._providers.get(name)

    # Providers that use account pools for round-robin key rotation
    _POOLED_PROVIDERS = frozenset({"nim", "deepseek_r1", "deepseek_v3"})

    def route(self, complexity: ComplexityTier) -> RoutingResult:
        """Select the best provider for a given complexity tier.

        Walks the fallback chain until an available provider is found.
        For pooled providers (NIM, DeepSeek), selects an account from
        the pool and sets the active API key on the provider.
        Ollama is always the final fallback.
        """
        chain = TIER_FALLBACK_CHAINS.get(complexity, TIER_FALLBACK_CHAINS[ComplexityTier.SIMPLE])
        is_fallback = False
        fallback_reason = None

        for i, provider_name in enumerate(chain):
            provider = self._providers.get(provider_name)
            if not provider:
                continue

            # For pooled providers, check pool availability
            account_index: int | None = None
            if provider_name in self._POOLED_PROVIDERS:
                pool = get_pool(provider_name)
                if pool:
                    if pool.is_exhausted:
                        # All accounts in pool exhausted — skip provider
                        continue
                    account = pool.next_account()
                    if account:
                        provider.api_key = account.key
                        account_index = account.index
                    else:
                        # next_account returned None (shouldn't happen if not exhausted)
                        continue
                else:
                    # No pool configured — fall back to static key check
                    if not provider.is_available():
                        continue
            elif not provider.is_available():
                continue

            if i > 0:
                is_fallback = True
                primary = chain[0]
                fallback_reason = f"{primary} unavailable, fell back to {provider_name}"
                logger.info(
                    "router.fallback",
                    complexity=complexity.value,
                    primary=primary,
                    selected=provider_name,
                    reason=fallback_reason,
                )

            logger.debug(
                "router.selected",
                complexity=complexity.value,
                provider=provider_name,
                model=provider.model,
                account_index=account_index,
            )
            return RoutingResult(
                provider=provider,
                complexity=complexity,
                is_fallback=is_fallback,
                fallback_reason=fallback_reason,
                account_index=account_index,
            )

        # Should never reach here — Ollama is always in chain and always available
        ollama = self._providers["ollama"]
        logger.error("router.all_providers_exhausted", complexity=complexity.value)
        return RoutingResult(
            provider=ollama,
            complexity=complexity,
            is_fallback=True,
            fallback_reason="all providers exhausted, using local Ollama",
        )

    def mark_provider_error(self, name: str, error: str) -> None:
        """Record a provider error for circuit-breaker logic."""
        provider = self._providers.get(name)
        if provider:
            provider.mark_error(error)

    def mark_provider_success(self, name: str) -> None:
        """Record a successful provider call."""
        provider = self._providers.get(name)
        if provider:
            provider.mark_success()

    def reset_provider(self, name: str) -> None:
        """Reset a provider's error state."""
        provider = self._providers.get(name)
        if provider:
            provider.reset()

    def get_status(self) -> dict[str, dict[str, Any]]:
        """Get status summary of all providers."""
        from typing import Any

        status: dict[str, dict[str, Any]] = {}
        for name, p in self._providers.items():
            status[name] = {
                "status": p.status.value,
                "model": p.model,
                "has_key": bool(p.api_key) or name == "ollama",
                "request_count": p.request_count,
                "error_count": p.error_count,
            }
        return status
