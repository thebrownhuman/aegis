"""Project Aegis — Account pool manager.

Sprint 1.2 Task 24b: Round-robin management for multi-account providers
(NIM × 4, DeepSeek × 4) with quota tracking and failover.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, UTC
from typing import Any

import structlog

from aegis.config import settings

logger = structlog.get_logger(__name__)


@dataclass
class AccountState:
    """Tracks usage and health for a single API account."""

    key: str
    index: int
    request_count: int = 0
    token_count: int = 0
    error_count: int = 0
    last_used: datetime | None = None
    last_error: datetime | None = None
    is_exhausted: bool = False

    @property
    def masked_key(self) -> str:
        """Return masked version of API key for logging."""
        if len(self.key) <= 8:
            return "***"
        return f"{self.key[:4]}...{self.key[-4:]}"

    def record_usage(self, tokens: int = 0) -> None:
        """Record a successful API call."""
        self.request_count += 1
        self.token_count += tokens
        self.last_used = datetime.now(UTC)
        self.error_count = 0  # Reset consecutive errors

    def record_error(self, error: str) -> None:
        """Record a failed API call."""
        self.error_count += 1
        self.last_error = datetime.now(UTC)
        if self.error_count >= 3:
            self.is_exhausted = True
            logger.warning(
                "account_pool.account_exhausted",
                index=self.index,
                key=self.masked_key,
                errors=self.error_count,
            )

    def mark_exhausted(self) -> None:
        """Manually mark account as exhausted (e.g., quota exceeded)."""
        self.is_exhausted = True

    def reset(self) -> None:
        """Reset account state (e.g., after quota refresh)."""
        self.is_exhausted = False
        self.error_count = 0


class AccountPool:
    """Round-robin pool for multi-account providers.

    Rotates through available accounts and skips exhausted ones.
    Falls through when all accounts in the pool are exhausted.
    """

    def __init__(self, provider_name: str, api_keys: list[str]) -> None:
        self._provider_name = provider_name
        self._accounts = [
            AccountState(key=key, index=i)
            for i, key in enumerate(api_keys)
            if key  # Skip empty keys
        ]
        self._current_index = 0

    @property
    def size(self) -> int:
        """Number of accounts in the pool."""
        return len(self._accounts)

    @property
    def available_count(self) -> int:
        """Number of non-exhausted accounts."""
        return sum(1 for a in self._accounts if not a.is_exhausted)

    @property
    def is_exhausted(self) -> bool:
        """Whether all accounts are exhausted."""
        return self.available_count == 0

    def next_account(self) -> AccountState | None:
        """Get the next available account via round-robin.

        Returns None if all accounts are exhausted.
        """
        if not self._accounts:
            return None

        # Try each account once starting from current index
        for _ in range(len(self._accounts)):
            account = self._accounts[self._current_index]
            self._current_index = (self._current_index + 1) % len(self._accounts)

            if not account.is_exhausted:
                logger.debug(
                    "account_pool.selected",
                    provider=self._provider_name,
                    index=account.index,
                    key=account.masked_key,
                )
                return account

        logger.warning(
            "account_pool.all_exhausted",
            provider=self._provider_name,
            total=len(self._accounts),
        )
        return None

    def record_usage(self, account_index: int, tokens: int = 0) -> None:
        """Record successful usage for an account."""
        if 0 <= account_index < len(self._accounts):
            self._accounts[account_index].record_usage(tokens)

    def record_error(self, account_index: int, error: str) -> None:
        """Record an error for an account."""
        if 0 <= account_index < len(self._accounts):
            self._accounts[account_index].record_error(error)

    def reset_all(self) -> None:
        """Reset all accounts (e.g., daily quota refresh)."""
        for account in self._accounts:
            account.reset()
        self._current_index = 0
        logger.info("account_pool.reset", provider=self._provider_name, accounts=len(self._accounts))

    def get_status(self) -> dict[str, Any]:
        """Get pool status summary."""
        return {
            "provider": self._provider_name,
            "total_accounts": len(self._accounts),
            "available": self.available_count,
            "exhausted": len(self._accounts) - self.available_count,
            "accounts": [
                {
                    "index": a.index,
                    "key": a.masked_key,
                    "requests": a.request_count,
                    "tokens": a.token_count,
                    "exhausted": a.is_exhausted,
                    "errors": a.error_count,
                }
                for a in self._accounts
            ],
        }


# --- Pool registry ---

_pools: dict[str, AccountPool] = {}


def get_pool(provider_name: str) -> AccountPool | None:
    """Get or create an account pool for a provider."""
    if provider_name in _pools:
        return _pools[provider_name]

    # Initialize from config
    cfg = settings.providers
    if provider_name == "nim" and cfg.nim.api_keys:
        _pools["nim"] = AccountPool("nim", cfg.nim.api_keys)
        return _pools["nim"]
    elif provider_name in ("deepseek_r1", "deepseek_v3") and cfg.deepseek.api_keys:
        # Both DeepSeek models share the same account pool
        if "deepseek" not in _pools:
            _pools["deepseek"] = AccountPool("deepseek", cfg.deepseek.api_keys)
        return _pools["deepseek"]

    return None


def get_all_pool_status() -> dict[str, Any]:
    """Get status of all active account pools."""
    return {name: pool.get_status() for name, pool in _pools.items()}
