"""Structured logging system for Project Aegis.

Uses structlog with JSON output, per-day log files, and correlation IDs.
Sprint 1.1 Task 4.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import structlog

from aegis.config import settings


def _ensure_log_dir() -> Path:
    """Create log directory if it doesn't exist."""
    log_dir = Path(settings.logging.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def setup_logging() -> None:
    """Configure structlog with JSON output and per-day log files.

    Log output goes to:
    - stdout (for Docker/dev visibility)
    - data/logs/aegis_YYYY-MM-DD.log (per-day rotation)
    """
    log_dir = _ensure_log_dir()
    log_level = getattr(logging, settings.logging.level.upper(), logging.INFO)

    # Per-day log file
    today = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
    log_file = log_dir / f"aegis_{today}.log"

    # Standard library logging setup (structlog wraps this)
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    file_handler = logging.FileHandler(str(log_file), encoding="utf-8")
    handlers.append(file_handler)

    logging.basicConfig(
        format="%(message)s",
        level=log_level,
        handlers=handlers,
        force=True,
    )

    # Structlog processors
    processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]

    if settings.logging.json_format:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def bind_correlation_ids(
    request_id: str | None = None,
    user_id: str | None = None,
    role: str | None = None,
) -> None:
    """Bind correlation IDs to the current context for all subsequent log calls.

    Called at the start of each request to ensure all logs within that
    request share the same correlation fields.
    """
    if request_id:
        structlog.contextvars.bind_contextvars(request_id=request_id)
    if user_id:
        structlog.contextvars.bind_contextvars(user_id=user_id)
    if role:
        structlog.contextvars.bind_contextvars(role=role)


def clear_correlation_ids() -> None:
    """Clear correlation IDs at the end of a request."""
    structlog.contextvars.clear_contextvars()
