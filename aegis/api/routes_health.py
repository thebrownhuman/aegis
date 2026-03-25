"""Health check endpoint for Project Aegis.

Sprint 1.1 Task 7: Returns JSON status for Ollama, Qdrant, and SQLite.
"""

from __future__ import annotations

from datetime import datetime, timezone

import httpx
import structlog
from fastapi import APIRouter
from sqlalchemy import text

from aegis.config import settings
from aegis.db.database import engine

router = APIRouter(tags=["health"])
logger = structlog.get_logger(__name__)


async def _check_ollama() -> dict[str, str]:
    """Ping Ollama API to check availability."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{settings.ollama.base_url}/api/tags")
            if resp.status_code == 200:
                data = resp.json()
                models = [m["name"] for m in data.get("models", [])]
                return {"status": "healthy", "models": ", ".join(models) if models else "none"}
            return {"status": "unhealthy", "error": f"HTTP {resp.status_code}"}
    except Exception as e:
        return {"status": "unreachable", "error": str(e)}


async def _check_qdrant() -> dict[str, str]:
    """Ping Qdrant health endpoint."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(
                f"http://{settings.qdrant.host}:{settings.qdrant.port}/healthz"
            )
            if resp.status_code == 200:
                return {"status": "healthy"}
            return {"status": "unhealthy", "error": f"HTTP {resp.status_code}"}
    except Exception as e:
        return {"status": "unreachable", "error": str(e)}


def _check_sqlite() -> dict[str, str]:
    """Run a simple query to verify SQLite is operational."""
    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            result.fetchone()
            # Also verify WAL mode is active
            wal_result = conn.execute(text("PRAGMA journal_mode"))
            wal_row = wal_result.fetchone()
            journal_mode = str(wal_row[0]) if wal_row else "unknown"
            return {"status": "healthy", "journal_mode": journal_mode}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


@router.get("/health")
async def health_check() -> dict:
    """System health check endpoint.

    Returns status of all core dependencies:
    - Ollama (local LLM runtime)
    - Qdrant (vector database)
    - SQLite (relational database)
    """
    ollama_status = await _check_ollama()
    qdrant_status = await _check_qdrant()
    sqlite_status = _check_sqlite()

    # Overall status: healthy only if all dependencies are healthy
    all_healthy = all(
        s["status"] == "healthy" for s in [ollama_status, qdrant_status, sqlite_status]
    )

    result = {
        "status": "healthy" if all_healthy else "degraded",
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "version": settings.app.version,
        "services": {
            "ollama": ollama_status,
            "qdrant": qdrant_status,
            "sqlite": sqlite_status,
        },
    }

    if not all_healthy:
        logger.warning("health.degraded", **result["services"])

    return result
