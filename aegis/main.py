"""Project Aegis — FastAPI application entrypoint.

Sprint 1.1: Boots the application, initializes database, and serves health endpoint.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator

import structlog
from fastapi import FastAPI

from aegis.api.routes_health import router as health_router
from aegis.config import settings
from aegis.db.database import check_integrity, engine, verify_pragmas
from aegis.utils.logger import setup_logging

logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifecycle: startup and shutdown hooks."""
    # --- Startup ---
    setup_logging()
    logger.info(
        "aegis.startup",
        version=settings.app.version,
        debug=settings.app.debug,
    )

    # Verify SQLite durability settings (NC6)
    pragmas = verify_pragmas()
    logger.info("aegis.database.pragmas", **pragmas)

    # Run integrity check
    integrity_ok = check_integrity()
    if not integrity_ok:
        logger.error("aegis.startup.integrity_failed")

    logger.info("aegis.ready", host=settings.app.host, port=settings.app.port)

    yield

    # --- Shutdown ---
    logger.info("aegis.shutdown")


app = FastAPI(
    title=settings.app.name,
    version=settings.app.version,
    lifespan=lifespan,
)

# Register routes
app.include_router(health_router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "aegis.main:app",
        host=settings.app.host,
        port=settings.app.port,
        reload=settings.app.debug,
    )
