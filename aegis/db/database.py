"""SQLite database engine with WAL mode and durability guarantees.

Implements NC6: WAL + synchronous=FULL for crash safety.
Includes startup integrity check.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Generator

import structlog
from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import Session, sessionmaker

from aegis.config import settings

logger = structlog.get_logger(__name__)

# Create engine with SQLite-specific settings
engine = create_engine(
    settings.database.url,
    echo=settings.database.echo,
    connect_args={"check_same_thread": False},  # Required for SQLite + FastAPI async
    pool_pre_ping=True,
)


@event.listens_for(engine, "connect")
def _set_sqlite_pragmas(dbapi_connection: sqlite3.Connection, _: object) -> None:
    """Set SQLite pragmas for durability and performance on every new connection.

    NC6: WAL mode + synchronous=FULL ensures crash safety.
    """
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA synchronous=FULL")
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.execute("PRAGMA busy_timeout=5000")  # 5s wait on lock contention
    cursor.close()


SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency-injectable database session.

    Usage with FastAPI:
        @app.get("/")
        def endpoint(db: Session = Depends(get_db)):
            ...

    Plain generator (no @contextmanager) so FastAPI's Depends() handles
    the yield protocol correctly.
    """
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def check_integrity() -> bool:
    """Run SQLite integrity check on startup.

    Returns True if database passes integrity check.
    """
    try:
        with engine.connect() as conn:
            result = conn.execute(text("PRAGMA integrity_check"))
            row = result.fetchone()
            if row and row[0] == "ok":
                logger.info("database.integrity_check", status="passed")
                return True
            logger.error("database.integrity_check", status="failed", result=str(row))
            return False
    except Exception as e:
        logger.error("database.integrity_check", status="error", error=str(e))
        return False


def verify_pragmas() -> dict[str, str]:
    """Verify that WAL and durability pragmas are active. Returns pragma values."""
    pragmas = {}
    with engine.connect() as conn:
        for pragma in ["journal_mode", "synchronous", "foreign_keys"]:
            result = conn.execute(text(f"PRAGMA {pragma}"))
            row = result.fetchone()
            pragmas[pragma] = str(row[0]) if row else "unknown"
    logger.info("database.pragmas_verified", **pragmas)
    return pragmas
