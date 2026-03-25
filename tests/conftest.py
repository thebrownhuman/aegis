"""Shared test fixtures for Project Aegis."""

from __future__ import annotations

import os
from collections.abc import Generator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from aegis.db.models import Base


@pytest.fixture(scope="session", autouse=True)
def _set_test_env() -> None:
    """Set environment variables for testing."""
    os.environ["AEGIS__APP__DEBUG"] = "true"
    os.environ["AEGIS__DATABASE__URL"] = "sqlite:///test_aegis.db"


@pytest.fixture
def test_db() -> Generator[Session, None, None]:
    """Create a fresh in-memory SQLite database for each test."""
    test_engine = create_engine("sqlite:///:memory:", echo=False)

    @event.listens_for(test_engine, "connect")
    def _set_pragmas(dbapi_connection, _):  # type: ignore[no-untyped-def]
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=FULL")
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    Base.metadata.create_all(bind=test_engine)
    TestSession = sessionmaker(bind=test_engine)
    session = TestSession()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=test_engine)


@pytest.fixture
def client() -> Generator[TestClient, None, None]:
    """Create a FastAPI test client.

    Creates tables via the app's engine since main.py no longer auto-creates
    (Codex finding #2: Alembic is the sole schema manager in production).
    """
    from aegis.db.database import engine as app_engine
    from aegis.main import app

    Base.metadata.create_all(bind=app_engine)
    with TestClient(app) as c:
        yield c
    Base.metadata.drop_all(bind=app_engine)
