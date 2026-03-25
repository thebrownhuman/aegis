"""Migration smoke test.

Verifies that `alembic upgrade head` on an empty database creates the
expected schema. This catches regressions where ORM models diverge from
migrations (Fix #5 from Codex review).
"""

from __future__ import annotations

import os
import sqlite3
import tempfile

import pytest
from alembic import command
from alembic.config import Config


EXPECTED_TABLES = {
    "alerts",
    "audit_events",
    "conversations",
    "guest_activity",
    "messages",
    "nim_usage",
    "policy_decisions",
    "portfolio_suggestions",
    "rate_limits",
    "sessions",
    "tool_executions",
    "upload_queue",
    "users",
}


class TestMigrationSmoke:
    """Validate Alembic migration creates complete schema on a fresh DB."""

    @pytest.fixture
    def fresh_db(self, tmp_path):
        """Create a temp DB path and run alembic upgrade head."""
        db_path = tmp_path / "test_migration.db"
        db_url = f"sqlite:///{db_path}"

        cfg = Config(os.path.join(os.path.dirname(__file__), "..", "alembic.ini"))
        cfg.set_main_option("sqlalchemy.url", db_url)
        command.upgrade(cfg, "head")

        yield db_path

    def test_all_expected_tables_exist(self, fresh_db) -> None:
        conn = sqlite3.connect(str(fresh_db))
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        tables = {row[0] for row in cursor.fetchall()}
        conn.close()

        # alembic_version is internal, not in our expected set
        app_tables = tables - {"alembic_version"}
        assert app_tables == EXPECTED_TABLES, (
            f"Missing: {EXPECTED_TABLES - app_tables}, "
            f"Extra: {app_tables - EXPECTED_TABLES}"
        )

    def test_alembic_version_stamped(self, fresh_db) -> None:
        conn = sqlite3.connect(str(fresh_db))
        cursor = conn.execute("SELECT version_num FROM alembic_version")
        row = cursor.fetchone()
        conn.close()

        assert row is not None, "alembic_version table is empty"
        assert row[0], "version_num is empty"

    def test_users_table_has_expected_columns(self, fresh_db) -> None:
        conn = sqlite3.connect(str(fresh_db))
        cursor = conn.execute("PRAGMA table_info(users)")
        columns = {row[1] for row in cursor.fetchall()}
        conn.close()

        expected_columns = {
            "id", "username", "password_hash", "role", "clearance",
            "owner_id", "display_name", "email", "company", "totp_secret",
            "created_at", "last_active", "is_active",
        }
        assert expected_columns.issubset(columns), (
            f"Missing columns in users: {expected_columns - columns}"
        )

    def test_audit_events_has_hash_chain_columns(self, fresh_db) -> None:
        conn = sqlite3.connect(str(fresh_db))
        cursor = conn.execute("PRAGMA table_info(audit_events)")
        columns = {row[1] for row in cursor.fetchall()}
        conn.close()

        assert "prev_hash" in columns, "audit_events missing prev_hash (ID-18)"
        assert "entry_hash" in columns, "audit_events missing entry_hash (ID-18)"

    def test_messages_foreign_key_to_conversations(self, fresh_db) -> None:
        conn = sqlite3.connect(str(fresh_db))
        cursor = conn.execute("PRAGMA foreign_key_list(messages)")
        fks = [row for row in cursor.fetchall()]
        conn.close()

        fk_targets = {row[2] for row in fks}  # column 2 = referenced table
        assert "conversations" in fk_targets

    def test_fresh_db_accepts_basic_insert(self, fresh_db) -> None:
        """Verify schema is usable — insert and query a user."""
        import uuid
        from datetime import datetime, UTC

        conn = sqlite3.connect(str(fresh_db))
        user_id = str(uuid.uuid4())
        conn.execute(
            "INSERT INTO users (id, username, password_hash, role, created_at, is_active) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (user_id, "testuser", "hash", "owner", datetime.now(UTC).isoformat(), 1),
        )
        conn.commit()

        cursor = conn.execute("SELECT username FROM users WHERE id = ?", (user_id,))
        row = cursor.fetchone()
        conn.close()

        assert row is not None
        assert row[0] == "testuser"
