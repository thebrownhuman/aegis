"""Tests for the health check endpoint.

Sprint 1.1 Task 7: Verify /health returns correct structure.
"""

from __future__ import annotations

from fastapi.testclient import TestClient


class TestHealthEndpoint:
    """Test /health endpoint responses."""

    def test_health_returns_200(self, client: TestClient) -> None:
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_has_required_fields(self, client: TestClient) -> None:
        response = client.get("/health")
        data = response.json()
        assert "status" in data
        assert "timestamp" in data
        assert "version" in data
        assert "services" in data

    def test_health_services_structure(self, client: TestClient) -> None:
        response = client.get("/health")
        services = response.json()["services"]
        assert "ollama" in services
        assert "qdrant" in services
        assert "sqlite" in services

    def test_health_sqlite_is_healthy(self, client: TestClient) -> None:
        """SQLite should always be healthy in test environment."""
        response = client.get("/health")
        sqlite_status = response.json()["services"]["sqlite"]
        assert sqlite_status["status"] == "healthy"

    def test_health_sqlite_wal_mode(self, client: TestClient) -> None:
        """Verify WAL mode is active (NC6 requirement)."""
        response = client.get("/health")
        sqlite_status = response.json()["services"]["sqlite"]
        assert sqlite_status.get("journal_mode") == "wal"
