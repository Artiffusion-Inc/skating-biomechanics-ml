"""Tests for backend/app/main.py."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Provide a FastAPI TestClient with external dependencies mocked."""
    with patch("app.main.configure_logging"):
        with patch("app.main.init_valkey_pool", new_callable=AsyncMock):
            with patch("app.main.close_valkey_pool", new_callable=AsyncMock):
                with patch("app.main.create_pool", new_callable=AsyncMock) as mock_create_pool:
                    mock_pool = AsyncMock()
                    mock_create_pool.return_value = mock_pool

                    with patch("app.main.get_settings") as mock_get:
                        settings = MagicMock()
                        settings.cors.origins = ["http://localhost:3000"]
                        settings.valkey.host = "localhost"
                        settings.valkey.port = 6379
                        settings.valkey.db = 0
                        settings.valkey.password.get_secret_value.return_value = ""
                        mock_get.return_value = settings

                        from app.main import app

                        with TestClient(app) as test_client:
                            yield test_client


def test_health_endpoint(client: TestClient):
    """GET /api/v1/health returns 200 and status ok."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_lifespan_initializes_arq_pool(client: TestClient):
    """Lifespan context should have started without raising."""
    # If lifespan failed, TestClient context manager would have raised.
    # A simple successful request proves the app is up.
    response = client.get("/api/v1/health")
    assert response.status_code == 200
