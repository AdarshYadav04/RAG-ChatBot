"""Integration tests for health endpoint."""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.fixture
def client():
    """Create test client with mocked app state."""
    with patch("app.db.vector_store.VectorStore.initialize", new_callable=AsyncMock), \
         patch("app.db.chat_history.ChatHistoryDB.initialize", new_callable=AsyncMock):
        from app.main import app
        mock_vs = MagicMock()
        mock_vs.count.return_value = 42
        mock_db = MagicMock()
        app.state.vector_store = mock_vs
        app.state.chat_db = mock_db
        with TestClient(app, raise_server_exceptions=False) as c:
            yield c


def test_health_endpoint(client):
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "version" in data
    assert "components" in data
    assert "uptime_seconds" in data
