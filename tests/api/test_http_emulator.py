"""
API-level tests for the HTTP emulator factory.

These tests create a FastAPI app from `create_http_app` with mock
dependencies, so we only verify the HTTP mapping logic
(FastAPI request → UnifiedRequest → UnifiedResponse → JSONResponse).
"""

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from server.core.models import UnifiedResponse
from server.emulators.http_emulator import create_http_app


@pytest.fixture
def mock_engine():
    engine = MagicMock()
    engine.process = AsyncMock()
    return engine


@pytest.fixture
def mock_rag():
    rag = MagicMock()
    rag.inspect_query = MagicMock(return_value={"latency_ms": 10, "chunks": []})
    return rag


@pytest.fixture
def client(mock_engine, mock_rag):
    app = create_http_app(mock_engine, mock_rag)
    return TestClient(app)


# ------------------------------------------------------------------
# RAG inspect endpoint
# ------------------------------------------------------------------


def test_rag_inspect_endpoint(client, mock_rag):
    response = client.post("/api/rag-inspect", json={"query": "test", "top_k": 3})

    assert response.status_code == 200
    mock_rag.inspect_query.assert_called_once_with("test", 3)


# ------------------------------------------------------------------
# Catch-all decoy routes
# ------------------------------------------------------------------


def test_decoy_endpoint_happy_path(client, mock_engine):
    mock_engine.process.return_value = UnifiedResponse(
        status_code=200,
        data={"message": "success"},
    )

    response = client.post("/api/test-endpoint", json={"dummy": "data"})

    assert response.status_code == 200
    assert response.json() == {"message": "success"}

    # Verify engine.process was called with a UnifiedRequest
    mock_engine.process.assert_called_once()
    call_args = mock_engine.process.call_args[0]
    unified_req = call_args[0]
    assert unified_req.protocol == "http"
    assert unified_req.method == "POST"
    assert unified_req.path == "/api/test-endpoint"


def test_decoy_endpoint_rate_limit(client, mock_engine):
    mock_engine.process.return_value = UnifiedResponse(
        status_code=429,
        data={
            "error": "Service Temporarily Unavailable (Rate Limit Exceeded)",
            "retry_after": "60s",
        },
    )

    response = client.get("/api/rate-limited")

    assert response.status_code == 429
    assert response.json()["retry_after"] == "60s"


def test_decoy_endpoint_internal_server_error(client, mock_engine):
    mock_engine.process.return_value = UnifiedResponse(
        status_code=500,
        data={"error": "An internal server error occurred."},
    )

    response = client.get("/api/broken")

    assert response.status_code == 500
    assert "error" in response.json()


def test_decoy_endpoint_get_request(client, mock_engine):
    """Ensure GET requests are correctly mapped to UnifiedRequest."""
    mock_engine.process.return_value = UnifiedResponse(
        status_code=200,
        data={"items": []},
    )

    response = client.get("/api/resources")

    assert response.status_code == 200
    unified_req = mock_engine.process.call_args[0][0]
    assert unified_req.method == "GET"
    assert unified_req.path == "/api/resources"
    assert unified_req.body == ""
