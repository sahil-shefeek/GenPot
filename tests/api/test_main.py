"""
API-level tests for the FastAPI adapter in server/main.py.

These tests mock GenPotEngine.process so we only verify the HTTP
mapping logic (FastAPI request → UnifiedRequest → UnifiedResponse →
JSONResponse).
"""

import pytest
from fastapi.testclient import TestClient

from server.core.models import UnifiedResponse
from server.main import app


@pytest.fixture
def mock_dependencies(mocker):
    mock_inspect = mocker.patch(
        "server.main.rag_system.inspect_query",
        return_value={"latency_ms": 10, "chunks": []},
    )
    mock_process = mocker.patch("server.main.engine.process")

    return {
        "inspect_query": mock_inspect,
        "process": mock_process,
    }


def test_rag_inspect_endpoint(mock_dependencies):
    client = TestClient(app)
    response = client.post("/api/rag-inspect", json={"query": "test", "top_k": 3})

    assert response.status_code == 200
    mock_dependencies["inspect_query"].assert_called_once_with("test", 3)


def test_decoy_endpoint_happy_path(mock_dependencies):
    mock_dependencies["process"].return_value = UnifiedResponse(
        status_code=200,
        data={"message": "success"},
    )
    client = TestClient(app)

    response = client.post("/api/test-endpoint", json={"dummy": "data"})

    assert response.status_code == 200
    assert response.json() == {"message": "success"}

    # Verify engine.process was called with a UnifiedRequest
    mock_dependencies["process"].assert_called_once()
    call_args = mock_dependencies["process"].call_args[0]
    unified_req = call_args[0]
    assert unified_req.protocol == "http"
    assert unified_req.method == "POST"
    assert unified_req.path == "/api/test-endpoint"


def test_decoy_endpoint_rate_limit(mock_dependencies):
    mock_dependencies["process"].return_value = UnifiedResponse(
        status_code=429,
        data={
            "error": "Service Temporarily Unavailable (Rate Limit Exceeded)",
            "retry_after": "60s",
        },
    )
    client = TestClient(app)

    response = client.get("/api/rate-limited")

    assert response.status_code == 429
    assert response.json()["retry_after"] == "60s"


def test_decoy_endpoint_internal_server_error(mock_dependencies):
    mock_dependencies["process"].return_value = UnifiedResponse(
        status_code=500,
        data={"error": "An internal server error occurred."},
    )
    client = TestClient(app)

    response = client.get("/api/broken")

    assert response.status_code == 500
    assert "error" in response.json()


def test_decoy_endpoint_get_request(mock_dependencies):
    """Ensure GET requests are correctly mapped to UnifiedRequest."""
    mock_dependencies["process"].return_value = UnifiedResponse(
        status_code=200,
        data={"items": []},
    )
    client = TestClient(app)

    response = client.get("/api/resources")

    assert response.status_code == 200
    unified_req = mock_dependencies["process"].call_args[0][0]
    assert unified_req.method == "GET"
    assert unified_req.path == "/api/resources"
    assert unified_req.body == ""
