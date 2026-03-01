import pytest
from fastapi.testclient import TestClient

from server.llm_client import LLMRateLimitError
from server.main import app


@pytest.fixture
def mock_dependencies(mocker):
    # Mock rag_system dependencies
    mocker.patch("server.main.rag_system.get_context", return_value="dummy context")
    mock_inspect = mocker.patch(
        "server.main.rag_system.inspect_query",
        return_value={"latency_ms": 10, "chunks": []},
    )
    mocker.patch("server.main.rag_system.compute_similarity", return_value=0.85)

    # Mock state_manager dependencies
    mocker.patch("server.main.state_manager.get_context", return_value="dummy state")
    mock_apply_updates = mocker.patch("server.main.state_manager.apply_updates")

    # Mock logger / interaction
    mock_log_interaction = mocker.patch("server.main.log_interaction")

    # Mock config loader
    mocker.patch(
        "server.main.load_config",
        return_value={"honeypot_provider": "gemini", "honeypot_model": "test-model"},
    )

    return {
        "inspect_query": mock_inspect,
        "apply_updates": mock_apply_updates,
        "log_interaction": mock_log_interaction,
    }


def test_rag_inspect_endpoint(mock_dependencies):
    client = TestClient(app)
    response = client.post("/api/rag-inspect", json={"query": "test", "top_k": 3})

    assert response.status_code == 200
    mock_dependencies["inspect_query"].assert_called_once_with("test", 3)


def test_decoy_endpoint_happy_path(mock_dependencies, mocker):
    mocker.patch(
        "server.main.generate_response",
        return_value='{"response": {"message": "success"}, "side_effects":[]}',
    )
    client = TestClient(app)

    response = client.post("/api/test-endpoint", json={"dummy": "data"})

    assert response.status_code == 200
    assert response.json() == {"message": "success"}

    mock_dependencies["log_interaction"].assert_called_once()
    call_kwargs = mock_dependencies["log_interaction"].call_args[1]
    assert call_kwargs["response_data"]["status_code"] == 200
    assert call_kwargs["protocol"] == "http"


def test_decoy_endpoint_side_effects(mock_dependencies, mocker):
    mocker.patch(
        "server.main.generate_response",
        return_value='{"response": {}, "side_effects":[{"action": "SET", "scope": "global", "key": "test", "value": "123"}]}',
    )
    client = TestClient(app)

    response = client.get("/api/another-endpoint")

    assert response.status_code == 200
    mock_dependencies["apply_updates"].assert_called_once_with(
        [{"action": "SET", "scope": "global", "key": "test", "value": "123"}]
    )


def test_decoy_endpoint_rate_limit(mock_dependencies, mocker):
    mocker.patch(
        "server.main.generate_response",
        side_effect=LLMRateLimitError("Quota Exceeded", retry_after="60s"),
    )
    client = TestClient(app)

    response = client.get("/api/rate-limited")

    assert response.status_code == 429
    assert "error" in response.json()
    assert response.json()["retry_after"] == "60s"

    mock_dependencies["log_interaction"].assert_called_once()
    call_kwargs = mock_dependencies["log_interaction"].call_args[1]
    assert call_kwargs["response_data"]["status_code"] == 429
    assert call_kwargs["error"] is not None


def test_decoy_endpoint_internal_server_error(mock_dependencies, mocker):
    mocker.patch(
        "server.main.generate_response", side_effect=Exception("Something broke")
    )
    client = TestClient(app)

    response = client.get("/api/broken")

    assert response.status_code == 500
    assert "error" in response.json()

    mock_dependencies["log_interaction"].assert_called_once()
    call_kwargs = mock_dependencies["log_interaction"].call_args[1]
    assert call_kwargs["response_data"]["status_code"] == 500
    assert call_kwargs["error"] is not None
