"""
Unit tests for GenPotEngine.process — happy path, 429, and 500.
"""

import pytest

from server.core.engine import GenPotEngine
from server.core.models import UnifiedRequest
from server.llm_client import LLMRateLimitError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_request(**overrides) -> UnifiedRequest:
    defaults = {
        "protocol": "http",
        "source_ip": "10.0.0.1",
        "method": "GET",
        "path": "/api/users",
        "headers": {"authorization": "Bearer tok123"},
        "body": "",
    }
    defaults.update(overrides)
    return UnifiedRequest(**defaults)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_engine(mocker):
    """Return a GenPotEngine with every external dependency mocked."""
    mock_rag = mocker.MagicMock()
    mock_rag.get_context.return_value = "dummy context"
    mock_rag.compute_similarity.return_value = 0.85

    mock_state = mocker.MagicMock()
    mock_state.get_context.return_value = "dummy state"

    mocker.patch(
        "server.core.engine.config_manager.load_config",
        return_value={
            "emulators": {"http": {"provider": "gemini", "model": "test-model"}}
        },
    )
    mocker.patch(
        "server.core.engine.config_manager.get_emulator_config",
        return_value={"provider": "gemini", "model": "test-model"},
    )

    mock_log = mocker.patch("server.core.engine.log_interaction")

    engine = GenPotEngine(rag_system=mock_rag, state_manager=mock_state)

    return {
        "engine": engine,
        "rag": mock_rag,
        "state": mock_state,
        "log_interaction": mock_log,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_process_happy_path(mock_engine, mocker):
    """200 response, side effects applied, interaction logged."""
    mocker.patch(
        "server.core.engine.generate_response",
        return_value='{"response": {"id": 1, "name": "Alice"}, "side_effects": [{"action": "SET", "scope": "global", "key": "/users/1", "value": {"id": 1}}]}',
    )

    result = await mock_engine["engine"].process(_make_request())

    assert result.status_code == 200
    assert result.data == {"id": 1, "name": "Alice"}

    # Side effects applied
    mock_engine["state"].apply_updates.assert_called_once_with(
        [{"action": "SET", "scope": "global", "key": "/users/1", "value": {"id": 1}}]
    )

    # Interaction logged with status 200
    mock_engine["log_interaction"].assert_called_once()
    log_kwargs = mock_engine["log_interaction"].call_args[1]
    assert log_kwargs["protocol"] == "http"
    assert log_kwargs["source_ip"] == "10.0.0.1"
    assert log_kwargs["response_data"]["status_code"] == 200


@pytest.mark.asyncio
async def test_process_rate_limit(mock_engine, mocker):
    """LLMRateLimitError → 429 UnifiedResponse, error logged."""
    mocker.patch(
        "server.core.engine.generate_response",
        side_effect=LLMRateLimitError("Quota Exceeded", retry_after="60s"),
    )

    result = await mock_engine["engine"].process(_make_request())

    assert result.status_code == 429
    assert result.data["retry_after"] == "60s"

    mock_engine["log_interaction"].assert_called_once()
    log_kwargs = mock_engine["log_interaction"].call_args[1]
    assert log_kwargs["response_data"]["status_code"] == 429
    assert log_kwargs["error"] is not None


@pytest.mark.asyncio
async def test_process_generic_error(mock_engine, mocker):
    """Generic Exception → 500 UnifiedResponse, error logged."""
    mocker.patch(
        "server.core.engine.generate_response",
        side_effect=Exception("Something broke"),
    )

    result = await mock_engine["engine"].process(_make_request())

    assert result.status_code == 500
    assert "error" in result.data

    mock_engine["log_interaction"].assert_called_once()
    log_kwargs = mock_engine["log_interaction"].call_args[1]
    assert log_kwargs["response_data"]["status_code"] == 500
    assert log_kwargs["error"] == "Something broke"


@pytest.mark.asyncio
async def test_process_no_side_effects(mock_engine, mocker):
    """When LLM returns no side effects, apply_updates is not called."""
    mocker.patch(
        "server.core.engine.generate_response",
        return_value='{"response": {"ok": true}, "side_effects": []}',
    )

    result = await mock_engine["engine"].process(_make_request())

    assert result.status_code == 200
    mock_engine["state"].apply_updates.assert_not_called()
