"""Tests for server.core.prompting — HttpPromptStrategy."""

import json

import pytest

from server.core.prompting import HttpPromptStrategy


@pytest.fixture
def strategy():
    return HttpPromptStrategy()


# ---------------------------------------------------------------------------
# build_prompt — Sandwich layout
# ---------------------------------------------------------------------------


def test_build_prompt_sandwich_layout(strategy):
    """Request details (Method, Path) must appear AFTER the RAG context,
    and the output schema section must appear at the very end."""
    request = {
        "method": "POST",
        "path": "/repos/octocat/hello-world/issues",
        "body": '{"title": "bug"}',
        "headers": {"authorization": "Bearer abc123"},
        "current_time": "2026-01-01T00:00:00+00:00",
    }
    context = "RELEVANT_DOCUMENTATION_BLOCK"
    state = "CURRENT_STATE_BLOCK"

    prompt = strategy.build_prompt(request, context, state)

    ctx_pos = prompt.index("RELEVANT_DOCUMENTATION_BLOCK")
    state_pos = prompt.index("CURRENT_STATE_BLOCK")
    method_pos = prompt.index("Method: POST")
    path_pos = prompt.index("Path: /repos/octocat/hello-world/issues")
    schema_pos = prompt.index("OUTPUT SCHEMA")
    generate_pos = prompt.index("GENERATE YOUR JSON RESPONSE NOW")

    # TOP: System role & rules come before context
    assert ctx_pos > 0
    # MIDDLE: Context and state come before request details
    assert ctx_pos < method_pos
    assert state_pos < method_pos
    # BOTTOM: Request comes before the output schema instruction
    assert method_pos < schema_pos
    assert path_pos < schema_pos
    # Schema and final instruction are at the very end
    assert schema_pos < generate_pos


def test_build_prompt_contains_all_fields(strategy):
    request = {
        "method": "GET",
        "path": "/api/v1/resource",
        "body": '{"key": "value"}',
        "headers": {"authorization": "Bearer token", "user-agent": "test-agent"},
        "current_time": "2026-01-01T00:00:00+00:00",
    }

    prompt = strategy.build_prompt(request, "API context docs", "{}")

    assert "Method: GET" in prompt
    assert "Path: /api/v1/resource" in prompt
    assert 'Body: {"key": "value"}' in prompt
    assert "Authorization: Bearer token" in prompt
    assert "User-Agent: test-agent" in prompt
    assert "API context docs" in prompt
    assert "2026-01-01T00:00:00+00:00" in prompt


# ---------------------------------------------------------------------------
# build_prompt — Header filtering
# ---------------------------------------------------------------------------


def test_build_prompt_header_filtering(strategy):
    headers = {
        "authorization": "Bearer token",  # allowlisted
        "accept": "application/json",  # allowlisted
        "x-custom-header": "custom",  # allowed (x- prefix)
        "host": "localhost:8000",  # NOT allowed
    }
    request = {"method": "GET", "path": "/", "body": "", "headers": headers}

    prompt = strategy.build_prompt(request, "", "")

    assert "Authorization: Bearer token" in prompt
    assert "Accept: application/json" in prompt
    assert "X-Custom-Header: custom" in prompt
    assert "Host:" not in prompt
    assert "localhost:8000" not in prompt


# ---------------------------------------------------------------------------
# build_prompt — Timestamp injection
# ---------------------------------------------------------------------------


def test_build_prompt_provided_timestamp(strategy):
    request = {
        "method": "GET",
        "path": "/",
        "body": "",
        "headers": {},
        "current_time": "2026-01-01T00:00:00+00:00",
    }
    prompt = strategy.build_prompt(request, "", "")
    assert "Current UTC Timestamp: 2026-01-01T00:00:00+00:00" in prompt


def test_build_prompt_auto_timestamp(strategy):
    request = {"method": "GET", "path": "/", "body": "", "headers": {}}
    prompt = strategy.build_prompt(request, "", "")
    assert "Current UTC Timestamp: None" not in prompt
    assert "Current UTC Timestamp:" in prompt


# ---------------------------------------------------------------------------
# parse_response — happy path
# ---------------------------------------------------------------------------


def test_parse_response_valid(strategy):
    raw = json.dumps({"response": {"id": 1}, "side_effects": [{"action": "SET"}]})
    result = strategy.parse_response(raw)

    assert result["response"] == {"id": 1}
    assert result["side_effects"] == [{"action": "SET"}]


# ---------------------------------------------------------------------------
# parse_response — fallback wrapping
# ---------------------------------------------------------------------------


def test_parse_response_missing_keys(strategy):
    """If the LLM returns valid JSON but without the canonical keys, wrap it."""
    raw = json.dumps({"id": 1, "name": "test"})
    result = strategy.parse_response(raw)

    assert result["response"] == {"id": 1, "name": "test"}
    assert result["side_effects"] == []


# ---------------------------------------------------------------------------
# parse_response — invalid JSON
# ---------------------------------------------------------------------------


def test_parse_response_invalid_json(strategy):
    """Non-JSON input should still return the canonical envelope."""
    result = strategy.parse_response("this is not json at all")

    assert "response" in result
    assert "side_effects" in result
    assert result["side_effects"] == []
    # The inner response should contain the error from clean_llm_response
    assert "error" in result["response"]
