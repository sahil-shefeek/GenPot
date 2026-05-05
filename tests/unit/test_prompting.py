"""Tests for server.core.prompting — HttpPromptStrategy & SmtpPromptStrategy."""

import json

import pytest

from server.core.prompting import (
    HttpPromptStrategy,
    SmtpPromptStrategy,
    HTTP_SYSTEM_PROMPT,
    SMTP_SYSTEM_PROMPT,
)


@pytest.fixture
def strategy():
    return HttpPromptStrategy()


@pytest.fixture
def smtp_strategy():
    return SmtpPromptStrategy()


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

    system_prompt, prompt = strategy.build_prompt(request, context, state)

    # System prompt is now returned separately
    assert "REST API server" in system_prompt

    ctx_pos = prompt.index("RELEVANT_DOCUMENTATION_BLOCK")
    state_pos = prompt.index("CURRENT_STATE_BLOCK")
    method_pos = prompt.index("Method: POST")
    path_pos = prompt.index("Path: /repos/octocat/hello-world/issues")
    schema_pos = prompt.index("OUTPUT SCHEMA")
    generate_pos = prompt.index("GENERATE YOUR JSON RESPONSE NOW")

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

    system_prompt, prompt = strategy.build_prompt(request, "API context docs", "{}")

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

    _system_prompt, prompt = strategy.build_prompt(request, "", "")

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
    _system_prompt, prompt = strategy.build_prompt(request, "", "")
    assert "Current UTC Timestamp: 2026-01-01T00:00:00+00:00" in prompt


def test_build_prompt_auto_timestamp(strategy):
    request = {"method": "GET", "path": "/", "body": "", "headers": {}}
    _system_prompt, prompt = strategy.build_prompt(request, "", "")
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


# ===========================================================================
# SmtpPromptStrategy tests
# ===========================================================================


# ---------------------------------------------------------------------------
# SMTP build_prompt — Sandwich layout
# ---------------------------------------------------------------------------


def test_smtp_build_prompt_sandwich_layout(smtp_strategy):
    """Context and state must appear BEFORE the command, and the output
    schema must appear at the very end."""
    request = {"command": "EHLO attacker.com"}
    context = "SMTP_DOCUMENTATION_BLOCK"
    state = "SESSION_STATE_BLOCK"

    system_prompt, prompt = smtp_strategy.build_prompt(request, context, state)

    # System prompt is now returned separately
    assert "SMTP server" in system_prompt

    ctx_pos = prompt.index("SMTP_DOCUMENTATION_BLOCK")
    state_pos = prompt.index("SESSION_STATE_BLOCK")
    cmd_pos = prompt.index("EHLO attacker.com")
    schema_pos = prompt.index("OUTPUT SCHEMA")
    generate_pos = prompt.index("GENERATE YOUR SMTP RESPONSE NOW")

    # MIDDLE: Context and state come before the command
    assert ctx_pos < cmd_pos
    assert state_pos < cmd_pos
    # BOTTOM: Command comes before the output schema
    assert cmd_pos < schema_pos
    # Schema and final instruction are at the very end
    assert schema_pos < generate_pos


# ---------------------------------------------------------------------------
# SMTP build_prompt — SMTP-specific instructions
# ---------------------------------------------------------------------------


def test_smtp_build_prompt_contains_smtp_instructions(smtp_strategy):
    """The prompt must contain SMTP-specific role text and the incoming
    command."""
    request = {"command": "MAIL FROM:<spammer@evil.com>"}

    system_prompt, prompt = smtp_strategy.build_prompt(request, "docs", "state")

    assert "SMTP server" in system_prompt
    assert "RFC 5321" in system_prompt
    assert "MAIL FROM:<spammer@evil.com>" in prompt
    assert "INCOMING SMTP COMMAND" in prompt


# ---------------------------------------------------------------------------
# SMTP parse_response — happy path
# ---------------------------------------------------------------------------


def test_smtp_parse_response_valid(smtp_strategy):
    raw = json.dumps({"response": "250 2.1.0 Ok", "side_effects": []})
    result = smtp_strategy.parse_response(raw)

    assert result["response"] == "250 2.1.0 Ok"
    assert result["side_effects"] == []


# ---------------------------------------------------------------------------
# SMTP parse_response — fallback wrapping
# ---------------------------------------------------------------------------


def test_smtp_parse_response_missing_keys(smtp_strategy):
    """If the LLM returns valid JSON but without the canonical keys, wrap it."""
    raw = json.dumps({"code": 250, "text": "Ok"})
    result = smtp_strategy.parse_response(raw)

    assert result["response"] == {"code": 250, "text": "Ok"}
    assert result["side_effects"] == []


# ---------------------------------------------------------------------------
# SMTP parse_response — invalid JSON
# ---------------------------------------------------------------------------


def test_smtp_parse_response_invalid_json(smtp_strategy):
    """Non-JSON input should still return the canonical envelope."""
    result = smtp_strategy.parse_response("250 Ok")

    assert "response" in result
    assert "side_effects" in result
    assert result["side_effects"] == []
