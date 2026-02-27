import pytest
from server.prompt_manager import craft_prompt


def test_craft_prompt_standard_inputs():
    method = "GET"
    path = "/api/v1/resource"
    body = '{"key": "value"}'
    headers = {"authorization": "Bearer token", "user-agent": "test-agent"}
    context = "API context docs"
    state_context = "{}"

    prompt = craft_prompt(
        method=method,
        path=path,
        body=body,
        headers=headers,
        context=context,
        state_context=state_context,
    )

    assert "Method: GET" in prompt
    assert "Path: /api/v1/resource" in prompt
    assert 'Body: {"key": "value"}' in prompt
    assert "Authorization: Bearer token" in prompt
    assert "User-Agent: test-agent" in prompt
    assert "API context docs" in prompt


def test_craft_prompt_header_filtering():
    headers = {
        "authorization": "Bearer token",  # allowlisted
        "accept": "application/json",  # allowlisted
        "x-custom-header": "custom",  # allowed (starts with x-)
        "host": "localhost:8000",  # not allowed
    }

    prompt = craft_prompt("GET", "/", "", headers, "", "")

    assert "Authorization: Bearer token" in prompt
    assert "Accept: application/json" in prompt
    assert "X-Custom-Header: custom" in prompt
    assert "Host:" not in prompt
    assert "localhost:8000" not in prompt


def test_craft_prompt_time_injection():
    # Test with provided time
    prompt_with_time = craft_prompt(
        "GET", "/", "", {}, "", "", current_time="2024-01-01T00:00:00+00:00"
    )
    assert "Current UTC Timestamp: 2024-01-01T00:00:00+00:00" in prompt_with_time

    # Test without provided time
    prompt_without_time = craft_prompt("GET", "/", "", {}, "", "")

    # Ensure a timestamp is generated and present
    assert "Current UTC Timestamp: None" not in prompt_without_time
    time_line = [
        line
        for line in prompt_without_time.split("\\n")
        if line.startswith("Current UTC Timestamp: ")
    ]
    if time_line:
        assert len(time_line[0]) > len("Current UTC Timestamp: ")
