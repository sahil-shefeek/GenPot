import pytest
from server.prompt_manager import craft_prompt, craft_ssh_prompt


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


# ── craft_ssh_prompt ──────────────────────────────────────────────────────────

def test_craft_ssh_prompt_contains_required_sections():
    prompt = craft_ssh_prompt(
        command="ls",
        cwd="/root",
        file_context='{"notes.txt": {"type": "file"}}',
        history=[],
    )
    assert "CURRENT ENVIRONMENT" in prompt
    assert "CWD:      /root" in prompt
    assert "VIRTUAL FILESYSTEM" in prompt
    assert "SIDE EFFECT PROTOCOL" in prompt
    assert "TERMINAL OUTPUT:" in prompt
    assert prompt.strip().endswith("ls\n\nTERMINAL OUTPUT:")


def test_craft_ssh_prompt_injects_file_context():
    file_context = '{"shifa": {"type": "directory"}, "notes.txt": {"type": "file"}}'
    prompt = craft_ssh_prompt("ls", "/root", file_context, [])
    assert "shifa" in prompt
    assert "notes.txt" in prompt


def test_craft_ssh_prompt_empty_vfs_shows_fallback():
    prompt = craft_ssh_prompt("ls", "/root", "", [])
    assert "empty" in prompt


def test_craft_ssh_prompt_history_injected_when_present():
    history = [
        {"command": "whoami", "output": "root"},
        {"command": "pwd",    "output": "/root"},
    ]
    prompt = craft_ssh_prompt("ls", "/root", "", history)
    assert "RECENT COMMAND HISTORY" in prompt
    assert "whoami" in prompt
    assert "pwd" in prompt


def test_craft_ssh_prompt_no_history_section_when_empty():
    prompt = craft_ssh_prompt("ls", "/root", "", [])
    assert "RECENT COMMAND HISTORY" not in prompt


def test_craft_ssh_prompt_history_capped_at_10():
    history = [{"command": f"cmd{i}", "output": f"out{i}"} for i in range(15)]
    prompt = craft_ssh_prompt("ls", "/root", "", history)
    assert "cmd14" in prompt
    assert "cmd4" not in prompt


def test_craft_ssh_prompt_cwd_reflected_in_env_block():
    prompt = craft_ssh_prompt("pwd", "/var/log", "", [])
    assert "CWD:      /var/log" in prompt


def test_craft_ssh_prompt_no_ai_persona_leakage():
    """Prompt must enforce the terminal persona, not an AI-assistant persona."""
    prompt = craft_ssh_prompt("ls", "/root", "", [])
    lower = prompt.lower()
    # The persona block must explicitly forbid assistant-style language
    assert "not an ai assistant" in lower
    # Phrases a helpful-assistant LLM might generate must not appear as instructions
    assert "i'm happy to help" not in lower
    assert "how can i assist" not in lower


def test_craft_ssh_prompt_side_effect_examples_present():
    prompt = craft_ssh_prompt("mkdir test", "/root", "", [])
    assert "<SIDE_EFFECT>" in prompt
    assert "</SIDE_EFFECT>" in prompt
