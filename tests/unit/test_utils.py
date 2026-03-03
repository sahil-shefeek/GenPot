import pytest
from server.utils import clean_llm_response, parse_ssh_response


def test_clean_llm_response_valid_json():
    raw = '{"key": "value"}'
    result = clean_llm_response(raw)
    assert result == {"key": "value"}


def test_clean_llm_response_markdown_fences():
    raw = '```json\n{"response": {}}\n```'
    result = clean_llm_response(raw)
    assert result == {"response": {}}


def test_clean_llm_response_malformed_json():
    raw = '{"key": "value"'
    result = clean_llm_response(raw)
    assert "error" in result
    assert "Invalid JSON from LLM:" in result["error"]
    assert result["raw"] == '{"key": "value"'


def test_clean_llm_response_non_string():
    raw = {"key": "value"}
    result = clean_llm_response(raw)
    assert "error" in result
    assert result["error"] == "LLM returned non-text content"
    assert "raw" in result


# ── parse_ssh_response ────────────────────────────────────────────────────────

def test_parse_ssh_response_no_tag_returns_full_output():
    """Read-only commands (ls, cat, echo) produce no side-effect tag."""
    result = parse_ssh_response("shifa\nnotes.txt")
    assert result["output"] == "shifa\nnotes.txt"
    assert result["side_effects"] == []


def test_parse_ssh_response_extracts_output_and_side_effects():
    raw = (
        "\n"
        "<SIDE_EFFECT>"
        '[{"action":"SET","scope":"filesystem","key":"/root/shifa",'
        '"value":{"type":"directory","owner":"root","permissions":"755"}}]'
        "</SIDE_EFFECT>"
    )
    result = parse_ssh_response(raw)
    assert result["output"] == ""
    assert len(result["side_effects"]) == 1
    fx = result["side_effects"][0]
    assert fx["action"] == "SET"
    assert fx["scope"] == "filesystem"
    assert fx["key"] == "/root/shifa"
    assert fx["value"]["type"] == "directory"


def test_parse_ssh_response_output_before_tag_is_preserved():
    raw = (
        "some output\n"
        '<SIDE_EFFECT>[{"action":"DELETE","scope":"filesystem","key":"/root/old.txt"}]</SIDE_EFFECT>'
    )
    result = parse_ssh_response(raw)
    assert result["output"] == "some output"
    assert result["side_effects"][0]["action"] == "DELETE"


def test_parse_ssh_response_malformed_tag_json():
    result = parse_ssh_response("<SIDE_EFFECT>not valid json</SIDE_EFFECT>")
    assert result["side_effects"] == []
    assert result["output"] == ""


def test_parse_ssh_response_single_dict_wrapped_in_list():
    """Gracefully handles the LLM returning a single object instead of an array."""
    raw = (
        '<SIDE_EFFECT>{"action":"SET","scope":"session",'
        '"key":"cwd","value":"/tmp"}</SIDE_EFFECT>'
    )
    result = parse_ssh_response(raw)
    assert len(result["side_effects"]) == 1
    assert result["side_effects"][0]["value"] == "/tmp"


def test_parse_ssh_response_multiple_mutations():
    raw = (
        "<SIDE_EFFECT>"
        '[{"action":"SET","scope":"filesystem","key":"/root/a","value":{"type":"directory"}},'
        '{"action":"SET","scope":"filesystem","key":"/root/a/b","value":{"type":"directory"}}]'
        "</SIDE_EFFECT>"
    )
    result = parse_ssh_response(raw)
    assert len(result["side_effects"]) == 2
    assert result["side_effects"][1]["key"] == "/root/a/b"


def test_parse_ssh_response_empty_string():
    result = parse_ssh_response("")
    assert result["output"] == ""
    assert result["side_effects"] == []


def test_parse_ssh_response_non_string_input():
    result = parse_ssh_response(None)  # type: ignore[arg-type]
    assert result["output"] == ""
    assert result["side_effects"] == []
