import pytest
from server.utils import clean_llm_response


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
