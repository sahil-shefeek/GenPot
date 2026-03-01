import json
import re
import pytest
from server import logger


@pytest.fixture(autouse=True)
def mock_paths(monkeypatch, tmp_path):
    logs_dir = tmp_path / "logs"
    log_file = logs_dir / "honeypot.jsonl"
    logs_dir.mkdir(exist_ok=True)
    monkeypatch.setattr(logger, "LOGS_DIR", logs_dir)
    monkeypatch.setattr(logger, "LOG_FILE", log_file)


def _make_call(**overrides):
    """Helper that returns a complete set of kwargs for log_interaction."""
    defaults = {
        "protocol": "http",
        "source_ip": "192.168.1.100",
        "request_data": {"method": "GET", "path": "/repos/owner/repo", "body": ""},
        "response_data": {"status_code": 200, "body": {"message": "ok"}},
        "genpot_metrics": {
            "rag_query": "GET /repos/owner/repo",
            "rag_context": "some context",
            "similarity_score": 0.85,
            "llm_provider": "gemini",
            "llm_model": "gemini-1.5-flash",
            "latency_ms": 123.4,
            "state_actions": [],
        },
    }
    defaults.update(overrides)
    return defaults


# ISO 8601 strict pattern: must end in Z
_ISO8601_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$")


def _read_last_entry():
    """Read and return the last JSON line from the log file."""
    with open(logger.LOG_FILE, "r") as f:
        lines = f.readlines()
    return json.loads(lines[-1])


# =========================================================================
# Test: ECS Base Fields
# =========================================================================


def test_ecs_base_fields():
    logger.log_interaction(**_make_call())
    entry = _read_last_entry()

    # @timestamp — strict ISO 8601 UTC ending in Z
    assert "@timestamp" in entry
    assert _ISO8601_PATTERN.match(entry["@timestamp"]), (
        f"Timestamp '{entry['@timestamp']}' does not match strict ISO 8601"
    )

    # event namespace
    assert entry["event"]["action"] == "honeypot_interaction"
    assert entry["event"]["outcome"] == "success"

    # source.ip
    assert entry["source"]["ip"] == "192.168.1.100"

    # network.protocol
    assert entry["network"]["protocol"] == "http"


# =========================================================================
# Test: HTTP Protocol Namespacing
# =========================================================================


def test_http_protocol_fields():
    logger.log_interaction(**_make_call())
    entry = _read_last_entry()

    assert entry["http"]["request"]["method"] == "GET"
    assert entry["http"]["response"]["status_code"] == 200
    assert entry["url"]["path"] == "/repos/owner/repo"


# =========================================================================
# Test: Custom GenPot Namespace
# =========================================================================


def test_genpot_custom_namespace():
    logger.log_interaction(**_make_call())
    entry = _read_last_entry()

    gp = entry["genpot"]
    assert gp["rag_query"] == "GET /repos/owner/repo"
    assert gp["rag_context"] == "some context"
    assert gp["similarity_score"] == 0.85
    assert gp["llm_provider"] == "gemini"
    assert gp["llm_model"] == "gemini-1.5-flash"
    assert gp["latency_ms"] == 123.4
    assert gp["state_actions"] == []


# =========================================================================
# Test: Error Event
# =========================================================================


def test_error_event():
    logger.log_interaction(**_make_call(error="Rate limit exceeded"))
    entry = _read_last_entry()

    assert entry["event"]["action"] == "error"
    assert entry["event"]["outcome"] == "failure"
    assert entry["error"]["message"] == "Rate limit exceeded"


# =========================================================================
# Test: Append Multiple Entries
# =========================================================================


def test_append_multiple_entries():
    logger.log_interaction(**_make_call(source_ip="10.0.0.1"))
    logger.log_interaction(**_make_call(source_ip="10.0.0.2"))

    with open(logger.LOG_FILE, "r") as f:
        lines = f.readlines()

    assert len(lines) == 2
    assert json.loads(lines[0])["source"]["ip"] == "10.0.0.1"
    assert json.loads(lines[1])["source"]["ip"] == "10.0.0.2"


# =========================================================================
# Test: Missing Optional Fields
# =========================================================================


def test_missing_optional_fields():
    """Passing empty/minimal dicts must not crash."""
    logger.log_interaction(
        protocol="http",
        source_ip="127.0.0.1",
        request_data={},
        response_data={},
        genpot_metrics={},
    )
    entry = _read_last_entry()

    # Base fields are always present
    assert "@timestamp" in entry
    assert entry["event"]["action"] == "honeypot_interaction"
    assert entry["source"]["ip"] == "127.0.0.1"
    assert entry["network"]["protocol"] == "http"

    # genpot block should be absent when no metrics provided
    assert "genpot" not in entry
