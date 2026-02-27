import json
import pytest
from datetime import datetime
from server import logger


@pytest.fixture(autouse=True)
def mock_paths(monkeypatch, tmp_path):
    logs_dir = tmp_path / "logs"
    log_file = logs_dir / "honeypot.jsonl"
    logs_dir.mkdir(exist_ok=True)
    monkeypatch.setattr(logger, "LOGS_DIR", logs_dir)
    monkeypatch.setattr(logger, "LOG_FILE", log_file)


def test_log_interaction():
    dummy_data = {"event": "test_event"}
    logger.log_interaction(dummy_data)

    assert logger.LOG_FILE.exists()

    with open(logger.LOG_FILE, "r") as f:
        lines = f.readlines()

    assert len(lines) == 1
    log_entry = json.loads(lines[0])

    assert log_entry["event"] == "test_event"
    assert "timestamp" in log_entry

    # Verify ISO format by parsing it
    try:
        datetime.fromisoformat(log_entry["timestamp"])
    except ValueError:
        pytest.fail("Timestamp is not in valid ISO format")


def test_log_interaction_append():
    logger.log_interaction({"event": "event1"})
    logger.log_interaction({"event": "event2"})

    with open(logger.LOG_FILE, "r") as f:
        lines = f.readlines()

    assert len(lines) == 2
    entry1 = json.loads(lines[0])
    entry2 = json.loads(lines[1])

    assert entry1["event"] == "event1"
    assert entry2["event"] == "event2"
