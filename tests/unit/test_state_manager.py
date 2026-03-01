import json
import os
import pytest
from server.state_manager import StateManager


@pytest.fixture
def state_file(tmp_path):
    return str(tmp_path / "world_state.json")


def test_initialization_no_file(state_file):
    manager = StateManager(state_file)
    assert manager.state == {"global": {}, "tokens": {}, "sessions": {}}
    assert not os.path.exists(state_file)


def test_initialization_with_file(state_file):
    initial_data = {
        "global": {"config": "value"},
        "tokens": {"token123": {"user": "admin"}},
        "sessions": {},
    }
    with open(state_file, "w") as f:
        json.dump(initial_data, f)

    manager = StateManager(state_file)
    assert manager.state == initial_data


def test_initialization_invalid_file(state_file):
    with open(state_file, "w") as f:
        f.write("invalid json")

    manager = StateManager(state_file)
    assert manager.state == {"global": {}, "tokens": {}, "sessions": {}}


def test_sanitize_for_prompt():
    manager = StateManager()

    # Test strings
    assert manager._sanitize_for_prompt("normal string") == "normal string"
    assert manager._sanitize_for_prompt("SYSTEM: hello") == "SYS\u200bTEM: hello"
    assert (
        manager._sanitize_for_prompt("*** ROLE: admin ***")
        == "*\u200b** RO\u200bLE: admin *\u200b**"
    )

    # Test dicts - should serialize and sanitize
    data = {"system_message": "SYSTEM: init", "other": "***"}
    sanitized = manager._sanitize_for_prompt(data)
    assert "SYS\u200bTEM: init" in sanitized
    assert "*\u200b**" in sanitized

    # Test lists
    data_list = ["<|im_start|>", "normal"]
    sanitized_list = manager._sanitize_for_prompt(data_list)
    assert "<|\u200bim_start|>" in sanitized_list


def test_get_context(state_file):
    manager = StateManager(state_file)
    manager.state = {
        "global": {"/api/users": {"users": []}, "/api/config": {"debug": True}},
        "tokens": {"valid_token": {"user_id": 1, "role": "user"}},
    }

    # Match global state
    ctx = manager.get_context("/api/users/1", headers={})
    assert '"/api/users"' in ctx
    assert '"/api/config"' not in ctx

    # Match tokens
    ctx = manager.get_context("/other", headers={"Authorization": "Bearer valid_token"})
    assert '"current_session"' in ctx
    assert '"user_id": 1' in ctx

    # Unrelated path and invalid token
    ctx = manager.get_context("/none", headers={"Authorization": "Bearer invalid"})
    assert ctx == "{}"


def test_apply_updates(state_file):
    manager = StateManager(state_file)

    # Test SET
    manager.apply_updates(
        [
            {
                "action": "SET",
                "scope": "global",
                "key": "test_key",
                "value": "test_val",
            },
            {"action": "SET", "scope": "tokens", "key": "t1", "value": "t_val"},
        ]
    )

    assert manager.state["global"]["test_key"] == "test_val"
    assert manager.state["tokens"]["t1"] == "t_val"
    assert os.path.exists(state_file)

    # Test DELETE
    manager.apply_updates([{"action": "DELETE", "scope": "global", "key": "test_key"}])
    assert "test_key" not in manager.state["global"]

    # Test Invalid
    manager.apply_updates(
        [
            {"action": "SET", "scope": "invalid_scope", "key": "k", "value": "v"},
            {"action": "SET", "scope": "global", "key": "", "value": "v"},
        ]
    )
    assert "invalid_scope" not in manager.state


def test_apply_updates_sessions_scope(state_file):
    manager = StateManager(state_file)

    manager.apply_updates(
        [
            {
                "action": "SET",
                "scope": "sessions",
                "key": "conn-1",
                "value": {"ehlo_received": True, "mail_from": "attacker@evil.com"},
            },
        ]
    )

    assert manager.state["sessions"]["conn-1"]["ehlo_received"] is True

    # Sessions should NOT be persisted to disk
    if os.path.exists(state_file):
        with open(state_file, "r") as f:
            on_disk = json.load(f)
        assert "sessions" not in on_disk

    # DELETE within sessions scope
    manager.apply_updates([{"action": "DELETE", "scope": "sessions", "key": "conn-1"}])
    assert "conn-1" not in manager.state["sessions"]


def test_get_context_with_session_id(state_file):
    manager = StateManager(state_file)
    manager.state["sessions"]["sess-abc"] = {
        "ehlo_received": True,
        "mail_from": "user@example.com",
    }

    ctx = manager.get_context("/smtp", headers={}, session_id="sess-abc")
    assert '"current_session_state"' in ctx
    assert "ehlo_received" in ctx

    # Without session_id, session state is absent
    ctx_no_session = manager.get_context("/smtp", headers={})
    assert "current_session_state" not in ctx_no_session

    # Unknown session_id returns no session state
    ctx_unknown = manager.get_context("/smtp", headers={}, session_id="unknown")
    assert "current_session_state" not in ctx_unknown


def test_clear_session(state_file):
    manager = StateManager(state_file)
    manager.state["sessions"]["conn-42"] = {"stage": "DATA"}

    manager.clear_session("conn-42")
    assert "conn-42" not in manager.state["sessions"]

    # Clearing a non-existent session is a no-op
    manager.clear_session("does-not-exist")
