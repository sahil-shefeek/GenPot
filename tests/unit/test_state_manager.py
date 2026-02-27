import json
import os
import pytest
from server.state_manager import StateManager


@pytest.fixture
def state_file(tmp_path):
    return str(tmp_path / "world_state.json")


def test_initialization_no_file(state_file):
    manager = StateManager(state_file)
    assert manager.state == {"global": {}, "tokens": {}}
    assert not os.path.exists(state_file)


def test_initialization_with_file(state_file):
    initial_data = {
        "global": {"config": "value"},
        "tokens": {"token123": {"user": "admin"}},
    }
    with open(state_file, "w") as f:
        json.dump(initial_data, f)

    manager = StateManager(state_file)
    assert manager.state == initial_data


def test_initialization_invalid_file(state_file):
    with open(state_file, "w") as f:
        f.write("invalid json")

    manager = StateManager(state_file)
    assert manager.state == {"global": {}, "tokens": {}}


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
