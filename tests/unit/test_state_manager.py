import json
import os
import pytest
from server.state_manager import StateManager


@pytest.fixture
def state_file(tmp_path):
    return str(tmp_path / "world_state.json")


def test_initialization_no_file(state_file):
    manager = StateManager(state_file)
    assert manager.state == {
        "global": {},
        "tokens": {},
        "filesystem": {},
        "sessions": {},
    }
    # Fresh install: skeleton file is written immediately
    assert os.path.exists(state_file)


def test_initialization_with_file(state_file):
    initial_data = {
        "global": {"config": "value"},
        "tokens": {"token123": {"user": "admin"}},
        "filesystem": {"/root/hello.txt": {"type": "file", "content": "hi"}},
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
    assert manager.state == {
        "global": {}, "tokens": {}, "filesystem": {}, "sessions": {}
    }


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


# ── Migration ──────────────────────────────────────────────────────────────────

def test_migration_adds_missing_keys(state_file):
    """Legacy world_state.json (only global+tokens) is upgraded transparently."""
    legacy = {"global": {"k": "v"}, "tokens": {"t": {}}}
    with open(state_file, "w") as f:
        json.dump(legacy, f)

    manager = StateManager(state_file)
    assert manager.state["filesystem"] == {}
    assert manager.state["sessions"] == {}
    # Migration must be persisted to disk
    with open(state_file) as f:
        on_disk = json.load(f)
    assert "filesystem" in on_disk
    assert "sessions" in on_disk


def test_migration_preserves_existing_data(state_file):
    """Existing global and token data survives migration."""
    legacy = {"global": {"repo": "linux"}, "tokens": {"tok": {"user": "root"}}}
    with open(state_file, "w") as f:
        json.dump(legacy, f)

    manager = StateManager(state_file)
    assert manager.state["global"]["repo"] == "linux"
    assert manager.state["tokens"]["tok"]["user"] == "root"


# ── VFS / get_ssh_context ──────────────────────────────────────────────────────

def test_get_ssh_context_empty(state_file):
    """Empty VFS returns an empty JSON object."""
    manager = StateManager(state_file)
    assert json.loads(manager.get_ssh_context("/root")) == {}


def test_get_ssh_context_direct_children(state_file):
    """Only direct children of cwd are returned."""
    manager = StateManager(state_file)
    manager.apply_updates([
        {"action": "SET", "scope": "filesystem", "key": "/root/notes.txt",
         "value": {"type": "file", "content": "hello", "owner": "root", "permissions": "644"}},
        {"action": "SET", "scope": "filesystem", "key": "/tmp/other.txt",
         "value": {"type": "file", "content": "", "owner": "root", "permissions": "644"}},
    ])
    ctx = json.loads(manager.get_ssh_context("/root"))
    assert "notes.txt" in ctx
    assert "other.txt" not in ctx


def test_get_ssh_context_nested_stub(state_file):
    """Deeper descendants appear as directory stubs, not full entries."""
    manager = StateManager(state_file)
    manager.apply_updates([
        {"action": "SET", "scope": "filesystem", "key": "/root/work/a.py",
         "value": {"type": "file", "content": "", "owner": "root", "permissions": "644"}},
    ])
    ctx = json.loads(manager.get_ssh_context("/root"))
    assert ctx == {"work": {"type": "directory"}}


def test_get_ssh_context_mkdir_then_ls(state_file):
    """mkdir followed by ls shows the new directory."""
    manager = StateManager(state_file)
    manager.apply_updates([
        {"action": "SET", "scope": "filesystem", "key": "/root/shifa",
         "value": {"type": "directory", "owner": "root", "permissions": "755"}},
    ])
    ctx = json.loads(manager.get_ssh_context("/root"))
    assert "shifa" in ctx
    assert ctx["shifa"]["type"] == "directory"


def test_get_ssh_context_delete(state_file):
    """DELETE action removes entries from the context."""
    manager = StateManager(state_file)
    manager.apply_updates([
        {"action": "SET", "scope": "filesystem", "key": "/root/tmp.txt",
         "value": {"type": "file", "content": "", "owner": "root", "permissions": "644"}},
    ])
    assert "tmp.txt" in json.loads(manager.get_ssh_context("/root"))

    manager.apply_updates([
        {"action": "DELETE", "scope": "filesystem", "key": "/root/tmp.txt"},
    ])
    assert "tmp.txt" not in json.loads(manager.get_ssh_context("/root"))


def test_get_ssh_context_cwd_normalisation(state_file):
    """Trailing slashes on cwd must not break path matching."""
    manager = StateManager(state_file)
    manager.apply_updates([
        {"action": "SET", "scope": "filesystem", "key": "/root/file.txt",
         "value": {"type": "file", "content": "", "owner": "root", "permissions": "644"}},
    ])
    ctx_trailing = json.loads(manager.get_ssh_context("/root/"))
    ctx_clean    = json.loads(manager.get_ssh_context("/root"))
    assert ctx_trailing == ctx_clean


# ── Session scope ──────────────────────────────────────────────────────────────

def test_session_scope_set_and_delete(state_file):
    """scope=\"session\" is stored under the \"sessions\" internal key."""
    manager = StateManager(state_file)
    manager.apply_updates([
        {"action": "SET", "scope": "session", "key": "cwd",
         "value": "/root/work"},
    ])
    assert manager.state["sessions"]["cwd"] == "/root/work"

    manager.apply_updates([
        {"action": "DELETE", "scope": "session", "key": "cwd"},
    ])
    assert "cwd" not in manager.state["sessions"]


def test_session_scope_persisted(state_file):
    """Session mutations are written to disk."""
    manager = StateManager(state_file)
    manager.apply_updates([
        {"action": "SET", "scope": "session", "key": "cwd", "value": "/tmp"},
    ])
    with open(state_file) as f:
        on_disk = json.load(f)
    assert on_disk["sessions"]["cwd"] == "/tmp"


# ── Scope isolation ────────────────────────────────────────────────────────────

def test_scopes_do_not_bleed(state_file):
    """Writing to one scope never affects another."""
    manager = StateManager(state_file)
    manager.apply_updates([
        {"action": "SET", "scope": "global",     "key": "g", "value": 1},
        {"action": "SET", "scope": "tokens",     "key": "t", "value": 2},
        {"action": "SET", "scope": "filesystem", "key": "/f", "value": {"type": "file"}},
        {"action": "SET", "scope": "session",    "key": "s", "value": 3},
    ])
    assert "g"  not in manager.state["tokens"]
    assert "t"  not in manager.state["global"]
    assert "/f" not in manager.state["global"]
    assert "s"  not in manager.state["filesystem"]