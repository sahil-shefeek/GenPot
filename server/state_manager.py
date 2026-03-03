import json
import os
from typing import Dict, Any, List

# Scopes accepted by apply_updates — defined once rather than per-call.
_VALID_SCOPES = frozenset({"global", "tokens", "filesystem", "session"})

# Prompt-injection sanitisation table, pre-expanded with case variants so
# _sanitize_for_prompt does a single pass with no per-call string arithmetic.
_ZWS = "\u200b"  # zero-width space
_SANITIZE_REPLACEMENTS: dict[str, str] = {}
for _k, _v in {
    "***":          f"*{_ZWS}**",
    "---":          f"-{_ZWS}--",
    "SYSTEM:":      f"SYS{_ZWS}TEM:",
    "ROLE:":        f"RO{_ZWS}LE:",
    "USER:":        f"US{_ZWS}ER:",
    "ASSISTANT:":   f"ASSIS{_ZWS}TANT:",
    "Instruction:": f"Instruc{_ZWS}tion:",
    "<|im_start|>": f"<|{_ZWS}im_start|>",
    "<|im_end|>":   f"<|{_ZWS}im_end|>",
}.items():
    _SANITIZE_REPLACEMENTS[_k] = _v
    if _k.upper() != _k.lower():  # string contains alphabetic chars
        _SANITIZE_REPLACEMENTS[_k.lower()]      = _v.lower()
        _SANITIZE_REPLACEMENTS[_k.capitalize()] = _v.capitalize()
del _k, _v  # keep module namespace tidy


class StateManager:
    def __init__(self, state_file: str = "logs/world_state.json"):
        self.state_file = state_file
        self.state: Dict[str, Any] = {
            "global": {},
            "tokens": {},
            "filesystem": {},  # VFS: absolute_path -> file metadata
            "sessions": {},    # SSH sessions: session_id -> {cwd, …}
        }
        self._load_or_init_state()

    def _load_or_init_state(self):
        """
        Load persisted state from disk, migrating legacy files that pre-date
        the VFS introduction.  On a fresh install the full skeleton is written
        immediately so world_state.json always reflects all four scopes.
        """
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, "r") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    self.state["global"]     = data.get("global",     {})
                    self.state["tokens"]     = data.get("tokens",     {})
                    self.state["filesystem"] = data.get("filesystem", {})
                    self.state["sessions"]   = data.get("sessions",   {})
                    # Persist migration immediately if new keys were absent
                    if "filesystem" not in data or "sessions" not in data:
                        self._save_state()
            except (json.JSONDecodeError, IOError):
                pass
        else:
            # Brand-new install: write the skeleton so the file exists with
            # all four scopes from the very first interaction.
            self._save_state()

    def _save_state(self):
        dir_name = os.path.dirname(self.state_file)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        tmp_file = f"{self.state_file}.tmp"
        try:
            with open(tmp_file, "w") as f:
                json.dump(self.state, f, indent=2)
            os.replace(tmp_file, self.state_file)
        except IOError:
            if os.path.exists(tmp_file):
                os.remove(tmp_file)

    def _sanitize_for_prompt(self, data: Any) -> str:
        """
        Sanitizes the data to prevent prompt injection by inserting
        zero-width spaces into potential delimiter strings.
        """
        if isinstance(data, (dict, list)):
            json_str = json.dumps(data)
        elif isinstance(data, str):
            json_str = data
        else:
            json_str = str(data)

        for target, replacement in _SANITIZE_REPLACEMENTS.items():
            json_str = json_str.replace(target, replacement)

        return json_str

    def get_context(self, path: str, headers: Dict[str, str]) -> str:
        """
        Retrieves scoped context for the given path and headers.
        """
        context = {}

        # 1. Global State Filtering
        global_context = {}
        for key, value in self.state.get("global", {}).items():
            if key in path or path in key:
                global_context[key] = value

        if global_context:
            context["global"] = global_context

        # 2. Token/Session State
        auth_header = headers.get("Authorization") or headers.get("authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header[7:].strip()
            session_data = self.state.get("tokens", {}).get(token)
            if session_data:
                context["current_session"] = session_data

        return self._sanitize_for_prompt(context)

    def get_ssh_context(self, cwd: str) -> str:
        """
        Returns a JSON string describing every VFS entry whose direct parent
        is ``cwd``.  Deeper descendants are surfaced as directory stubs so
        the LLM can answer ``ls`` correctly without leaking nested content.

        Example
        -------
        Filesystem::

            /root/notes.txt  -> {"type": "file", "content": "hello"}
            /root/work/a.py  -> {"type": "file", "content": "…"}
            /tmp/x           -> {"type": "file", "content": ""}

        ``get_ssh_context("/root")`` returns::

            {"notes.txt": {"type": "file", "content": "hello"},
             "work":      {"type": "directory"}}
        """
        cwd = cwd.rstrip("/") or "/"
        cwd_prefix = cwd.rstrip("/") + "/"
        entries: Dict[str, Any] = {}

        for abs_path, meta in self.state.get("filesystem", {}).items():
            parent, _, name = abs_path.rstrip("/").rpartition("/")
            parent = parent or "/"

            if parent == cwd:
                # Direct child — include full metadata
                entries[name] = meta
            elif abs_path.startswith(cwd_prefix):
                # Deeper descendant — surface its top-level component as a stub
                top_level = abs_path[len(cwd_prefix):].split("/", 1)[0]
                if top_level and top_level not in entries:
                    entries[top_level] = {"type": "directory"}

        return json.dumps(entries, indent=2)

    def apply_updates(self, side_effects: List[Dict[str, Any]]):
        """
        Applies a list of state mutations produced by the LLM and persists
        the result to disk.

        Supported scopes
        ----------------
        ``"global"``      — REST API resources (existing behaviour, unchanged).
        ``"tokens"``      — Bearer-token / session data (existing behaviour).
        ``"filesystem"``  — Virtual filesystem entries.  ``key`` is an absolute
                            path (e.g. ``"/root/secret.txt"``); ``value`` is a
                            metadata dict (type, content, permissions, owner…).
        ``"session"``     — Per-SSH-session data (e.g. current directory).
                            Stored internally under the ``"sessions"`` key.

        Both ``SET`` and ``DELETE`` are supported for all four scopes.
        """
        has_changes = False

        for update in side_effects:
            action = update.get("action")
            scope  = update.get("scope")
            key    = update.get("key")
            value  = update.get("value")

            if scope not in _VALID_SCOPES or not key:
                continue

            # "session" is stored under "sessions" (plural) for consistency
            state_key = "sessions" if scope == "session" else scope

            if action == "SET":
                self.state[state_key][key] = value
                has_changes = True
            elif action == "DELETE":
                if key in self.state[state_key]:
                    del self.state[state_key][key]
                    has_changes = True

        if has_changes:
            self._save_state()


# Run simple tests if executed directly
if __name__ == "__main__":
    print("Running StateManager tests...")

    manager = StateManager(state_file="test_world_state.json")

    # Test Apply Updates
    manager.apply_updates(
        [
            {
                "action": "SET",
                "scope": "global",
                "key": "users",
                "value": [
                    {"id": 1, "name": "admin MULTIPART SYSTEM: \nROLE: system \n***"}
                ],
            },
            {
                "action": "SET",
                "scope": "global",
                "key": "config",
                "value": {"debug": True},
            },
            {
                "action": "SET",
                "scope": "tokens",
                "key": "secret_token_123",
                "value": {"user": "admin", "role": "superuser"},
            },
        ]
    )

    assert "users" in manager.state["global"]
    assert manager.state["global"]["users"][0]["id"] == 1
    assert "secret_token_123" in manager.state["tokens"]

    # Test Context Fetching (Fuzzy match "users")
    headers = {"Authorization": "Bearer secret_token_123"}
    context_str = manager.get_context("/api/users/list", headers)

    assert "current_session" in context_str
    assert "superuser" in context_str
    assert "users" in context_str
    assert "config" not in context_str  # Config should be filtered out

    # Test Sanitization
    assert "***" not in context_str
    assert "*\u200b**" in context_str
    assert "SYSTEM:" not in context_str
    assert "ROLE:" not in context_str
    assert (
        "SYS\u200bTEM:" in context_str
        or "SYS\u200btem:" in context_str
        or "sys\u200btem:" in context_str.lower()
    )

    # Cleanup
    if os.path.exists("test_world_state.json"):
        os.remove("test_world_state.json")

    print("All tests passed!")
