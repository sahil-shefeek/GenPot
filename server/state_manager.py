import json
import os
from typing import Dict, Any, List


class StateManager:
    def __init__(self, state_file: str = "logs/world_state.json"):
        self.state_file = state_file
        self.state = {"global": {}, "tokens": {}, "sessions": {}}
        self._load_state()

    def _load_state(self):
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, "r") as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        self.state["global"] = data.get("global", {})
                        self.state["tokens"] = data.get("tokens", {})
                        self.state["sessions"] = data.get("sessions", {})
            except (json.JSONDecodeError, IOError):
                pass

    def _save_state(self):
        dir_name = os.path.dirname(self.state_file)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        tmp_file = f"{self.state_file}.tmp"
        try:
            with open(tmp_file, "w") as f:
                persistent = {k: v for k, v in self.state.items() if k != "sessions"}
                json.dump(persistent, f, indent=2)
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

        # Zero-width space
        zws = "\u200b"

        replacements = {
            "***": f"*{zws}**",
            "---": f"-{zws}--",
            "SYSTEM:": f"SYS{zws}TEM:",
            "ROLE:": f"RO{zws}LE:",
            "USER:": f"US{zws}ER:",
            "ASSISTANT:": f"ASSIS{zws}TANT:",
            "Instruction:": f"Instruc{zws}tion:",
            "<|im_start|>": f"<|{zws}im_start|>",
            "<|im_end|>": f"<|{zws}im_end|>",
        }

        for target, replacement in replacements.items():
            json_str = json_str.replace(target, replacement)
            # Handle case variations for common labels
            if target.upper() != target.lower():
                json_str = json_str.replace(target.lower(), replacement.lower())
                json_str = json_str.replace(
                    target.capitalize(), replacement.capitalize()
                )

        return json_str

    def get_context(
        self, path: str, headers: Dict[str, str], session_id: str = None
    ) -> str:
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

        # 3. TCP Session State
        if session_id:
            tcp_session = self.state.get("sessions", {}).get(session_id)
            if tcp_session:
                context["current_session_state"] = tcp_session

        return self._sanitize_for_prompt(context)

    def apply_updates(self, side_effects: List[Dict[str, Any]]):
        """
        Applies a list of updates to the state and saves.
        """
        has_changes = False
        for update in side_effects:
            action = update.get("action")
            scope = update.get("scope")
            key = update.get("key")
            value = update.get("value")

            if scope not in ["global", "tokens", "sessions"] or not key:
                continue

            if action == "SET":
                self.state[scope][key] = value
                has_changes = True
            elif action == "DELETE":
                if key in self.state[scope]:
                    del self.state[scope][key]
                    has_changes = True

        if has_changes:
            self._save_state()

    def clear_session(self, session_id: str) -> None:
        """Remove transient session state (called on TCP disconnect)."""
        self.state.get("sessions", {}).pop(session_id, None)


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
