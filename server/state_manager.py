import json
import os
import threading
from typing import Dict, Any, List, Optional
import copy


class StateManager:
    def __init__(self, state_file: str = "logs/world_state.json"):
        """
        Initialize the StateManager.
        :param state_file: Path to the JSON file for persistence.
        """
        self.state_file = state_file
        # Ensure the directory exists
        os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
        self.lock = threading.Lock()  # Ensure thread safety for file writes
        self.state = self._load_state()

    def _load_state(self) -> Dict[str, Any]:
        """
        Load state from the JSON file. If file doesn't exist or is invalid, return default structure.
        Default structure: {'global': {}, 'sessions': {}}
        """
        default_state = {"global": {}, "sessions": {}}
        if not os.path.exists(self.state_file):
            return default_state

        try:
            with open(self.state_file, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            # Fallback if file is corrupted
            return default_state

    def _save_state(self):
        """
        Save the current state to the JSON file.
        Uses a thread lock to prevent race conditions during writes.
        """
        try:
            with open(self.state_file, "w") as f:
                json.dump(self.state, f, indent=2)
        except IOError as e:
            print(f"Error saving state: {e}")

    def get_context(self, path: str, auth_token: Optional[str] = None) -> str:
        """
        Retrieve the combined context (Global + Session) for the LLM.
        :param path: The path of the request (e.g., /api/login).
        :param auth_token: The Bearer token from the Authorization header (acts as session ID).
        :return: JSON string of the combined state, sanitized.
        """
        with self.lock:
            # Deep copy to avoid accidental modification of internal state
            context = {
                "global": copy.deepcopy(self.state.get("global", {})),
                "session": {},
            }

            if auth_token:
                session_data = self.state.get("sessions", {}).get(auth_token, {})
                context["session"] = copy.deepcopy(session_data)

        # Sanitize before returning
        return json.dumps(self._sanitize_for_prompt(context), indent=2)

    def apply_updates(self, updates: List[Dict[str, Any]]):
        """
        Apply a list of updates to the state.
        :param updates: List of dicts, e.g.,
               {'action': 'SET', 'scope': 'global', 'key': 'users', 'value': [...]}
               {'action': 'SET', 'scope': 'session', 'session_id': 'abc', 'key': 'last_login', 'value': '...'}
        Supported actions: SET (create/update), DELETE
        """
        with self.lock:
            modified = False
            for update in updates:
                action = update.get("action")
                scope = update.get("scope")  # 'global' or 'session'
                key = update.get("key")
                value = update.get("value")

                if not action or not scope or not key:
                    continue

                target_dict = None
                if scope == "global":
                    target_dict = self.state.setdefault("global", {})
                elif scope == "session":
                    session_id = update.get("session_id")
                    if session_id:
                        target_dict = self.state.setdefault("sessions", {}).setdefault(
                            session_id, {}
                        )

                if target_dict is None:
                    continue

                if action == "SET":
                    target_dict[key] = value
                    modified = True
                elif action == "DELETE":
                    if key in target_dict:
                        del target_dict[key]
                        modified = True

            if modified:
                self._save_state()

    def _sanitize_for_prompt(self, data: Any) -> Any:
        """
        Recursively sanitize data to prevent prompt injection.
        Removes/escapes dangerous delimiters like '*** SYSTEM INSTRUCTION ***'.
        """
        if isinstance(data, dict):
            return {k: self._sanitize_for_prompt(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._sanitize_for_prompt(item) for item in data]
        elif isinstance(data, str):
            # Simple sanitization strategies:
            # 1. Replace potentially dangerous sequences
            sanitized = data.replace("*** SYSTEM INSTRUCTION ***", "[REDACTED]")
            # We can add more specific rules here as needed
            return sanitized
        else:
            return data
