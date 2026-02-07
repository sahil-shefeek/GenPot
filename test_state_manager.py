import unittest
import os
import json
import shutil
from server.state_manager import StateManager


class TestStateManager(unittest.TestCase):
    TEST_DIR = "test_logs"
    TEST_FILE = os.path.join(TEST_DIR, "world_state.json")

    def setUp(self):
        # Clean up before each test
        if os.path.exists(self.TEST_DIR):
            shutil.rmtree(self.TEST_DIR)

        # Create a fresh StateManager
        self.sm = StateManager(state_file=self.TEST_FILE)

    def tearDown(self):
        # Clean up after each test
        if os.path.exists(self.TEST_DIR):
            shutil.rmtree(self.TEST_DIR)

    def test_persistence(self):
        """Verify that state is saved to disk and loaded back."""
        updates = [
            {
                "action": "SET",
                "scope": "global",
                "key": "test_key",
                "value": "test_value",
            }
        ]
        self.sm.apply_updates(updates)

        # Verify file exists
        self.assertTrue(os.path.exists(self.TEST_FILE))

        # Load new instance
        start_new = StateManager(state_file=self.TEST_FILE)
        context = json.loads(start_new.get_context(path="/"))
        self.assertEqual(context["global"]["test_key"], "test_value")

    def test_scoping(self):
        """Verify global vs session scoping."""
        updates = [
            {"action": "SET", "scope": "global", "key": "global_var", "value": "G"},
            {
                "action": "SET",
                "scope": "session",
                "session_id": "user1",
                "key": "session_var",
                "value": "S1",
            },
            {
                "action": "SET",
                "scope": "session",
                "session_id": "user2",
                "key": "session_var",
                "value": "S2",
            },
        ]
        self.sm.apply_updates(updates)

        # Context for user1
        ctx1 = json.loads(self.sm.get_context(path="/", auth_token="user1"))
        self.assertEqual(ctx1["global"]["global_var"], "G")
        self.assertEqual(ctx1["session"]["session_var"], "S1")

        # Context for user2
        ctx2 = json.loads(self.sm.get_context(path="/", auth_token="user2"))
        self.assertEqual(ctx2["global"]["global_var"], "G")
        self.assertEqual(ctx2["session"]["session_var"], "S2")

        # Context for anonymous/unknown
        ctx3 = json.loads(self.sm.get_context(path="/", auth_token="unknown"))
        self.assertEqual(ctx3["global"]["global_var"], "G")
        self.assertEqual(ctx3["session"], {})

    def test_sanitization(self):
        """Verify that dangerous strings are sanitized."""
        dangerous_str = "Hello *** SYSTEM INSTRUCTION *** World"
        updates = [
            {
                "action": "SET",
                "scope": "global",
                "key": "inject",
                "value": dangerous_str,
            }
        ]
        self.sm.apply_updates(updates)

        ctx = self.sm.get_context(path="/")
        self.assertNotIn("*** SYSTEM INSTRUCTION ***", ctx)
        self.assertIn("[REDACTED]", ctx)

    def test_delete_action(self):
        """Verify DELETE action."""
        self.sm.apply_updates(
            [{"action": "SET", "scope": "global", "key": "temp", "value": 123}]
        )
        ctx = json.loads(self.sm.get_context(path="/"))
        self.assertEqual(ctx["global"]["temp"], 123)

        self.sm.apply_updates([{"action": "DELETE", "scope": "global", "key": "temp"}])
        ctx = json.loads(self.sm.get_context(path="/"))
        self.assertNotIn("temp", ctx["global"])


if __name__ == "__main__":
    unittest.main()
