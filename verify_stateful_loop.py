import os
import json
import shutil
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

# Clean up previous state
if os.path.exists("logs/world_state.json"):
    os.remove("logs/world_state.json")

# Import app after cleanup to ensure fresh state load if possible (though StateManager loads in __init__)
from server.main import app, state_manager

client = TestClient(app)


def test_stateful_flow():
    print("--- Starting Verification ---")

    # Mock RAG to return static context + metadata
    with patch(
        "server.main.rag_system.get_context", return_value=("Mock RAG Context", {})
    ):
        # Test 1: Successful Stateful Update
        print("\n[Test 1] Stateful Update (POST /users)")
        stateful_response = {
            "response": {"id": 1, "status": "created"},
            "side_effects": [
                {
                    "action": "SET",
                    "scope": "global",
                    "key": "users",
                    "value": [{"name": "Alice", "id": 1}],
                }
            ],
        }

        with patch(
            "server.main.generate_response", return_value=json.dumps(stateful_response)
        ) as mock_llm:
            response = client.post(
                "/users",
                json={"name": "Alice"},
                headers={"Authorization": "Bearer session_123"},
            )

            print(f"Status: {response.status_code}")
            print(f"Response: {response.json()}")

            assert response.status_code == 200
            assert response.json() == {"id": 1, "status": "created"}

            # Check if state was updated
            current_state = state_manager.get_context("/users", "session_123")
            print(f"Current State: {current_state}")
            assert "Alice" in current_state

        # Test 2: Fallback (LLM returns just body)
        print("\n[Test 2] Fallback - Just Body (GET /users)")
        fallback_response = {"users": [{"name": "Alice", "id": 1}]}

        with patch(
            "server.main.generate_response", return_value=json.dumps(fallback_response)
        ):
            response = client.get(
                "/users", headers={"Authorization": "Bearer session_123"}
            )

            print(f"Status: {response.status_code}")
            print(f"Response: {response.json()}")

            assert response.status_code == 200
            assert response.json() == fallback_response

        # Test 3: Error Fallback (LLM returns invalid JSON)
        print("\n[Test 3] Error Fallback - Invalid JSON")
        invalid_json = "I am not a JSON object."

        with patch("server.main.generate_response", return_value=invalid_json):
            response = client.post(
                "/test", json={}, headers={"Authorization": "Bearer session_123"}
            )

            print(f"Status: {response.status_code}")
            print(f"Response: {response.json()}")

            assert response.status_code == 200
            assert response.json() == {"raw_response": invalid_json}

    print("\n--- Verification Successful ---")


if __name__ == "__main__":
    test_stateful_flow()
