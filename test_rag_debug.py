import json
import sys
from fastapi.testclient import TestClient

# Add project root to sys.path to ensure imports work
import os

sys.path.append(os.getcwd())

from server.main import app

client = TestClient(app)


def test_rag_debug():
    print("Testing POST /api/debug/rag using TestClient...")

    payload = {"query": "login issues", "top_k": 3, "threshold": 0.0}

    try:
        response = client.post("/api/debug/rag", json=payload)

        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            context = data.get("context", "")
            metadata = data.get("metadata", {})
            retrieved = metadata.get("retrieved_chunks", [])

            print(f"Chunks Retrieved: {len(retrieved)}")
            print("Metadata Keys:", list(metadata.keys()))

            if len(retrieved) > 0:
                print("First Chunk Score:", retrieved[0].get("score"))
                print("First Chunk Text Snippet:", retrieved[0].get("text")[:50])
            else:
                print("WARN: No chunks retrieved (index might be empty).")

            return True
        else:
            print("Error Response:", response.text)
            return False

    except Exception as e:
        print(f"An error occurred: {e}")
        return False


if __name__ == "__main__":
    success = test_rag_debug()
    sys.exit(0 if success else 1)
