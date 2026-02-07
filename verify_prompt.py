# verify_prompt.py
import sys
import os

# Ensure server module is importable
sys.path.append(os.path.join(os.getcwd(), "server"))

try:
    from server.prompt_manager import craft_prompt
except ImportError:
    # Adjust import if running from root
    from server.prompt_manager import craft_prompt


def test_prompt_creation():
    method = "POST"
    path = "/api/resource"
    body = '{"foo": "bar"}'
    context = "API Documentation..."
    state_context = '{"global": {"users": ["Alice"]}}'

    prompt = craft_prompt(method, path, body, context, state_context)

    print("--- PROMPT START ---")
    print(prompt)
    print("--- PROMPT END ---")

    # Assertions
    if "**--- CURRENT DATABASE STATE ---**" not in prompt:
        print("FAIL: State section missing")
        return False

    if '"alice"' in prompt.lower() or "Alice" in prompt:
        print("PASS: State injected correctly")
    else:
        print("FAIL: State value missing")
        return False

    if '"side_effects"' not in prompt:
        print("FAIL: Side effects instructions missing")
        return False

    return True


if __name__ == "__main__":
    if test_prompt_creation():
        print("\nVerification Successful!")
    else:
        print("\nVerification Failed!")
        sys.exit(1)
