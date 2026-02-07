# server/prompt_manager.py
"""
Manages the construction of prompts for the LLM, incorporating state context
and enforcing strict JSON output for stateful interactions.
"""

PROMPT_TEMPLATE = """
**ROLE:**
You are a stateful REST API server. You must maintain internal consistency with the provided DATABASE STATE.
Your goal is to process the request, update the state if necessary, and return a JSON response.

**TASK:**
1. Analyze the Incoming Request and the Current Database State.
2. Determine if the request creates, updates, or deletes any resources.
3. Construct a JSON object containing the HTTP response and any state side effects.

**--- METADATA ---**
Current UTC Timestamp: {current_time}
(Use this timestamp for any 'created_at' or 'updated_at' fields)

**--- CURRENT DATABASE STATE ---**
{state_context}

**--- INCOMING REQUEST ---**
Method: {method}
Path: {path}
Body: {body}

**--- RELEVANT DOCUMENTATION (CONTEXT) ---**
{context}

**--- OUTPUT INSTRUCTIONS ---**
1. You must return a SINGLE valid JSON object.
2. DO NOT include any markdown formatting (like ```json ... ```) or explanatory text.
3. The JSON object must have exactly two keys: "response" and "side_effects".

**STRUCTURE:**
{{
  "response": {{
    // The actual HTTP response body to send to the client.
    // Must follow the API documentation schemas.
  }},
  "side_effects": [
    // A list of state updates to apply.
    {{
      "action": "SET",       // or "DELETE"
      "scope": "global",     // or "session"
      "key": "resource_name", // e.g., "users", "todos"
      "value": ...           // The NEW complete value for this key (snapshot).
    }}
  ]
}}

**RULES FOR SIDE EFFECTS:**
- If a resource is modified, provide the ENTIRE new state for that resource key (Snapshot approach).
- Example: If "users" is ["A"] and you add "B", value must be ["A", "B"].
- If no state changes are needed (e.g., GET request), "side_effects" should be an empty list [].
- "scope" must be either "global" or "session".
- For "session" scope, the system handles the session ID mapping automatically.

**ONE-SHOT EXAMPLE:**
Request: POST /users {{ "name": "Bob" }}
Current State: {{ "global": {{ "users": [{{ "name": "Alice" }}] }} }}

Output:
{{
  "response": {{
    "id": "user_123",
    "name": "Bob",
    "status": "created"
  }},
  "side_effects": [
    {{
      "action": "SET",
      "scope": "global",
      "key": "users",
      "value": [
        {{ "name": "Alice" }},
        {{ "name": "Bob" }}
      ]
    }}
  ]
}}

**GENERATED RESPONSE:**
"""


def craft_prompt(
    method: str,
    path: str,
    body: str,
    context: str,
    state_context: str | None = None,
    current_time: str | None = None,
) -> str:
    """
    Builds the prompt with current UTC metadata and state context.

    Args:
        method: HTTP method (GET, POST, etc.)
        path: Request path
        body: Request body (JSON string)
        context: API documentation context
        state_context: JSON string representing the current state (Global + Session).
                       Defaults to "{}" if None or empty.
        current_time: ISO 8601 timestamp. Defaults to current UTC time.

    Returns:
        The formatted prompt string.
    """
    # Lazy import to avoid adding module-level dependencies unnecessarily
    import datetime as _dt

    # If the body is empty, indicate that in the prompt
    body_str = body if body else "None"

    # Ensure state_context is a valid JSON object string string if empty
    if not state_context:
        state_context = "{}"

    # Current UTC time in ISO 8601 format
    if current_time is None:
        current_time = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")

    return PROMPT_TEMPLATE.format(
        method=method,
        path=path,
        body=body_str,
        context=context,
        state_context=state_context,
        current_time=current_time,
    )
