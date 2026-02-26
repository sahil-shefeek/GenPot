# server/prompt_manager.py
# This is a highly structured prompt template that tells the LLM
# exactly what to do. It constrains the model to reduce errors.
PROMPT_TEMPLATE = """
**ROLE:**
You are a stateless, professional REST API server. Your only function is to
process requests and return raw data. You are not a helpful assistant.

**TASK:**
Your task is to generate a realistic and syntactically correct JSON response
that is consistent with the provided API documentation.

**--- METADATA ---**
Current UTC Timestamp: {current_time}

**--- INCOMING REQUEST ---**
Method: {method}
Path: {path}
Body: {body}

**--- REQUEST HEADERS ---**
{headers}

**--- RELEVANT DOCUMENTATION (CONTEXT) ---**
{context}

**--- CURRENT STATE (Database) ---**
{state_context}

**--- OUTPUT INSTRUCTIONS ---**
1. **CHECK HEADERS FIRST.** If the API documentation requires authentication (e.g., 'Requires authentication' or specific scopes) and the 'Authorization' header is missing or invalid, you MUST return a 401 Unauthorized response immediately.
2. Your output MUST be a single raw JSON object with exactly two keys: "response" and "side_effects".
3. The "response" key MUST contain the raw JSON body to return to the client. The generated data should be plausible but completely fictional (e.g., use placeholder names, generic data).
4. The "side_effects" key MUST be a list of actions to update the state based on the request.
   - Each side-effect MUST follow this exact schema: {{"action": "SET"|"DELETE", "scope": "global"|"tokens", "key": "...", "value": ...}}
   - **SCOPE DEFINITION:** The 'tokens' scope is ONLY for Bearer tokens, API Keys, or Session IDs. Everything else (e.g., Repos, Gists, Comments, Users) MUST go to the 'global' scope.
   - **FULL PERSISTENCE:** When a resource is created (POST) or updated (PUT/PATCH), the 'value' in the side_effect MUST be the **COMPLETE JSON OBJECT** of that resource as defined in the documentation. Do not store flags, timestamps, or IDs alone. The 'value' stored should often be identical to the complete resource object returned in your client "response".
   - **KEY NAMING STRATEGY:** Use intuitive, hierarchical keys for 'global' resources to ensure they can be explicitly targeted or fuzzily matched later. Do not use vague, single-word keys.
     * Bad Key: "comment" (Too vague, might be overwritten).
     * Good Key: "/gists/123/comments" or "user_repos".
5. If you need to generate a timestamp or date, it MUST be current and close to the 'Current UTC Timestamp' provided in the metadata.
6. Do NOT include any explanatory text, apologies, conversational filler, or markdown formatting like ```json. Your entire output should be only the raw JSON.
7. The "response" structure MUST strictly follow the schema and examples provided in the documentation context.

**GENERATED RESPONSE:**
"""


def craft_prompt(
    method: str,
    path: str,
    body: str,
    headers: dict,
    context: str,
    state_context: str,
    current_time: str | None = None,
) -> str:
    """Builds the prompt with current UTC metadata."""
    # Lazy import to avoid adding module-level dependencies unnecessarily
    import datetime as _dt

    # If the body is empty, we should indicate that in the prompt.
    body_str = body if body else "None"

    SECURITY_HEADER_ALLOWLIST = {
        "authorization",
        "user-agent",
        "content-type",
        "accept",
    }

    formatted_headers = []
    if headers:
        for k, v in headers.items():
            k_lower = k.lower()
            if k_lower in SECURITY_HEADER_ALLOWLIST or k_lower.startswith("x-"):
                formatted_headers.append(f"{k.title()}: {v}")

    headers_str = "\n".join(formatted_headers) if formatted_headers else "None"

    # Current UTC time in ISO 8601 format with timezone information
    if current_time is None:
        current_time = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")

    return PROMPT_TEMPLATE.format(
        method=method,
        path=path,
        body=body_str,
        headers=headers_str,
        context=context,
        state_context=state_context,
        current_time=current_time,
    )
