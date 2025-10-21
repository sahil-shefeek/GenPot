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

**--- RELEVANT DOCUMENTATION (CONTEXT) ---**
{context}

**--- OUTPUT INSTRUCTIONS ---**
1. The response MUST be in raw JSON format.
2. The generated data should be plausible but completely fictional (e.g., use
   placeholder names, generic data).
3. If you need to generate a timestamp or date, it MUST be current and close to
   the 'Current UTC Timestamp' provided in the metadata.
4. Do NOT include any explanatory text, apologies, conversational filler, or
   markdown formatting like ```json. Your entire output should be only the raw
   JSON.
5. The response structure MUST strictly follow the schema and examples provided
   in the documentation context.

**GENERATED RESPONSE:**
"""


def craft_prompt(
    method: str,
    path: str,
    body: str,
    context: str,
    current_time: str | None = None,
) -> str:
    """Builds the prompt with current UTC metadata."""
    # Lazy import to avoid adding module-level dependencies unnecessarily
    import datetime as _dt

    # If the body is empty, we should indicate that in the prompt.
    body_str = body if body else "None"

    # Current UTC time in ISO 8601 format with timezone information
    if current_time is None:
        current_time = _dt.datetime.now(_dt.timezone.utc).isoformat(
            timespec="seconds"
        )

    return PROMPT_TEMPLATE.format(
        method=method,
        path=path,
        body=body_str,
        context=context,
        current_time=current_time,
    )
