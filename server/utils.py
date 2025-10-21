# server/utils.py
import json
import re
from typing import Any, Dict


def clean_llm_response(raw_text: str) -> Dict[str, Any]:
    """
    Strips markdown code fences from LLM output and parses JSON into a dict.
    Returns a fallback error payload if JSON parsing fails.
    """
    if not isinstance(raw_text, str):
        return {"error": "LLM returned non-text content", "raw": repr(raw_text)}

    text = raw_text.strip()

    # Remove enclosing triple-backtick fence
    # (with optional language, e.g., ```json)
    fence_pattern = re.compile(r"^\s*```(?:[a-zA-Z]+)?\s*([\s\S]*?)\s*```\s*$")
    match = fence_pattern.match(text)
    if match:
        text = match.group(1).strip()
    else:
        # Fallback removal if fences exist but not strictly wrapped
        text = (
            text.replace("```json", "")
            .replace("```JSON", "")
            .replace("```", "")
            .strip()
        )

    # Attempt strict JSON parsing
    try:
        return json.loads(text)
    except Exception as e:
        # Provide a safe, inspectable payload for the caller
        return {"error": f"Invalid JSON from LLM: {e}", "raw": text}
