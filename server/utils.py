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


def parse_ssh_response(raw_text: str) -> Dict[str, Any]:
    """
    Splits raw LLM SSH terminal output from the optional ``<SIDE_EFFECT>`` tag.

    ``craft_ssh_prompt`` instructs the LLM to append a
    ``<SIDE_EFFECT>[…]</SIDE_EFFECT>`` tag after any command that mutates the
    filesystem or SSH session state.  This function extracts it so the caller
    can:

    * Write only the clean ``"output"`` text to the SSH channel.
    * Pass ``"side_effects"`` directly to ``StateManager.apply_updates()``.

    Returns
    -------
    dict
        ``{"output": str, "side_effects": list[dict]}``.
        Both keys are always present; ``side_effects`` is an empty list when
        the tag is absent or unparseable.
    """
    if not isinstance(raw_text, str):
        return {"output": "", "side_effects": []}

    pattern = re.compile(r"<SIDE_EFFECT>(.*?)</SIDE_EFFECT>", re.DOTALL)
    match = pattern.search(raw_text)

    side_effects: list = []
    if match:
        try:
            parsed = json.loads(match.group(1).strip())
            if isinstance(parsed, list):
                side_effects = parsed
            elif isinstance(parsed, dict):
                # Gracefully handle a single object instead of a list.
                side_effects = [parsed]
        except (json.JSONDecodeError, ValueError):
            pass
        # Strip the tag (and any surrounding whitespace) from the output.
        output = raw_text[: match.start()].strip()
    else:
        output = raw_text.strip()

    return {"output": output, "side_effects": side_effects}
