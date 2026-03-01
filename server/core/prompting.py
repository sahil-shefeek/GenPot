# server/core/prompting.py
"""
Prompt Strategy Pattern for GenPot.

Provides a pluggable interface for building LLM prompts and parsing responses
across different protocols. The HttpPromptStrategy applies the "Sandwich
Technique" to mitigate Lost-in-the-Middle LLM degradation.
"""

import datetime as _dt
from abc import ABC, abstractmethod
from typing import Any, Dict

from server.utils import clean_llm_response

# ---------------------------------------------------------------------------
# Strategy Interface
# ---------------------------------------------------------------------------


class PromptStrategy(ABC):
    """Abstract base for protocol-specific prompt strategies."""

    @abstractmethod
    def build_prompt(self, request: dict, context: str, state: str) -> str:
        """Build a complete LLM prompt from request data, RAG context, and
        world state."""

    @abstractmethod
    def parse_response(self, raw_text: str) -> dict:
        """Parse raw LLM output into a normalised ``{"response": …,
        "side_effects": …}`` dict."""


# ---------------------------------------------------------------------------
# Sandwich-Technique Prompt Template
# ---------------------------------------------------------------------------
# TOP    → System role + strict instructions  (Primacy Bias)
# MIDDLE → RAG context + world state          (noisy data)
# BOTTOM → Exact request + output schema      (Recency Bias)
# ---------------------------------------------------------------------------

_HTTP_PROMPT_TEMPLATE = """\
**SYSTEM ROLE:**
You are a stateless, professional REST API server. Your only function is to \
process requests and return raw data. You are not a helpful assistant.

**TASK:**
Your task is to generate a realistic and syntactically correct JSON response \
that is consistent with the provided API documentation.

**STRICT RULES (read carefully):**
1. CHECK HEADERS FIRST. If the documentation requires authentication and the \
'Authorization' header is missing or invalid, return a 401 Unauthorized \
response immediately.
2. Your output MUST be a single raw JSON object with exactly two keys: \
"response" and "side_effects".
3. Do NOT include any explanatory text, apologies, conversational filler, or \
markdown formatting like ```json.

**METADATA:**
Current UTC Timestamp: {current_time}

**--- RELEVANT DOCUMENTATION (CONTEXT) ---**
{context}

**--- CURRENT STATE (Database) ---**
{state_context}

**--- INCOMING REQUEST ---**
Method: {method}
Path: {path}
Headers:
{headers}
Body: {body}

**--- OUTPUT SCHEMA (you MUST follow this exactly) ---**
Your entire output must be ONLY the following raw JSON structure:
{{
  "response": <the raw JSON body to return to the client>,
  "side_effects": [
    {{"action": "SET"|"DELETE", "scope": "global"|"tokens", "key": "...", "value": ...}}
  ]
}}

SCOPE DEFINITION: The 'tokens' scope is ONLY for Bearer tokens, API Keys, or \
Session IDs. Everything else MUST go to the 'global' scope.
FULL PERSISTENCE: When a resource is created (POST) or updated (PUT/PATCH), \
the 'value' MUST be the COMPLETE JSON OBJECT of that resource.
KEY NAMING: Use intuitive, hierarchical keys (e.g., "/gists/123/comments").
TIMESTAMPS: Any generated timestamp MUST be close to the Current UTC Timestamp.
The "response" structure MUST strictly follow the schema in the documentation.

**GENERATE YOUR JSON RESPONSE NOW:**
"""


# Header allowlist (lowercase) — kept identical to legacy craft_prompt
_SECURITY_HEADER_ALLOWLIST = frozenset(
    {"authorization", "user-agent", "content-type", "accept"}
)


# ---------------------------------------------------------------------------
# HTTP Strategy
# ---------------------------------------------------------------------------


class HttpPromptStrategy(PromptStrategy):
    """Builds HTTP-specific prompts using the Sandwich Technique and parses
    LLM responses into normalised dicts."""

    # ----- prompt building --------------------------------------------------

    def build_prompt(self, request: dict, context: str, state: str) -> str:
        """Return a fully-rendered prompt string.

        ``request`` must contain keys: ``method``, ``path``, ``body``,
        ``headers`` and optionally ``current_time``.
        """
        method = request.get("method", "GET")
        path = request.get("path", "/")
        body = request.get("body") or "None"
        headers = request.get("headers") or {}
        current_time = request.get("current_time")

        if current_time is None:
            current_time = _dt.datetime.now(_dt.timezone.utc).isoformat(
                timespec="seconds"
            )

        headers_str = self._format_headers(headers)

        return _HTTP_PROMPT_TEMPLATE.format(
            current_time=current_time,
            context=context,
            state_context=state,
            method=method,
            path=path,
            headers=headers_str,
            body=body,
        )

    # ----- response parsing -------------------------------------------------

    def parse_response(self, raw_text: str) -> Dict[str, Any]:
        """Clean raw LLM text and guarantee the canonical schema.

        Always returns ``{"response": …, "side_effects": […]}``.
        """
        parsed = clean_llm_response(raw_text)

        if (
            isinstance(parsed, dict)
            and "response" in parsed
            and "side_effects" in parsed
        ):
            return parsed

        # Wrap non-conforming output in the canonical envelope
        return {"response": parsed, "side_effects": []}

    # ----- internal helpers -------------------------------------------------

    @staticmethod
    def _format_headers(headers: dict) -> str:
        """Filter and format headers using the security allowlist."""
        lines: list[str] = []
        for key, value in headers.items():
            k_lower = key.lower()
            if k_lower in _SECURITY_HEADER_ALLOWLIST or k_lower.startswith("x-"):
                lines.append(f"{key.title()}: {value}")
        return "\n".join(lines) if lines else "None"
