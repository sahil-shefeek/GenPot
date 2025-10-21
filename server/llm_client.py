import os
from typing import Final

from dotenv import load_dotenv

try:
    # google-genai official SDK (pip install google-genai)
    from google import genai  # type: ignore
except Exception as _imp_err:
    raise RuntimeError(
        "google-genai is not installed. Add 'google-genai' to your "
        "dependencies or install it in your environment."
    ) from _imp_err

load_dotenv()

API_KEY: Final[str | None] = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise RuntimeError("GOOGLE_API_KEY is not set in environment (.env)")

MODEL_NAME: Final[str] = os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash")

# Initialize the official Google GenAI Python SDK client
try:
    CLIENT: Final = genai.Client(api_key=API_KEY)
except Exception as e:
    raise RuntimeError(f"Failed to initialize Google GenAI client: {e}") from e


def generate_response(prompt: str) -> str:
    """
    Call Gemini with the given prompt and return the raw text response.
    """
    try:
        resp = CLIENT.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
        )
        text = getattr(resp, "text", None)
        return (text or "").strip()
    except Exception as e:
        raise RuntimeError(f"Gemini generation failed: {e}") from e
