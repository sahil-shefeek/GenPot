import os
from abc import ABC, abstractmethod
from typing import List, Optional

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- Interfaces ---


class LLMProvider(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generates text based on the prompt."""
        pass

    @abstractmethod
    def list_models(self) -> List[str]:
        """Lists available models for this provider."""
        pass


# --- Exceptions ---


class LLMError(Exception):
    """Base exception for all LLM errors."""

    pass


class LLMRateLimitError(LLMError):
    """Raised when the LLM provider's rate limit is exceeded."""

    def __init__(self, message: str, retry_after: Optional[str] = None):
        super().__init__(message)
        self.retry_after = retry_after


class LLMContextError(LLMError):
    """Raised when the prompt exceeds the model's context window."""

    pass


class LLMProviderError(LLMError):
    """Raised for generic provider errors."""

    pass


# --- Implementations ---


class GeminiProvider(LLMProvider):
    def __init__(self, model_name: Optional[str] = None):
        try:
            from google import genai
        except ImportError as e:
            raise RuntimeError(
                "google-genai is not installed. Run 'uv add google-genai'."
            ) from e

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY is not set in environment (.env)")

        # Default to a safe fallback if not provided
        self.model_name = model_name or os.getenv(
            "GEMINI_MODEL_NAME", "gemini-2.0-flash-exp"
        )
        try:
            self.client = genai.Client(api_key=api_key)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Google GenAI client: {e}") from e

    def generate(self, prompt: str) -> str:
        try:
            resp = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
            )
            # Handle potential different response structures if needed,
            # but usually .text access is standard for this SDK.
            text = getattr(resp, "text", None)
            return (text or "").strip()
        except Exception as e:
            error_msg = str(e)
            
            # Check for Rate Limit / Quota Exhausted
            if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg or "Resource exhausted" in error_msg:
                # Try to extract retry delay
                retry_after = None
                import re
                # Pattern for "Please retry in 43s" or similar
                match = re.search(r"retry in (\d+(\.\d+)?[sm])", error_msg, re.IGNORECASE)
                if match:
                    retry_after = match.group(1)
                
                raise LLMRateLimitError(
                    f"Gemini quota exceeded: {error_msg}", 
                    retry_after=retry_after
                ) from e
            
            # Generic Provider Error
            raise LLMProviderError(f"Gemini generation failed: {e}") from e

    def list_models(self) -> List[str]:
        # Static list as requested for now
        return [
            "gemini-3-pro-preview",
            "gemini-2.5-flash",
            "gemini-2.5-pro",
            "gemini-2.0-flash",
            "gemini-1.5-pro",
            "gemini-1.5-flash",
        ]


class OllamaProvider(LLMProvider):
    def __init__(self, model_name: str = "phi4-mini"):
        try:
            import ollama
        except ImportError as e:
            raise RuntimeError("ollama is not installed. Run 'uv add ollama'.") from e

        # Default to localhost for local dev; Docker will override this env var.
        host = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.client = ollama.Client(host=host)
        self.model_name = model_name

    def generate(self, prompt: str) -> str:
        try:
            resp = self.client.generate(model=self.model_name, prompt=prompt)
            return resp.get("response", "").strip()
        except Exception as e:
            raise RuntimeError(f"Ollama generation failed: {e}") from e

    def list_models(self) -> List[str]:
        try:
            models_info = self.client.list()
            # print(f"DEBUG OLLAMA LIST: {models_info}") # Uncomment for deep debugging

            # Handle response (it might be a dict or an object with a .models attribute)
            models_list = (
                getattr(models_info, "models", [])
                if not isinstance(models_info, dict)
                else models_info.get("models", [])
            )

            model_names = []
            for m in models_list:
                # Try specific object attributes first (new library version), then dict key (old version)
                # 'model' is often the key in newer versions, 'name' in older or different contexts.
                name = (
                    getattr(m, "model", None)
                    or getattr(m, "name", None)
                    or (m.get("model") if isinstance(m, dict) else None)
                    or (m.get("name") if isinstance(m, dict) else None)
                )
                if name:
                    model_names.append(name)
            return model_names
        except Exception as e:
            # Fallback or empty if connection fails
            print(f"Warning: Failed to list Ollama models: {e}")
            return []


# --- Factory / Wrapper ---


def generate_response(
    prompt: str, provider_type: str = "gemini", model_name: Optional[str] = None
) -> str:
    """
    Unified entry point for LLM generation.
    """
    provider: LLMProvider

    if provider_type.lower() == "gemini":
        provider = GeminiProvider(model_name=model_name)
    elif provider_type.lower() == "ollama":
        # Default to phi4-mini if not specified for Ollama
        provider = OllamaProvider(model_name=model_name or "phi4-mini")
    else:
        raise ValueError(f"Unknown provider type: {provider_type}")

    return provider.generate(prompt)


def list_available_models(provider_type: str = "gemini") -> List[str]:
    """
    Helper to list models for a given provider.
    """
    if provider_type.lower() == "gemini":
        return GeminiProvider().list_models()
    elif provider_type.lower() == "ollama":
        return OllamaProvider().list_models()
    return []
