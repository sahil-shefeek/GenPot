import pytest
from unittest.mock import MagicMock, patch

from server.llm_client import (
    GeminiProvider,
    OllamaProvider,
    LLMRateLimitError,
    LLMProviderError,
    generate_response,
    list_available_models,
)


@pytest.fixture
def mock_genai():
    # We patch sys.modules to return a MagicMock for google.genai
    # We yield the mock so we can configure it in the test
    mock_google = MagicMock()
    mock_genai_module = MagicMock()
    mock_google.genai = mock_genai_module
    with patch.dict(
        "sys.modules", {"google": mock_google, "google.genai": mock_genai_module}
    ):
        yield mock_genai_module


@pytest.fixture
def mock_ollama():
    mock_ollama_module = MagicMock()
    with patch.dict("sys.modules", {"ollama": mock_ollama_module}):
        yield mock_ollama_module


# --- Step 1: Testing GeminiProvider ---


def test_gemini_successful_generation(monkeypatch, mock_genai):
    monkeypatch.setenv("GOOGLE_API_KEY", "dummy_key")

    mock_client_instance = mock_genai.Client.return_value
    mock_response = MagicMock()
    mock_response.text = "mocked gemini response"
    mock_client_instance.models.generate_content.return_value = mock_response

    provider = GeminiProvider()
    result = provider.generate("test")

    assert result == "mocked gemini response"
    mock_client_instance.models.generate_content.assert_called_once_with(
        model=provider.model_name, contents="test"
    )


def test_gemini_rate_limit_handling(monkeypatch, mock_genai):
    monkeypatch.setenv("GOOGLE_API_KEY", "dummy_key")

    mock_client_instance = mock_genai.Client.return_value
    mock_client_instance.models.generate_content.side_effect = Exception(
        "429 Resource exhausted: Please retry in 45s"
    )

    provider = GeminiProvider()

    with pytest.raises(LLMRateLimitError) as exc_info:
        provider.generate("test")

    assert exc_info.value.retry_after == "45s"


def test_gemini_generic_error_handling(monkeypatch, mock_genai):
    monkeypatch.setenv("GOOGLE_API_KEY", "dummy_key")

    mock_client_instance = mock_genai.Client.return_value
    mock_client_instance.models.generate_content.side_effect = Exception(
        "Unknown server error"
    )

    provider = GeminiProvider()

    with pytest.raises(LLMProviderError):
        provider.generate("test")


def test_gemini_missing_api_key(monkeypatch, mock_genai):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        GeminiProvider()


def test_gemini_list_models(monkeypatch, mock_genai):
    monkeypatch.setenv("GOOGLE_API_KEY", "dummy_key")
    provider = GeminiProvider()
    models = provider.list_models()

    assert isinstance(models, list)
    assert "gemini-1.5-flash" in models


# --- Step 2: Testing OllamaProvider ---


def test_ollama_successful_generation(mock_ollama):
    mock_client_instance = mock_ollama.Client.return_value
    mock_client_instance.generate.return_value = {"response": "mocked ollama response"}

    provider = OllamaProvider()
    result = provider.generate("test")

    assert result == "mocked ollama response"
    mock_client_instance.generate.assert_called_once_with(
        model="phi4-mini", prompt="test"
    )


def test_ollama_exception_handling(mock_ollama):
    mock_client_instance = mock_ollama.Client.return_value
    mock_client_instance.generate.side_effect = Exception("Connection refused")

    provider = OllamaProvider()

    with pytest.raises(RuntimeError):
        provider.generate("test")


def test_ollama_list_models_dict_response(mock_ollama):
    mock_client_instance = mock_ollama.Client.return_value
    mock_client_instance.list.return_value = {
        "models": [{"name": "phi4-mini"}, {"name": "llama3"}]
    }

    provider = OllamaProvider()
    models = provider.list_models()

    assert models == ["phi4-mini", "llama3"]


def test_ollama_list_models_object_response(mock_ollama):
    mock_client_instance = mock_ollama.Client.return_value

    class MockModel:
        def __init__(self, model):
            self.model = model

    class MockListResponse:
        def __init__(self):
            self.models = [MockModel("phi4-mini"), MockModel("llama3")]

    mock_client_instance.list.return_value = MockListResponse()

    provider = OllamaProvider()
    models = provider.list_models()

    assert models == ["phi4-mini", "llama3"]


# --- Step 3: Testing Factory Methods ---


@patch("server.llm_client.GeminiProvider.__init__", return_value=None)
@patch("server.llm_client.GeminiProvider.generate", return_value="gemini response")
@patch("server.llm_client.OllamaProvider.__init__", return_value=None)
@patch("server.llm_client.OllamaProvider.generate", return_value="ollama response")
def test_generate_response_router(
    mock_ollama_generate, mock_ollama_init, mock_gemini_generate, mock_gemini_init
):
    # Test Gemini
    result_gemini = generate_response("test", provider_type="gemini")
    assert result_gemini == "gemini response"
    mock_gemini_generate.assert_called_once_with("test")

    # Test Ollama
    result_ollama = generate_response("test", provider_type="ollama")
    assert result_ollama == "ollama response"
    mock_ollama_generate.assert_called_once_with("test")

    # Test Unknown
    with pytest.raises(ValueError):
        generate_response("test", provider_type="unknown")


@patch("server.llm_client.GeminiProvider.__init__", return_value=None)
@patch("server.llm_client.GeminiProvider.list_models", return_value=["gemini-1"])
@patch("server.llm_client.OllamaProvider.__init__", return_value=None)
@patch("server.llm_client.OllamaProvider.list_models", return_value=["ollama-1"])
def test_list_available_models_router(
    mock_ollama_list, mock_ollama_init, mock_gemini_list, mock_gemini_init
):
    # Test Gemini
    models_gemini = list_available_models(provider_type="gemini")
    assert models_gemini == ["gemini-1"]
    mock_gemini_list.assert_called_once()

    # Test Ollama
    models_ollama = list_available_models(provider_type="ollama")
    assert models_ollama == ["ollama-1"]
    mock_ollama_list.assert_called_once()
