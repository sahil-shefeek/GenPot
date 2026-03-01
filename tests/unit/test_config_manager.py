import pytest
from pathlib import Path
from unittest.mock import mock_open
from server import config_manager

VALID_YAML = """\
core:
  log_level: "INFO"

llm_defaults:
  provider: "gemini"
  model: "gemini-2.5-flash"
  analysis_provider: "gemini"
  analysis_model: "gemini-2.5-flash"

emulators:
  http:
    enabled: true
    port: 8000

  smtp:
    enabled: false
    port: 8025
    provider: "ollama"
    model: "phi4-mini"
"""


def test_load_config_valid(monkeypatch, mocker):
    """load_config() parses a valid YAML structure correctly."""
    # Point CONFIG_PATH to a tmp path that 'exists'
    fake_path = Path("/tmp/fake_genpot.yaml")
    monkeypatch.setattr(config_manager, "CONFIG_PATH", fake_path)
    mocker.patch("builtins.open", mock_open(read_data=VALID_YAML))
    mocker.patch.object(Path, "exists", return_value=True)

    config = config_manager.load_config()

    assert config["core"]["log_level"] == "INFO"
    assert config["llm_defaults"]["provider"] == "gemini"
    assert config["emulators"]["http"]["enabled"] is True
    assert config["emulators"]["smtp"]["port"] == 8025


def test_get_emulator_config_fallback(monkeypatch, mocker):
    """HTTP emulator inherits provider and model from llm_defaults."""
    fake_path = Path("/tmp/fake_genpot.yaml")
    monkeypatch.setattr(config_manager, "CONFIG_PATH", fake_path)
    mocker.patch("builtins.open", mock_open(read_data=VALID_YAML))
    mocker.patch.object(Path, "exists", return_value=True)

    http_config = config_manager.get_emulator_config("http")

    assert http_config["enabled"] is True
    assert http_config["port"] == 8000
    assert http_config["provider"] == "gemini"
    assert http_config["model"] == "gemini-2.5-flash"


def test_get_emulator_config_override(monkeypatch, mocker):
    """SMTP emulator overrides llm_defaults with its own provider/model."""
    fake_path = Path("/tmp/fake_genpot.yaml")
    monkeypatch.setattr(config_manager, "CONFIG_PATH", fake_path)
    mocker.patch("builtins.open", mock_open(read_data=VALID_YAML))
    mocker.patch.object(Path, "exists", return_value=True)

    smtp_config = config_manager.get_emulator_config("smtp")

    assert smtp_config["enabled"] is False
    assert smtp_config["port"] == 8025
    assert smtp_config["provider"] == "ollama"
    assert smtp_config["model"] == "phi4-mini"


def test_load_config_missing_file(monkeypatch, mocker):
    """load_config() raises FileNotFoundError when genpot.yaml is missing."""
    fake_path = Path("/tmp/nonexistent_genpot.yaml")
    monkeypatch.setattr(config_manager, "CONFIG_PATH", fake_path)
    mocker.patch.object(Path, "exists", return_value=False)

    with pytest.raises(FileNotFoundError, match="Configuration file not found"):
        config_manager.load_config()
