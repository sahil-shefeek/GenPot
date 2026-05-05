from pathlib import Path
from unittest.mock import mock_open
from server import config_manager

VALID_YAML = """\
core:
  log_level: "INFO"

llm_defaults:
  provider: "gemini"
  model: "gemini-2.5-flash"
  temperature: 0.7
  thinking: false

emulators:
  http:
    enabled: true
    port: 8000

  smtp:
    enabled: false
    port: 8025
    provider: "ollama"
    model: "phi4-mini"
    temperature: 0.3
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
    """HTTP emulator inherits all LLM params from llm_defaults."""
    fake_path = Path("/tmp/fake_genpot.yaml")
    monkeypatch.setattr(config_manager, "CONFIG_PATH", fake_path)
    mocker.patch("builtins.open", mock_open(read_data=VALID_YAML))
    mocker.patch.object(Path, "exists", return_value=True)

    http_config = config_manager.get_emulator_config("http")

    assert http_config["enabled"] is True
    assert http_config["port"] == 8000
    assert http_config["provider"] == "gemini"
    assert http_config["model"] == "gemini-2.5-flash"
    assert http_config["temperature"] == 0.7
    assert http_config["thinking"] is False


def test_get_emulator_config_override(monkeypatch, mocker):
    """SMTP emulator overrides llm_defaults with its own values."""
    fake_path = Path("/tmp/fake_genpot.yaml")
    monkeypatch.setattr(config_manager, "CONFIG_PATH", fake_path)
    mocker.patch("builtins.open", mock_open(read_data=VALID_YAML))
    mocker.patch.object(Path, "exists", return_value=True)

    smtp_config = config_manager.get_emulator_config("smtp")

    assert smtp_config["enabled"] is False
    assert smtp_config["port"] == 8025
    assert smtp_config["provider"] == "ollama"
    assert smtp_config["model"] == "phi4-mini"
    # temperature is overridden in the SMTP block
    assert smtp_config["temperature"] == 0.3
    # thinking is NOT set in the SMTP block, so it cascades from llm_defaults
    assert smtp_config["thinking"] is False


def test_load_config_missing_file_creates_template(monkeypatch, tmp_path):
    """load_config() should auto-generate the file if it is missing."""
    config_dir = tmp_path / "config"
    fake_path = config_dir / "genpot.yaml"

    monkeypatch.setattr(config_manager, "CONFIG_DIR", config_dir)
    monkeypatch.setattr(config_manager, "CONFIG_PATH", fake_path)

    # Ensure it doesn't exist yet
    assert not fake_path.exists()

    # Trigger load, which should invoke generation
    config = config_manager.load_config()

    # Verify file was written and successfully parsed
    assert fake_path.exists()
    assert config["core"]["log_level"] == "INFO"
    assert config["llm_defaults"]["thinking"] is False
