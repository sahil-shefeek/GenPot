import json
import pytest
from pathlib import Path
from server import config_manager


@pytest.fixture(autouse=True)
def mock_paths(monkeypatch, tmp_path):
    log_dir = tmp_path / "logs"
    config_file = log_dir / "app_config.json"
    monkeypatch.setattr(config_manager, "LOG_DIR", log_dir)
    monkeypatch.setattr(config_manager, "CONFIG_FILE", config_file)


def test_load_config_no_file(monkeypatch, tmp_path):
    # Verify file doesn't exist initially
    assert not config_manager.CONFIG_FILE.exists()

    config = config_manager.load_config()

    assert config == config_manager.DEFAULT_CONFIG
    assert config_manager.CONFIG_FILE.exists()
    with open(config_manager.CONFIG_FILE, "r") as f:
        saved_config = json.load(f)
    assert saved_config == config_manager.DEFAULT_CONFIG


def test_save_config():
    new_config = {"custom_key": "custom_value"}
    config_manager.save_config(new_config)

    assert config_manager.CONFIG_FILE.exists()
    with open(config_manager.CONFIG_FILE, "r") as f:
        saved_config = json.load(f)
    assert saved_config == new_config


def test_load_config_partial_merge():
    # Write a partial config
    config_manager.CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(config_manager.CONFIG_FILE, "w") as f:
        json.dump({"honeypot_model": "custom-model"}, f)

    config = config_manager.load_config()

    # Should merge DEFAULT_CONFIG and the partial config
    expected = config_manager.DEFAULT_CONFIG.copy()
    expected["honeypot_model"] = "custom-model"
    assert config == expected
