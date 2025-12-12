import json
import os
from pathlib import Path
from typing import Dict, Any

# Define paths relative to this file
BASE_DIR = Path(__file__).resolve().parent.parent
LOG_DIR = BASE_DIR / "logs"
CONFIG_FILE = LOG_DIR / "app_config.json"

DEFAULT_CONFIG = {
    "honeypot_provider": "gemini",
    "honeypot_model": "gemini-1.5-flash",
    "analysis_provider": "gemini",
    "analysis_model": "gemini-1.5-flash",
}


def load_config() -> Dict[str, Any]:
    """
    Load configuration from disk.
    If file doesn't exist, create it with defaults.
    """
    if not CONFIG_FILE.exists():
        save_config(DEFAULT_CONFIG)
        return DEFAULT_CONFIG

    try:
        with open(CONFIG_FILE, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        # Fallback to defaults if corrupted
        return DEFAULT_CONFIG


def save_config(new_config: Dict[str, Any]) -> None:
    """Save configuration to disk."""
    # Ensure logs directory exists
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    with open(CONFIG_FILE, "w") as f:
        json.dump(new_config, f, indent=4)
