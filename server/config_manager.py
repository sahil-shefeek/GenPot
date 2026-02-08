import json

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
    "generator_provider": "gemini",
    "generator_model": "gemini-2.5-flash",
    "rag_top_k": 3,
    "rag_similarity_threshold": 0.75,
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
            loaded_config = json.load(f)
            # Merge defaults with loaded config (defaults serve as base, loaded overrides)
            return {**DEFAULT_CONFIG, **loaded_config}
    except (json.JSONDecodeError, IOError):
        # Fallback to defaults if corrupted
        return DEFAULT_CONFIG


def save_config(new_config: Dict[str, Any]) -> None:
    """Save configuration to disk."""
    # Ensure logs directory exists
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    with open(CONFIG_FILE, "w") as f:
        json.dump(new_config, f, indent=4)
