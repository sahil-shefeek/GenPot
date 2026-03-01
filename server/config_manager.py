import yaml

from pathlib import Path
from typing import Dict, Any

# Define paths relative to this file
BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = BASE_DIR / "config" / "genpot.yaml"


def load_config() -> Dict[str, Any]:
    """
    Load configuration from the YAML config file.
    Raises FileNotFoundError if genpot.yaml is missing.
    """
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {CONFIG_PATH}. "
            "Please create config/genpot.yaml before starting the server."
        )

    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    return config


def get_emulator_config(protocol: str) -> Dict[str, Any]:
    """
    Return the configuration for a specific emulator protocol.

    If the emulator block does not specify 'provider' or 'model',
    the values are injected from 'llm_defaults'.

    Raises:
        KeyError: If the requested protocol is not defined under 'emulators'.
    """
    config = load_config()
    emulators = config.get("emulators", {})

    if protocol not in emulators:
        raise KeyError(
            f"Emulator '{protocol}' is not defined in the configuration. "
            f"Available emulators: {list(emulators.keys())}"
        )

    emulator = dict(emulators[protocol])  # shallow copy
    defaults = config.get("llm_defaults", {})

    # Inject fallback values from llm_defaults
    if "provider" not in emulator:
        emulator["provider"] = defaults.get("provider")
    if "model" not in emulator:
        emulator["model"] = defaults.get("model")

    return emulator
