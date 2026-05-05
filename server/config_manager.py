import yaml

from pathlib import Path
from typing import Dict, Any

# Define paths relative to this file
BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = BASE_DIR / "config"
CONFIG_PATH = CONFIG_DIR / "genpot.yaml"

DEFAULT_YAML_TEMPLATE = """\
# =============================================================================
# GenPot Main Configuration
# =============================================================================
core:
  log_level: "INFO"

# =============================================================================
# Global LLM Parameters
# These apply to all emulators globally unless specifically overridden below.
# =============================================================================
llm_defaults:
  provider: "gemini"               # Supported: "gemini", "ollama"
  model: "gemini-2.5-flash"
  temperature: 0.7                 # 0.0 (deterministic) to 2.0 (highly creative)
  thinking: false                  # Enable/Disable chain-of-thought (<think> blocks)

# =============================================================================
# Protocol Emulators
# =============================================================================
emulators:
  http:
    enabled: true
    port: 8000
    # --- Optional Overrides ---
    # provider: "ollama"
    # model: "phi4-mini"
    # temperature: 0.5
    # thinking: false

  smtp:
    enabled: true
    port: 8025
    # --- Optional Overrides ---
    # provider: "ollama"
    # model: "phi4-mini"
    # temperature: 0.3
    # thinking: true
"""

_REQUIRED_TOP_LEVEL_KEYS = {"core", "llm_defaults", "emulators"}


def _write_default_template():
    """Write the default YAML template to CONFIG_PATH."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        f.write(DEFAULT_YAML_TEMPLATE)


def _ensure_config_exists():
    """Create the default config file if it does not exist."""
    if not CONFIG_PATH.exists():
        _write_default_template()
        print(f"[*] Auto-generated default configuration at {CONFIG_PATH}")


def _validate_config(config: Any) -> bool:
    """Return True if *config* is a dict containing all required top-level keys."""
    return isinstance(config, dict) and _REQUIRED_TOP_LEVEL_KEYS.issubset(config)


def _repair_config():
    """Back up the broken config file and regenerate the default template."""
    backup_path = CONFIG_PATH.with_suffix(".yaml.bak")
    try:
        CONFIG_PATH.rename(backup_path)
        print(
            f"[!] Corrupt configuration backed up to {backup_path}"
        )
    except OSError:
        pass  # If rename fails, overwrite in place
    _write_default_template()
    print(f"[*] Regenerated default configuration at {CONFIG_PATH}")


def load_config() -> Dict[str, Any]:
    """Load configuration from the YAML config file.

    * If the file is missing, a default template is generated.
    * If the file is empty, corrupt, or structurally invalid, it is backed
      up to ``genpot.yaml.bak`` and a fresh template is written.
    """
    _ensure_config_exists()

    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError:
        config = None

    if not _validate_config(config):
        _repair_config()
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

    return config


def get_emulator_config(protocol: str) -> Dict[str, Any]:
    """
    Return the configuration for a specific emulator protocol.
    Cascades missing LLM parameters from 'llm_defaults'.
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

    # Neatly inject fallback values for all LLM parameters if omitted
    # in the emulator block
    llm_keys = ["provider", "model", "temperature", "thinking"]
    for key in llm_keys:
        if key not in emulator and key in defaults:
            emulator[key] = defaults[key]

    return emulator
