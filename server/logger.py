# server/logger.py
import datetime
import json
from pathlib import Path

# --- CONFIGURATION ---
# Use pathlib for robust path management. This finds the project root
# and creates the logs folder if it doesn't exist.
LOGS_DIR = Path(__file__).resolve().parents[1] / "logs"
LOGS_DIR.mkdir(exist_ok=True)
LOG_FILE = LOGS_DIR / "honeypot.jsonl"


def log_interaction(log_data: dict):
    """
    Appends a new interaction log as a single JSON line to the log file.
    This function is the single point of entry for all logging in the application.
    """
    try:
        # Ensure every log entry has a consistent, timezone-aware timestamp
        log_data["timestamp"] = datetime.datetime.now(
            datetime.timezone.utc
        ).isoformat()

        # Open the log file in "append" mode and write the new log entry
        with open(LOG_FILE, "a") as f:
            # The json.dumps() function converts a Python dictionary to a JSON string.
            # The newline character '\n' is what makes it a JSON Lines (.jsonl) file.
            f.write(json.dumps(log_data) + "\n")

    except Exception as e:
        # It's important to print errors related to logging to the console
        # so that we know if the logging system itself fails.
        print(f"[!] CRITICAL: Logging failed! Error: {e}")
        print(f"    Failed to log data: {log_data}")
