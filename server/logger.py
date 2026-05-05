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


def _build_ecs_entry(
    protocol: str,
    source_ip: str,
    request_data: dict,
    response_data: dict,
    genpot_metrics: dict,
    error: str = None,
) -> dict:
    """
    Assembles an Elastic Common Schema (ECS) compliant log entry.

    This is kept as a pure function (no I/O) to simplify unit testing.
    """
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%S.%f"
    )[:-3] + "Z"

    entry = {
        "@timestamp": timestamp,
        "event": {
            "action": "error" if error else "honeypot_interaction",
            "outcome": "failure" if error else "success",
        },
        "network": {"protocol": protocol},
    }

    # FIX: Omit the source block entirely if IP is missing or empty
    if source_ip:
        entry["source"] = {"ip": source_ip}

    if error:
        entry["error"] = {"message": error}

    # --- Protocol-specific namespacing ---
    if protocol == "http":
        http_block = {}

        method = request_data.get("method")
        body = request_data.get("body")
        if method or body:
            http_request = {}
            if method:
                http_request["method"] = method
            if body:
                http_request["body"] = {"content": body}
            http_block["request"] = http_request

        status_code = response_data.get("status_code")
        response_body = response_data.get("body")
        if status_code is not None or response_body is not None:
            http_response = {}
            if status_code is not None:
                http_response["status_code"] = status_code
            if response_body is not None:
                if isinstance(response_body, (dict, list)):
                    content_str = json.dumps(response_body)
                else:
                    content_str = str(response_body)
                http_response["body"] = {"content": content_str}
            http_block["response"] = http_response

        if http_block:
            entry["http"] = http_block

        path = request_data.get("path")
        if path:
            entry["url"] = {"path": path}

        headers = request_data.get("headers")
        if headers:
            # FIX: Safely initialize nested dicts
            entry.setdefault("http", {}).setdefault("request", {})["headers"] = headers

    elif protocol == "smtp":
        smtp_block = {}
        # FIX 2: Correctly map the engine's keys so SMTP data actually appears
        command = request_data.get("command")
        if command:
            smtp_block["command"] = command
        
        body = request_data.get("body")
        if body:
            smtp_block["data"] = body
            
        if smtp_block:
            entry["smtp"] = smtp_block

    # --- Custom GenPot namespace ---
    genpot_block = {}
    genpot_fields = (
        "rag_query",
        "rag_context",
        "similarity_score",
        "llm_provider",
        "llm_model",
        "llm_temperature",
        "llm_thinking",
        "system_prompt",
        "latency_ms",
        "state_actions",
    )
    for field in genpot_fields:
        value = genpot_metrics.get(field)
        if value is not None:
            # FIX: Prevent mapping explosions by stringifying dynamic LLM payloads
            if field == "state_actions" and isinstance(value, list):
                sanitized_actions = []
                for action in value:
                    safe_action = dict(action)
                    if "value" in safe_action and isinstance(safe_action["value"], (dict, list)):
                        safe_action["value"] = json.dumps(safe_action["value"])
                    else:
                        safe_action["value"] = str(safe_action.get("value", ""))
                    sanitized_actions.append(safe_action)
                genpot_block[field] = sanitized_actions
            else:
                genpot_block[field] = value
    if genpot_block:
        entry["genpot"] = genpot_block

    return entry


def log_interaction(
    protocol: str,
    source_ip: str,
    request_data: dict,
    response_data: dict,
    genpot_metrics: dict,
    error: str = None,
):
    """
    Assembles an ECS-compliant log entry and appends it as a single JSON
    line to the log file.

    This function is the single point of entry for all logging in the
    application.
    """
    try:
        entry = _build_ecs_entry(
            protocol=protocol,
            source_ip=source_ip,
            request_data=request_data or {},
            response_data=response_data or {},
            genpot_metrics=genpot_metrics or {},
            error=error,
        )

        with open(LOG_FILE, "a") as f:
            f.write(json.dumps(entry) + "\n")

    except Exception as e:
        print(f"[!] CRITICAL: Logging failed! Error: {e}")
