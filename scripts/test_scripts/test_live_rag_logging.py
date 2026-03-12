"""
Live Integration Test: RAG Retrieval & Log Verification

Validates two critical server subsystems against the live GenPot server:
  Phase 1 — RAG system returns relevant chunks for a known GitHub API query.
  Phase 2 — Server writes structured JSONL telemetry to disk after handling
             a request.

Usage:
    uv run python -m scripts.test_scripts.test_live_rag_logging
"""

import json
import sys
import time
import uuid
from pathlib import Path

import requests

# =============================================================================
# Configuration
# =============================================================================

BASE_URL = "http://localhost:8000"
REQUEST_TIMEOUT = 60  # seconds — LLM generation can be slow

# Resolve logs/honeypot.jsonl relative to this script's location
# scripts/test_scripts/test_live_rag_logging.py → project root is 2 levels up
PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOG_FILE = PROJECT_ROOT / "logs" / "honeypot.jsonl"


# =============================================================================
# Pre-flight
# =============================================================================


def preflight_check():
    """
    Verify the GenPot server is reachable before running tests.
    Exits immediately with a clear error if not.
    """
    print(f"[*] Pre-flight: Checking server at {BASE_URL} ...")
    try:
        requests.get(f"{BASE_URL}/api/health", timeout=10)
        print("[*] Pre-flight: Server is reachable. ✅\n")
    except requests.exceptions.ConnectionError:
        print(
            f"\n[!] ERROR: GenPot server is not running at {BASE_URL}. "
            "Please start it using 'uv run uvicorn server.main:app' "
            "before running live tests."
        )
        sys.exit(1)
    except requests.exceptions.Timeout:
        print(f"\n[!] ERROR: Server at {BASE_URL} timed out on pre-flight check.")
        sys.exit(1)


# =============================================================================
# Phase 1: RAG Inspection
# =============================================================================


def test_rag_inspection():
    """
    Verify the RAG system retrieves relevant context for a known
    GitHub API query via the /api/rag-inspect endpoint.
    """
    print("═" * 70)
    print("  PHASE 1: Live RAG Inspection")
    print("═" * 70)

    payload = {
        "query": "GET /repos/{owner}/{repo}/issues",
        "top_k": 2,
    }

    print("\n  → POST /api/rag-inspect")
    print(f"    Query: {payload['query']}")

    try:
        start = time.time()
        resp = requests.post(
            f"{BASE_URL}/api/rag-inspect",
            json=payload,
            timeout=REQUEST_TIMEOUT,
        )
        latency_ms = (time.time() - start) * 1000
    except requests.exceptions.ConnectionError:
        print("    ❌ Connection Error during RAG inspection.")
        return False
    except requests.exceptions.Timeout:
        print(f"    ❌ Request timed out after {REQUEST_TIMEOUT}s.")
        return False

    # Assert status code
    if resp.status_code != 200:
        print(f"    ❌ FAIL — Expected 200, got {resp.status_code}")
        print(f"       Response: {resp.text[:200]}")
        return False

    # Parse and validate response structure
    try:
        data = resp.json()
    except json.JSONDecodeError:
        print(f"    ❌ FAIL — Response is not valid JSON: {resp.text[:200]}")
        return False

    chunks = data.get("chunks")
    if not isinstance(chunks, list):
        print(f"    ❌ FAIL — 'chunks' is not a list. Got: {type(chunks)}")
        return False

    if len(chunks) < 1:
        print("    ❌ FAIL — 'chunks' list is empty. RAG returned no results.")
        return False

    server_latency = data.get("latency_ms", "?")
    print(f"    ✅ PASS — RAG returned {len(chunks)} chunk(s)")
    print(f"       Server-side latency: {server_latency}ms")
    print(f"       Round-trip latency:  {latency_ms:.0f}ms")
    print(f"       First chunk preview: {chunks[0].get('text', '')[:80]}...")

    return True


# =============================================================================
# Phase 2: Log Verification
# =============================================================================


def test_log_writing():
    """
    Fire a uniquely identifiable request at the server, then verify that
    a structured log entry appears in logs/honeypot.jsonl on disk.
    """
    print("\n" + "═" * 70)
    print("  PHASE 2: Live File Logging Verification")
    print("═" * 70)

    # Generate a unique test path so we can find it in the log file
    test_id = uuid.uuid4().hex[:12]
    test_path = f"/repos/test-org/logging-probe-{test_id}/issues"

    print(f"\n  → POST {test_path}")
    print(f"    Unique marker: {test_id}")

    try:
        resp = requests.post(
            f"{BASE_URL}{test_path}",
            json={"test": "logging", "marker": test_id},
            headers={
                "User-Agent": "LogVerifier/1.0",
                "Accept": "application/vnd.github.v3+json",
            },
            timeout=REQUEST_TIMEOUT,
        )
        print(f"    Server responded: {resp.status_code}")
    except requests.exceptions.ConnectionError:
        print("    ❌ Connection Error while sending probe request.")
        return False
    except requests.exceptions.Timeout:
        print(f"    ❌ Probe request timed out after {REQUEST_TIMEOUT}s.")
        return False

    # Wait for file I/O to complete
    print("    Waiting 2s for log flush...")
    time.sleep(2)

    # Verify log file exists
    if not LOG_FILE.exists():
        print(f"    ❌ FAIL — Log file not found at {LOG_FILE}")
        return False

    # Search for our unique path in recent log entries
    print(f"  → Scanning {LOG_FILE.name} for marker...")

    found_entry = None
    try:
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            # Read all lines and search from the end (most recent first)
            lines = f.readlines()

        # Check the last 20 lines (our entry should be very recent)
        for line in reversed(lines[-20:]):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                # ECS schema: path is nested under url.path
                if entry.get("url", {}).get("path") == test_path:
                    found_entry = entry
                    break
            except json.JSONDecodeError:
                continue
    except IOError as e:
        print(f"    ❌ FAIL — Could not read log file: {e}")
        return False

    if not found_entry:
        print(f"    ❌ FAIL — No log entry found with path '{test_path}'")
        print(f"       Searched last {min(20, len(lines))} lines of {LOG_FILE.name}")
        return False

    # Verify required ECS telemetry structure
    genpot = found_entry.get("genpot", {})
    required_genpot_keys = ["rag_query", "similarity_score", "latency_ms"]
    missing_keys = [k for k in required_genpot_keys if k not in genpot]

    http_response = found_entry.get("http", {}).get("response", {})
    if not http_response:
        missing_keys.append("http.response")

    if missing_keys:
        print(f"    ❌ FAIL — Log entry missing keys: {missing_keys}")
        print(f"       Available top-level keys: {list(found_entry.keys())}")
        return False

    response_body = http_response.get("body", {}).get("content", {})
    print("    ✅ PASS — Log entry found and verified (ECS schema)!")
    print(f"       genpot.rag_query:        {genpot['rag_query']}")
    print(f"       genpot.similarity_score: {genpot['similarity_score']}")
    print(f"       genpot.latency_ms:       {genpot['latency_ms']:.0f}ms")
    print(
        f"       response body keys:      {list(response_body.keys()) if isinstance(response_body, dict) else type(response_body).__name__}"
    )

    return True


# =============================================================================
# Main
# =============================================================================


def run_tests():
    """Execute all RAG & logging verification phases."""

    preflight_check()

    results = {}
    results["Phase 1: RAG Inspection"] = "PASS" if test_rag_inspection() else "FAIL"
    results["Phase 2: Log Verification"] = "PASS" if test_log_writing() else "FAIL"

    # -- Summary --
    print("\n" + "═" * 70)
    print("  SUMMARY")
    print("═" * 70)

    passed = sum(1 for v in results.values() if v == "PASS")
    total = len(results)
    for name, status in results.items():
        icon = "✅" if status == "PASS" else "❌"
        print(f"  {icon} {name}: {status}")

    print(f"\n  Total: {passed}/{total} passed")
    print("═" * 70)


if __name__ == "__main__":
    run_tests()
