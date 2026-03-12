"""
Live Integration Test: State Leakage Detection

Tests whether the GenPot honeypot correctly maintains stateful context
across independent attacker sessions, validating two key scenarios using
real GitHub API endpoints (matching the RAG knowledge base):

  1. Global Resource Persistence — POST /repos/:owner/:repo/issues → GET
  2. Auth Token Persistence — extract token, reuse cross-session

Usage:
    uv run python -m scripts.test_scripts.test_statefulness
"""

import re
import sys
import time

import requests

# =============================================================================
# Configuration
# =============================================================================

BASE_URL = "http://localhost:8000"
REQUEST_TIMEOUT = 120  # seconds — generous for LLM generation latency

# GitHub-style headers for two distinct "attacker" sessions
HEADERS_A = {
    "User-Agent": "AttackerA-Scanner/1.0",
    "X-Forwarded-For": "10.0.0.1",
    "Accept": "application/vnd.github.v3+json",
    "Authorization": "Bearer ghp_mockHoneypotTestTokenAttackerA123",
}
HEADERS_B = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "X-Forwarded-For": "10.0.0.2",
    "Accept": "application/vnd.github.v3+json",
    "Authorization": "Bearer ghp_mockHoneypotTestTokenAttackerB456",
}


# =============================================================================
# Helpers
# =============================================================================


def preflight_check():
    """
    Verify the GenPot server is reachable before running tests.
    Exits immediately with a clear error message if not.
    """
    print(f"[*] Pre-flight: Checking server at {BASE_URL} ...")
    try:
        requests.get(f"{BASE_URL}/api/health", timeout=10)
        print("[*] Pre-flight: Server is reachable. ✅\n")
    except requests.exceptions.ConnectionError:
        print(
            f"\n[!] ERROR: DecoyPot server is not running at {BASE_URL}. "
            "Please start it using 'uv run uvicorn server.main:app' "
            "before running live tests."
        )
        sys.exit(1)
    except requests.exceptions.Timeout:
        print(
            f"\n[!] ERROR: Server at {BASE_URL} timed out on pre-flight check. "
            "The server may be overloaded or unresponsive."
        )
        sys.exit(1)


def extract_token(data):
    """
    3-Tiered token extraction strategy to handle LLM response variability.

    Tier 1: Check common top-level token key names.
    Tier 2: Check one level of nesting for common token key names.
    Tier 3: Regex fallback — scan all string values for JWT patterns
            or long alphanumeric strings.
    """
    # Tier 1: Direct key lookup
    token_keys = ["token", "access_token", "jwt", "session_token"]
    for key in token_keys:
        if key in data and isinstance(data[key], str):
            return data[key]

    # Tier 2: Nested key lookup (one level deep)
    for _, value in data.items():
        if isinstance(value, dict):
            for key in token_keys:
                if key in value and isinstance(value[key], str):
                    return value[key]

    # Tier 3: Regex fallback (JWT or generic long alphanumeric string)
    def search_dict(d):
        for v in d.values():
            if isinstance(v, str):
                # JWT pattern: header.payload.signature
                if re.match(r"^[A-Za-z0-9-_]+\.[A-Za-z0-9-_]+\.[A-Za-z0-9-_]+$", v):
                    return v
                # Any alphanumeric string longer than 20 chars
                if len(v) > 20 and re.match(r"^[A-Za-z0-9-_]+$", v):
                    return v
            elif isinstance(v, dict):
                res = search_dict(v)
                if res:
                    return res
        return None

    return search_dict(data)


# =============================================================================
# Test Scenarios
# =============================================================================


def run_tests():
    """Execute all statefulness test scenarios against the live server."""

    preflight_check()

    print("═" * 70)
    print("  LIVE STATEFULNESS TEST SUITE  (GitHub API Endpoints)")
    print("═" * 70)

    results = {}

    # ═════════════════════════════════════════════════════════════════════════
    # SCENARIO 1: Global Resource Persistence
    #   Attacker A creates a GitHub Issue via POST,
    #   Attacker B lists issues via GET and sees it.
    # ═════════════════════════════════════════════════════════════════════════
    print("\n" + "═" * 70)
    print("  SCENARIO 1: Global Resource Persistence")
    print("  Attacker A: POST /repos/test_owner/test_repo/issues")
    print("  Attacker B: GET  /repos/test_owner/test_repo/issues")
    print("═" * 70)

    scenario_1_pass = False
    issue_endpoint = "/repos/test_owner/test_repo/issues"

    # Step 1: Attacker A creates a GitHub Issue
    print(f"\n  → Attacker A: POST {issue_endpoint}")
    issue_payload = {
        "title": "Bug: state_leak_marker_8832",
        "body": "This issue was created by Attacker A to test state persistence.",
        "labels": ["bug"],
    }
    try:
        resp_a = requests.post(
            f"{BASE_URL}{issue_endpoint}",
            json=issue_payload,
            headers=HEADERS_A,
            timeout=REQUEST_TIMEOUT,
        )
        if resp_a.status_code in [200, 201]:
            print(f"    ✅ Issue created (Status: {resp_a.status_code})")
        else:
            print(f"    ❌ POST failed (Status: {resp_a.status_code})")
            print(f"       Response: {resp_a.text[:200]}")
    except requests.exceptions.ConnectionError:
        print("    ❌ Connection Error: Server went down mid-test.")
        results["Scenario 1"] = "ERROR"
        print(_format_summary(results))
        return
    except requests.exceptions.Timeout:
        print(f"    ❌ Request timed out after {REQUEST_TIMEOUT}s.")
        results["Scenario 1"] = "TIMEOUT"
        print(_format_summary(results))
        return

    time.sleep(2)  # Allow state to propagate

    # Step 2: Attacker B lists issues on the same repo
    print(f"  → Attacker B: GET {issue_endpoint}")
    try:
        resp_b = requests.get(
            f"{BASE_URL}{issue_endpoint}",
            headers=HEADERS_B,
            timeout=REQUEST_TIMEOUT,
        )
        if "state_leak_marker_8832" in resp_b.text:
            print(
                f"    ✅ PASS — Issue data visible to Attacker B! "
                f"(Status: {resp_b.status_code})"
            )
            scenario_1_pass = True
        else:
            print(
                f"    ❌ FAIL — Marker not found in response. "
                f"(Status: {resp_b.status_code})"
            )
            print(f"       Response snippet: {resp_b.text[:200]}")
    except requests.exceptions.ConnectionError:
        print("    ❌ Connection Error on GET request.")
    except requests.exceptions.Timeout:
        print(f"    ❌ GET request timed out after {REQUEST_TIMEOUT}s.")

    results["Scenario 1"] = "PASS" if scenario_1_pass else "FAIL"

    # ═════════════════════════════════════════════════════════════════════════
    # SCENARIO 2: Auth Token Persistence
    #   Attacker A creates an installation access token via the GitHub
    #   App API, Attacker B reuses it to access an authenticated endpoint.
    # ═════════════════════════════════════════════════════════════════════════
    print("\n" + "═" * 70)
    print("  SCENARIO 2: Auth Token Persistence")
    print("  Attacker A: POST /app/installations/1/access_tokens")
    print("  Attacker B: GET  /user  (with stolen token)")
    print("═" * 70)

    scenario_2_pass = False
    token_endpoint = "/app/installations/1/access_tokens"

    # Step 1: Attacker A requests an installation access token
    print(f"\n  → Attacker A: POST {token_endpoint}")
    token_payload = {
        "repositories": ["test_repo"],
        "permissions": {"issues": "write", "contents": "read"},
    }
    try:
        resp_token = requests.post(
            f"{BASE_URL}{token_endpoint}",
            json=token_payload,
            headers=HEADERS_A,
            timeout=REQUEST_TIMEOUT,
        )
    except requests.exceptions.ConnectionError:
        print("    ❌ Connection Error on token request.")
        results["Scenario 2"] = "ERROR"
        print(_format_summary(results))
        return
    except requests.exceptions.Timeout:
        print(f"    ❌ Token request timed out after {REQUEST_TIMEOUT}s.")
        results["Scenario 2"] = "TIMEOUT"
        print(_format_summary(results))
        return

    if resp_token.status_code not in [200, 201]:
        print(f"    ❌ Token creation failed (Status: {resp_token.status_code})")
        print(f"       Response: {resp_token.text[:200]}")
        results["Scenario 2"] = "FAIL"
        print(_format_summary(results))
        return

    # Step 2: Extract token using robust 3-tier extraction
    try:
        token_data = resp_token.json()
        token = extract_token(token_data)
        if token:
            print(f"    ✅ Token extracted: {token[:15]}...")
        else:
            print(
                f"    ❌ Could not extract token from LLM response.\n"
                f"       Raw response: {token_data}"
            )
            results["Scenario 2"] = "FAIL"
            print(_format_summary(results))
            return
    except Exception as e:
        print(f"    ❌ Failed to parse token JSON response: {e}")
        results["Scenario 2"] = "FAIL"
        print(_format_summary(results))
        return

    time.sleep(2)

    # Step 3: Attacker B — unauthenticated access to /user (should fail)
    print("  → Attacker B: GET /user (No Auth)")
    headers_b_noauth = {k: v for k, v in HEADERS_B.items() if k != "Authorization"}
    try:
        resp_unauth = requests.get(
            f"{BASE_URL}/user",
            headers=headers_b_noauth,
            timeout=REQUEST_TIMEOUT,
        )
        if resp_unauth.status_code in [401, 403]:
            print(f"    ✅ Correctly blocked (Status: {resp_unauth.status_code})")
        else:
            print(
                f"    ⚠️  Expected 401/403, got {resp_unauth.status_code} "
                f"(honeypot may not enforce auth here)"
            )
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
        print("    ❌ Request failed (connection/timeout).")

    # Step 4: Attacker B — authenticated with Attacker A's stolen token
    print("  → Attacker B: GET /user (With Stolen Token)")
    headers_b_stolen = {k: v for k, v in HEADERS_B.items() if k != "Authorization"}
    headers_b_stolen["Authorization"] = f"Bearer {token}"
    try:
        resp_auth = requests.get(
            f"{BASE_URL}/user",
            headers=headers_b_stolen,
            timeout=REQUEST_TIMEOUT,
        )
        if resp_auth.status_code in [200, 201]:
            print(
                f"    ✅ PASS — Token persistence works! "
                f"(Status: {resp_auth.status_code})"
            )
            scenario_2_pass = True
        else:
            print(f"    ❌ FAIL — Expected 2xx, got {resp_auth.status_code}")
            print(f"       Response snippet: {resp_auth.text[:200]}")
    except requests.exceptions.ConnectionError:
        print("    ❌ Connection Error on authenticated request.")
    except requests.exceptions.Timeout:
        print(f"    ❌ Authenticated request timed out after {REQUEST_TIMEOUT}s.")

    results["Scenario 2"] = "PASS" if scenario_2_pass else "FAIL"

    # ═════════════════════════════════════════════════════════════════════════
    # Summary
    # ═════════════════════════════════════════════════════════════════════════
    print(_format_summary(results))


def _format_summary(results):
    """Format the final test summary block."""
    lines = [
        "\n" + "═" * 70,
        "  SUMMARY",
        "═" * 70,
    ]
    passed = sum(1 for v in results.values() if v == "PASS")
    total = len(results)
    for name, status in results.items():
        icon = "✅" if status == "PASS" else "❌"
        lines.append(f"  {icon} {name}: {status}")
    lines.append(f"\n  Total: {passed}/{total} passed")
    lines.append("═" * 70)
    return "\n".join(lines)


if __name__ == "__main__":
    run_tests()
