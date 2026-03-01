"""
Live Integration Test: Automated Attack Simulation

Generates AI-crafted malicious payloads via the TestGenerator (LLM-powered)
and fires them at the live GenPot server to validate honeypot responses.

Usage:
    uv run python -m scripts.test_scripts.test_live_attacks
"""

import json
import os
import sys
import time
from pathlib import Path

import requests

# =============================================================================
# Configuration
# =============================================================================

BASE_URL = "http://localhost:8000"
REQUEST_TIMEOUT = 60  # seconds — LLM generation can be slow
NUM_ATTACKS = 3
LLM_PROVIDER = "gemini"
LLM_MODEL = "gemini-2.5-flash"


# =============================================================================
# Pre-flight
# =============================================================================


def preflight_check():
    """
    Verify the GenPot server is reachable before running attacks.
    Exits immediately with a clear error if not.
    """
    print(f"[*] Pre-flight: Checking server at {BASE_URL} ...")
    try:
        requests.get(f"{BASE_URL}/", timeout=REQUEST_TIMEOUT)
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
# Attack Execution
# =============================================================================


def run_attacks():
    """Generate AI attack payloads and fire them at the live server."""

    preflight_check()

    # -- Initialize TestGenerator with absolute spec path --
    # Resolve from project root (assuming script is run via `python -m`)
    from dashboard.test_generator import TestGenerator

    default_spec = (
        Path(__file__).resolve().parents[2]
        / "data"
        / "api.github.com.2022-11-28.deref.yaml"
    )
    spec_path = Path(os.getenv("OPENAPI_SPEC_PATH", str(default_spec)))

    if not spec_path.exists():
        print(f"[!] ERROR: OpenAPI spec not found at {spec_path}")
        sys.exit(1)

    generator = TestGenerator(spec_path=str(spec_path))

    # -- Generate attack cases via LLM --
    print("═" * 70)
    print("  LIVE ATTACK SIMULATION")
    print(
        f"  Generating {NUM_ATTACKS} attack payloads via {LLM_PROVIDER}/{LLM_MODEL}..."
    )
    print("═" * 70)

    test_cases = generator.generate_test_cases(
        n=NUM_ATTACKS, provider=LLM_PROVIDER, model=LLM_MODEL
    )

    if not test_cases:
        print("\n[!] ERROR: TestGenerator returned 0 valid test cases.")
        print("    The LLM may have returned malformed JSON. Try again.")
        sys.exit(1)

    print(f"\n[*] Generated {len(test_cases)} valid attack cases.\n")

    # -- Fire each attack --
    results = []
    for i, case in enumerate(test_cases, 1):
        method = case.get("method", "GET").upper()
        path = case.get("path", "/")
        headers = case.get("headers", {})
        body = case.get("body")
        description = case.get("description", "No description")

        url = f"{BASE_URL}{path}"

        # Ensure body is a string for requests
        if isinstance(body, dict):
            body_str = json.dumps(body)
            content_type = "application/json"
        elif isinstance(body, str):
            body_str = body
            content_type = headers.get("Content-Type", "text/plain")
        else:
            body_str = None
            content_type = None

        if content_type and "Content-Type" not in headers:
            headers["Content-Type"] = content_type

        print("─" * 70)
        print(f"  ATTACK [{i}/{len(test_cases)}]")
        print(f"  Intent: {description}")
        print(f"  Target: {method} {path}")
        print("─" * 70)

        try:
            start_time = time.time()
            resp = requests.request(
                method=method,
                url=url,
                headers=headers,
                data=body_str,
                timeout=REQUEST_TIMEOUT,
            )
            latency_ms = (time.time() - start_time) * 1000

            snippet = resp.text[:100]
            if len(resp.text) > 100:
                snippet += "..."

            print(f"  Status:  {resp.status_code}")
            print(f"  Latency: {latency_ms:.0f}ms")
            print(f"  Response: {snippet}")

            results.append(
                {
                    "attack": i,
                    "description": description,
                    "method": method,
                    "path": path,
                    "status": resp.status_code,
                    "latency_ms": round(latency_ms),
                }
            )

        except requests.exceptions.ConnectionError:
            print("  ❌ Connection Error — server may have crashed.")
            results.append(
                {"attack": i, "description": description, "status": "CONN_ERR"}
            )
        except requests.exceptions.Timeout:
            print(f"  ❌ Timed out after {REQUEST_TIMEOUT}s.")
            results.append(
                {"attack": i, "description": description, "status": "TIMEOUT"}
            )

    # -- Summary --
    print("\n" + "═" * 70)
    print("  ATTACK SUMMARY")
    print("═" * 70)
    print(f"  {'#':<4} {'Status':<10} {'Latency':<10} {'Method':<8} {'Description'}")
    print("  " + "─" * 66)

    for r in results:
        status = str(r.get("status", "?"))
        latency = f"{r.get('latency_ms', '?')}ms" if "latency_ms" in r else "N/A"
        method = r.get("method", "?")
        desc = r.get("description", "")[:45]
        print(f"  {r['attack']:<4} {status:<10} {latency:<10} {method:<8} {desc}")

    responded = sum(1 for r in results if isinstance(r.get("status"), int))
    print(f"\n  Total: {responded}/{len(results)} received HTTP responses")
    print("═" * 70)


if __name__ == "__main__":
    run_attacks()
