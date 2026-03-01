"""
RAG Retrieval Accuracy Benchmark

Parses the raw OpenAPI knowledge base, uses Faker to substitute path parameters
with realistic values, queries the live RAG inspection endpoint, and calculates
a final retrieval accuracy score with semantic similarity analysis.

Features:
    - Deterministic or random Faker seeding via --seed.
    - Semantic similarity scoring: compares each endpoint's source description
      from the YAML against retrieved RAG chunks using sentence-transformers.
    - Configurable endpoint sample size and missed-query diagnostics.

Usage:
    # Run full benchmark (random faker values each time)
    uv run python -m scripts.test_scripts.test_rag_accuracy

    # Deterministic run with seed 42
    uv run python -m scripts.test_scripts.test_rag_accuracy --seed 42

    # Test only 100 randomly-sampled endpoints
    uv run python -m scripts.test_scripts.test_rag_accuracy --limit 100

    # Show 10 missed-query diagnostics instead of the default 3
    uv run python -m scripts.test_scripts.test_rag_accuracy --show-misses 10

    # Combine options
    uv run python -m scripts.test_scripts.test_rag_accuracy --seed 42 --limit 200 --show-misses 5

CLI Arguments:
    --seed INT        Faker seed for reproducible runs. Omit for random values
                      each execution.
    --limit INT       Max number of endpoints to test. Omit to test all
                      endpoints in the spec.
    --show-misses INT Number of missed-query diagnostics to print at the end.
                      Default: 3.
    --top-k INT       Number of RAG chunks to retrieve per query. Default: 3.
"""

import argparse
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
import requests
import yaml
from faker import Faker
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# =============================================================================
# Configuration
# =============================================================================

BASE_URL = "http://localhost:8000"
REQUEST_TIMEOUT = 30  # seconds per request
ENCODER_MODEL = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"

SPEC_PATH = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "api.github.com.2022-11-28.deref.yaml"
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="RAG Retrieval Accuracy Benchmark for GenPot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                       Full benchmark, random faker values
  %(prog)s --seed 42             Deterministic, reproducible run
  %(prog)s --limit 100           Test only 100 randomly-sampled endpoints
  %(prog)s --show-misses 10      Show 10 detailed miss diagnostics
  %(prog)s --seed 42 --limit 200 --show-misses 5
        """,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Faker seed for reproducible runs. Omit for random values.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max number of endpoints to test. Omit to test all.",
    )
    parser.add_argument(
        "--show-misses",
        type=int,
        default=3,
        help="Number of missed-query diagnostics to display (default: 3).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of RAG chunks to retrieve per query (default: 3).",
    )
    return parser.parse_args()


# =============================================================================
# Pre-flight Check
# =============================================================================


def preflight_check():
    """Verify the GenPot server is reachable before benchmarking."""
    print(f"[*] Pre-flight: Checking server at {BASE_URL} ...")
    try:
        requests.get(f"{BASE_URL}/", timeout=REQUEST_TIMEOUT)
        print("[*] Pre-flight: Server is reachable. ✅\n")
    except requests.exceptions.ConnectionError:
        print(
            f"\n[!] ERROR: GenPot server is not running at {BASE_URL}. "
            "Please start it with 'uv run uvicorn server.main:app' "
            "before running this benchmark."
        )
        sys.exit(1)
    except requests.exceptions.Timeout:
        print(f"\n[!] ERROR: Server at {BASE_URL} timed out on pre-flight check.")
        sys.exit(1)


# =============================================================================
# Faker-based Path Parameter Substitution
# =============================================================================

# Category mappings for GitHub-specific parameter types
IDENTITY_PARAMS = {
    "owner",
    "org",
    "username",
    "target_user",
    "account_id",
    "enterprise",
}
REPO_GIT_PARAMS = {
    "repo",
    "branch",
    "base",
    "head",
    "path",
    "file_path",
}
HASH_PARAMS = {
    "sha",
    "commit_sha",
    "tree_sha",
    "tag_sha",
}
NUMBER_PARAMS = {
    "issue_number",
    "pull_number",
    "number",
    "comment_id",
    "run_id",
    "job_id",
    "artifact_id",
    "installation_id",
}
SLUG_NAME_PARAMS = {
    "team_slug",
    "app_slug",
    "secret_name",
    "variable_name",
    "label",
}


def _fake_value(fake: Faker, param_name: str) -> str:
    """Return a realistic fake value for a given path parameter name."""
    name = param_name.lstrip("+")  # handle {+param} syntax

    if name in IDENTITY_PARAMS:
        return fake.user_name()
    if name in REPO_GIT_PARAMS:
        return fake.word()
    if name in HASH_PARAMS:
        return fake.sha1()
    if name in NUMBER_PARAMS:
        return str(fake.random_int(min=1, max=9999))
    if name in SLUG_NAME_PARAMS:
        return fake.slug()

    # --- Smart fallback ---
    if "id" in name or "number" in name:
        return str(fake.random_int(min=1, max=9999))
    return fake.word()


def fake_path(fake: Faker, original_path: str) -> str:
    """Replace all {param} and {+param} tags with Faker-generated values."""
    return re.sub(
        r"\{\+?([^}]+)\}",
        lambda m: _fake_value(fake, m.group(1)),
        original_path,
    )


# =============================================================================
# Endpoint Extraction (with source descriptions)
# =============================================================================


def extract_endpoints(fake: Faker) -> list[dict]:
    """
    Load the OpenAPI YAML and build a list of faked query endpoints.
    Also extracts the source summary/description for semantic comparison.
    """
    if not SPEC_PATH.exists():
        print(f"[!] ERROR: OpenAPI spec not found at {SPEC_PATH}")
        sys.exit(1)

    print(f"[*] Loading OpenAPI spec from:\n    {SPEC_PATH}\n")
    with open(SPEC_PATH, "r") as f:
        spec = yaml.safe_load(f)

    paths = spec.get("paths", {})
    endpoints = []

    for path_template, methods in paths.items():
        if not isinstance(methods, dict):
            continue
        for method, operation in methods.items():
            if method.lower() not in {"get", "post", "put", "patch", "delete"}:
                continue

            faked = fake_path(fake, path_template)

            # Extract source description for semantic similarity
            source_desc = ""
            if isinstance(operation, dict):
                summary = operation.get("summary", "")
                description = operation.get("description", "")
                source_desc = f"{summary}. {description}".strip(". ")

            endpoints.append(
                {
                    "method": method.upper(),
                    "original_path": path_template,
                    "faked_query": f"{method.upper()} {faked}",
                    "source_description": source_desc,
                }
            )

    print(f"[*] Extracted {len(endpoints)} endpoints from spec.\n")
    return endpoints


# =============================================================================
# Semantic Similarity
# =============================================================================


def compute_cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


# =============================================================================
# Mass Query & Validation
# =============================================================================


def run_benchmark(
    endpoints: list[dict], encoder: SentenceTransformer, top_k: int
) -> tuple[int, int, list[dict], list[float]]:
    """
    Query the RAG inspect endpoint for every extracted endpoint.

    Validates two things per endpoint:
      1. Path match:  Does the original_path appear in any returned chunk?
      2. Semantic sim: Cosine similarity between the source description and
                       the best-matching retrieved chunk.

    Returns (total, successes, missed_list, similarity_scores).
    """
    total_queries = 0
    successful_retrievals = 0
    missed = []
    similarity_scores: list[float] = []

    for ep in tqdm(endpoints, desc="Querying RAG", unit="endpoint", ncols=80):
        total_queries += 1
        try:
            resp = requests.post(
                f"{BASE_URL}/api/rag-inspect",
                json={"query": ep["faked_query"], "top_k": top_k},
                timeout=REQUEST_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()

            chunks = data.get("chunks", [])

            # --- Path-match check ---
            found = False
            for chunk in chunks:
                if ep["original_path"] in chunk.get("text", ""):
                    found = True
                    break

            if found:
                successful_retrievals += 1
            else:
                # Store retrieved chunks for detailed miss diagnostics
                ep_miss = {**ep, "retrieved_chunks": chunks}
                missed.append(ep_miss)

            # --- Semantic similarity ---
            source_desc = ep.get("source_description", "")
            if source_desc and chunks:
                chunk_texts = [c.get("text", "") for c in chunks]

                # Encode source description + all chunk texts in one batch
                all_texts = [source_desc] + chunk_texts
                embeddings = encoder.encode(all_texts, normalize_embeddings=True)

                source_vec = embeddings[0]
                chunk_vecs = embeddings[1:]

                # Take the best (max) similarity across chunks
                best_sim = max(
                    compute_cosine_similarity(source_vec, cv) for cv in chunk_vecs
                )
                similarity_scores.append(best_sim)
            else:
                similarity_scores.append(0.0)

        except requests.exceptions.RequestException as exc:
            tqdm.write(f"  [WARN] Request failed for {ep['faked_query']}: {exc}")
            missed.append({**ep, "retrieved_chunks": [], "error": str(exc)})
            similarity_scores.append(0.0)

    return total_queries, successful_retrievals, missed, similarity_scores


# =============================================================================
# Reporting
# =============================================================================


def _truncate(text: str, max_len: int = 256) -> str:
    """Truncate text with an ellipsis if it exceeds max_len."""
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def print_report(
    total: int,
    successes: int,
    missed: list[dict],
    similarity_scores: list[float],
    elapsed: float,
    seed: int | None,
    show_misses: int,
) -> None:
    """Print a highly visible summary report with detailed miss diagnostics."""
    misses = total - successes
    accuracy = (successes / total) * 100 if total > 0 else 0.0
    avg_sim = (
        (sum(similarity_scores) / len(similarity_scores)) * 100
        if similarity_scores
        else 0.0
    )

    seed_label = str(seed) if seed is not None else "random"

    print("\n")
    print("═" * 70)
    print("  RAG RETRIEVAL ACCURACY REPORT")
    print("═" * 70)
    print(f"  Faker Seed:               {seed_label}")
    print(f"  Encoder Model:            {ENCODER_MODEL}")
    print("─" * 70)
    print(f"  Total Endpoints Tested:   {total}")
    print(f"  Successful Retrievals:    {successes}")
    print(f"  Missed Retrievals:        {misses}")
    print(f"  Elapsed Time:             {elapsed:.1f}s")
    print("─" * 70)
    print(f"  PATH-MATCH ACCURACY:      {accuracy:.2f}%")
    print(f"  AVG SEMANTIC SIMILARITY:  {avg_sim:.2f}%")
    print("═" * 70)

    if missed and show_misses > 0:
        sample = missed[:show_misses]
        print(f"\n  Missed Query Diagnostics ({len(sample)} of {len(missed)}):\n")

        for i, ep in enumerate(sample, 1):
            print(f"  {'─' * 66}")
            print(f"  Miss #{i}")
            print(f"    Endpoint:     [{ep['method']}] {ep['original_path']}")
            print(f"    Faked Query:  {ep['faked_query']}")

            if ep.get("source_description"):
                print(f"    Expected:     {_truncate(ep['source_description'])}")

            if ep.get("error"):
                print(f"    Error:        {ep['error']}")
                continue

            chunks = ep.get("retrieved_chunks", [])
            if not chunks:
                print("    Retrieved:    (no chunks returned)")
            else:
                print(f"    Retrieved {len(chunks)} chunk(s):")
                for j, chunk in enumerate(chunks, 1):
                    dist = chunk.get("faiss_distance", "?")
                    text = _truncate(chunk.get("text", ""), 100)
                    print(f"      Chunk {j} (dist={dist:.4f}): {text}")

        print(f"  {'─' * 66}\n")


# =============================================================================
# Main
# =============================================================================


def main():
    args = parse_args()

    # --- Faker initialization ---
    if args.seed is not None:
        Faker.seed(args.seed)
        random.seed(args.seed)
        print(f"[*] Faker seed set to {args.seed} (deterministic mode)\n")
    else:
        print("[*] No seed provided — using random Faker values\n")

    fake = Faker()

    preflight_check()
    endpoints = extract_endpoints(fake)

    # --- Optional endpoint limit ---
    if args.limit is not None and args.limit < len(endpoints):
        random.shuffle(endpoints)
        endpoints = endpoints[: args.limit]
        print(f"[*] Sampled {args.limit} endpoints for testing.\n")

    if not endpoints:
        print("[!] No endpoints extracted. Aborting.")
        sys.exit(1)

    print(f"[*] Loading encoder model: {ENCODER_MODEL} ...")
    encoder = SentenceTransformer(ENCODER_MODEL)
    print("[*] Encoder loaded. ✅\n")

    start = time.time()
    total, successes, missed, sim_scores = run_benchmark(endpoints, encoder, args.top_k)
    elapsed = time.time() - start

    print_report(
        total, successes, missed, sim_scores, elapsed, args.seed, args.show_misses
    )


if __name__ == "__main__":
    main()
