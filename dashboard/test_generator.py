import os
import yaml
import random
from typing import List, Dict, Any
from server.llm_client import generate_response
from server.utils import clean_llm_response

DEFAULT_SPEC_PATH = "data/api.github.com.2022-11-28.deref.yaml"


class TestGenerator:
    """
    Generates synthetic attack test cases for the honeypot using an LLM.
    """

    def __init__(self, spec_path: str = None):
        self.spec_path = spec_path or os.getenv("OPENAPI_SPEC_PATH", DEFAULT_SPEC_PATH)
        self.api_endpoints = self._load_github_endpoints()

    def _load_github_endpoints(self) -> List[str]:
        """Parses the OpenAPI spec to extract available paths and methods."""
        try:
            with open(self.spec_path, "r", encoding="utf-8") as f:
                spec = yaml.safe_load(f)

            endpoints = []
            # Extract paths and HTTP methods from the loaded YAML
            for path, path_items in spec.get("paths", {}).items():
                for method in path_items.keys():
                    if method.lower() in ["get", "post", "put", "patch", "delete"]:
                        endpoints.append(f"{method.upper()} {path}")
            return endpoints
        except Exception as e:
            print(f"Error loading OpenAPI spec: {e}")
            # Fallback endpoints just in case the file is missing
            return ["GET /", "POST /repos/{owner}/{repo}/issues"]

    def generate_test_cases(
        self, n: int, provider: str, model: str
    ) -> List[Dict[str, Any]]:
        """
        Generates N test cases acting as a Red Team Penetration Tester.
        """
        vectors = [
            "SQL Injection (SQLi)",
            "Cross-Site Scripting (XSS)",
            "Path Traversal",
            "Command Injection",
            "Broken Authentication",
            "Insecure Direct Object References (IDOR)",
            "Server-Side Request Forgery (SSRF)",
            "Mass Assignment",
        ]

        # Select a random sample of endpoints to keep the prompt context focused
        # and avoid exceeding the LLM's context window.
        sample_size = min(n * 3, len(self.api_endpoints))
        target_endpoints = random.sample(self.api_endpoints, sample_size)
        endpoints_str = "\n".join(target_endpoints)

        prompt = f"""
System: You are a Red Team Penetration Tester.
Objective: Generate {n} diverse synthetic HTTP test cases to test a honeypot's detection capabilities.
Context: The target is a Python FastAPI application simulating the GitHub v3 REST API.

Use the following valid GitHub API endpoints as targets for your attacks:
{endpoints_str}

Available Attack Vectors:
{", ".join(vectors)}

Instruction:
Select {n} DISTINCT vectors from the list above. For each selected vector, create a malicious request targeting one of the provided GitHub API endpoints. 
IMPORTANT: If the endpoint contains path parameters (e.g., {{owner}} or {{repo}}), replace them with realistic mock values (e.g., 'octocat' or 'Hello-World') before injecting the attack payload.

Return ONLY a valid JSON array of {n} objects. Do not include markdown formatting or explanations.

Each object MUST have the following keys:
- "method": HTTP method (must match the targeted endpoint)
- "path": URL path (e.g., /repos/octocat/Hello-World/issues/1?q=<script>alert(1)</script>)
- "headers": Dictionary of HTTP headers
- "body": The payload string (or JSON object). Use null if empty.
- "description": A short sentence explaining the intent of the test.

Generate exactly {n} cases now.
"""
        try:
            raw_response = generate_response(
                prompt, provider_type=provider, model_name=model
            )
            parsed_data = clean_llm_response(raw_response)

            if isinstance(parsed_data, list):
                valid_cases = []
                required_keys = {"method", "path", "headers", "body", "description"}
                for item in parsed_data:
                    if isinstance(item, dict) and required_keys.issubset(item.keys()):
                        valid_cases.append(item)
                return valid_cases
            elif isinstance(parsed_data, dict) and "error" in parsed_data:
                print(f"Generator Error: {parsed_data.get('error')}")
                return []
            else:
                print(f"Unexpected response format: {type(parsed_data)}")
                return []

        except Exception as e:
            print(f"Error generating test cases: {e}")
            return []
