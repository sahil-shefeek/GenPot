import os
import re
import yaml
import time
import requests

BASE_URL = "http://localhost:8000"


class SpecTester:
    def __init__(self, spec_path):
        self.spec_path = spec_path
        self.spec = {}
        self.base_url = BASE_URL

        # We will use a Mock Token Strategy injected directly into our headers.
        # Honeypots generally accept syntactically valid tokens.
        self.headers_a = {
            "User-Agent": "AttackerA/1.0",
            "Authorization": "Bearer ghp_mockHoneypotTestToken1234567890",
            "Accept": "application/vnd.github.v3+json",
        }
        self.headers_b = {
            "User-Agent": "AttackerB/1.0",
            "Authorization": "Bearer ghp_mockHoneypotTestToken1234567890",
            "Accept": "application/vnd.github.v3+json",
        }

    def load_spec(self):
        print(f"Loading OpenAPI specification from {self.spec_path}...")
        with open(self.spec_path, "r", encoding="utf-8") as f:
            self.spec = yaml.safe_load(f)
        print("Spec loaded successfully.")

    def substitute_path_params(self, path):
        """
        Replaces {param} with test_param so POST and GET paths resolve to the exact same URL.
        e.g., /repos/{owner}/{repo}/issues -> /repos/test_owner/test_repo/issues
        """
        return re.sub(r"\{([^}]+)\}", r"test_\1", path)

    def find_crud_pairs(self):
        print("Discovering CRUD pairs...")
        pairs = []
        paths = self.spec.get("paths", {})
        for path, operations in paths.items():
            if "post" in operations and "get" in operations:
                # Same path has GET and POST
                endpoint = self.substitute_path_params(path)
                post_op = operations["post"]

                # Extract schema if available
                schema = None
                try:
                    content = post_op.get("requestBody", {}).get("content", {})
                    if "application/json" in content:
                        schema = content["application/json"].get("schema")
                except Exception:
                    pass

                pairs.append(
                    {
                        "original_path": path,
                        "endpoint": endpoint,
                        "schema": schema,
                        "post_op": post_op,
                        "get_op": operations["get"],
                    }
                )

        print(f"Discovered {len(pairs)} GET/POST pairs.\n")
        return pairs

    def _resolve_ref(self, ref):
        if not ref.startswith("#/"):
            return {}
        parts = ref[2:].split("/")
        curr = self.spec
        for part in parts:
            if part in curr:
                curr = curr[part]
            else:
                return {}
        return curr

    def _resolve_schema(self, schema):
        if not schema:
            return {}
        # Simple circular reference protection could be added here if needed
        # but for this script, standard $ref resolution is fine.
        while "$ref" in schema:
            schema = self._resolve_ref(schema["$ref"])
        return schema

    def generate_payload(self, schema):
        schema = self._resolve_schema(schema)

        # Strategy 1: Look for top-level example
        if "example" in schema:
            return schema["example"]
        if "examples" in schema and schema["examples"]:
            # Take the first example
            first_key = list(schema["examples"].keys())[0]
            example_ref = schema["examples"][first_key]
            if "$ref" in example_ref:
                example_obj = self._resolve_ref(example_ref["$ref"])
                if "value" in example_obj:
                    return example_obj["value"]
            elif "value" in example_ref:
                return example_ref["value"]

        # Strategy 2: Generate from properties based on types
        payload = {}
        properties = schema.get("properties", {})

        # If no properties are found, provide a dummy generic payload
        if not properties:
            return {"name": "test_data_8832"}

        for prop, prop_schema in properties.items():
            prop_schema = self._resolve_schema(prop_schema)

            if "example" in prop_schema:
                payload[prop] = prop_schema["example"]
            elif "default" in prop_schema:
                payload[prop] = prop_schema["default"]
            else:
                p_type = prop_schema.get("type")
                if p_type == "string":
                    payload[prop] = f"test_{prop}_8832"
                elif p_type == "integer":
                    payload[prop] = 1
                elif p_type == "boolean":
                    payload[prop] = True
                elif p_type == "array":
                    payload[prop] = []
                elif p_type == "object":
                    payload[prop] = {}
                else:
                    # Fallback
                    payload[prop] = f"test_{prop}_8832"

        return payload

    def run_state_test(self, endpoint, payload):
        url = f"{self.base_url}{endpoint}"

        # Identify our stateful marker: specific identifier from the payload
        marker = None
        for k, v in payload.items():
            if isinstance(v, str) and "test_" in v:
                marker = v
                break

        # Fallback marker selection
        if not marker:
            for k, v in payload.items():
                if isinstance(v, str):
                    marker = v
                    break

        print(f" -> Attacker A: POST {url}")
        try:
            resp_a = requests.post(url, json=payload, headers=self.headers_a)
            print(f"      Status: {resp_a.status_code}")
            if resp_a.status_code == 401:
                print(
                    "      ❌ The honeypot rejected the mock token. Might need dynamic token extraction."
                )
        except requests.exceptions.ConnectionError:
            print("      ❌ Connection Error: Is the server running?")
            return False

        time.sleep(1)

        print(f" -> Attacker B: GET {url}")
        try:
            resp_b = requests.get(url, headers=self.headers_b)
            print(f"      Status: {resp_b.status_code}")
        except requests.exceptions.ConnectionError:
            print("      ❌ Connection Error: Is the server running?")
            return False

        if marker and marker in resp_b.text:
            print(
                f"      ✅ Success! Identifier ({marker}) appeared in the GET request for Attacker B."
            )
            return True
        elif not marker:
            print(
                f"      ⚠️ Warning: No string marker found in payload to verify. Status: {resp_b.status_code}"
            )
            return False
        else:
            print(
                f"      ❌ Failed! Identifier ({marker}) did not appear in the GET request for Attacker B."
            )
            return False


def main():
    print("--- OpenAPI Statefulness Testing ---")
    default_spec = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "data",
        "api.github.com.2022-11-28.deref.yaml",
    )
    spec_path = os.getenv("OPENAPI_SPEC_PATH", default_spec)

    if not os.path.exists(spec_path):
        print(f"❌ Error: OpenAPI spec not found at {spec_path}")
        return

    tester = SpecTester(spec_path)

    try:
        tester.load_spec()
    except Exception as e:
        print(f"❌ Failed to load OpenAPI spec: {e}")
        return

    pairs = tester.find_crud_pairs()

    if not pairs:
        print("❌ No CRUD pairs found.")
        return

    # User requires testing Top 3 pairs to avoid flooding logs.
    top_pairs = pairs[:3]
    print(f"Evaluating Top {len(top_pairs)} Discovered Endpoints:")

    success_count = 0
    for i, pair in enumerate(top_pairs, 1):
        print(f"\n[{i}/{len(top_pairs)}] Target: {pair['original_path']}")

        payload = tester.generate_payload(pair["schema"])

        is_success = tester.run_state_test(pair["endpoint"], payload)
        if is_success:
            success_count += 1

    print("\n--- Summary ---")
    print(f"Total Evaluated: {len(top_pairs)}")
    print(f"Successful State Leakage (State Maintained): {success_count} ✅")
    print(
        f"Failed State Leakage (No State Maintained): {len(top_pairs) - success_count} ❌"
    )


if __name__ == "__main__":
    main()
