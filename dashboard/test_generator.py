from typing import List, Dict, Any
from server.llm_client import generate_response
from server.utils import clean_llm_response


class TestGenerator:
    """
    Generates synthetic attack test cases for the honeypot using an LLM.
    """

    def generate_test_cases(
        self, n: int, provider: str, model: str
    ) -> List[Dict[str, Any]]:
        """
        Generates N test cases acting as a Red Team Penetration Tester.

        Args:
            n: Number of test cases to generate.
            provider: LLM provider (e.g., 'gemini', 'ollama').
            model: Model name.

        Returns:
            A list of dictionary objects, each representing a test case.
            Returns an empty list on failure.
        """
        prompt = f"""
System: You are a Red Team Penetration Tester.
Objective: Generate {n} diverse synthetic HTTP test cases to test a honeypot's detection capabilities.
Output Format: STRICTLY a JSON Array of objects. Do not include markdown formatting or explanations.

Each object MUST have the following keys:
- "method": HTTP method (GET, POST, PUT, DELETE, PATCH, etc.)
- "path": URL path (e.g., /admin, /api/login, /etc/passwd)
- "headers": Dictionary of HTTP headers (User-Agent, Content-Type, custom headers)
- "body": The payload string (or JSON object) representing the request body. Use null if empty.
- "description": A short sentence explaining the intent of the test (e.g., "SQL Injection attempt", "Normal user login").

Examples of intent: SQL Injection, XSS, Directory Traversal, Command Injection, Credential Stuffing, Reconnaissance.

Generate exactly {n} cases now.
"""
        try:
            raw_response = generate_response(
                prompt, provider_type=provider, model_name=model
            )
            parsed_data = clean_llm_response(raw_response)

            if isinstance(parsed_data, list):
                # Validate keys in each item
                valid_cases = []
                required_keys = {"method", "path", "headers", "body", "description"}
                for item in parsed_data:
                    if isinstance(item, dict) and required_keys.issubset(item.keys()):
                        valid_cases.append(item)
                return valid_cases
            elif isinstance(parsed_data, dict) and "error" in parsed_data:
                # Logic to handle if clean_llm_response returned an error dict
                print(f"Generator Error: {parsed_data.get('error')}")
                return []
            else:
                print(f"Unexpected response format: {type(parsed_data)}")
                return []

        except Exception as e:
            print(f"Error generating test cases: {e}")
            return []
