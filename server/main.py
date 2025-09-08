import os
import json
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
import google.generativeai as genai
from dotenv import load_dotenv

# from rag_system import RAGSystem
from .prompt_manager import craft_prompt

class MockRAGSystem:
    """
    This is a temporary, fake RAGSystem.
    It returns a hardcoded piece of API documentation for testing purposes.
    """
    def __init__(self):
        print("Mock RAG System is ready.")
        # In a real system, this would load a FAISS index. We don't need that for the mock.
        pass

    def get_context(self, query: str) -> str:
        """
        Ignores the query and returns a sample context for "GET /users/{username}".
        """
        print(f"Mock RAG System received query: '{query}'")
        # 3. PASTE THE SAMPLE RAG OUTPUT HERE
        return """
Endpoint: GET /users/{username}
Description: Provides publicly available information about someone with a GitHub account.

Response Schema (200 OK):
- Type: object
- Properties:
  - login: string
  - id: integer
  - node_id: string
  - avatar_url: string
  - url: string
  - html_url: string
  - followers_url: string
  - following_url: string
  - gists_url: string
  - starred_url: string
  - subscriptions_url: string
  - organizations_url: string
  - repos_url: string
  - type: string
  - site_admin: boolean
  - name: string
  - company: string
  - blog: string
  - location: string
  - email: string or null
  - public_repos: integer
  - followers: integer
  - following: integer
  - created_at: string
  - updated_at: string
"""

load_dotenv()

try:
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    llm_model = genai.GenerativeModel('gemini-2.5-flash')
except Exception as e:
    print(f"Error configuring Gemini API: {e}")
    exit()


# rag_system = RAGSystem()
rag_system = MockRAGSystem()
app = FastAPI()
@app.api_route("/{full_path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"])
async def decoy_api_endpoint(request: Request, full_path: str):
    method = request.method
    path = "/" + full_path
    body_bytes = await request.body()
    body_str = body_bytes.decode('utf-8', errors='ignore')
    print(f"[*] Received request: {method} {path}")
    rag_query = f"{method} {path}"
    context = rag_system.get_context(rag_query)
    prompt = craft_prompt(method, path, body_str, context)
    try:
        llm_response = llm_model.generate_content(prompt)
        raw_response_text = llm_response.text
        cleaned_text = raw_response_text.strip().removeprefix("```json").removesuffix("```").strip()
        response_json = json.loads(cleaned_text)
        print(f"[+] Generated Response: {response_json}")
        return JSONResponse(content=response_json)
    except Exception as e:
            print(f"[!] Error during LLM generation or JSON parsing: {e}")
            return JSONResponse(content={"error": "An internal server error occurred."}, status_code=500)
