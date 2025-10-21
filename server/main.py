from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .llm_client import generate_response
from .logger import log_interaction
from .prompt_manager import craft_prompt
from .rag_system import RAGSystem
from .utils import clean_llm_response

# Initialize real RAG (loads FAISS + mapping from knowledge_base/)
rag_system = RAGSystem()

app = FastAPI()


@app.api_route(
    "/{full_path:path}",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"],
)
async def decoy_api_endpoint(request: Request, full_path: str):
    method = request.method
    path = "/" + full_path
    body_bytes = await request.body()
    body_str = body_bytes.decode("utf-8", errors="ignore")

    client_ip = request.client.host if request.client else None
    base_event = {
        "event": "request_received",
        "ip": client_ip,
        "method": method,
        "path": path,
        "body": body_str,
        "headers": dict(request.headers),
    }
    log_interaction(base_event)

    rag_query = f"{method} {path}"
    context = rag_system.get_context(rag_query)
    prompt = craft_prompt(method, path, body_str, context)

    try:
        raw_response_text = generate_response(prompt)
        response_json = clean_llm_response(raw_response_text)

        log_interaction(
            {
                **base_event,
                "event": "response_generated",
                "rag_query": rag_query,
                "context": context,
                "response": response_json,
                "status_code": 200,
            }
        )
        return JSONResponse(content=response_json)
    except Exception as e:
        log_interaction(
            {
                **base_event,
                "event": "error",
                "rag_query": rag_query,
                "context": context,
                "error": str(e),
                "status_code": 500,
            }
        )
        return JSONResponse(
            content={"error": "An internal server error occurred."},
            status_code=500,
        )
