import time

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .config_manager import load_config
from .llm_client import LLMRateLimitError, generate_response
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

    start_time = time.time()
    client_ip = request.client.host if request.client else None
    base_event = {
        "event": "interaction",
        "ip": client_ip,
        "method": method,
        "path": path,
        "body": body_str,
        "headers": dict(request.headers),
    }
    # Removed early log_interaction call to consolidate logs

    rag_query = f"{method} {path}"
    context = rag_system.get_context(rag_query)
    prompt = craft_prompt(method, path, body_str, context)

    try:
        config = load_config()
        provider = config.get("honeypot_provider", "gemini")
        model = config.get("honeypot_model", "gemini-1.5-flash")

        raw_response_text = generate_response(
            prompt,
            provider_type=provider,
            model_name=model,
        )
        response_json = clean_llm_response(raw_response_text)

        response_time_ms = (time.time() - start_time) * 1000

        log_interaction(
            {
                **base_event,
                "rag_query": rag_query,
                "context": context,
                "response": response_json,
                "status_code": 200,
                "response_time_ms": response_time_ms,
                "provider": provider,
                "model": model,
            }
        )
        return JSONResponse(content=response_json)
    except LLMRateLimitError as e:
        response_time_ms = (time.time() - start_time) * 1000
        log_interaction(
            {
                **base_event,
                "rag_query": rag_query,
                "context": context,
                "error": str(e),
                "event_status": "error_rate_limit",
                "status_code": 429,
                "response_time_ms": response_time_ms,
            }
        )
        return JSONResponse(
            status_code=429,
            content={
                "error": "Service Temporarily Unavailable (Rate Limit Exceeded)",
                "retry_after": e.retry_after,
            },
        )
    except Exception as e:
        response_time_ms = (time.time() - start_time) * 1000
        log_interaction(
            {
                **base_event,
                "rag_query": rag_query,
                "context": context,
                "error": str(e),
                "status_code": 500,
                "response_time_ms": response_time_ms,
            }
        )
        return JSONResponse(
            content={"error": "An internal server error occurred."},
            status_code=500,
        )
