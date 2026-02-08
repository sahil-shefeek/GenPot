import time

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .config_manager import load_config
from .llm_client import LLMRateLimitError, generate_response
from .logger import log_interaction
from .prompt_manager import craft_prompt
from .rag_system import RAGSystem
from .state_manager import StateManager
from .utils import clean_llm_response

# Initialize real RAG (loads FAISS + mapping from knowledge_base/)
rag_system = RAGSystem()
# Initialize State Manager
state_manager = StateManager()

app = FastAPI()


class RAGDebugRequest(BaseModel):
    query: str
    top_k: int = 3
    threshold: float = 0.75


@app.post("/api/debug/rag")
async def debug_rag_endpoint(request: RAGDebugRequest):
    """
    Debug endpoint to retrieve RAG context without triggering LLM or State Manager.
    """
    start_time = time.time()

    # Call RAG System
    context, metadata = rag_system.get_context(
        request.query, top_k=request.top_k, threshold=request.threshold
    )

    response_time_ms = (time.time() - start_time) * 1000

    # Log debug event
    log_interaction(
        {
            "event": "debug_rag",
            "query": request.query,
            "rag_metadata": metadata,
            "response_time_ms": response_time_ms,
        }
    )

    return {"context": context, "metadata": metadata}


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

    # Load Config for RAG parameters
    config = load_config()
    rag_top_k = config.get("rag_top_k", 3)
    rag_threshold = config.get("rag_similarity_threshold", 0.75)

    rag_query = f"{method} {path}"
    # Unpack context and metadata
    context, rag_metadata = rag_system.get_context(
        rag_query, top_k=rag_top_k, threshold=rag_threshold
    )

    # State Management: Get Context
    auth_token = request.headers.get("Authorization", "").replace("Bearer ", "").strip()
    if not auth_token:
        auth_token = None

    state_context = state_manager.get_context(path, auth_token=auth_token)

    prompt = craft_prompt(method, path, body_str, context, state_context=state_context)

    try:
        # config is already loaded above
        provider = config.get("honeypot_provider", "gemini")
        model = config.get("honeypot_model", "gemini-1.5-flash")

        raw_response_text = generate_response(
            prompt,
            provider_type=provider,
            model_name=model,
        )

        # Robust Parsing & Branching Logic
        parsed_output = clean_llm_response(raw_response_text)
        final_response = {}
        side_effects = []
        is_fallback = False

        if "response" in parsed_output and "side_effects" in parsed_output:
            # Case A: Success - Valid Stateful Response
            final_response = parsed_output["response"]
            side_effects = parsed_output["side_effects"]
        else:
            # Case B: Fallback
            is_fallback = True
            # Check if parsing failed (clean_llm_response returns error dict)
            if (
                "error" in parsed_output
                and "raw" in parsed_output
                and len(parsed_output) == 2
            ):
                # JSON parsing failed, treat raw text as response body (wrapped)
                final_response = {"raw_response": raw_response_text}
                # Log the raw text for debugging
                print(
                    f"WARN: LLM JSON parsing failed. Raw: {raw_response_text[:200]}..."
                )
            else:
                # Valid JSON but missing schema (e.g. just the body)
                final_response = parsed_output

        # Apply State Updates
        if side_effects:
            state_manager.apply_updates(side_effects)

        response_time_ms = (time.time() - start_time) * 1000

        log_interaction(
            {
                **base_event,
                "rag_query": rag_query,
                "context": context,
                "rag_metadata": rag_metadata,
                "response": final_response,
                "state_actions": side_effects,
                "is_fallback": is_fallback,
                "status_code": 200,
                "response_time_ms": response_time_ms,
                "provider": provider,
                "model": model,
            }
        )
        return JSONResponse(content=final_response)
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
