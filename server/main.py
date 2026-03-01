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
state_manager = StateManager()

app = FastAPI()


class RAGInspectRequest(BaseModel):
    query: str
    top_k: int | None = None


@app.post("/api/rag-inspect")
async def inspect_rag(request: RAGInspectRequest):
    result = rag_system.inspect_query(request.query, request.top_k)
    return result


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

    headers_dict = dict(request.headers)

    request_data = {
        "method": method,
        "path": path,
        "body": body_str,
        "headers": headers_dict,
    }

    rag_query = f"{method} {path}"
    context = rag_system.get_context(rag_query)
    state_context = state_manager.get_context(path, headers_dict)
    prompt = craft_prompt(
        method=method,
        path=path,
        body=body_str,
        headers=headers_dict,
        context=context,
        state_context=state_context,
    )

    try:
        config = load_config()
        provider = config.get("honeypot_provider", "gemini")
        model = config.get("honeypot_model", "gemini-1.5-flash")

        raw_response_text = generate_response(
            prompt,
            provider_type=provider,
            model_name=model,
        )
        parsed_llm_output = clean_llm_response(raw_response_text)

        api_response = {}
        side_effects = []

        if (
            isinstance(parsed_llm_output, dict)
            and "response" in parsed_llm_output
            and "side_effects" in parsed_llm_output
        ):
            api_response = parsed_llm_output.get("response", {})
            side_effects_raw = parsed_llm_output.get("side_effects", [])

            if isinstance(side_effects_raw, list):
                side_effects = side_effects_raw
        else:
            api_response = parsed_llm_output

        SIMILARITY_THRESHOLD = 0.5
        similarity_score = rag_system.compute_similarity(context, api_response)

        if similarity_score < SIMILARITY_THRESHOLD:
            print(
                f"[WARN] Response rejected due to low similarity (score: {similarity_score:.4f})"
            )

        if side_effects:
            state_manager.apply_updates(side_effects)

        response_time_ms = (time.time() - start_time) * 1000

        log_interaction(
            protocol="http",
            source_ip=client_ip,
            request_data=request_data,
            response_data={"status_code": 200, "body": api_response},
            genpot_metrics={
                "rag_query": rag_query,
                "rag_context": context,
                "similarity_score": round(similarity_score, 4),
                "llm_provider": provider,
                "llm_model": model,
                "latency_ms": response_time_ms,
                "state_actions": side_effects,
            },
        )
        return JSONResponse(content=api_response)
    except LLMRateLimitError as e:
        response_time_ms = (time.time() - start_time) * 1000
        log_interaction(
            protocol="http",
            source_ip=client_ip,
            request_data=request_data,
            response_data={"status_code": 429},
            genpot_metrics={
                "rag_query": rag_query,
                "rag_context": context,
                "latency_ms": response_time_ms,
            },
            error=str(e),
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
            protocol="http",
            source_ip=client_ip,
            request_data=request_data,
            response_data={"status_code": 500},
            genpot_metrics={
                "rag_query": rag_query,
                "rag_context": context,
                "latency_ms": response_time_ms,
            },
            error=str(e),
        )
        return JSONResponse(
            content={"error": "An internal server error occurred."},
            status_code=500,
        )
