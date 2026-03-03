import asyncio
import functools
import time

import asyncssh
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .config_manager import load_config
from .llm_client import LLMRateLimitError, generate_response
from .logger import log_interaction
from .prompt_manager import craft_prompt
from .rag_system import RAGSystem
from .ssh_adapter import SSHServer, handle_ssh_session
from .ssh_server import _KEY_PATH
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

    base_event = {
        "event": "interaction",
        "ip": client_ip,
        "method": method,
        "path": path,
        "body": body_str,
        "headers": headers_dict,
    }
    # Removed early log_interaction call to consolidate logs

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
            {
                **base_event,
                "rag_query": rag_query,
                "context": context,
                "response": api_response,
                "state_actions": side_effects,
                "status_code": 200,
                "response_time_ms": response_time_ms,
                "provider": provider,
                "model": model,
                "similarity_score": round(similarity_score, 4),
            }
        )
        return JSONResponse(content=api_response)
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


# ── Concurrent server launch ──────────────────────────────────────────────────

async def start_api_server(host: str = "0.0.0.0", port: int = 8000) -> None:
    """
    Serve the FastAPI application with uvicorn inside the running event loop.

    ``loop="none"`` instructs uvicorn not to spin up its own event loop — it
    uses the one already managed by ``asyncio.run()`` in ``main()``.
    """
    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_level="info",
        loop="none",
    )
    server = uvicorn.Server(config)
    print(f"[HTTP] FastAPI honeypot listening on http://{host}:{port}")
    await server.serve()


async def start_ssh_server(host: str = "0.0.0.0", port: int = 2222) -> None:
    """
    Start the AsyncSSH honeypot using the process-factory integration path.

    *Server factory* – ``SSHServer`` (the public alias for ``_GenPotSSHServer``)
    handles connection-level auth: it logs every credential attempt via
    ``validate_password`` and accepts all of them so attackers walk straight in.

    *Process factory* – ``handle_ssh_session`` drives each interactive session:
    MOTD banner, local ``cd`` resolution, LLM round-trip via
    ``run_in_executor``, ``<SIDE_EFFECT>`` parsing, typewriter output, and
    ``log_interaction`` logging.

    The host key is persisted at ``logs/ssh_host_key``.  A fresh ED25519 key is
    generated on first boot; subsequent restarts reload it so clients don't see
    a host-key-changed warning.
    """
    _KEY_PATH.parent.mkdir(parents=True, exist_ok=True)

    if _KEY_PATH.exists():
        host_key = asyncssh.read_private_key(str(_KEY_PATH))
        print(f"[SSH]  Loaded persistent host key from {_KEY_PATH}")
    else:
        host_key = asyncssh.generate_private_key("ssh-ed25519")
        host_key.write_private_key(str(_KEY_PATH))
        print(f"[SSH]  Generated new ED25519 host key \u2192 {_KEY_PATH}")

    server = await asyncssh.create_server(
        lambda: SSHServer(state_manager),
        host,
        port,
        server_host_keys=[host_key],
        process_factory=functools.partial(
            handle_ssh_session, state_manager=state_manager
        ),
    )
    print(f"[SSH]  AsyncSSH honeypot listening on {host}:{port}")
    async with server:
        await server.wait_closed()


async def main(
    host: str = "0.0.0.0",
    http_port: int = 8000,
    ssh_port: int = 2222,
) -> None:
    """
    Launch the HTTP and SSH honeypot servers concurrently.

    Both servers run inside the **same** asyncio event loop and share the
    module-level ``state_manager`` instance, so filesystem mutations from either
    protocol are immediately visible on the other.

    ``asyncio.gather`` propagates the first fatal exception, bringing the whole
    process down cleanly rather than silently continuing with one server dead.
    """
    print(f"[*]   Starting GenPot — HTTP on :{http_port}, SSH on :{ssh_port}")
    print("[*]   Press Ctrl-C to stop both servers.\n")
    await asyncio.gather(
        start_api_server(host, http_port),
        start_ssh_server(host, ssh_port),
    )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[*] Shutting down GenPot \u2014 goodbye.")
