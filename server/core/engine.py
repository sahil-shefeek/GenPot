# server/core/engine.py
"""
GenPotEngine — the protocol-agnostic orchestration core.

Receives a ``UnifiedRequest`` from any network adapter, drives the
RAG → Prompt → LLM → Parse → State → Log lifecycle, and returns a
``UnifiedResponse``.
"""

import time

from server import config_manager
from server.core.models import UnifiedRequest, UnifiedResponse
from server.core.prompting import HttpPromptStrategy, SmtpPromptStrategy
from server.llm_client import LLMRateLimitError, generate_response
from server.logger import log_interaction
from server.rag_system import RAGSystem
from server.state_manager import StateManager

SIMILARITY_THRESHOLD = 0.5


class GenPotEngine:
    """Central honeypot lifecycle engine."""

    def __init__(self, rag_system: RAGSystem, state_manager: StateManager) -> None:
        self.rag_system = rag_system
        self.state_manager = state_manager
        self.config = config_manager.load_config()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def process(self, request: UnifiedRequest) -> UnifiedResponse:
        """Run the full honeypot lifecycle for *request*."""

        start_time = time.time()

        # --- Build request data dict for prompt / logging ---
        request_data = {
            "method": request.method,
            "path": request.path,
            "body": request.body,
            "headers": request.headers,
            "command": request.command,
        }

        # --- Gather context ---
        rag_query = f"{request.method} {request.path}"
        context = self.rag_system.get_context(rag_query)
        state_context = self.state_manager.get_context(
            request.path, request.headers, session_id=request.session_id
        )

        # --- Strategy (protocol-aware) ---
        if request.protocol.lower() == "smtp":
            strategy = SmtpPromptStrategy()
        else:
            strategy = HttpPromptStrategy()

        # Unpack the two prompts
        system_prompt, prompt = strategy.build_prompt(request_data, context, state_context)

        try:
            # --- LLM config ---
            emulator_cfg = config_manager.get_emulator_config(request.protocol)
            provider = emulator_cfg.get("provider", "gemini")
            model = emulator_cfg.get("model", "gemini-2.5-flash")

            # Optional: Add a "thinking: true/false" key to your genpot.yaml
            use_thinking = emulator_cfg.get("thinking", True)

            # --- LLM call ---
            raw_response_text = generate_response(
                prompt=prompt,
                system_prompt=system_prompt,
                provider_type=provider,
                model_name=model,
                thinking=use_thinking,
            )
            parsed_llm_output = strategy.parse_response(raw_response_text)

            api_response = parsed_llm_output.get("response", {})
            side_effects_raw = parsed_llm_output.get("side_effects", [])
            side_effects = (
                side_effects_raw if isinstance(side_effects_raw, list) else []
            )

            # --- RAG similarity check ---
            similarity_score = self.rag_system.compute_similarity(context, api_response)
            if similarity_score < SIMILARITY_THRESHOLD:
                print(
                    f"[WARN] Response rejected due to low similarity "
                    f"(score: {similarity_score:.4f})"
                )

            # --- State side effects ---
            if side_effects:
                self.state_manager.apply_updates(side_effects)

            # --- Logging ---
            response_time_ms = (time.time() - start_time) * 1000
            log_interaction(
                protocol=request.protocol,
                source_ip=request.source_ip,
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

            return UnifiedResponse(status_code=200, data=api_response)

        except LLMRateLimitError as e:
            response_time_ms = (time.time() - start_time) * 1000
            log_interaction(
                protocol=request.protocol,
                source_ip=request.source_ip,
                request_data=request_data,
                response_data={"status_code": 429},
                genpot_metrics={
                    "rag_query": rag_query,
                    "rag_context": context,
                    "latency_ms": response_time_ms,
                },
                error=str(e),
            )
            return UnifiedResponse(
                status_code=429,
                data={
                    "error": "Service Temporarily Unavailable (Rate Limit Exceeded)",
                    "retry_after": e.retry_after,
                },
            )

        except Exception as e:
            response_time_ms = (time.time() - start_time) * 1000
            log_interaction(
                protocol=request.protocol,
                source_ip=request.source_ip,
                request_data=request_data,
                response_data={"status_code": 500},
                genpot_metrics={
                    "rag_query": rag_query,
                    "rag_context": context,
                    "latency_ms": response_time_ms,
                },
                error=str(e),
            )
            return UnifiedResponse(
                status_code=500,
                data={"error": "An internal server error occurred."},
            )
