# server/emulators/http_emulator.py
"""
HTTP emulator — FastAPI application factory.

Constructs a fully-wired FastAPI app whose routes close over the
injected ``GenPotEngine`` and ``RAGSystem`` instances.
"""

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from server.core.engine import GenPotEngine
from server.core.models import UnifiedRequest
from server.rag_system import RAGSystem


class RAGInspectRequest(BaseModel):
    query: str
    top_k: int | None = None


def create_http_app(engine: GenPotEngine, rag_system: RAGSystem) -> FastAPI:
    """Build and return a FastAPI application wired to *engine* and *rag_system*."""

    app = FastAPI()

    @app.post("/api/rag-inspect")
    async def inspect_rag(request: RAGInspectRequest):
        result = rag_system.inspect_query(request.query, request.top_k)
        return result

    @app.api_route(
        "/{full_path:path}",
        methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"],
    )
    async def decoy_api_endpoint(request: Request, full_path: str):
        body_bytes = await request.body()

        unified_req = UnifiedRequest(
            protocol="http",
            source_ip=request.client.host if request.client else "",
            method=request.method,
            path="/" + full_path,
            headers=dict(request.headers),
            body=body_bytes.decode("utf-8", errors="ignore"),
        )

        response = await engine.process(unified_req)

        return JSONResponse(
            status_code=response.status_code,
            content=response.data,
        )

    return app
