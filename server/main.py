from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .core.engine import GenPotEngine
from .core.models import UnifiedRequest
from .rag_system import RAGSystem
from .state_manager import StateManager

# Initialize subsystems and engine
rag_system = RAGSystem()
state_manager = StateManager()
engine = GenPotEngine(rag_system=rag_system, state_manager=state_manager)

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
