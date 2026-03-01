# server/main.py
"""
Bootstrap script — wires core dependencies and starts the HTTP emulator.
"""

from server.core.engine import GenPotEngine
from server.emulators.http_emulator import create_http_app
from server.rag_system import RAGSystem
from server.state_manager import StateManager

# Initialize subsystems and engine
rag_system = RAGSystem()
state_manager = StateManager()
engine = GenPotEngine(rag_system=rag_system, state_manager=state_manager)

# Create the HTTP application
app = create_http_app(engine, rag_system)

if __name__ == "__main__":
    import uvicorn

    # Read port from config later, default 8000 for now
    uvicorn.run("server.main:app", host="0.0.0.0", port=8000, reload=True)
