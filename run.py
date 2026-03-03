#!/usr/bin/env python3
"""
run.py — GenPot dual-protocol entry point.

Starts the FastAPI HTTP honeypot (via uvicorn) and the AsyncSSH honeypot
concurrently inside the **same** asyncio event loop, sharing a single
StateManager instance so mutations made through either protocol are
immediately visible on the other.

                ┌─────────────────────────────────────┐
                │           asyncio event loop         │
                │                                      │
                │   ┌──────────────┐  ┌─────────────┐ │
  HTTP ─────────┼──▶│  FastAPI     │  │  AsyncSSH   │◀─┼── SSH
  (port 8000)   │   │  (uvicorn)   │  │  (port 2222)│ │  (port 2222)
                │   └──────┬───────┘  └──────┬──────┘ │
                │          │                 │         │
                │          ▼                 ▼         │
                │       ┌──────────────────────┐       │
                │       │    StateManager        │       │
                │       │  (logs/world_state.json)│      │
                │       └──────────────────────┘       │
                └─────────────────────────────────────┘

Usage
─────
    python run.py                         # defaults: HTTP :8000, SSH :2222
    python run.py --http-port 80 --ssh-port 22
    python run.py --host 127.0.0.1 --http-port 8080 --ssh-port 2222
"""
from __future__ import annotations

import argparse
import asyncio

import uvicorn

# Import the FastAPI app AND the already-initialised StateManager instance
# from server.main so both protocols operate on the exact same object.
from server.main import app, state_manager
from server.ssh_server import start_ssh_server


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GenPot — LLM-powered dual-protocol honeypot (HTTP + SSH)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Bind address for both servers.",
    )
    parser.add_argument(
        "--http-port",
        type=int,
        default=8000,
        metavar="PORT",
        help="TCP port for the FastAPI / HTTP honeypot.",
    )
    parser.add_argument(
        "--ssh-port",
        type=int,
        default=2222,
        metavar="PORT",
        help="TCP port for the AsyncSSH honeypot.",
    )
    return parser.parse_args()


# ── Async main ────────────────────────────────────────────────────────────────

async def _main(host: str, http_port: int, ssh_port: int) -> None:
    # ── uvicorn (FastAPI) ─────────────────────────────────────────────────
    # `loop="none"` tells uvicorn not to create its own event loop; we hand
    # it the running loop managed by asyncio.run().
    http_config = uvicorn.Config(
        app,
        host=host,
        port=http_port,
        log_level="info",
        loop="none",
    )
    http_server = uvicorn.Server(http_config)

    print(f"[HTTP] FastAPI honeypot starting on http://{host}:{http_port}")
    print(f"[SSH]  SSH    honeypot starting on      {host}:{ssh_port}")
    print("[*]   Press Ctrl-C to stop both servers.\n")

    # ── Run both servers concurrently ─────────────────────────────────────
    # asyncio.gather propagates the first exception so a fatal error in
    # either server brings the whole process down cleanly.
    await asyncio.gather(
        http_server.serve(),
        start_ssh_server(state_manager, host=host, port=ssh_port),
    )


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = _parse_args()
    try:
        asyncio.run(
            _main(
                host=args.host,
                http_port=args.http_port,
                ssh_port=args.ssh_port,
            )
        )
    except KeyboardInterrupt:
        print("\n[*] GenPot shutting down — goodbye.")
