# server/emulators/smtp_emulator.py
"""
SMTP emulator — asynchronous TCP server.

Handles raw SMTP over ``asyncio`` streams.  Each client connection gets a
unique session, and every SMTP command is forwarded through the
``GenPotEngine`` as a ``UnifiedRequest``.  The multi-line ``DATA`` payload
is buffered locally until the lone-period terminator is received.
"""

import asyncio
import uuid

from server.core.engine import GenPotEngine
from server.core.models import UnifiedRequest
from server.state_manager import StateManager


class SmtpEmulator:
    """Async TCP server that speaks just enough SMTP to fool scanners."""

    def __init__(
        self,
        engine: GenPotEngine,
        state_manager: StateManager,
        host: str = "0.0.0.0",
        port: int = 8025,
    ) -> None:
        self.engine = engine
        self.state_manager = state_manager
        self.host = host
        self.port = port

    async def start(self) -> asyncio.AbstractServer:
        """Start listening and return the ``asyncio`` server handle."""
        server = await asyncio.start_server(self.handle_client, self.host, self.port)
        return server

    # ------------------------------------------------------------------
    # Per-connection handler
    # ------------------------------------------------------------------

    async def handle_client(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        """Manage a single SMTP client connection from greeting to QUIT."""

        peername = writer.get_extra_info("peername")
        client_ip = peername[0] if peername else ""
        session_id = uuid.uuid4().hex

        # Send the SMTP greeting immediately.
        writer.write(b"220 ESMTP GenPot Honeypot\r\n")
        await writer.drain()

        in_data_mode = False
        data_buffer: list[str] = []

        try:
            while True:
                raw = await reader.readline()
                if not raw:
                    # EOF — client disconnected.
                    break

                line = raw.decode("utf-8", errors="ignore").rstrip("\r\n")

                if in_data_mode:
                    if line == ".":
                        # End-of-data marker — flush the buffer.
                        body_payload = "\r\n".join(data_buffer)
                        req = UnifiedRequest(
                            protocol="smtp",
                            source_ip=client_ip,
                            method="SMTP",
                            path="/smtp",
                            headers={},
                            command="[DATA PAYLOAD]",
                            body=body_payload,
                            session_id=session_id,
                        )
                        response = await self.engine.process(req)
                        reply = str(response.data)
                        writer.write(f"{reply}\r\n".encode())
                        await writer.drain()

                        in_data_mode = False
                        data_buffer = []
                    else:
                        data_buffer.append(line)
                    continue

                # --- Normal command mode ---
                req = UnifiedRequest(
                    protocol="smtp",
                    source_ip=client_ip,
                    method="SMTP",
                    path="/smtp",
                    headers={},
                    command=line,
                    body="",
                    session_id=session_id,
                )
                response = await self.engine.process(req)
                reply = str(response.data)
                writer.write(f"{reply}\r\n".encode())
                await writer.drain()

                # Transition into DATA mode when the engine replies with 3xx.
                if line.upper().startswith("DATA") and reply.startswith("3"):
                    in_data_mode = True
                    data_buffer = []

                # Honour QUIT — send the response then close.
                if line.upper().startswith("QUIT"):
                    break

        finally:
            self.state_manager.clear_session(session_id)
            writer.close()
