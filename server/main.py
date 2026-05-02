"""
Bootstrap script — wires core dependencies and starts the enabled emulators.
"""

import asyncio
import logging
import signal

import uvicorn

from server import config_manager
from server.core.engine import GenPotEngine
from server.emulators.http_emulator import create_http_app
from server.emulators.smtp_emulator import SmtpEmulator
from server.rag_system import RAGSystem
from server.state_manager import StateManager

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


async def serve_smtp_forever(
    smtp_emulator: SmtpEmulator,
    stop_event: asyncio.Event,
) -> None:
    """Start the SMTP server and wait until *stop_event* is set."""
    try:
        server = await smtp_emulator.start()
        logger.info(
            "SMTP Emulator listening on %s:%s",
            smtp_emulator.host,
            smtp_emulator.port,
        )
        async with server:
            # Replace serve_forever() with an event-driven wait so the task
            # responds immediately to cancellation / shutdown signals.
            await stop_event.wait()
    except asyncio.CancelledError:
        logger.info("SMTP Emulator task cancelled.")
    except Exception as e:
        logger.error("Failed to start SMTP Emulator: %s", e)
    finally:
        logger.info("SMTP Emulator stopped.")


async def main() -> None:
    """Initialize and run all enabled emulators concurrently."""

    logger.info("Starting GenPot Honeypot...")

    try:
        conf = config_manager.load_config()
    except FileNotFoundError as e:
        logger.error(str(e))
        return

    # ── Core subsystems ────────────────────────────────────────────────
    rag_system = RAGSystem()
    state_manager = StateManager()
    engine = GenPotEngine(rag_system=rag_system, state_manager=state_manager)

    tasks: list[asyncio.Task] = []
    stop_event = asyncio.Event()
    uvicorn_server: uvicorn.Server | None = None

    # ── HTTP Emulator ──────────────────────────────────────────────────
    http_conf = conf.get("emulators", {}).get("http", {})
    if http_conf.get("enabled", False):
        port = http_conf.get("port", 8000)
        app = create_http_app(engine, rag_system)
        u_config = uvicorn.Config(
            app=app,
            host="0.0.0.0",
            port=port,
            log_level="info",
        )
        uvicorn_server = uvicorn.Server(u_config)
        tasks.append(asyncio.create_task(uvicorn_server.serve(), name="http-emulator"))
        logger.info("HTTP Emulator scheduled to start on port %d", port)
    else:
        logger.info("HTTP Emulator is disabled.")

    # ── SMTP Emulator ──────────────────────────────────────────────────
    smtp_conf = conf.get("emulators", {}).get("smtp", {})
    if smtp_conf.get("enabled", False):
        port = smtp_conf.get("port", 8025)
        smtp = SmtpEmulator(engine, state_manager, host="0.0.0.0", port=port)
        tasks.append(
            asyncio.create_task(
                serve_smtp_forever(smtp, stop_event), name="smtp-emulator"
            )
        )
        logger.info("SMTP Emulator scheduled to start on port %d", port)
    else:
        logger.info("SMTP Emulator is disabled.")

    if not tasks:
        logger.warning("No emulators are enabled in configuration! Exiting.")
        return

    # ── Signal handling ────────────────────────────────────────────────
    # Use loop.add_signal_handler (Unix only) so shutdown is handled
    # inside the event loop — no KeyboardInterrupt race conditions.
    def _handle_shutdown() -> None:
        logger.info("Shutdown signal received. Stopping GenPot gracefully...")
        # 1. Tell uvicorn to drain and exit.
        if uvicorn_server is not None:
            uvicorn_server.should_exit = True
        # 2. Wake the SMTP stop_event so serve_smtp_forever() returns.
        stop_event.set()
        # 3. Cancel all tasks so asyncio.gather() returns promptly.
        for t in tasks:
            t.cancel()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _handle_shutdown)

    # ── Run ────────────────────────────────────────────────────────────
    try:
        await asyncio.gather(*tasks, return_exceptions=True)
    finally:
        logger.info("GenPot stopped.")


if __name__ == "__main__":
    asyncio.run(main())
