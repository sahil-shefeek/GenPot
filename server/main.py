# server/main.py
"""
Bootstrap script — wires core dependencies and starts the enabled emulators (HTTP and/or SMTP).
"""

import asyncio
import logging

import uvicorn

from server import config_manager
from server.core.engine import GenPotEngine
from server.emulators.http_emulator import create_http_app
from server.emulators.smtp_emulator import SmtpEmulator
from server.rag_system import RAGSystem
from server.state_manager import StateManager

# Set up logging for the bootstrap process
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


async def serve_smtp_forever(smtp_emulator: SmtpEmulator):
    """Helper to start the SMTP server and block until it completes."""
    try:
        server = await smtp_emulator.start()
        logger.info(
            f"SMTP Emulator listening on {smtp_emulator.host}:{smtp_emulator.port}"
        )
        async with server:
            await server.serve_forever()
    except Exception as e:
        logger.error(f"Failed to start SMTP Emulator: {e}")


async def main():
    """Main entry point to initialize and run the honeypot concurrently."""

    logger.info("Starting GenPot Honeypot...")

    # 1. Load configuration
    try:
        conf = config_manager.load_config()
    except FileNotFoundError as e:
        logger.error(str(e))
        return

    # 2. Initialize Core Subsystems
    rag_system = RAGSystem()
    state_manager = StateManager()
    engine = GenPotEngine(rag_system=rag_system, state_manager=state_manager)

    tasks = []

    # 3. Conditionally start HTTP Emulator
    http_conf = conf.get("emulators", {}).get("http", {})
    if http_conf.get("enabled", False):
        port = http_conf.get("port", 8000)
        app = create_http_app(engine, rag_system)

        # Configure uvicorn programmatically
        u_config = uvicorn.Config(
            app=app,
            host="0.0.0.0",
            port=port,
            log_level="info",
        )
        server = uvicorn.Server(u_config)
        tasks.append(server.serve())
        logger.info(f"HTTP Emulator scheduled to start on port {port}")
    else:
        logger.info("HTTP Emulator is disabled.")

    # 4. Conditionally start SMTP Emulator
    smtp_conf = conf.get("emulators", {}).get("smtp", {})
    if smtp_conf.get("enabled", False):
        port = smtp_conf.get("port", 8025)
        smtp = SmtpEmulator(engine, state_manager, host="0.0.0.0", port=port)
        tasks.append(serve_smtp_forever(smtp))
        logger.info(f"SMTP Emulator scheduled to start on port {port}")
    else:
        logger.info("SMTP Emulator is disabled.")

    if not tasks:
        logger.warning("No emulators are enabled in configuration! Exiting.")
        return

    # 5. Run all emulator tasks concurrently
    try:
        await asyncio.gather(*tasks)
    except KeyboardInterrupt:
        logger.info("GenPot stopped by user.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
