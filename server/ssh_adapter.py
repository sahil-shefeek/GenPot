"""
server/ssh_adapter.py

Public-facing adapter layer for GenPot's AsyncSSH server.

This module exposes two integration points:

1.  **``SSHServer``** – a stable public subclass of ``_GenPotSSHServer``
    suitable for ``asyncssh.create_server(lambda: SSHServer(sm), ...)``.
    All session dispatch, credential logging, and MOTD logic live in the
    parent class; see ``ssh_server.py`` for implementation details.

2.  **``handle_ssh_session``** – an async coroutine compatible with
    asyncssh's *process* API::

        await asyncssh.create_server(
            ...,
            process_factory=functools.partial(
                handle_ssh_session, state_manager=sm
            ),
        )

    It implements the same Ubuntu 22.04 shell simulation – VFS context
    injection, LLM round-trip via ``run_in_executor``, ``<SIDE_EFFECT>``
    parsing, and typewriter streaming output – as the session-class path,
    offering an alternative integration point for callers that prefer the
    process-handler style.

3.  **``typewrite``** – a standalone async helper that streams a string
    to any write callable one character at a time with a small delay,
    creating the typewriter effect used by both integration paths.
"""
from __future__ import annotations

import asyncio
import time
from typing import Callable

import asyncssh

from .config_manager import load_config
from .llm_client import LLMRateLimitError, generate_response
from .logger import log_interaction
from .prompt_manager import craft_ssh_prompt
from .ssh_server import (
    _EXIT_COMMANDS,
    _GenPotSSHServer,
    _get_command_context,
    _MOTD,
    start_ssh_server,
)
from .state_manager import StateManager
from .utils import parse_ssh_response

# ── Public re-exports ─────────────────────────────────────────────────────────
__all__ = [
    "SSHServer",
    "handle_ssh_session",
    "start_ssh_server",
    "typewrite",
]

# ── Typewriter constants ──────────────────────────────────────────────────────
# Delay between characters when streaming LLM output.
# 0.008 s ≈ 125 chars/sec — fast enough to feel snappy, slow enough to convey
# the illusion that a real shell is computing and printing its answer.
_TYPEWRITER_DELAY: float = 0.008


# ─────────────────────────────────────────────────────────────────────────────
# Public SSHServer alias
# ─────────────────────────────────────────────────────────────────────────────
class SSHServer(_GenPotSSHServer):
    """
    Public name for GenPot's AsyncSSH honeypot server.

    Thin subclass of the internal ``_GenPotSSHServer`` that exposes a
    stable, importable name.  Instantiate with a shared ``StateManager``::

        server = asyncssh.create_server(
            lambda: SSHServer(state_manager),
            host="0.0.0.0",
            port=2222,
            server_host_keys=[str(key_path)],
        )

    All credential capture, session dispatch, command history, and MOTD
    logic are implemented in the parent; this subclass exists solely to
    provide a clean public API surface.
    """


# ─────────────────────────────────────────────────────────────────────────────
# Typewriter helper
# ─────────────────────────────────────────────────────────────────────────────
async def typewrite(
    text: str,
    write_fn: Callable[[str], None],
    delay: float = _TYPEWRITER_DELAY,
) -> None:
    """
    Stream *text* to a terminal one character at a time with a small delay
    between writes, creating a typewriter effect.

    Parameters
    ----------
    text:
        The string to stream.  May contain ``\\r\\n`` line endings.
    write_fn:
        Any callable that accepts a single ``str`` and writes it to the SSH
        channel – typically ``process.stdout.write`` or ``chan.write``.
    delay:
        Seconds to sleep between characters.  Defaults to
        ``_TYPEWRITER_DELAY`` (0.008 s).
    """
    for char in text:
        write_fn(char)
        await asyncio.sleep(delay)


# ─────────────────────────────────────────────────────────────────────────────
# Process-based session handler
# ─────────────────────────────────────────────────────────────────────────────

def _prompt(cwd: str) -> str:
    """Return the shell prompt string for *cwd*."""
    short = "~" if cwd == "/root" else cwd
    return f"root@genpot:{short}# "


def _resolve_cd(command: str, cwd: str) -> str:
    """Compute the new CWD for a ``cd`` command without an LLM call."""
    parts  = command.split(maxsplit=1)
    target = parts[1].strip() if len(parts) > 1 else "~"
    if target in ("", "~"):
        return "/root"
    if target.startswith("/"):
        return target.rstrip("/") or "/"
    if target == "..":
        parent = cwd.rstrip("/").rsplit("/", 1)[0]
        return parent or "/"
    if target == ".":
        return cwd
    return cwd.rstrip("/") + "/" + target


async def handle_ssh_session(
    process: asyncssh.SSHServerProcess,  # type: ignore[name-defined]
    state_manager: StateManager,
) -> None:
    """
    Drive a single SSH session via asyncssh's *process* API.

    This coroutine is an alternative integration path to the ``SSHServer``
    class.  Pass a partially-applied version as ``process_factory`` when
    creating the server::

        import functools
        await asyncssh.create_server(
            ...,
            process_factory=functools.partial(
                handle_ssh_session, state_manager=state_manager
            ),
        )

    Pipeline
    --------
    1.  Display the Ubuntu 22.04 MOTD banner.
    2.  Prompt ``root@genpot:<cwd># `` and read commands from stdin.
    3.  Handle ``cd`` locally (no LLM call) so the prompt stays accurate.
    4.  For all other commands: ``craft_ssh_prompt`` →
        ``generate_response`` (in a thread-pool executor) →
        ``parse_ssh_response`` → typewriter-stream the output.
    5.  Persist any ``<SIDE_EFFECT>`` mutations via
        ``state_manager.apply_updates()``.
    6.  Log every interaction through ``log_interaction``.
    """
    # ── Connection metadata ───────────────────────────────────────────────
    conn = process.get_extra_info("connection")
    peer = conn.get_extra_info("peername") if conn else None
    client_ip: str = peer[0] if peer else "unknown"
    username: str = process.get_extra_info("username") or "unknown"

    config = load_config()
    provider: str = config.get("honeypot_provider", "gemini")
    model: str = config.get("honeypot_model", "gemini-1.5-flash")

    cwd: str = "/root"
    history: list = []

    # ── Greet ─────────────────────────────────────────────────────────────
    process.stdout.write(_MOTD)
    process.stdout.write(_prompt(cwd))

    try:
        async for raw_line in process.stdin:
            command: str = raw_line.rstrip("\r\n").strip()

            if not command:
                process.stdout.write(_prompt(cwd))
                continue

            # ── Built-ins (no LLM call) ───────────────────────────────────
            if command in _EXIT_COMMANDS:
                process.stdout.write("exit\r\n")
                break

            if command == "clear":
                process.stdout.write("\033[2J\033[H")
                process.stdout.write(_prompt(cwd))
                continue

            if command.startswith("cd"):
                cwd = _resolve_cd(command, cwd)
                history.append({"command": command, "output": ""})
                log_interaction(
                    {
                        "event": "ssh_interaction",
                        "ip": client_ip,
                        "username": username,
                        "command": command,
                        "cwd": cwd,
                        "response": "",
                        "event_status": "local_cd",
                        "response_time_ms": 0,
                        "provider": "local",
                        "model": "builtin",
                    }
                )
                process.stdout.write(_prompt())
                continue

            # ── LLM call ──────────────────────────────────────────────────
            start_time = time.time()
            event_status = "success"
            response_text = ""

            try:
                file_context = _get_command_context(command, cwd, state_manager)
                prompt = craft_ssh_prompt(command, cwd, file_context, history)

                loop = asyncio.get_running_loop()
                raw = await loop.run_in_executor(
                    None,
                    lambda: generate_response(
                        prompt, provider_type=provider, model_name=model
                    ),
                )

                parsed = parse_ssh_response(raw)
                output_text: str = parsed["output"]
                side_effects: list = parsed["side_effects"]

                # Persist VFS / session mutations reported by the LLM.
                if side_effects:
                    state_manager.apply_updates(side_effects)

                # Honour an explicit cwd side-effect from the LLM.
                new_cwd = next(
                    (
                        e.get("value")
                        for e in side_effects
                        if e.get("scope") == "session" and e.get("key") == "cwd"
                    ),
                    None,
                )
                if new_cwd:
                    cwd = new_cwd

                # Maintain rolling command/output history (max 20 entries).
                history.append({"command": command, "output": output_text})
                if len(history) > 20:
                    history = history[-20:]

                response_text = output_text

                if output_text:
                    terminal_output = (
                        output_text.replace("\n", "\r\n").rstrip("\r\n") + "\r\n"
                    )
                    await typewrite(terminal_output, process.stdout.write)

            except LLMRateLimitError as exc:
                process.stdout.write(
                    "bash: service temporarily unavailable, try again later\r\n"
                )
                event_status = "error_rate_limit"
                response_text = str(exc)

            except Exception as exc:
                cmd_name = command.split()[0] if command.split() else command
                process.stdout.write(f"bash: {cmd_name}: command not found\r\n")
                event_status = "error"
                response_text = str(exc)

            finally:
                response_time_ms = (time.time() - start_time) * 1_000
                log_interaction(
                    {
                        "event": "ssh_interaction",
                        "ip": client_ip,
                        "username": username,
                        "command": command,
                        "cwd": cwd,
                        "response": response_text,
                        "event_status": event_status,
                        "response_time_ms": round(response_time_ms, 2),
                        "provider": provider,
                        "model": model,
                    }
                )

            process.stdout.write(_prompt(cwd))

    except asyncssh.BreakReceived:
        pass

    finally:
        process.exit(0)
