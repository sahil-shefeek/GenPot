"""
server/ssh_server.py

AsyncSSH-based honeypot SSH server for GenPot.

Design goals
─────────────
• Accept **every** credential attempt and log the username + password so
  attackers reveal their wordlists.
• Simulate a realistic interactive Linux root shell (and non-interactive
  exec mode) backed by the same LLM already used for the HTTP honeypot.
• Share the *single* StateManager instance with the FastAPI component so
  that state mutations from either protocol are immediately visible on
  the other (e.g. a file "created" via POST /api/files appears in `ls`).
• Log every command, its LLM-generated response, and timing data through
  the existing log_interaction utility.
"""
from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Optional

import asyncssh

from .config_manager import load_config
from .llm_client import LLMProviderError, LLMRateLimitError, generate_response
from .logger import log_interaction
from .prompt_manager import craft_ssh_prompt
from .state_manager import StateManager
from .utils import parse_ssh_response

# ── Persistent host-key location ──────────────────────────────────────────────
# Generated on first boot; reused on subsequent restarts so SSH clients don't
# complain about a changed host key.
_KEY_PATH = Path(__file__).resolve().parents[1] / "logs" / "ssh_host_key"

# ── Fake MOTD banner shown after successful authentication ─────────────────────
_MOTD = (
    "Ubuntu 22.04.3 LTS (GNU/Linux 5.15.0-91-generic x86_64)\r\n"
    "\r\n"
    " * Documentation:  https://help.ubuntu.com\r\n"
    " * Management:     https://landscape.canonical.com\r\n"
    " * Support:        https://ubuntu.com/advantage\r\n"
    "\r\n"
    "Last login: Sat Mar  1 04:23:11 2026 from 203.0.113.42\r\n"
    "\r\n"
)

# ── LLM prompt construction ───────────────────────────────────────────────────
# Prompt building is delegated to craft_ssh_prompt() in prompt_manager.py.
# Side-effect parsing (the <SIDE_EFFECT> tag) lives in parse_ssh_response()
# in utils.py.  Neither the template string nor JSON parsing belong here.

# Commands that terminate the session without an LLM round-trip.
_EXIT_COMMANDS = frozenset({"exit", "logout", "quit"})

# Commands that consume file content — we pre-fetch absolute-path arguments
# from the VFS and inject them into the LLM context so the model sees the
# actual file contents rather than hallucinating them.
_READ_CMDS = frozenset({
    "cat", "tail", "head", "less", "more",
    "grep", "egrep", "fgrep", "zgrep",
    "stat", "file", "wc", "strings", "xxd", "od", "nl", "tac",
    "diff", "vimdiff", "view",
})


def _get_command_context(command: str, cwd: str, state_manager: StateManager) -> str:
    """
    Build the VFS context string passed to ``craft_ssh_prompt``.

    For read-oriented commands (``cat``, ``tail``, ``grep``, …) that
    reference an absolute path, the target file's full metadata block is
    merged into the standard CWD directory listing under the key
    ``"[file] /abs/path"`` so the LLM sees the real file content and can
    return accurate output.

    Example::

        cat /etc/passwd
        → context includes both the CWD listing **and**
          { "[file] /etc/passwd": {"type": "file", "content": "root:x:…"} }
    """
    base: dict = json.loads(state_manager.get_ssh_context(cwd))

    parts = command.split()
    if len(parts) >= 2 and parts[0] in _READ_CMDS:
        fs = state_manager.state.get("filesystem", {})
        for arg in parts[1:]:
            if arg.startswith("-"):
                continue  # skip flags
            if arg.startswith("/"):
                if arg in fs:
                    base[f"[file] {arg}"] = fs[arg]
                # Only enrich the first absolute-path argument per command.
                break

    return json.dumps(base, indent=2)


# Delay between characters when streaming LLM output in interactive mode.
# 0.012 s ≈ 80 chars/sec — fast enough to feel snappy; slow enough to look
# like a real shell computing and printing its answer.
_TYPEWRITER_DELAY: float = 0.012

# Octal permission string → symbolic form used by ``ls -l``.
_PERM_MAP: dict[str, str] = {
    "755": "rwxr-xr-x", "644": "rw-r--r--", "700": "rwx------",
    "640": "rw-r-----", "600": "rw-------", "555": "r-xr-xr-x",
    "444": "r--r--r--", "777": "rwxrwxrwx",
}


# ─────────────────────────────────────────────────────────────────────────────
# Session — one instance per SSH channel
# ─────────────────────────────────────────────────────────────────────────────
class _GenPotSSHSession(asyncssh.SSHServerSession):
    """
    Handles a single SSH channel.

    Two modes are supported:
    • Interactive shell  – the attacker types commands; we echo input,
                           buffer until Enter, then dispatch to the LLM.
    • Non-interactive exec – the attacker runs `ssh host cmd`; we handle
                             the command and close the channel with an
                             appropriate exit code.
    """

    def __init__(
        self,
        state_manager: StateManager,
        client_ip: str,
        username: str,
        password: str = "",
    ) -> None:
        self._state_manager = state_manager
        self._client_ip = client_ip
        self._username = username
        self._password = password  # credential used to open this session

        # Each user lands in their own home directory.
        self._home: str = "/root" if username == "root" else f"/home/{username}"
        self._cwd: str  = self._home

        # Ensure the user's home directory exists in the VFS so ls works
        # immediately even for first-time connections.
        fs = state_manager.state.get("filesystem", {})
        if self._home not in fs:
            state_manager.apply_updates([{
                "action": "SET", "scope": "filesystem", "key": self._home,
                "value": {"type": "directory", "owner": username, "permissions": "700" if username == "root" else "755"},
            }])

        self._chan: Optional[asyncssh.SSHServerChannel] = None
        self._input_buffer: str = ""

        # Queue used to serialise commands in interactive mode.
        self._command_queue: asyncio.Queue[str] = asyncio.Queue()
        self._processor_task: Optional[asyncio.Task] = None
        # Rolling window of past commands injected into future prompts.
        self._command_history: list = []
        # Track whether the last character was \r so we can swallow the \n
        # in a \r\n pair (clients in cooked mode send both).
        self._last_was_cr: bool = False

    # ── Channel lifecycle ──────────────────────────────────────────────────

    def connection_made(self, chan: asyncssh.SSHServerChannel) -> None:  # type: ignore[override]
        self._chan = chan

    def pty_requested(self, term_type, term_size, term_modes) -> bool:  # type: ignore[override]
        """Accept the PTY allocation request.

        Returning ``True`` tells asyncssh to accept the PTY and put the
        client's terminal into raw mode (no local echo, character-at-a-time
        delivery).  Without this, asyncssh *rejects* the PTY, the client
        stays in cooked mode and echoes locally — causing every command to
        appear twice in the terminal output.
        """
        return True

    def shell_requested(self) -> bool:
        """Client requested an interactive shell (e.g. plain `ssh host`)."""
        assert self._chan is not None
        self._chan.write(_MOTD)
        self._write_prompt()
        self._processor_task = asyncio.ensure_future(self._command_processor())
        return True

    def exec_requested(self, command: str) -> bool:
        """Client requested a single command (e.g. `ssh host ls -la`)."""
        asyncio.ensure_future(self._handle_exec(command.strip()))
        return True

    def eof_received(self) -> None:
        if self._chan:
            self._chan.write("\r\nexit\r\n")
            self._chan.exit(0)

    def connection_lost(self, exc: Optional[Exception]) -> None:
        if self._processor_task:
            self._processor_task.cancel()

    # ── Keystroke handling (interactive mode only) ─────────────────────────

    def data_received(self, data: str, datatype) -> None:  # type: ignore[override]
        assert self._chan is not None
        for char in data:
            if char == "\r":
                # Carriage return (Enter).  Flush the buffer as a command and
                # remember that \r arrived so a trailing \n can be ignored.
                self._chan.write("\r\n")
                command = self._input_buffer.strip()
                self._input_buffer = ""
                self._command_queue.put_nowait(command)
                self._last_was_cr = True

            elif char == "\n":
                # \n immediately after \r is the second half of a \r\n pair —
                # skip it.  A bare \n (e.g. from a script) is treated as Enter
                # only when there is content in the buffer.
                if self._last_was_cr:
                    self._last_was_cr = False
                    continue
                if self._input_buffer:
                    self._chan.write("\r\n")
                    command = self._input_buffer.strip()
                    self._input_buffer = ""
                    self._command_queue.put_nowait(command)

            elif char in ("\x7f", "\x08"):  # DEL / Backspace
                self._last_was_cr = False
                if self._input_buffer:
                    self._input_buffer = self._input_buffer[:-1]
                    self._chan.write("\b \b")  # erase character on terminal

            elif char == "\x03":  # Ctrl-C
                self._last_was_cr = False
                self._input_buffer = ""
                self._chan.write("^C\r\n")
                self._write_prompt()

            elif char == "\x04":  # Ctrl-D (EOF on empty line → exit)
                self._last_was_cr = False
                if not self._input_buffer:
                    self._chan.write("exit\r\n")
                    self._chan.exit(0)

            else:
                self._last_was_cr = False
                self._input_buffer += char
                self._chan.write(char)  # local echo

    # ── Interactive command processor ──────────────────────────────────────

    async def _command_processor(self) -> None:
        """Dequeues and processes commands one-at-a-time in interactive mode."""
        try:
            while True:
                command = await self._command_queue.get()
                if command:
                    await self._handle_command(command)
                else:
                    # Empty line — just re-draw the prompt.
                    self._write_prompt()
        except asyncio.CancelledError:
            pass  # Connection was closed; exit cleanly.

    # ── Shared LLM I/O pipeline ──────────────────────────────────────────

    async def _call_llm(
        self, command: str, provider: str, model: str
    ) -> tuple[str, list]:
        """Build the prompt, call the LLM, and parse the response.

        Returns ``(terminal_output, side_effects)``.
        Kept separate so both interactive and exec handlers share one path
        without duplicating the prompt-build / executor / parse boilerplate.
        """
        prompt = self._build_prompt(command)
        loop   = asyncio.get_running_loop()
        # generate_response is synchronous (blocking HTTP); run it in a
        # thread-pool executor so we don't stall the event loop.
        raw    = await loop.run_in_executor(
            None,
            lambda: generate_response(prompt, provider_type=provider, model_name=model),
        )
        parsed = parse_ssh_response(raw)
        return parsed["output"], parsed["side_effects"]

    # ── Core LLM-backed command handler (interactive) ─────────────────────

    async def _handle_command(self, command: str) -> None:
        assert self._chan is not None

        # ── Handle built-ins without burning an LLM call ──────────────────
        if command in _EXIT_COMMANDS:
            self._chan.write("exit\r\n")
            self._chan.exit(0)
            return

        if command == "clear":
            self._chan.write("\033[2J\033[H")  # ANSI: erase screen + cursor home
            self._write_prompt()
            return

        # ── Deterministic VFS mutations (no LLM needed) ───────────────────
        # touch, mkdir, rm are handled locally so the VFS is always updated
        # even when the LLM forgets to emit a <SIDE_EFFECT> tag.
        handled = self._handle_vfs_builtin(command)
        if handled is not None:
            if handled:
                # Normalise to \r\n so every line starts at column 0.
                self._chan.write(handled.replace("\n", "\r\n") + "\r\n")
            self._write_prompt()
            return

        # ── LLM call ──────────────────────────────────────────────────────
        config        = load_config()
        provider      = config.get("honeypot_provider", "gemini")
        model         = config.get("honeypot_model", "gemini-1.5-flash")
        start_time    = time.time()
        response_text = ""
        event_status  = "success"

        try:
            # Split clean terminal output from filesystem/session mutations.
            terminal_output, side_effects = await self._call_llm(command, provider, model)

            # Some models echo the command as the first line — strip it.
            for prefix in (command, command.strip()):
                if terminal_output.startswith(prefix):
                    terminal_output = terminal_output[len(prefix):].lstrip("\n\r")
                    break

            # Persist any VFS or session mutations the LLM reported.
            if side_effects:
                self._state_manager.apply_updates(side_effects)

            # Update cwd: prefer an explicit session side-effect over a
            # local parse of the cd command.
            new_cwd = next(
                (e.get("value") for e in side_effects
                 if e.get("scope") == "session" and e.get("key") == "cwd"),
                None,
            )
            if new_cwd:
                self._cwd = new_cwd
            elif command.strip().startswith("cd"):
                self._cwd = self._resolve_cd(command.strip())

            # Keep a rolling window of history for future prompt injections.
            self._command_history.append(
                {"command": command, "output": terminal_output}
            )
            if len(self._command_history) > 20:
                self._command_history = self._command_history[-20:]

            response_text = terminal_output
            output = terminal_output.replace("\n", "\r\n").rstrip("\r\n")
            if output:
                await self._typewrite(output + "\r\n")

        except LLMRateLimitError as e:
            self._chan.write("bash: service temporarily unavailable, try again later\r\n")
            event_status = "error_rate_limit"
            response_text = str(e)

        except LLMProviderError as e:
            self._chan.write("bash: service temporarily unavailable, try again later\r\n")
            event_status = "error_provider"
            response_text = str(e)

        except Exception as e:
            cmd_name = command.split()[0] if command.split() else command
            self._chan.write(f"bash: {cmd_name}: command not found\r\n")
            event_status = "error"
            response_text = str(e)

        finally:
            response_time_ms = (time.time() - start_time) * 1000
            log_interaction(
                {
                    "event": "ssh_interaction",
                    "ip": self._client_ip,
                    "username": self._username,
                    "password": self._password,
                    "command": command,
                    "cwd": self._cwd,
                    "response": response_text,
                    "event_status": event_status,
                    "response_time_ms": round(response_time_ms, 2),
                    "provider": provider,
                    "model": model,
                }
            )
            self._write_prompt()

    # ── Non-interactive exec handler ───────────────────────────────────────

    async def _handle_exec(self, command: str) -> None:
        """Handles `ssh host <command>` style invocations."""
        assert self._chan is not None

        if not command:
            self._chan.exit(0)
            return

        config        = load_config()
        provider      = config.get("honeypot_provider", "gemini")
        model         = config.get("honeypot_model", "gemini-1.5-flash")
        start_time    = time.time()
        response_text = ""
        event_status  = "success"
        exit_code     = 0

        try:
            terminal_output, side_effects = await self._call_llm(command, provider, model)

            if side_effects:
                self._state_manager.apply_updates(side_effects)

            response_text = terminal_output
            output = terminal_output.replace("\n", "\r\n").rstrip("\r\n")
            if output:
                self._chan.write(output + "\r\n")

        except LLMRateLimitError as e:
            self._chan.write("bash: service temporarily unavailable, try again later\r\n")
            event_status = "error_rate_limit"
            response_text = str(e)
            exit_code = 1

        except LLMProviderError as e:
            self._chan.write("bash: service temporarily unavailable, try again later\r\n")
            event_status = "error_provider"
            response_text = str(e)
            exit_code = 1

        except Exception as e:
            cmd_name = command.split()[0] if command.split() else command
            self._chan.write(f"bash: {cmd_name}: command not found\r\n")
            event_status = "error"
            response_text = str(e)
            exit_code = 127

        finally:
            response_time_ms = (time.time() - start_time) * 1000
            log_interaction(
                {
                    "event": "ssh_exec",
                    "ip": self._client_ip,
                    "username": self._username,
                    "password": self._password,
                    "command": command,
                    "cwd": self._cwd,
                    "response": response_text,
                    "event_status": event_status,
                    "exit_code": exit_code,
                    "response_time_ms": round(response_time_ms, 2),
                    "provider": provider,
                    "model": model,
                }
            )
            self._chan.exit(exit_code)

    # ── Helpers ────────────────────────────────────────────────────────────

    def _write_prompt(self) -> None:
        assert self._chan is not None
        short_cwd   = "~" if self._cwd == self._home else self._cwd
        prompt_char = "#" if self._username == "root" else "$"
        self._chan.write(f"{self._username}@genpot:{short_cwd}{prompt_char} ")

    def _handle_vfs_builtin(self, command: str) -> Optional[str]:
        """
        Deterministically handle VFS-mutating commands so the filesystem state
        is always updated even if the LLM omits the ``<SIDE_EFFECT>`` tag.

        Returns
        -------
        str
            Error message to display (e.g. "mkdir: cannot create directory…").
        ""  (empty string)
            Command succeeded with no output.
        None
            Command is not a VFS built-in; caller should forward to the LLM.
        """
        parts = command.split()
        if not parts:
            return None
        cmd = parts[0]

        if cmd == "ls":
            return self._vfs_ls(parts)

        if cmd == "touch" and len(parts) >= 2:
            for name in parts[1:]:
                path = name if name.startswith("/") else self._cwd.rstrip("/") + "/" + name
                if path not in self._state_manager.state["filesystem"]:
                    self._state_manager.apply_updates([{
                        "action": "SET", "scope": "filesystem", "key": path,
                        "value": {"type": "file", "content": "", "owner": self._username, "permissions": "644"},
                    }])
            return ""

        if cmd == "mkdir" and len(parts) >= 2:
            flags = [p for p in parts[1:] if p.startswith("-")]
            targets = [p for p in parts[1:] if not p.startswith("-")]
            for name in targets:
                path = name if name.startswith("/") else self._cwd.rstrip("/") + "/" + name
                if path in self._state_manager.state["filesystem"] and "-p" not in flags:
                    return f"mkdir: cannot create directory '{name}': File exists"
                self._state_manager.apply_updates([{
                    "action": "SET", "scope": "filesystem", "key": path,
                    "value": {"type": "directory", "owner": self._username, "permissions": "755"},
                }])
            return ""

        if cmd == "rm" and len(parts) >= 2:
            flags   = [p for p in parts[1:] if p.startswith("-")]
            targets = [p for p in parts[1:] if not p.startswith("-")]
            fs      = self._state_manager.state["filesystem"]
            updates = []
            for name in targets:
                path = name if name.startswith("/") else self._cwd.rstrip("/") + "/" + name
                if path not in fs:
                    return f"rm: cannot remove '{name}': No such file or directory"
                if fs[path].get("type") == "directory" and "-r" not in flags and "-rf" not in flags:
                    return f"rm: cannot remove '{name}': Is a directory"
                # Remove the entry and all children
                prefix = path.rstrip("/") + "/"
                for k in list(fs):
                    if k == path or k.startswith(prefix):
                        updates.append({"action": "DELETE", "scope": "filesystem", "key": k})
            if updates:
                self._state_manager.apply_updates(updates)
            return ""

        if cmd == "rmdir" and len(parts) >= 2:
            fs = self._state_manager.state["filesystem"]
            for name in parts[1:]:
                path = name if name.startswith("/") else self._cwd.rstrip("/") + "/" + name
                if path not in fs:
                    return f"rmdir: failed to remove '{name}': No such file or directory"
                # Check for children
                prefix = path.rstrip("/") + "/"
                if any(k.startswith(prefix) for k in fs):
                    return f"rmdir: failed to remove '{name}': Directory not empty"
                self._state_manager.apply_updates([{"action": "DELETE", "scope": "filesystem", "key": path}])
            return ""

        return None

    def _vfs_ls(self, parts: list) -> str:
        """Generate ``ls`` / ``ls -la`` output directly from the VFS.

        Handles the most common flag combinations: plain, ``-l``, ``-a``,
        ``-la`` / ``-al``, ``-lah``.  Always reads from the single source of
        truth (StateManager) so the output is perfectly consistent with what
        was actually created/removed.
        """
        flags_raw = [p for p in parts[1:] if p.startswith("-")]
        args      = [p for p in parts[1:] if not p.startswith("-")]
        flags     = "".join(flags_raw).replace("-", "")

        show_all  = "a" in flags   # include . and ..
        long_fmt  = "l" in flags   # long listing
        target    = args[0] if args else self._cwd
        target    = target if target.startswith("/") else self._cwd.rstrip("/") + "/" + target
        target    = target.rstrip("/") or "/"

        # Build a snapshot: name -> meta for direct children of target
        fs = self._state_manager.state.get("filesystem", {})
        target_prefix = target.rstrip("/") + "/"
        entries: dict = {}
        for abs_path, meta in fs.items():
            stripped       = abs_path.rstrip("/")
            parent, _, name = stripped.rpartition("/")
            parent         = parent or "/"
            if parent == target:
                entries[name] = meta
            elif abs_path.startswith(target_prefix):
                top = abs_path[len(target_prefix):].split("/", 1)[0]
                if top and top not in entries:
                    entries[top] = {"type": "directory", "owner": "root", "permissions": "755"}

        # Check if target is an explicit file rather than a directory
        if target in fs and fs[target].get("type") == "file":
            if not long_fmt:
                return target.rsplit("/", 1)[-1]
            meta  = fs[target]
            raw_p = meta.get("permissions", "644")
            pstr  = "-" + _PERM_MAP.get(raw_p, raw_p)
            size  = len(meta.get("content", ""))
            return f"{pstr}  1 root root {size:>6} Jan  1 08:00 {target.rsplit('/', 1)[-1]}"

        # Determine which entries are visible (hide dotfiles unless -a).
        visible: dict = {
            name: meta for name, meta in entries.items()
            if show_all or not name.startswith(".")
        }

        if not long_fmt:
            # short form: one name per column
            names = sorted(visible.keys())
            if show_all:
                names = [".", ".."] + names
            return "  ".join(names) if names else ""

        # long form ── helpers ────────────────────────────────────────────────
        # nlink of the target directory itself (used for '.' and '..').
        target_nlink = 2 + sum(1 for m in entries.values() if m.get("type") == "directory")

        def _blocks_1k(meta: dict) -> int:
            """1 K-blocks used by an entry (matches default ls output)."""
            if meta.get("type") == "directory":
                return 4  # one 4 096-byte block = 4 × 1 K
            size = len(meta.get("content", ""))
            # Round up to the nearest 4 096-byte filesystem block.
            return max(4, ((size + 4095) // 4096) * 4)

        dot_blocks = 8 if show_all else 0  # 4 each for '.' and '..'
        total_1k   = dot_blocks + sum(_blocks_1k(m) for m in visible.values())

        def _fmt_entry(name: str, meta: dict, nlink: int = 0) -> str:
            ftype = meta.get("type", "file")
            raw_p = meta.get("permissions", "755" if ftype == "directory" else "644")
            pstr  = ("d" if ftype == "directory" else "-") + _PERM_MAP.get(raw_p, raw_p)
            nl    = nlink or (target_nlink if name in (".", "..") else
                              (2 if ftype == "directory" else 1))
            owner = meta.get("owner", "root")
            size  = len(meta.get("content", "")) if ftype == "file" else 4096
            return f"{pstr}  {nl} {owner} {owner} {size:>6} Jan  1 08:00 {name}"

        lines: list[str] = [f"total {total_1k}"]

        if show_all:
            lines.append(_fmt_entry(".",  {"type": "directory", "permissions": "755"}))
            lines.append(_fmt_entry("..", {"type": "directory", "permissions": "755"}))

        for name in sorted(visible.keys()):
            lines.append(_fmt_entry(name, visible[name]))

        return "\n".join(lines)

    async def _typewrite(self, text: str) -> None:
        """Stream *text* to the channel one character at a time.

        The small per-character delay creates a typewriter effect that makes
        the simulated shell feel like a real remote process printing output,
        rather than a response that appears all at once.
        """
        assert self._chan is not None
        for char in text:
            self._chan.write(char)
            await asyncio.sleep(_TYPEWRITER_DELAY)

    def _build_prompt(self, command: str) -> str:
        """
        Delegates prompt construction to ``craft_ssh_prompt()``, injecting
        the live VFS context for the current directory and this session's
        command history.

        For read-oriented commands (``cat``, ``tail``, ``grep``, …) that
        reference an absolute path the target file's content is pre-fetched
        from the VFS and merged into the context so the LLM never has to
        guess the file's contents.
        """
        return craft_ssh_prompt(
            command=command,
            cwd=self._cwd,
            file_context=_get_command_context(command, self._cwd, self._state_manager),
            history=self._command_history,
        )

    def _resolve_cd(self, command: str) -> str:
        """
        Updates the simulated CWD after a `cd` command without calling the
        LLM — keeps the prompt accurate without spending a generation.
        """
        parts = command.split(maxsplit=1)
        target = parts[1].strip() if len(parts) > 1 else "~"

        if target in ("", "~"):
            return self._home
        if target.startswith("/"):
            return target.rstrip("/") or "/"
        if target == "..":
            parent = self._cwd.rstrip("/").rsplit("/", 1)[0]
            return parent or "/"
        if target == ".":
            return self._cwd
        return self._cwd.rstrip("/") + "/" + target


# ─────────────────────────────────────────────────────────────────────────────
# Server — one instance per TCP connection
# ─────────────────────────────────────────────────────────────────────────────
class _GenPotSSHServer(asyncssh.SSHServer):
    """
    Handles connection-level SSH events.

    • Records the client IP from the TCP peername.
    • Enables password authentication so attackers submit credentials we
      can log (username + password).
    • Accepts *every* credential — this is a honeypot; we want to let
      attackers in so we can observe their behaviour.
    • Creates a new _GenPotSSHSession for each channel request.
    """

    def __init__(self, state_manager: StateManager) -> None:
        self._state_manager = state_manager
        self._conn: asyncssh.SSHServerConnection
        self._client_ip: str = "unknown"
        self._username: str = "unknown"
        self._password: str = ""  # set by validate_password, read by session_requested

    def connection_made(self, conn: asyncssh.SSHServerConnection) -> None:  # type: ignore[override]
        self._conn = conn
        peer = conn.get_extra_info("peername")
        self._client_ip = peer[0] if peer else "unknown"

    def begin_auth(self, username: str) -> bool:
        self._username = username
        self._conn.send_auth_banner("Unauthorized access is strictly prohibited.\r\n")
        return True  # Return True → require auth (so we can capture creds).

    def password_auth_supported(self) -> bool:
        return True

    def validate_password(self, username: str, password: str) -> bool:
        """Log credential attempt and always grant access (honeypot)."""
        log_interaction(
            {
                "event": "ssh_auth_attempt",
                "ip": self._client_ip,
                "username": username,
                "password": password,
            }
        )
        self._password = password  # carry into the session for correlation
        return True  # Accept every attempt.

    def session_requested(self) -> asyncssh.SSHServerSession:  # type: ignore[override]
        return _GenPotSSHSession(
            self._state_manager,
            self._client_ip,
            self._username,
            self._password,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────
async def start_ssh_server(
    state_manager: StateManager,
    host: str = "0.0.0.0",
    port: int = 2222,
) -> None:
    """
    Start the AsyncSSH honeypot and run until the task is cancelled.

    The *state_manager* argument MUST be the same instance used by the
    FastAPI application so both protocols share a single world state.

    Host-key persistence
    ────────────────────
    An ED25519 host key is generated on the first boot and written to
    ``logs/ssh_host_key``.  On subsequent restarts the same key is reloaded
    so SSH clients don't encounter a "host identification has changed" error.
    """
    _KEY_PATH.parent.mkdir(parents=True, exist_ok=True)

    if _KEY_PATH.exists():
        host_key = asyncssh.read_private_key(str(_KEY_PATH))
        print(f"[SSH] Loaded persistent host key from {_KEY_PATH}")
    else:
        host_key = asyncssh.generate_private_key("ssh-ed25519")
        host_key.write_private_key(str(_KEY_PATH))
        print(f"[SSH] Generated new ED25519 host key → {_KEY_PATH}")

    server = await asyncssh.create_server(
        lambda: _GenPotSSHServer(state_manager),
        host,
        port,
        server_host_keys=[host_key],
    )

    print(f"[SSH] Honeypot listening on {host}:{port}")

    async with server:
        await server.wait_closed()
