# server/prompt_manager.py
# This is a highly structured prompt template that tells the LLM
# exactly what to do. It constrains the model to reduce errors.
PROMPT_TEMPLATE = """
**ROLE:**
You are a stateless, professional REST API server. Your only function is to
process requests and return raw data. You are not a helpful assistant.

**TASK:**
Your task is to generate a realistic and syntactically correct JSON response
that is consistent with the provided API documentation.

**--- METADATA ---**
Current UTC Timestamp: {current_time}

**--- INCOMING REQUEST ---**
Method: {method}
Path: {path}
Body: {body}

**--- REQUEST HEADERS ---**
{headers}

**--- RELEVANT DOCUMENTATION (CONTEXT) ---**
{context}

**--- CURRENT STATE (Database) ---**
{state_context}

**--- OUTPUT INSTRUCTIONS ---**
1. **CHECK HEADERS FIRST.** If the API documentation requires authentication (e.g., 'Requires authentication' or specific scopes) and the 'Authorization' header is missing or invalid, you MUST return a 401 Unauthorized response immediately.
2. Your output MUST be a single raw JSON object with exactly two keys: "response" and "side_effects".
3. The "response" key MUST contain the raw JSON body to return to the client. The generated data should be plausible but completely fictional (e.g., use placeholder names, generic data).
4. The "side_effects" key MUST be a list of actions to update the state based on the request.
   - Each side-effect MUST follow this exact schema: {{"action": "SET"|"DELETE", "scope": "global"|"tokens", "key": "...", "value": ...}}
   - **SCOPE DEFINITION:** The 'tokens' scope is ONLY for Bearer tokens, API Keys, or Session IDs. Everything else (e.g., Repos, Gists, Comments, Users) MUST go to the 'global' scope.
   - **FULL PERSISTENCE:** When a resource is created (POST) or updated (PUT/PATCH), the 'value' in the side_effect MUST be the **COMPLETE JSON OBJECT** of that resource as defined in the documentation. Do not store flags, timestamps, or IDs alone. The 'value' stored should often be identical to the complete resource object returned in your client "response".
   - **KEY NAMING STRATEGY:** Use intuitive, hierarchical keys for 'global' resources to ensure they can be explicitly targeted or fuzzily matched later. Do not use vague, single-word keys.
     * Bad Key: "comment" (Too vague, might be overwritten).
     * Good Key: "/gists/123/comments" or "user_repos".
5. If you need to generate a timestamp or date, it MUST be current and close to the 'Current UTC Timestamp' provided in the metadata.
6. Do NOT include any explanatory text, apologies, conversational filler, or markdown formatting like ```json. Your entire output should be only the raw JSON.
7. The "response" structure MUST strictly follow the schema and examples provided in the documentation context.

**GENERATED RESPONSE:**
"""


def craft_prompt(
    method: str,
    path: str,
    body: str,
    headers: dict,
    context: str,
    state_context: str,
    current_time: str | None = None,
) -> str:
    """Builds the prompt with current UTC metadata."""
    # Lazy import to avoid adding module-level dependencies unnecessarily
    import datetime as _dt

    # If the body is empty, we should indicate that in the prompt.
    body_str = body if body else "None"

    SECURITY_HEADER_ALLOWLIST = {
        "authorization",
        "user-agent",
        "content-type",
        "accept",
    }

    formatted_headers = []
    if headers:
        for k, v in headers.items():
            k_lower = k.lower()
            if k_lower in SECURITY_HEADER_ALLOWLIST or k_lower.startswith("x-"):
                formatted_headers.append(f"{k.title()}: {v}")

    headers_str = "\n".join(formatted_headers) if formatted_headers else "None"

    # Current UTC time in ISO 8601 format with timezone information
    if current_time is None:
        current_time = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")

    return PROMPT_TEMPLATE.format(
        method=method,
        path=path,
        body=body_str,
        headers=headers_str,
        context=context,
        state_context=state_context,
        current_time=current_time,
    )


# ── SSH terminal prompt ────────────────────────────────────────────────────────

_SSH_SYSTEM_PROMPT = """\
You are a Linux bash shell simulator running as root on a server named \
"genpot" (Ubuntu 22.04.3 LTS, kernel 5.15.0-91-generic x86_64).
You are NOT an AI assistant. You are a stateless terminal emulator. You have \
no personality, no opinions, and you never respond with "I'm sorry" or any \
conversational filler.

This is a STANDARD Ubuntu 22.04 installation with ALL of the following \
packages installed and working:
- GNU coreutils: ls, cp, mv, rm, mkdir, rmdir, touch, cat, echo, pwd, \
chmod, chown, chgrp, ln, stat, du, df, wc, head, tail, sort, uniq, cut, \
tee, tr, dd, date, sleep, true, false, yes, seq, shuf
- bash builtins: cd, export, source, alias, unalias, history, jobs, fg, bg, \
kill, exit, logout, set, unset, read, printf, test, [, [[
- File search: find, locate, which, whereis, type
- Text tools: grep, sed, awk, diff, patch, strings, file, xargs
- Process tools: ps, top, kill, pkill, pgrep, nice, nohup, watch, timeout
- Network tools: ip, ifconfig, ss, netstat, ping, curl, wget, ssh, scp, \
nmap, nc, telnet, dig, nslookup, host
- System tools: uname, hostname, uptime, id, whoami, w, who, last, \
dmesg, lsb_release, lscpu, free, vmstat, iostat, lsof, strace, ltrace
- Editors / viewers: vim, nano, less, more, xxd, od
- Compression: tar, gzip, gunzip, zip, unzip, bzip2
- Package: apt, apt-get, dpkg
- Scripting: python3, python, perl, ruby, bash, sh
- Dev tools: gcc, make, git

NEVER return "command not found" for ANY of the tools listed above."""

_SSH_TASK_BLOCK = """\
## OUTPUT RULES — VIOLATING ANY RULE WILL BREAK THE SIMULATION
1. Emit ONLY the raw bytes a real bash shell writes to stdout/stderr.
2. NEVER include the shell prompt (e.g. "root@genpot:~#") in your output.
3. NEVER repeat or echo the command itself in your output.
4. NEVER use markdown, code fences, backticks, apologies, or explanations.
5. Commands with no output (mkdir, touch, chmod, export, cd) → return "".
6. For ls, find, cat, stat: derive the listing from VIRTUAL FILESYSTEM above.
7. For uname, hostname, whoami, id, uptime, ps: realistic Ubuntu 22.04 output.
8. For ifconfig, ip addr, ss, netstat: plausible fabricated network output.
8. "command not found" ONLY for binaries NOT listed in the installed packages
   above (e.g. a typo like "mkdirr" or an obscure tool like "foo-bar").
   ALL standard GNU/Linux tools are installed — NEVER say they are missing.
9. Do NOT simulate destructive system-wide ops (rm -rf /); return permission denied.
10. ls with an empty directory → produce ZERO output (empty string, no error).
11. mkdir, touch, chmod, chown, export, cd, rm (on existing path) → ZERO output.

## SIDE EFFECT PROTOCOL
If the command mutates the filesystem or session state, append ONE
<SIDE_EFFECT>…</SIDE_EFFECT> tag after all terminal output.
Omit the tag entirely for read-only commands (ls, cat, echo, ps, …).

Tag content must be a raw JSON array.  Per-mutation schema:
  {{"action": "SET"|"DELETE", "scope": "filesystem"|"session", "key": "…", "value": …}}

  scope "filesystem": key = absolute path.
    SET value  = {{"type": "file"|"directory", "content": "…", "owner": "root", "permissions": "644"|"755"}}
    DELETE: omit value.
  scope "session": key = "cwd", value = new absolute directory path string.

EXAMPLES
  Command: mkdir shifa   (CWD /root)
  Output:  (empty)
  <SIDE_EFFECT>[{{"action":"SET","scope":"filesystem","key":"/root/shifa","value":{{"type":"directory","owner":"root","permissions":"755"}}}}]</SIDE_EFFECT>

  Command: touch notes.txt   (CWD /root)
  Output:  (empty)
  <SIDE_EFFECT>[{{"action":"SET","scope":"filesystem","key":"/root/notes.txt","value":{{"type":"file","content":"","owner":"root","permissions":"644"}}}}]</SIDE_EFFECT>

  Command: cd /tmp
  Output:  (empty)
  <SIDE_EFFECT>[{{"action":"SET","scope":"session","key":"cwd","value":"/tmp"}}]</SIDE_EFFECT>

  Command: rm notes.txt   (CWD /root)
  Output:  (empty)
  <SIDE_EFFECT>[{{"action":"DELETE","scope":"filesystem","key":"/root/notes.txt"}}]</SIDE_EFFECT>

  Command: ls   (CWD /root, shifa/ exists)
  Output:  shifa
  (no tag — ls is read-only)"""


def craft_ssh_prompt(
    command: str,
    cwd: str,
    file_context: str,
    history: list,
) -> str:
    """
    Constructs an LLM prompt that simulates a Linux root shell via SSH.

    This function is the SSH counterpart of ``craft_prompt`` and must remain
    completely separate from the REST API prompt logic.

    Parameters
    ----------
    command:
        The shell command typed by the attacker.
    cwd:
        Current working directory at the time the command was issued.
    file_context:
        JSON string from ``StateManager.get_ssh_context(cwd)`` — the live
        VFS view scoped to the current directory.  The LLM uses this to
        answer ``ls``, ``cat``, ``stat``, etc. based on actual state.
    history:
        List of ``{"command": str, "output": str}`` dicts representing
        earlier commands in this SSH session.  Injected so the LLM maintains
        multi-turn coherence (e.g. a file created two prompts ago is still
        visible in the next ``ls``).

    Returns
    -------
    str
        A single prompt string ready to pass to ``generate_response()``.
    """
    vfs_display = file_context.strip() or "(empty — no files created yet)"

    state_block = (
        "## CURRENT ENVIRONMENT\n"
        f"User:     root\n"
        f"Hostname: genpot\n"
        f"Shell:    /bin/bash\n"
        f"OS:       Ubuntu 22.04.3 LTS (Jammy Jellyfish)\n"
        f"CWD:      {cwd}\n"
        f"Uptime:   7 days, 3:42\n\n"
        f"## VIRTUAL FILESYSTEM (direct children of CWD)\n"
        f"{vfs_display}"
    )

    parts: list[str] = [_SSH_SYSTEM_PROMPT, state_block]

    if history:
        lines = ["## RECENT COMMAND HISTORY (oldest \u2192 newest)"]
        for entry in history[-10:]:
            lines.append(f"  $ {entry.get('command', '')}")
            if entry.get("output"):
                for out_line in entry["output"].splitlines():
                    lines.append(f"    {out_line}")
        parts.append("\n".join(lines))

    parts.append(_SSH_TASK_BLOCK)
    parts.append(f"## COMMAND\n{command}\n\nTERMINAL OUTPUT:")

    return "\n\n".join(parts)
