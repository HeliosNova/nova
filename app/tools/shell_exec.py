"""Shell execution tool — run shell commands with safety guardrails."""

from __future__ import annotations

import logging
import re
import shlex
import subprocess

from app.config import config
from app.core.access_tiers import get_blocked_shell_commands, _tier
from app.tools.base import BaseTool, ToolResult

logger = logging.getLogger(__name__)

# --- Security: Blocked command patterns (all tiers) ---

_BLOCKED_PATTERNS = [
    r"rm\s+(-[a-zA-Z]*\s+)*/\s",       # rm -rf / (followed by space/args)
    r"rm\s+(-[a-zA-Z]*\s+)*/$",        # rm -rf / (end of string)
    r"rm\s+(-[a-zA-Z]*\s+)*/\*",       # rm -rf /*
    r">\s*/dev/sd",                      # write to block devices
    r"dd\s+.*of=/dev/",                  # dd to block devices
    r"chmod\s+(-R\s+)?777\s+/",         # chmod 777 on root
    r":\(\)\s*\{",                       # fork bomb
    r"/proc/|/sys/",                     # kernel interfaces
    r">\s*/etc/",                        # overwriting system config
    r"curl.*\|\s*(sh|bash)",            # pipe curl to shell
    r"wget.*\|\s*(sh|bash)",            # pipe wget to shell
    r"\$\(",                             # command substitution $(...) — also catches $((..)) arithmetic
    r"`[^`]+`",                          # backtick command substitution
    r"\$\{",                             # ${} expansion syntax
    r"(?:^|[\s;|&])eval(?:\s|$)",        # eval command (word boundary, not substring)
    r"\bsource\s+/dev/",                # source from devices
    r"<\(",                              # process substitution <(...)
    r">\(",                              # process substitution >(...)
]

# Prefixes to skip when extracting the base command
_SKIP_PREFIXES = {"sudo", "env", "nice", "nohup", "timeout"}


def _check_command_safety(command: str) -> str | None:
    """Check if a shell command is safe to execute.

    Returns error message if blocked, None if safe.
    Splits on pipe/chain operators to check EACH sub-command.
    """
    command = command.strip()
    if not command:
        return "Empty command"

    # "none" tier: no restrictions
    if _tier() == "none":
        return None

    # Split on pipe/chain operators to check each sub-command
    sub_commands = re.split(r'\s*(?:\|{1,2}|;|&&)\s*', command)
    blocked = get_blocked_shell_commands()

    for sub_cmd in sub_commands:
        sub_cmd = sub_cmd.strip()
        if not sub_cmd:
            continue

        # Use shlex for proper quote handling; fall back to str.split on malformed input
        try:
            tokens = shlex.split(sub_cmd)
        except ValueError:
            tokens = sub_cmd.split()

        # Extract the base command (skip sudo/env/nice/etc.)
        idx = 0
        while idx < len(tokens) and tokens[idx] in _SKIP_PREFIXES:
            idx += 1
        base_cmd = tokens[idx] if idx < len(tokens) else tokens[0]

        # Strip path prefix (e.g., /usr/bin/rm -> rm)
        base_cmd = base_cmd.rsplit("/", 1)[-1]

        if base_cmd in blocked:
            return f"Command '{base_cmd}' is blocked for security."

        # If the base command is a shell (sh/bash/zsh/dash), also check the -c argument
        if base_cmd in ("sh", "bash", "zsh", "dash") and "-c" in tokens:
            c_idx = tokens.index("-c")
            if c_idx + 1 < len(tokens):
                # shlex already stripped quotes; recursively check the inner command
                inner_cmd = tokens[c_idx + 1]
                inner_err = _check_command_safety(inner_cmd)
                if inner_err:
                    return f"{inner_err} (via {base_cmd} -c)"

    for pattern in _BLOCKED_PATTERNS:
        if re.search(pattern, command, re.DOTALL):
            return "Command matches a blocked pattern for security."

    return None


class ShellExecTool(BaseTool):
    name = "shell_exec"
    description = (
        "Execute a shell command on the system. "
        "Use for: listing files, checking system info, running scripts, "
        "package management, git operations. Returns stdout+stderr."
    )
    parameters = "command: str"

    async def execute(self, *, command: str = "", **kwargs) -> ToolResult:
        if not command:
            return ToolResult(output="", success=False, error="No command provided")

        if not config.ENABLE_SHELL_EXEC:
            return ToolResult(
                output="",
                success=False,
                error="Shell execution is disabled. Set ENABLE_SHELL_EXEC=true to enable.",
            )

        safety_error = _check_command_safety(command)
        if safety_error:
            return ToolResult(output="", success=False, error=safety_error)

        try:
            result = subprocess.run(
                ["sh", "-c", command],
                capture_output=True,
                text=True,
                timeout=config.SHELL_EXEC_TIMEOUT,
                cwd="/data",
                env={
                    "PATH": "/usr/local/bin:/usr/bin:/bin",
                    "HOME": "/home/nova",
                    "USER": "nova",
                    "LANG": "C.UTF-8",
                },
            )

            output = result.stdout
            if result.stderr:
                output += f"\n[stderr]: {result.stderr}"

            if result.returncode != 0:
                return ToolResult(
                    output=output or result.stderr,
                    success=False,
                    error=f"Command exited with code {result.returncode}",
                )

            if not output.strip():
                output = "[Command executed successfully with no output]"

            # Truncate long output
            if len(output) > 8000:
                total_len = len(output)
                output = output[:8000] + f"\n[... truncated: showing 8000 of {total_len} chars]"

            return ToolResult(output=output, success=True)

        except subprocess.TimeoutExpired:
            return ToolResult(
                output="",
                success=False,
                error=f"Command timed out after {config.SHELL_EXEC_TIMEOUT}s",
            )
        except Exception as e:
            return ToolResult(output="", success=False, error=f"Execution failed: {e}")
