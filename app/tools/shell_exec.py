"""Shell execution tool — run shell commands with safety guardrails."""

from __future__ import annotations

import asyncio
import logging
import re
import shlex
import subprocess

from app.config import config
from app.core.access_tiers import get_blocked_shell_commands, is_command_blocked, _tier
from app.core.platform import get_safe_env, get_safe_cwd, get_shell_command
from app.tools.base import BaseTool, ToolResult, ErrorCategory

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
    r"<<['\"]?\w",                       # heredoc injection (<<EOF, <<'EOF', <<"EOF")
]

# Prefixes to skip when extracting the base command
_SKIP_PREFIXES = {"sudo", "env", "nice", "nohup", "timeout"}


def _check_command_safety(command: str, *, depth: int = 0, max_depth: int = 5) -> str | None:
    """Check if a shell command is safe to execute.

    Returns error message if blocked, None if safe.
    Splits on pipe/chain operators to check EACH sub-command.
    """
    if depth >= max_depth:
        return "Command nesting too deep (possible recursion attack)"

    command = command.strip()
    if not command:
        return "Empty command"

    # "none" tier: no restrictions
    if _tier() == "none":
        return None

    # Split on pipe/chain operators to check each sub-command
    sub_commands = re.split(r'\s*(?:\|{1,2}|;|&&|\n)\s*', command)
    blocked = get_blocked_shell_commands()

    for sub_cmd in sub_commands:
        sub_cmd = sub_cmd.strip()
        if not sub_cmd:
            continue

        # Use shlex for proper quote handling; reject malformed input
        try:
            tokens = shlex.split(sub_cmd)
        except ValueError:
            return "Command rejected: malformed shell quoting"

        # Extract the base command (skip sudo/env/nice/etc.)
        idx = 0
        while idx < len(tokens) and tokens[idx] in _SKIP_PREFIXES:
            idx += 1
        base_cmd = tokens[idx] if idx < len(tokens) else tokens[0]

        # Strip path prefix (e.g., /usr/bin/rm -> rm)
        base_cmd = base_cmd.rsplit("/", 1)[-1]

        blocked_match = is_command_blocked(base_cmd)
        if blocked_match:
            return f"Command '{base_cmd}' is blocked for security."

        # If the base command is a shell (sh/bash/zsh/dash), also check the -c argument
        if base_cmd in ("sh", "bash", "zsh", "dash") and "-c" in tokens:
            c_idx = tokens.index("-c")
            if c_idx + 1 < len(tokens):
                # shlex already stripped quotes; recursively check the inner command
                inner_cmd = tokens[c_idx + 1]
                inner_err = _check_command_safety(inner_cmd, depth=depth + 1, max_depth=max_depth)
                if inner_err:
                    return f"{inner_err} (via {base_cmd} -c)"

    for pattern in _BLOCKED_PATTERNS:
        if re.search(pattern, command, re.DOTALL):
            return "Command matches a blocked pattern for security."

    return None


class ShellExecTool(BaseTool):
    name = "shell_exec"
    description = (
        "Execute shell commands on the host system. Returns stdout and stderr. "
        "Use for listing files, checking system info, running scripts, package management, and git operations. "
        "Commands are safety-checked against blocked patterns and tier-based restrictions. "
        "Output is truncated to TOOL_OUTPUT_MAX_CHARS. Do NOT use for Python code (use code_exec) or HTTP requests (use http_fetch)."
    )
    parameters = "command: str"
    input_schema = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "Shell command to execute. Runs via sh -c in /data directory.",
            },
        },
        "required": ["command"],
    }

    def trim_output(self, output: str) -> str:
        """Keep tail of output — most relevant for command results."""
        if len(output) <= 2000:
            return output
        return '[...truncated]\n' + output[-1500:]

    async def execute(self, *, command: str = "", **kwargs) -> ToolResult:
        if not command:
            return ToolResult(output="", success=False, error="No command provided", error_category=ErrorCategory.VALIDATION)

        if not config.ENABLE_SHELL_EXEC:
            return ToolResult(
                output="",
                success=False,
                error="Shell execution is disabled. Set ENABLE_SHELL_EXEC=true to enable.",
                error_category=ErrorCategory.PERMISSION,
            )

        safety_error = _check_command_safety(command)
        if safety_error:
            return ToolResult(output="", success=False, error=safety_error, error_category=ErrorCategory.VALIDATION)

        try:
            result = await asyncio.to_thread(
                subprocess.run,
                get_shell_command(command),
                capture_output=True,
                text=True,
                timeout=config.SHELL_EXEC_TIMEOUT,
                cwd=get_safe_cwd(),
                env=get_safe_env(),
            )

            output = result.stdout
            if result.stderr:
                output += f"\n[stderr]: {result.stderr}"

            if result.returncode != 0:
                return ToolResult(
                    output=output or result.stderr,
                    success=False,
                    error=f"Command exited with code {result.returncode}",
                    error_category=ErrorCategory.INTERNAL,
                )

            if not output.strip():
                output = "[Command executed successfully with no output]"

            # Truncate long output
            max_chars = config.TOOL_OUTPUT_MAX_CHARS
            if len(output) > max_chars:
                total_len = len(output)
                output = output[:max_chars] + f"\n[... truncated: showing {max_chars} of {total_len} chars]"

            return ToolResult(output=output, success=True)

        except subprocess.TimeoutExpired:
            return ToolResult(
                output="",
                success=False,
                error=f"Command timed out after {config.SHELL_EXEC_TIMEOUT}s",
                error_category=ErrorCategory.TRANSIENT,
            )
        except Exception as e:
            return ToolResult(output="", success=False, error=f"Execution failed: {e}", error_category=ErrorCategory.INTERNAL)
