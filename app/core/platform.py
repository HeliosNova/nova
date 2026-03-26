"""Platform utilities — safe env/cwd helpers for subprocess sandboxing."""

from __future__ import annotations

import os
import sys

IS_WINDOWS = sys.platform == "win32"


def get_safe_env() -> dict[str, str]:
    """Return a minimal environment dict for subprocess sandboxing.

    On Windows: inherit PATH, SYSTEMROOT, TEMP (required for process creation).
    On Linux: hardcoded minimal PATH only.
    Never includes API keys, tokens, or other sensitive variables.
    """
    if IS_WINDOWS:
        return {
            "PATH": os.environ.get("PATH", ""),
            "SYSTEMROOT": os.environ.get("SYSTEMROOT", r"C:\Windows"),
            "TEMP": os.environ.get("TEMP", os.path.join(os.environ.get("SYSTEMROOT", r"C:\Windows"), "Temp")),
            "LANG": "C.UTF-8",
        }
    return {
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "HOME": "/tmp",
        "LANG": "C.UTF-8",
    }


def get_safe_cwd() -> str:
    """Return a safe working directory for subprocess execution."""
    if IS_WINDOWS:
        return os.environ.get("USERPROFILE", os.environ.get("TEMP", "."))
    return "/data"


def get_shell_command(command: str) -> list[str]:
    """Return the shell invocation list for running a command string."""
    if IS_WINDOWS:
        return ["cmd", "/c", command]
    return ["sh", "-c", command]
