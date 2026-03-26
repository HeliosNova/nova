"""System access tiers — sandboxed / standard / full / none.

Central module that defines what each tool is allowed to do at each tier.
shell_exec, file_ops, and code_exec call into this module instead of
maintaining their own hardcoded deny lists.

Tiers:
  sandboxed (default) — most restrictive, safe for untrusted use
  standard            — moderate, blocks system commands but allows more file/code access
  full                — minimal restrictions, only blocks container-escape and ctypes
  none                — all restrictions disabled, sandbox fully off
"""

from __future__ import annotations

import contextvars
import functools
import logging
import sys
from pathlib import Path

from app.config import config

logger = logging.getLogger(__name__)

VALID_TIERS = {"sandboxed", "standard", "full", "none"}

# ContextVar override for access tier — used by delegate to constrain sub-agents.
# When set, _tier() returns this value instead of config.SYSTEM_ACCESS_LEVEL.
_access_tier_override: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "access_tier_override", default=None
)


def get_access_tier_override() -> str | None:
    return _access_tier_override.get()


def set_access_tier_override(tier: str | None) -> None:
    _access_tier_override.set(tier)


def _tier() -> str:
    # Check ContextVar override first (set by delegate for sub-agents)
    override = _access_tier_override.get()
    if override is not None:
        t = override.lower()
        if t in VALID_TIERS:
            return t
        logger.warning("Invalid access tier override '%s', falling back to config", override)

    t = config.SYSTEM_ACCESS_LEVEL.lower()
    if t not in VALID_TIERS:
        logger.warning("Invalid SYSTEM_ACCESS_LEVEL '%s', falling back to 'sandboxed'", t)
        return "sandboxed"
    return t


def requires_tier(*allowed_tiers: str):
    """Decorator to restrict tool execution to specific access tiers."""
    from app.tools.base import ToolResult, ErrorCategory

    def decorator(execute_method):
        @functools.wraps(execute_method)
        async def wrapper(self, **kwargs):
            current = _tier()
            if current not in allowed_tiers and current != "none":
                return ToolResult(
                    output="", success=False,
                    error=f"Tool '{self.name}' requires tier {'/'.join(allowed_tiers)}, current: {current}",
                    error_category=ErrorCategory.PERMISSION,
                )
            return await execute_method(self, **kwargs)
        return wrapper
    return decorator


# ---------------------------------------------------------------------------
# Shell access
# ---------------------------------------------------------------------------

# Container-escape commands blocked at ALL tiers
_ALWAYS_BLOCKED_COMMANDS = {
    "docker", "podman", "nsenter", "chroot", "unshare",
}

# Destructive system commands blocked at sandboxed + standard
_SYSTEM_COMMANDS = {
    "shutdown", "reboot", "poweroff", "halt", "init",
    "mkfs", "fdisk", "parted",
    "iptables", "ip6tables", "nft", "ufw",
    "useradd", "userdel", "usermod", "passwd", "groupadd", "groupdel",
    "systemctl", "service",
    "mount", "umount", "swapon", "swapoff",
}

# Interpreters blocked only at sandboxed tier
_INTERPRETER_COMMANDS = {
    "python", "python3", "node", "ruby", "perl",
}


def get_blocked_shell_commands() -> set[str]:
    """Return the set of blocked shell commands for the current tier."""
    tier = _tier()

    if tier == "none":
        logger.warning("Shell running with NO restrictions (sandbox disabled)")
        return set()

    blocked = set(_ALWAYS_BLOCKED_COMMANDS)

    if tier == "sandboxed":
        blocked |= _SYSTEM_COMMANDS | _INTERPRETER_COMMANDS
    elif tier == "standard":
        blocked |= _SYSTEM_COMMANDS
    # full: only container-escape commands blocked

    if tier == "full":
        logger.warning("Shell running at FULL access tier")

    return blocked


def is_command_blocked(base_cmd: str) -> str | None:
    """Check if a base command is blocked, with prefix matching for interpreters.

    Returns the matched blocked command name if blocked, None if allowed.
    Handles versioned interpreters like python3.11, node18, ruby3.2, etc.
    """
    blocked = get_blocked_shell_commands()

    # Exact match first (covers all non-interpreter commands too)
    if base_cmd in blocked:
        return base_cmd

    # Prefix match for interpreters: e.g. python3.11 matches python3
    # Only match if the suffix after the blocked name is digits/dots (version suffix)
    tier = _tier()
    if tier == "sandboxed":
        for interp in _INTERPRETER_COMMANDS:
            if base_cmd.startswith(interp) and len(base_cmd) > len(interp):
                suffix = base_cmd[len(interp):]
                # Allow only version-like suffixes (digits and dots)
                if all(c in "0123456789." for c in suffix):
                    return interp

    return None


# ---------------------------------------------------------------------------
# Filesystem access
# ---------------------------------------------------------------------------

# Paths that are NEVER writable (any tier)
_ALWAYS_PROTECTED_PATHS = {
    "/etc/shadow", "/root/.ssh",
    "/etc/passwd", "/etc/sudoers", "/etc/sudoers.d",
    "/root/.bashrc", "/root/.profile",
    "/etc/crontab", "/etc/fstab",
}
# Directories where no path under them is writable (prefix matching)
_ALWAYS_PROTECTED_DIRS = {"/proc", "/sys"}

if sys.platform == "win32":
    import os
    _system_root = Path(os.environ.get("SYSTEMROOT", "C:\\Windows"))
    _ALWAYS_PROTECTED_PATHS.update({
        _system_root / "System32" / "config" / "SAM",
        _system_root / "System32" / "config" / "SYSTEM",
        _system_root / "System32" / "config" / "SECURITY",
        _system_root / "System32" / "config" / "SOFTWARE",
        _system_root / "System32" / "config" / "DEFAULT",
        Path(os.environ.get("USERPROFILE", "C:\\Users\\Default")) / ".ssh",
    })
    _ALWAYS_PROTECTED_DIRS.update({
        _system_root / "System32" / "drivers",
        _system_root / "System32" / "config" / "RegBack",
        _system_root / "Boot",
    })


def get_allowed_read_roots() -> list[Path]:
    """Return allowed read roots for file_ops."""
    tier = _tier()
    if tier == "sandboxed":
        return [Path("/data")]
    # standard + full + none can read anywhere
    return [Path("/")]


def get_allowed_write_roots() -> list[Path]:
    """Return allowed write roots for file_ops."""
    tier = _tier()
    if tier == "sandboxed":
        return [Path("/data")]
    elif tier == "standard":
        return [Path("/data"), Path("/tmp"), Path("/home/nova")]
    else:
        # full + none: write anywhere
        return [Path("/")]


def is_path_allowed(path: Path, write: bool = False) -> bool:
    """Check if a resolved path is allowed under the current tier."""
    tier = _tier()
    resolved = path.resolve()

    # Always block protected paths for writes (except "none" tier)
    if write and tier != "none":
        for protected in _ALWAYS_PROTECTED_PATHS:
            protected_resolved = Path(protected).resolve()
            if resolved == protected_resolved or resolved.is_relative_to(protected_resolved):
                return False
        for protected_dir in _ALWAYS_PROTECTED_DIRS:
            protected_resolved = Path(protected_dir).resolve()
            if resolved == protected_resolved or resolved.is_relative_to(protected_resolved):
                return False

    roots = get_allowed_write_roots() if write else get_allowed_read_roots()
    for root in roots:
        root_resolved = root.resolve()
        if str(root_resolved) == "/":
            # Root allows everything
            return True
        if resolved == root_resolved or resolved.is_relative_to(root_resolved):
            return True
    return False


# ---------------------------------------------------------------------------
# Code execution
# ---------------------------------------------------------------------------

# Imports blocked at ALL tiers
_ALWAYS_BLOCKED_IMPORTS = {"ctypes", "multiprocessing"}

# Full sandboxed set (current behavior)
_SANDBOXED_BLOCKED_IMPORTS = {
    "os", "subprocess", "shutil", "sys", "importlib",
    "ctypes", "socket", "http", "urllib", "requests", "httpx",
    "pathlib", "glob", "signal", "multiprocessing",
}

# Standard: allow os.path, pathlib, glob, sys but block dangerous ones
_STANDARD_BLOCKED_IMPORTS = {
    "subprocess", "shutil", "importlib",
    "ctypes", "socket", "http", "urllib", "requests", "httpx",
    "signal", "multiprocessing",
}


def get_blocked_imports() -> set[str]:
    """Return blocked imports for code_exec at the current tier."""
    tier = _tier()
    if tier == "none":
        return set()
    if tier == "sandboxed":
        return set(_SANDBOXED_BLOCKED_IMPORTS)
    elif tier == "standard":
        return set(_STANDARD_BLOCKED_IMPORTS)
    else:
        # full: only block truly dangerous ones
        return set(_ALWAYS_BLOCKED_IMPORTS)


def get_blocked_builtins() -> list[str]:
    """Return blocked builtins for code_exec at the current tier."""
    tier = _tier()
    if tier == "none":
        return []
    if tier in ("sandboxed", "standard"):
        return [
            "open(", "getattr(", "compile(", "globals(", "locals(",
            "vars(", "dir(", "breakpoint(",
            "__builtins__", "__import__", "eval(", "exec(",
        ]
    else:
        # full: minimal restrictions
        return [
            "__builtins__", "__import__",
            "breakpoint(",
        ]
