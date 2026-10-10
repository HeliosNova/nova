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

# ContextVar for tool whitelist — restricts which tools a sub-agent can call.
# When set, only tools in the whitelist can execute; all others get PERMISSION error.
_tool_whitelist: contextvars.ContextVar[frozenset[str] | None] = contextvars.ContextVar(
    "tool_whitelist", default=None
)


def get_tool_whitelist() -> frozenset[str] | None:
    """Get the current tool whitelist (None = all tools allowed)."""
    return _tool_whitelist.get()


def set_tool_whitelist(tools: list[str] | frozenset[str] | None) -> None:
    """Set the tool whitelist for the current async context.

    Args:
        tools: Frozenset of allowed tool names, or None to allow all.
    """
    _tool_whitelist.set(frozenset(tools) if tools is not None else None)


def is_tool_allowed(tool_name: str) -> bool:
    """Check if a tool is allowed under the current whitelist (if any)."""
    whitelist = _tool_whitelist.get()
    if whitelist is None:
        return True  # No whitelist = all tools allowed
    return tool_name in whitelist


# Predefined whitelist sets for different isolation modes
MAINTENANCE_TOOLS = frozenset({
    "memory_search", "knowledge_search", "context_detail", "calculator",
})
CURIOSITY_TOOLS = MAINTENANCE_TOOLS | frozenset({
    "web_search", "http_fetch", "browser",
})
# Everything an unattended research generation (curiosity research, monitor
# queries) legitimately needs — and nothing with side effects. 2026-09-01:
# curiosity research ran think() with the whole registry (shell_exec,
# file_ops, desktop, tool_create) on web-derived topics, 44 times a week.
RESEARCH_TOOLS = CURIOSITY_TOOLS | frozenset({
    "deep_research", "active_memory", "code_understand", "screenshot",
})


class research_scope:
    """Context manager: run the block under RESEARCH_TOOLS, intersected with
    any whitelist already in force, and restore the previous whitelist."""

    def __enter__(self):
        self._prev = get_tool_whitelist()
        scope = RESEARCH_TOOLS if self._prev is None else (self._prev & RESEARCH_TOOLS)
        set_tool_whitelist(scope)
        return scope

    def __exit__(self, *exc):
        set_tool_whitelist(self._prev)
        return False


def _tier() -> str:

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
