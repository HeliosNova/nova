"""Dynamic Tool Creation — create, store, and execute user-defined tools.

Tools are Python scripts persisted in SQLite. They execute in a hardened
subprocess sandbox with forced sandboxed-level safety checks (regardless of
the system access tier) and a runtime preamble that blocks network access
and filesystem writes outside the sandbox directory.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from app.config import config
from app.core.access_tiers import requires_tier
from app.core.platform import get_safe_env
from app.tools.base import BaseTool, ErrorCategory, ToolResult
from app.tools.code_exec import _check_code_safety

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Forced-sandbox safety check (ignores system tier — always max restriction)
# ---------------------------------------------------------------------------

# Imports ALWAYS blocked for dynamic tools, regardless of system access tier.
# This is the sandboxed tier's blocklist, permanently enforced for user-created code.
_DYNAMIC_TOOL_BLOCKED_IMPORTS = frozenset({
    "os", "subprocess", "shutil", "sys", "importlib",
    "ctypes", "socket", "http", "urllib", "requests", "httpx",
    "pathlib", "glob", "signal", "multiprocessing",
    # Additional network/process modules not in the base sandboxed set
    "asyncio", "aiohttp", "websockets", "xmlrpc", "ftplib",
    "smtplib", "poplib", "imaplib", "telnetlib", "ssl",
    "webbrowser", "tempfile", "io",
    "pickle", "shelve", "marshal",
})

_DYNAMIC_TOOL_BLOCKED_BUILTINS = frozenset({
    "open", "getattr", "compile", "globals", "locals",
    "vars", "dir", "breakpoint",
    "__builtins__", "__import__", "eval", "exec",
})

# Dunder attributes that enable sandbox escape
_DYNAMIC_TOOL_BLOCKED_DUNDERS = frozenset({
    "__loader__", "__spec__", "__builtins__",
    "__class__", "__bases__", "__mro__", "__subclasses__",
    "__globals__", "__code__",
})


def _check_dynamic_tool_safety(code: str) -> str | None:
    """Per owner directive 2026-04-25: no policy filtering on custom tools.

    Nova is sovereign. He owns the box. When he writes a tool, it runs.
    Only hard-fail on syntax errors so we don't store broken code.
    The previous tier-conditional / always-blocked logic is removed entirely.
    """
    try:
        import ast as _ast
        _ast.parse(code)
    except SyntaxError as e:
        return f"Syntax error: {e}"
    return None


def _legacy_blocklist_disabled(code: str) -> str | None:
    """Retained only as documentation of what USED to be blocked. Never called."""
    import ast

    # --- AST-based analysis ---
    try:
        tree = ast.parse(code)
    except SyntaxError:
        # Fall back to text-based checks
        return _check_dynamic_tool_safety_text(code)

    blocked_builtin_names = {b.rstrip("(") for b in _DYNAMIC_TOOL_BLOCKED_BUILTINS}

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top_module = alias.name.split(".")[0]
                if top_module in _DYNAMIC_TOOL_BLOCKED_IMPORTS:
                    return f"Import '{top_module}' is blocked in dynamic tools."

        elif isinstance(node, ast.ImportFrom):
            if node.module:
                top_module = node.module.split(".")[0]
                if top_module in _DYNAMIC_TOOL_BLOCKED_IMPORTS:
                    return f"Import '{top_module}' is blocked in dynamic tools."

        elif isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in blocked_builtin_names:
                return f"'{func.id}' is blocked in dynamic tools."
            if isinstance(func, ast.Attribute) and func.attr in blocked_builtin_names:
                return f"'{func.attr}' is blocked in dynamic tools."
            if isinstance(func, ast.Name) and func.id == "getattr" and len(node.args) >= 2:
                second_arg = node.args[1]
                if isinstance(second_arg, ast.Constant) and isinstance(second_arg.value, str):
                    if second_arg.value in blocked_builtin_names:
                        return f"getattr() with '{second_arg.value}' is blocked in dynamic tools."
                    if second_arg.value in _DYNAMIC_TOOL_BLOCKED_DUNDERS:
                        return f"getattr() with '{second_arg.value}' is blocked in dynamic tools."

        elif isinstance(node, ast.Name):
            if node.id in blocked_builtin_names and node.id.startswith("__"):
                return f"'{node.id}' is blocked in dynamic tools."
            if node.id == "builtins":
                return "Access to 'builtins' is blocked in dynamic tools."

        elif isinstance(node, ast.Attribute):
            if node.attr in _DYNAMIC_TOOL_BLOCKED_DUNDERS:
                return f"Access to '{node.attr}' is blocked in dynamic tools."

    return None


def _check_dynamic_tool_safety_text(code: str) -> str | None:
    """Fallback text-based check for code that doesn't parse."""
    for blocked in _DYNAMIC_TOOL_BLOCKED_IMPORTS:
        if f"import {blocked}" in code or f"from {blocked}" in code:
            return f"Import '{blocked}' is blocked in dynamic tools."
    for builtin in _DYNAMIC_TOOL_BLOCKED_BUILTINS:
        token = builtin + "(" if not builtin.startswith("__") else builtin
        if token in code:
            return f"'{builtin}' is blocked in dynamic tools."
    return None


# Runtime sandbox preamble — injected into every dynamic tool script.
# Defence-in-depth for sandboxed/standard tiers. At 'full'/'none' the
# preamble is replaced with a no-op so tools can hit network and disk freely.
_SANDBOX_PREAMBLE_STRICT = '''\
import sys as _sys

# --- Block network access at runtime ---
class _BlockedSocket:
    def __init__(self, *a, **kw):
        raise PermissionError("Network access is blocked in dynamic tools")
    def __getattr__(self, name):
        raise PermissionError("Network access is blocked in dynamic tools")

# Poison the socket module so any network library fails
class _FakeSocketModule:
    socket = _BlockedSocket
    AF_INET = AF_INET6 = SOCK_STREAM = SOCK_DGRAM = 0
    def __getattr__(self, name):
        if callable(getattr(type(self), name, None)):
            return lambda *a, **kw: (_ for _ in ()).throw(
                PermissionError("Network access is blocked in dynamic tools"))
        raise PermissionError("Network access is blocked in dynamic tools")

_sys.modules["socket"] = _FakeSocketModule()
_sys.modules["http"] = type(_sys)("http")
_sys.modules["http.client"] = type(_sys)("http.client")
_sys.modules["urllib"] = type(_sys)("urllib")
_sys.modules["urllib.request"] = type(_sys)("urllib.request")

# --- Restrict open() to read-only in sandbox dir ---
_sandbox_dir = _sys.modules["__main__"].__dict__.get("_SANDBOX_DIR", ".")
_original_open = open

def _restricted_open(file, mode="r", *args, **kwargs):
    if any(m in str(mode) for m in ("w", "a", "x", "+")):
        from pathlib import Path as _P
        resolved = _P(str(file)).resolve()
        sandbox = _P(_sandbox_dir).resolve()
        if not (resolved == sandbox or str(resolved).startswith(str(sandbox) + _sys.modules["os.path"].sep if "os.path" in _sys.modules else str(sandbox))):
            raise PermissionError(
                f"Write access outside sandbox dir is blocked: {file}"
            )
    return _original_open(file, mode, *args, **kwargs)

try:
    import builtins as _builtins
    _builtins.open = _restricted_open
except Exception:
    pass

del _sys, _BlockedSocket, _FakeSocketModule
'''

_SANDBOX_PREAMBLE_PERMISSIVE = "# (no runtime sandbox — owner directive)\n"


def _sandbox_preamble() -> str:
    """Per owner directive 2026-04-25: no runtime sandbox. Sovereign machine."""
    return _SANDBOX_PREAMBLE_PERMISSIVE


# Use the no-op preamble at module-level too (DynamicTool.execute reads this)
_SANDBOX_PREAMBLE = _SANDBOX_PREAMBLE_PERMISSIVE


# ---------------------------------------------------------------------------
# Schema + Data types
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS custom_tools (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,
    description TEXT NOT NULL,
    parameters TEXT NOT NULL,
    code TEXT NOT NULL,
    times_used INTEGER DEFAULT 0,
    success_rate REAL DEFAULT 1.0,
    enabled BOOLEAN DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""


@dataclass
class CustomToolRecord:
    id: int
    name: str
    description: str
    parameters: str  # JSON string: [{"name": "x", "type": "str"}, ...]
    code: str
    times_used: int = 0
    success_rate: float = 1.0
    enabled: bool = True


# ---------------------------------------------------------------------------
# CustomToolStore — CRUD for custom tools
# ---------------------------------------------------------------------------

class CustomToolStore:
    """SQLite-backed store for user-created tools."""

    @property
    def MAX_CODE_LENGTH(self):
        return config.MAX_CUSTOM_TOOL_CODE_LENGTH

    @property
    def MAX_TOOLS(self):
        return config.MAX_CUSTOM_TOOLS

    def __init__(self, db):
        self._db = db
        self._db.execute(_SCHEMA.strip())

    def create_tool(
        self,
        name: str,
        description: str,
        parameters: str,
        code: str,
    ) -> int:
        """Create a new custom tool. Returns tool ID, or -1 on failure."""
        name = name.strip().lower().replace(" ", "_")
        if not name or len(name) > 50:
            logger.warning("Invalid tool name: %r", name)
            return -1

        # Check name uniqueness
        existing = self._db.fetchone(
            "SELECT id FROM custom_tools WHERE name = ?", (name,)
        )
        if existing:
            logger.warning("Tool '%s' already exists", name)
            return -1

        # Validate code safety — always use forced-sandbox restrictions
        safety_error = _check_dynamic_tool_safety(code)
        if safety_error:
            logger.warning("Tool '%s' code blocked: %s", name, safety_error)
            return -1

        # Size limits
        if len(code) > self.MAX_CODE_LENGTH:
            logger.warning("Tool '%s' code too long: %d chars", name, len(code))
            return -1

        # Placeholder guard — Nova sometimes proposes tools whose body is
        # "# Placeholder for X" / "actual tool calls are not permitted in this
        # function" / hardcoded f-strings masquerading as results. These are
        # useless and misleading when invoked. Reject before storing.
        _code_lower = code.lower()
        _placeholder_markers = (
            "placeholder for", "placeholder implementation",
            "actual tool calls are not permitted",
            "this would invoke", "would be invoked",
            "not actually execute", "fake implementation",
        )
        if any(m in _code_lower for m in _placeholder_markers):
            logger.warning(
                "Tool '%s' rejected: body contains placeholder markers (%s)",
                name, [m for m in _placeholder_markers if m in _code_lower][:2],
            )
            return -1

        # Tool count limit
        count = self._db.fetchone("SELECT COUNT(*) as c FROM custom_tools")
        if count and count["c"] >= self.MAX_TOOLS:
            logger.warning("Tool limit reached (%d)", self.MAX_TOOLS)
            return -1

        # Validate parameters JSON
        try:
            if isinstance(parameters, list):
                parameters = json.dumps(parameters)
            elif isinstance(parameters, str):
                json.loads(parameters)  # validate it's valid JSON
        except (json.JSONDecodeError, TypeError):
            parameters = "[]"

        cursor = self._db.execute(
            "INSERT INTO custom_tools (name, description, parameters, code) VALUES (?, ?, ?, ?)",
            (name, description[:500], parameters, code),
        )
        return cursor.lastrowid

    def get_tool(self, name: str) -> CustomToolRecord | None:
        """Get a tool by name."""
        row = self._db.fetchone(
            "SELECT * FROM custom_tools WHERE name = ? AND enabled = 1", (name,)
        )
        if not row:
            return None
        return CustomToolRecord(
            id=row["id"], name=row["name"], description=row["description"],
            parameters=row["parameters"], code=row["code"],
            times_used=row["times_used"], success_rate=row["success_rate"],
            enabled=bool(row["enabled"]),
        )

    def get_all_tools(self) -> list[CustomToolRecord]:
        """Get all enabled tools."""
        rows = self._db.fetchall(
            "SELECT * FROM custom_tools WHERE enabled = 1 ORDER BY name"
        )
        return [
            CustomToolRecord(
                id=r["id"], name=r["name"], description=r["description"],
                parameters=r["parameters"], code=r["code"],
                times_used=r["times_used"], success_rate=r["success_rate"],
                enabled=bool(r["enabled"]),
            )
            for r in rows
        ]

    def record_use(self, name: str, success: bool) -> str | None:
        """Record a tool usage. Auto-disables if success rate drops below 0.3 after 5+ uses.

        Returns a warning message if the tool was auto-disabled, None otherwise.
        """
        # Use EMA (alpha=0.15) for consistency with skills
        alpha = 0.15
        success_val = 1.0 if success else 0.0
        self._db.execute(
            "UPDATE custom_tools SET times_used = times_used + 1, "
            "success_rate = ? * ? + (1 - ?) * success_rate WHERE name = ?",
            (alpha, success_val, alpha, name),
        )
        # Re-read for auto-disable check
        row = self._db.fetchone(
            "SELECT times_used, success_rate FROM custom_tools WHERE name = ?", (name,)
        )
        if not row:
            return None

        new_uses = row["times_used"]
        new_rate = row["success_rate"]

        # Auto-disable if consistently failing
        if new_uses >= 5 and new_rate < 0.3:
            self._db.execute(
                "UPDATE custom_tools SET enabled = 0 WHERE name = ?", (name,)
            )
            msg = f"Tool '{name}' auto-disabled (success rate {new_rate*100:.0f}% after {new_uses} uses)"
            logger.info(msg)
            return msg
        return None

    def delete_tool(self, name: str) -> bool:
        """Delete a tool by name."""
        cursor = self._db.execute("DELETE FROM custom_tools WHERE name = ?", (name,))
        return cursor.rowcount > 0

    def toggle_tool(self, name: str, enabled: bool) -> bool:
        """Enable or disable a tool."""
        cursor = self._db.execute(
            "UPDATE custom_tools SET enabled = ? WHERE name = ?", (int(enabled), name)
        )
        return cursor.rowcount > 0


# ---------------------------------------------------------------------------
# DynamicTool — wraps a CustomToolRecord as a BaseTool
# ---------------------------------------------------------------------------

class DynamicTool(BaseTool):
    """Executes a user-defined Python tool in a subprocess sandbox."""

    def __init__(self, record: CustomToolRecord, store: CustomToolStore):
        self.name = record.name
        self.description = record.description
        self.parameters = record.parameters
        self._code = record.code
        self._store = store

    @requires_tier("standard", "full")
    async def execute(self, **kwargs) -> ToolResult:
        """Build and execute the tool's Python script in a hardened sandbox."""
        # FORCED sandbox-level safety check — ignores system tier.
        # Dynamic tools are user-created code and must always be maximally restricted.
        safety_error = _check_dynamic_tool_safety(self._code)
        if safety_error:
            return ToolResult(output="", success=False, error=safety_error, error_category=ErrorCategory.PERMISSION)

        # Validate args against declared schema
        try:
            declared = json.loads(self.parameters) if self.parameters else []
            declared_names = {p["name"] for p in declared if isinstance(p, dict)}
            if declared_names:
                unexpected = set(kwargs.keys()) - declared_names
                if unexpected:
                    return ToolResult(output="", success=False,
                        error=f"Unexpected parameters: {unexpected}. Expected: {declared_names}",
                        error_category=ErrorCategory.VALIDATION)
        except (json.JSONDecodeError, KeyError) as _schema_err:
            logger.warning(
                "[CustomTool %s] malformed parameter schema (%s); proceeding without validation",
                getattr(self, "name", "?"), _schema_err,
            )

        # Create isolated sandbox directory for this execution
        sandbox_dir = None
        script_path = None
        try:
            sandbox_dir = tempfile.mkdtemp(prefix="nova_tool_")

            # Build script with sandbox preamble + user code + invocation.
            # Always emit a UTF-8 BOM/declaration so Python doesn't reject
            # files containing non-ASCII bytes from user code (Windows cp1252
            # default + Python's PEP 263 strict-encoding rule was rejecting
            # any bytes > 0x7f without declaration).
            args_json = json.dumps(kwargs)
            script = (
                f"# -*- coding: utf-8 -*-\n"
                f"_SANDBOX_DIR = {repr(sandbox_dir)}\n"
                f"{_SANDBOX_PREAMBLE}\n"
                f"# --- User tool code ---\n"
                f"{self._code}\n\n"
                f"import json as _json\n"
                f"_args = _json.loads({repr(args_json)})\n"
                f"_result = run(**_args)\n"
                f"print(_result)\n"
            )

            script_file = os.path.join(sandbox_dir, "_tool.py")
            with open(script_file, "w", encoding="utf-8") as f:
                f.write(script)
            script_path = script_file

            result = await asyncio.to_thread(
                subprocess.run,
                [sys.executable, "-I", script_path],
                capture_output=True,
                text=True,
                timeout=config.CODE_EXEC_TIMEOUT,
                cwd=sandbox_dir,
                env=get_safe_env(),
            )

            success = result.returncode == 0
            output = result.stdout
            if result.stderr:
                output += f"\n[stderr]: {result.stderr}"

            if not output.strip():
                output = "[Tool executed successfully with no output]"

            if len(output) > 5000:
                output = output[:5000] + "\n[... output truncated]"

            # Record usage
            self._store.record_use(self.name, success=success)

            if not success:
                return ToolResult(
                    output=output or result.stderr,
                    success=False,
                    error=f"Tool exited with code {result.returncode}",
                    error_category=ErrorCategory.INTERNAL,
                )

            return ToolResult(output=output, success=True)

        except subprocess.TimeoutExpired:
            self._store.record_use(self.name, success=False)
            return ToolResult(
                output="", success=False,
                error=f"Tool timed out after {config.CODE_EXEC_TIMEOUT}s",
                error_category=ErrorCategory.TRANSIENT,
            )
        except Exception as e:
            self._store.record_use(self.name, success=False)
            return ToolResult(output="", success=False, error=f"Tool failed: {e}", error_category=ErrorCategory.INTERNAL)
        finally:
            # Clean up the entire sandbox directory
            if sandbox_dir:
                import shutil
                try:
                    shutil.rmtree(sandbox_dir, ignore_errors=True)
                except OSError:
                    pass


# Tool-create prompt addition for the system prompt
TOOL_CREATE_DESCRIPTION = (
    "tool_create(name: str, description: str, parameters: str, code: str) "
    "— Create a new reusable tool. Write the tool as a Python function named 'run' "
    "that takes the declared parameters and returns a string result. "
    "Only create tools for capabilities you'll need repeatedly."
)
