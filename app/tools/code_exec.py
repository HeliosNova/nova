"""Code execution tool — tier-aware Python sandbox."""

from __future__ import annotations

import ast
import asyncio
import logging
import subprocess
import sys
import tempfile
from pathlib import Path

from app.config import config
from app.core.access_tiers import get_blocked_builtins, get_blocked_imports
from app.core.platform import get_safe_env
from app.tools.base import BaseTool, ToolResult, ErrorCategory

logger = logging.getLogger(__name__)


def _check_code_safety(code: str) -> str | None:
    """Check code against tier-aware blocked imports and builtins using AST analysis.

    Returns error message or None if safe.
    """
    blocked_imports = get_blocked_imports()
    blocked_builtins = get_blocked_builtins()

    # "none" tier: no restrictions
    if not blocked_imports and not blocked_builtins:
        return None

    # Extract builtin function names (strip trailing parens)
    blocked_builtin_names = {b.rstrip("(") for b in blocked_builtins}

    # --- AST-based analysis ---
    try:
        tree = ast.parse(code)
    except SyntaxError:
        # If it doesn't parse, fall back to text checks (the code will fail to run anyway)
        return _check_code_safety_text(code, blocked_imports, blocked_builtins)

    # Expanded dunder attribute blocklist for sandbox escape prevention
    _blocked_dunder_attrs = frozenset({
        "__loader__", "__spec__", "__builtins__",
        "__class__", "__bases__", "__mro__", "__subclasses__",
        "__globals__", "__code__",
    })

    for node in ast.walk(tree):
        # Check import statements
        if isinstance(node, ast.Import):
            for alias in node.names:
                top_module = alias.name.split(".")[0]
                if top_module in blocked_imports:
                    return f"Import '{top_module}' is blocked for security."

        elif isinstance(node, ast.ImportFrom):
            if node.module:
                top_module = node.module.split(".")[0]
                if top_module in blocked_imports:
                    return f"Import '{top_module}' is blocked for security."

        # Check calls to __import__, eval, exec, compile, etc.
        elif isinstance(node, ast.Call):
            func = node.func
            # Direct call: eval(...), exec(...), __import__(...)
            if isinstance(func, ast.Name) and func.id in blocked_builtin_names:
                return f"'{func.id}' is blocked for security."
            # Attribute access: builtins.__import__(...)
            if isinstance(func, ast.Attribute) and func.attr in blocked_builtin_names:
                return f"'{func.attr}' is blocked for security."
            # getattr() bypass: getattr(obj, "blocked_builtin") or getattr(obj, "__dunder__")
            if isinstance(func, ast.Name) and func.id == "getattr" and len(node.args) >= 2:
                second_arg = node.args[1]
                if isinstance(second_arg, ast.Constant) and isinstance(second_arg.value, str):
                    if second_arg.value in blocked_builtin_names:
                        return f"getattr() with '{second_arg.value}' is blocked for security."
                    if second_arg.value in _blocked_dunder_attrs:
                        return f"getattr() with '{second_arg.value}' is blocked for security."

        # Check bare Name references to __builtins__, __import__, builtins
        elif isinstance(node, ast.Name):
            if node.id in blocked_builtin_names and node.id.startswith("__"):
                return f"'{node.id}' is blocked for security."
            if node.id == "builtins":
                return "Access to 'builtins' is blocked for security."

        # Block access to module internals for sandbox escape
        elif isinstance(node, ast.Attribute):
            if node.attr in _blocked_dunder_attrs:
                return f"Access to '{node.attr}' is blocked for security."

    return None


def _check_code_safety_text(
    code: str, blocked_imports: set[str], blocked_builtins: list[str]
) -> str | None:
    """Fallback text-based check for code that doesn't parse as valid Python."""
    for blocked in blocked_imports:
        if f"import {blocked}" in code or f"from {blocked}" in code:
            return f"Import '{blocked}' is blocked for security."

    for builtin in blocked_builtins:
        if builtin in code:
            return f"'{builtin.rstrip('(')}' is blocked for security."

    return None


class CodeExecTool(BaseTool):
    name = "code_exec"
    description = (
        "Execute Python code in a sandboxed subprocess. Returns stdout and stderr. "
        "Use for data processing, complex calculations, formatting, or any task requiring code execution. "
        "Code runs in an isolated environment with minimal PATH and no access to environment variables. "
        "Imports are restricted based on the SYSTEM_ACCESS_LEVEL tier. "
        "Do NOT use for shell commands (use shell_exec instead)."
    )
    parameters = "code: str"
    input_schema = {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "Python code to execute. Must be valid Python 3 syntax.",
            },
        },
        "required": ["code"],
    }

    def trim_output(self, output: str) -> str:
        """Keep tail of output — most relevant for code results."""
        if len(output) <= 2000:
            return output
        return '[...truncated]\n' + output[-1500:]

    async def execute(self, *, code: str = "", **kwargs) -> ToolResult:
        if not code:
            return ToolResult(output="", success=False, error="No code provided", error_category=ErrorCategory.VALIDATION)

        # Safety check
        safety_error = _check_code_safety(code)
        if safety_error:
            return ToolResult(output="", success=False, error=safety_error, error_category=ErrorCategory.PERMISSION)

        script_path = None
        try:
            # Write code to temp file
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", delete=False
            ) as f:
                f.write(code)
                script_path = f.name

            # Execute in subprocess with timeout and minimal env (no token leakage)
            result = await asyncio.to_thread(
                subprocess.run,
                [sys.executable, script_path],
                capture_output=True,
                text=True,
                timeout=config.CODE_EXEC_TIMEOUT,
                cwd=tempfile.gettempdir(),
                env=get_safe_env(),
            )

            if result.stdout and result.stderr:
                output = f"[stdout]\n{result.stdout}\n[stderr]\n{result.stderr}"
            elif result.stdout:
                output = result.stdout
            elif result.stderr:
                output = f"[stderr]\n{result.stderr}"
            else:
                output = ""

            if result.returncode != 0:
                return ToolResult(
                    output=output or result.stderr,
                    success=False,
                    error=f"Script exited with code {result.returncode}",
                    error_category=ErrorCategory.INTERNAL,
                )

            if not output.strip():
                output = "[Code executed successfully with no output]"

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
                error=f"Code execution timed out after {config.CODE_EXEC_TIMEOUT}s",
                error_category=ErrorCategory.TRANSIENT,
            )
        except Exception as e:
            return ToolResult(output="", success=False, error=f"Execution failed: {e}", error_category=ErrorCategory.INTERNAL)
        finally:
            if script_path:
                try:
                    Path(script_path).unlink()
                except OSError:
                    pass
