"""Code execution tool — tier-aware Python sandbox."""

from __future__ import annotations

import ast
import asyncio
import logging
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from app.config import config
from app.core.access_tiers import get_blocked_builtins, get_blocked_imports
from app.core.platform import IS_WINDOWS, get_safe_env
from app.tools.base import BaseTool, ToolResult, ErrorCategory

logger = logging.getLogger(__name__)

# Resource ceilings for the execution subprocess (POSIX only). These are
# tier-independent DoS guards — not new config knobs (Rule 2). Generous enough
# for numpy/pandas, tight enough to stop runaway allocation or disk-fill.
_CODE_EXEC_MAX_ADDRESS_SPACE = 1024 * 1024 * 1024  # 1 GiB virtual memory
_CODE_EXEC_MAX_FILE_BYTES = 50 * 1024 * 1024        # 50 MiB per written file

# Builtins we physically remove from the user namespace at runtime (when the
# tier blocks them) — the execution + file-open primitives. Static AST checks
# alone never held: `e = exec; e(...)` rebinds past them. Removing the names
# means the alias has nothing to point at. Introspection builtins
# (getattr/globals/dir/...) are left to the AST layer; pulling them breaks too
# much normal code for little gain once import is guarded and eval/exec are gone.
_RUNTIME_REMOVABLE_BUILTINS = frozenset({"eval", "exec", "compile", "open", "breakpoint"})

# Bootstrap that wraps user code in the subprocess. It installs a runtime
# import guard (every import spelling funnels through builtins.__import__, so
# blocking there is spelling-proof) and runs user code under a builtins mapping
# with the dangerous primitives stripped. The runner itself keeps full builtins
# — only the *user* globals are restricted. Imports only sys+builtins so the
# subclass-walk escape has no os/subprocess/socket gadget already loaded.
_RUNNER_BODY = '''
import sys as _sys, builtins as _builtins

# Modules the interpreter already loaded at startup (os, io, sys, ...). We must
# NOT block these or stdlib's own transitive imports (json -> re -> enum ->
# import sys) break. They are also impossible to truly contain in-process, so
# source-level use of them is the AST layer's job. Fresh imports of dangerous
# modules that are NOT preloaded (socket, urllib, subprocess, ctypes,
# requests, httpx) are the exfil/escape vectors this hook actually stops.
_PRELOADED = frozenset(_sys.modules)

_real_import = _builtins.__import__

def _guarded_import(name, *a, **k):
    top = name.split(".", 1)[0]
    if top in _BLOCKED_IMPORTS and top not in _PRELOADED:
        raise ImportError("import of %r is blocked by the code sandbox" % top)
    return _real_import(name, *a, **k)

_builtins.__import__ = _guarded_import

_user_builtins = _builtins.__dict__.copy()
for _n in _REMOVED_BUILTINS:
    _user_builtins.pop(_n, None)
_user_builtins["__import__"] = _guarded_import

_script = _sys.argv[1]
with open(_script, "r", encoding="utf-8") as _fh:
    _source = _fh.read()

_user_globals = {"__name__": "__main__", "__file__": _script, "__builtins__": _user_builtins}
exec(compile(_source, _script, "exec"), _user_globals)
'''


def _build_guarded_runner() -> str:
    """Return runner source with the current tier's block lists baked in."""
    blocked_imports = sorted(get_blocked_imports())
    blocked_builtin_names = {b.rstrip("(") for b in get_blocked_builtins()}
    removed = sorted(_RUNTIME_REMOVABLE_BUILTINS & blocked_builtin_names)
    header = "_BLOCKED_IMPORTS = set(%r)\n_REMOVED_BUILTINS = set(%r)\n" % (
        blocked_imports,
        removed,
    )
    return header + _RUNNER_BODY


def _posix_rlimits() -> None:
    """preexec_fn: cap CPU/memory/file-size/core for the child (POSIX only)."""
    import resource

    cpu = int(config.CODE_EXEC_TIMEOUT) + 2
    resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu + 1))
    resource.setrlimit(resource.RLIMIT_FSIZE, (_CODE_EXEC_MAX_FILE_BYTES, _CODE_EXEC_MAX_FILE_BYTES))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    try:
        resource.setrlimit(resource.RLIMIT_AS, (_CODE_EXEC_MAX_ADDRESS_SPACE, _CODE_EXEC_MAX_ADDRESS_SPACE))
    except (ValueError, OSError):
        # Some platforms refuse RLIMIT_AS; the other limits still apply.
        pass


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

    # First pass: collect simple aliases of blocked builtins, e.g. `e = exec`
    # or `o = open`. The original checker only matched calls whose func.id was
    # *literally* a blocked name, so `e = exec; e(code)` slipped straight
    # through (audit 2026-06-12). We now flag both the binding and any later
    # call through the alias. This is a cheap first layer — the real boundary
    # is the runtime guard in the subprocess (see _build_guarded_runner).
    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Name):
            src = node.value.id
            if src in blocked_builtin_names:
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        aliases[target.id] = src

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
            # Call through an alias: e = exec; e(...)
            if isinstance(func, ast.Name) and func.id in aliases:
                return f"'{aliases[func.id]}' (aliased as '{func.id}') is blocked for security."
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
            # Aliasing a blocked builtin at all is suspicious — flag the binding.
            if node.id in aliases:
                return f"'{aliases[node.id]}' (aliased as '{node.id}') is blocked for security."

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

        sandbox_dir = None
        script_path = None
        try:
            # Per-request isolated sandbox directory — prevents cross-request file leakage.
            # Everything created by user code lands here and is wiped in finally.
            sandbox_dir = tempfile.mkdtemp(prefix="nova_code_")
            script_path = str(Path(sandbox_dir) / "_script.py")
            with open(script_path, "w", encoding="utf-8") as f:
                f.write(code)

            # The runner enforces the tier's block lists at RUNTIME (import hook
            # + stripped builtins), so source-spelling tricks that slip past the
            # static AST screen still can't import os or call exec.
            runner_path = str(Path(sandbox_dir) / "_runner.py")
            with open(runner_path, "w", encoding="utf-8") as f:
                f.write(_build_guarded_runner())

            # Execute in subprocess with timeout and minimal env (no token leakage).
            # POSIX gets hard CPU/memory/file-size ceilings via preexec_fn.
            result = await asyncio.to_thread(
                subprocess.run,
                [sys.executable, "-I", runner_path, script_path],
                capture_output=True,
                text=True,
                timeout=config.CODE_EXEC_TIMEOUT,
                cwd=sandbox_dir,
                env=get_safe_env(),
                preexec_fn=None if IS_WINDOWS else _posix_rlimits,
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
            if sandbox_dir:
                try:
                    shutil.rmtree(sandbox_dir, ignore_errors=True)
                except OSError:
                    pass
