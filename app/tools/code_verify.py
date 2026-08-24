"""Code verification tool — run code against test cases, return pass/fail per case.

Distinct from `code_exec` (which runs arbitrary code and returns stdout):
  - `code_exec` answers "what does this code do?"
  - `code_verify` answers "does this code give the right answer for these cases?"

Closes the code-debug loop — Nova can write a function, verify it against cases,
and iterate. Also produces a structured pass/fail reward signal usable for
future GRPO training on code tasks.

Safety: the harness (user code + cases) runs through the SAME isolation as
`code_exec` — the tier-aware AST screen, the runtime import-guard + stripped-
builtins runner (`python -I`), hard CPU/memory/file rlimits, and the
network-isolated nova-exec sidecar when it is mounted. (Before 2026-08-22 this
module ran model code with only the AST screen and no resource ceilings,
despite the docstring's claim — audit fix.)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path

from app.config import config
from app.core.platform import IS_WINDOWS, get_safe_env
from app.tools.base import BaseTool, ToolResult, ErrorCategory
from app.tools.code_exec import (
    _build_guarded_runner,
    _check_code_safety,
    _posix_rlimits,
    _sidecar_run_raw,
)

logger = logging.getLogger(__name__)


_HARNESS_TEMPLATE = """
import json, sys, traceback

# --- User code ---
{user_code}

# --- Harness ---
_RESULTS = []
_CASES = json.loads({cases_json!r})
_FN_NAME = {fn_name!r}

_fn = globals().get(_FN_NAME)
if _fn is None or not callable(_fn):
    print(json.dumps({{"error": f"function {{_FN_NAME!r}} not defined in code"}}))
    sys.exit(1)

for case in _CASES:
    name = case.get("name", "<unnamed>")
    inp = case.get("input", [])
    expected = case.get("expected")
    try:
        if isinstance(inp, list):
            actual = _fn(*inp)
        elif isinstance(inp, dict):
            actual = _fn(**inp)
        else:
            actual = _fn(inp)
        ok = (actual == expected)
        _RESULTS.append({{
            "name": name,
            "pass": bool(ok),
            "input": inp,
            "expected": expected,
            "actual": actual,
        }})
    except Exception as e:
        _RESULTS.append({{
            "name": name,
            "pass": False,
            "input": inp,
            "expected": expected,
            "error": f"{{type(e).__name__}}: {{e}}",
            "traceback": traceback.format_exc(limit=5),
        }})

print(json.dumps({{"results": _RESULTS}}))
"""


class CodeVerifyTool(BaseTool):
    name = "code_verify"
    description = (
        "Run Python code against a list of test cases and return pass/fail per case. "
        "Each case has {name, input, expected}. Input can be a list (positional args), "
        "a dict (keyword args), or a single value. Returns structured results showing "
        "which cases passed, which failed, and actual vs expected for failures. "
        "Use this after writing a function to verify it works before reporting back to "
        "the owner. Prefer over code_exec when the goal is correctness checking (not "
        "arbitrary execution)."
    )
    parameters = (
        'code: the Python code to test (function definition), '
        'function_name: name of the function to call from the code, '
        'test_cases: list of {name, input, expected} dicts'
    )
    input_schema = {
        "type": "object",
        "properties": {
            "code": {"type": "string", "description": "Python source defining the function."},
            "function_name": {"type": "string", "description": "Name of the function to call."},
            "test_cases": {
                "type": "array",
                "description": "List of test cases. Each: {name, input, expected}.",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "input": {},
                        "expected": {},
                    },
                    "required": ["input", "expected"],
                },
            },
        },
        "required": ["code", "function_name", "test_cases"],
    }

    async def execute(
        self,
        code: str | None = None,
        function_name: str | None = None,
        test_cases: list | None = None,
        **_ignored,
    ) -> ToolResult:
        if not code or not isinstance(code, str):
            return ToolResult(
                output="", success=False,
                error="`code` argument missing or not a string",
                error_category=ErrorCategory.VALIDATION,
            )
        if not function_name or not isinstance(function_name, str):
            return ToolResult(
                output="", success=False,
                error="`function_name` argument missing or not a string",
                error_category=ErrorCategory.VALIDATION,
            )
        if not test_cases or not isinstance(test_cases, list):
            return ToolResult(
                output="", success=False,
                error="`test_cases` must be a non-empty list",
                error_category=ErrorCategory.VALIDATION,
            )

        # Tier-aware safety check on user code
        safety_err = _check_code_safety(code)
        if safety_err:
            return ToolResult(
                output="", success=False, error=safety_err,
                error_category=ErrorCategory.PERMISSION,
            )

        # Normalize cases to serializable JSON (reject non-serializable input/expected)
        try:
            cases_json = json.dumps(test_cases)
        except (TypeError, ValueError) as e:
            return ToolResult(
                output="", success=False,
                error=f"test_cases must be JSON-serializable: {e}",
                error_category=ErrorCategory.VALIDATION,
            )

        harness = _HARNESS_TEMPLATE.format(
            user_code=code,
            cases_json=cases_json,
            fn_name=function_name,
        )

        # Run the harness with full code_exec isolation: the network-isolated
        # nova-exec sidecar when mounted, else an in-process guarded runner
        # (python -I + import hook + stripped builtins) under hard CPU/memory
        # rlimits. Model-authored code is never run unsandboxed here.
        queue_dir = Path(os.environ.get("EXEC_QUEUE_DIR", "/exec_queue"))
        if queue_dir.is_dir():
            try:
                res = await _sidecar_run_raw(harness, queue_dir, config.TOOL_TIMEOUT)
            except Exception as e:
                return ToolResult(
                    output="", success=False,
                    error=f"verifier sidecar failed: {e}",
                    error_category=ErrorCategory.TRANSIENT, retriable=True,
                )
            if res.get("timed_out"):
                return ToolResult(
                    output="", success=False,
                    error=f"verifier timed out after {config.TOOL_TIMEOUT}s",
                    error_category=ErrorCategory.TRANSIENT, retriable=True,
                )
            return self._parse_harness_output(
                (res.get("stdout") or "").strip(),
                (res.get("stderr") or "").strip(),
                int(res.get("returncode") or 0),
            )

        # In-process guarded fallback (dev / no sidecar mounted).
        sandbox_dir = tempfile.mkdtemp(prefix="nova_verify_")
        try:
            harness_path = Path(sandbox_dir) / "_harness.py"
            harness_path.write_text(harness, encoding="utf-8")
            runner_path = Path(sandbox_dir) / "_runner.py"
            runner_path.write_text(_build_guarded_runner(), encoding="utf-8")

            proc = await asyncio.create_subprocess_exec(
                sys.executable, "-I", str(runner_path), str(harness_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=sandbox_dir,
                env=get_safe_env(),
                preexec_fn=None if IS_WINDOWS else _posix_rlimits,
            )
            try:
                stdout_b, stderr_b = await asyncio.wait_for(
                    proc.communicate(), timeout=config.TOOL_TIMEOUT,
                )
            except asyncio.TimeoutError:
                proc.kill()
                return ToolResult(
                    output="", success=False,
                    error=f"verifier timed out after {config.TOOL_TIMEOUT}s",
                    error_category=ErrorCategory.TRANSIENT, retriable=True,
                )
            return self._parse_harness_output(
                stdout_b.decode("utf-8", errors="replace").strip(),
                stderr_b.decode("utf-8", errors="replace").strip(),
                proc.returncode,
            )
        finally:
            shutil.rmtree(sandbox_dir, ignore_errors=True)

    def _parse_harness_output(self, stdout: str, stderr: str, returncode: int) -> ToolResult:
        """Turn the harness's JSON stdout into a compact pass/fail summary."""
        if returncode != 0:
            return ToolResult(
                output=stdout, success=False,
                error=f"verifier exited {returncode}: {stderr[:500]}",
                error_category=ErrorCategory.INTERNAL,
            )
        try:
            payload = json.loads(stdout)
        except json.JSONDecodeError:
            return ToolResult(
                output=stdout, success=False,
                error="verifier output not valid JSON",
                error_category=ErrorCategory.INTERNAL,
            )
        if "error" in payload:
            return ToolResult(
                output="", success=False, error=payload["error"],
                error_category=ErrorCategory.VALIDATION,
            )
        results = payload.get("results", [])
        passed = sum(1 for r in results if r.get("pass"))
        failed = len(results) - passed
        lines = [f"code_verify: {passed}/{len(results)} passed"]
        for r in results:
            mark = "OK  " if r.get("pass") else "FAIL"
            nm = r.get("name", "<unnamed>")
            if r.get("pass"):
                lines.append(f"  {mark} {nm}")
            elif "error" in r:
                lines.append(f"  {mark} {nm} — {r['error']}")
            else:
                lines.append(
                    f"  {mark} {nm} — expected {r.get('expected')!r}, got {r.get('actual')!r}"
                )
        summary = "\n".join(lines)
        return ToolResult(
            output=summary,
            success=(failed == 0),
            error=None if failed == 0 else f"{failed} case(s) failed",
            error_category=None if failed == 0 else ErrorCategory.VALIDATION,
        )
