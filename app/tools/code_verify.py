"""Code verification tool — run code against test cases, return pass/fail per case.

Distinct from `code_exec` (which runs arbitrary code and returns stdout):
  - `code_exec` answers "what does this code do?"
  - `code_verify` answers "does this code give the right answer for these cases?"

Closes the code-debug loop — Nova can write a function, verify it against cases,
and iterate. Also produces a structured pass/fail reward signal usable for
future GRPO training on code tasks.

Safety: reuses `code_exec`'s AST-based blocked-import/builtin checks + subprocess
isolation with tier-aware `SYSTEM_ACCESS_LEVEL`.
"""

from __future__ import annotations

import asyncio
import json
import logging
import subprocess
import sys
import tempfile
from pathlib import Path

from app.config import config
from app.core.platform import get_safe_env
from app.tools.base import BaseTool, ToolResult, ErrorCategory
from app.tools.code_exec import _check_code_safety

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

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as f:
            f.write(harness)
            script_path = Path(f.name)

        try:
            proc = await asyncio.create_subprocess_exec(
                sys.executable, str(script_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=get_safe_env(),
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
                    error_category=ErrorCategory.TRANSIENT,
                    retriable=True,
                )

            stdout = stdout_b.decode("utf-8", errors="replace").strip()
            stderr = stderr_b.decode("utf-8", errors="replace").strip()

            if proc.returncode != 0:
                return ToolResult(
                    output=stdout, success=False,
                    error=f"verifier exited {proc.returncode}: {stderr[:500]}",
                    error_category=ErrorCategory.INTERNAL,
                )

            # Parse harness output
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

            # Format compact summary
            lines = [f"code_verify: {passed}/{len(results)} passed"]
            for r in results:
                mark = "OK  " if r.get("pass") else "FAIL"
                nm = r.get("name", "<unnamed>")
                if r.get("pass"):
                    lines.append(f"  {mark} {nm}")
                else:
                    if "error" in r:
                        lines.append(f"  {mark} {nm} — {r['error']}")
                    else:
                        lines.append(
                            f"  {mark} {nm} — expected {r.get('expected')!r}, "
                            f"got {r.get('actual')!r}"
                        )

            summary = "\n".join(lines)
            return ToolResult(
                output=summary,
                success=(failed == 0),
                error=None if failed == 0 else f"{failed} case(s) failed",
                error_category=None if failed == 0 else ErrorCategory.VALIDATION,
            )
        finally:
            try:
                script_path.unlink(missing_ok=True)
            except OSError:
                pass
