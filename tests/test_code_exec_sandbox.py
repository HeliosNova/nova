"""Runtime sandbox guarantees for code_exec.

The static AST screen (`_check_code_safety`) is a fast first layer, not the
boundary — it was bypassable via `e = exec; e(...)` (audit 2026-06-12). The
real enforcement is the runtime guard inside the execution subprocess: a
spelling-proof import hook plus a stripped builtins namespace. These tests
pin both layers at the sandboxed tier.
"""

from __future__ import annotations

import asyncio
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from app.core.access_tiers import set_access_tier_override
from app.tools.code_exec import CodeExecTool, _build_guarded_runner, _check_code_safety


@pytest.fixture
def sandboxed_tier():
    set_access_tier_override("sandboxed")
    yield
    set_access_tier_override(None)


def _run(code: str):
    return asyncio.run(CodeExecTool().execute(code=code))


def _run_via_runner(code: str) -> subprocess.CompletedProcess:
    """Run code through the guarded runner directly, bypassing the AST screen.

    This isolates the RUNTIME enforcement layer: whatever the static check
    would or wouldn't catch, the in-subprocess guard must still hold.
    """
    d = tempfile.mkdtemp(prefix="nova_test_")
    runner = Path(d, "_runner.py")
    runner.write_text(_build_guarded_runner(), encoding="utf-8")
    script = Path(d, "_script.py")
    script.write_text(code, encoding="utf-8")
    return subprocess.run(
        [sys.executable, "-I", str(runner), str(script)],
        capture_output=True, text=True, timeout=30,
    )


# ---- AST layer: the documented alias bypass is now caught ----------------

def test_alias_of_exec_is_blocked_statically(sandboxed_tier):
    # The exact bypass from the audit: rebind exec, then call the alias.
    assert _check_code_safety("e = exec\ne('import os')") is not None


def test_alias_of_open_is_blocked_statically(sandboxed_tier):
    assert _check_code_safety("o = open\no('/etc/passwd')") is not None


def test_plain_math_still_passes_ast(sandboxed_tier):
    assert _check_code_safety("import math\nprint(math.factorial(5))") is None


# ---- Runtime layer: enforcement survives an AST miss ---------------------

def test_runtime_blocks_blocked_import_end_to_end(sandboxed_tier):
    # Even reaching execution, the import hook denies a blocked module.
    result = _run("import os\nprint(os.getcwd())")
    assert result.success is False
    assert "block" in (result.error or result.output or "").lower()


def test_runtime_strips_exec_from_user_namespace(sandboxed_tier):
    # If a craft slipped past AST, exec is simply absent at runtime -> NameError,
    # never a working code-execution primitive.
    result = _run("print(type(exec))")
    assert result.success is False
    assert "exec" in (result.output or result.error or "")


def test_runtime_hook_blocks_fresh_socket_import(sandboxed_tier):
    # socket is not preloaded -> the import hook denies it even with the AST
    # screen out of the picture. This is the exfil vector that matters.
    proc = _run_via_runner("import socket\nprint(socket.gethostname())")
    assert proc.returncode != 0
    assert "blocked by the code sandbox" in proc.stderr


def test_runtime_hook_blocks_fresh_subprocess_import(sandboxed_tier):
    proc = _run_via_runner("import subprocess\nsubprocess.run(['echo', 'hi'])")
    assert proc.returncode != 0
    assert "blocked by the code sandbox" in proc.stderr


def test_runtime_hook_allows_preloaded_and_stdlib(sandboxed_tier):
    # sys is preloaded (and needed transitively by re/json/enum) -> must pass,
    # or we would break every legitimate stdlib import.
    proc = _run_via_runner("import sys, json, re\nprint(json.dumps({'p': sys.version_info[0]}))")
    assert proc.returncode == 0, proc.stderr
    assert '"p"' in proc.stdout


def test_runtime_exec_is_gone_even_bypassing_ast(sandboxed_tier):
    # Directly through the runner: exec/eval/open are absent from user builtins.
    proc = _run_via_runner("print('exec' in dir(__builtins__))")
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "False"


# ---- Legit code keeps working --------------------------------------------

def test_legit_data_processing_runs(sandboxed_tier):
    result = _run("nums = [i*i for i in range(5)]\nprint(sum(nums))")
    assert result.success is True
    assert "30" in result.output


def test_allowed_import_runs(sandboxed_tier):
    result = _run("import json\nprint(json.dumps({'ok': True}))")
    assert result.success is True
    assert "true" in result.output.lower()
