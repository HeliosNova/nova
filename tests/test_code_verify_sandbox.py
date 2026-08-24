"""code_verify runtime isolation (audit 2026-08-22).

code_verify used to run model-authored code with only the static AST screen —
no runtime import guard and, critically, no resource ceilings — despite a
docstring claiming code_exec's isolation. These tests pin the resource-limit
guarantee (the child is capped, a runaway can't OOM the host) and that normal
verification still works. The in-process guarded path is exercised by forcing
the sidecar off.
"""

import asyncio

import pytest

from app.core.access_tiers import set_access_tier_override
from app.tools.code_verify import CodeVerifyTool


@pytest.fixture
def sandboxed_no_sidecar(monkeypatch):
    set_access_tier_override("sandboxed")
    # Force the in-process guarded path so the rlimits under test are ours.
    monkeypatch.setenv("EXEC_QUEUE_DIR", "/nonexistent_queue_for_test")
    yield
    set_access_tier_override(None)


def _verify(code, fn, cases):
    return asyncio.run(CodeVerifyTool().execute(code=code, function_name=fn, test_cases=cases))


def test_normal_verification_passes(sandboxed_no_sidecar):
    r = _verify(
        "def add(a, b):\n    return a + b",
        "add",
        [{"name": "c1", "input": [2, 3], "expected": 5}],
    )
    assert r.success, f"expected pass, got error={r.error!r} output={r.output!r}"
    assert "1/1 passed" in r.output


def test_failing_case_reported(sandboxed_no_sidecar):
    r = _verify(
        "def add(a, b):\n    return a + b",
        "add",
        [{"name": "wrong", "input": [2, 3], "expected": 99}],
    )
    assert r.success is False
    assert "FAIL" in r.output


def test_verifier_enforces_memory_rlimit(sandboxed_no_sidecar):
    # RLIMIT_AS caps the child near 1 GiB; a 2 GiB allocation must surface as a
    # MemoryError inside the case — not actually allocate 2 GiB in the host.
    r = _verify(
        "def hog(n):\n    x = bytearray(2 * 1024**3)\n    return len(x)",
        "hog",
        [{"name": "big", "input": [1], "expected": 123}],
    )
    assert r.success is False
    combined = (r.output or "") + (r.error or "")
    assert "MemoryError" in combined, f"rlimit not enforced; got {combined!r}"
