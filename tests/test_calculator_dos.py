"""Calculator DoS guard (audit 2026-08-22).

The `**` power-tower (`10**10**10**10`) and huge factorial arguments must be
rejected *instantly* with a clear "too large" error — not evaluated on a shared
`asyncio.to_thread` worker that then keeps computing a multi-million-digit integer
after `wait_for` gives up (the leaked-worker DoS that poisons the pool DB writes
share). Large-but-cheap magnitudes (`9**9**9`, `10**10**10`) still resolve via
mpmath's floating-point path and are allowed.
"""

import asyncio
import time

import pytest

from app.tools.calculator import CalculatorTool


def _run(expr: str):
    return asyncio.run(CalculatorTool().execute(expression=expr))


@pytest.mark.parametrize(
    "expr,needle",
    [
        ("2**10", "1024"),
        ("47*89+156", "4339"),
        ("sqrt(144)", "12"),
        ("factorial(10)", "3628800"),
    ],
)
def test_normal_calculations_still_work(expr, needle):
    r = _run(expr)
    assert r.success, f"{expr} should succeed, got error={r.error!r}"
    assert needle in r.output, f"{expr} -> {r.output!r} missing {needle!r}"


@pytest.mark.parametrize("expr", ["10**10**10**10", "factorial(1000000)", "9**9**9**9"])
def test_dos_inputs_rejected_fast_not_timeout(expr):
    start = time.monotonic()
    r = _run(expr)
    elapsed = time.monotonic() - start
    # Must be a structural rejection, not a 10s wait_for timeout.
    assert elapsed < 5.0, f"{expr} took {elapsed:.1f}s — guard should reject instantly"
    assert not r.success, f"{expr} should be rejected, got output={r.output!r}"
    assert "too large" in (r.error or "").lower(), f"{expr} error={r.error!r}"
    assert "timed out" not in (r.error or "").lower(), f"{expr} timed out instead of guarding"


@pytest.mark.parametrize("expr", ["9**9**9", "10**10**10", "factorial(100000)"])
def test_large_but_cheap_inputs_allowed(expr):
    start = time.monotonic()
    r = _run(expr)
    elapsed = time.monotonic() - start
    assert elapsed < 5.0, f"{expr} took {elapsed:.1f}s — should resolve fast via mpmath"
    assert r.success, f"{expr} should succeed (compact scientific-notation), got error={r.error!r}"
