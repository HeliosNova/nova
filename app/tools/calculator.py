"""Calculator tool — safe SymPy math evaluation."""

from __future__ import annotations

import asyncio
import logging
import math
import re
from concurrent.futures import ThreadPoolExecutor

from app.tools.base import BaseTool, ToolResult, ErrorCategory

logger = logging.getLogger(__name__)

# Dedicated worker pool for the (CPU-bound, occasionally-runaway) SymPy eval.
# A calculation that slips the magnitude guard below and pegs a worker is
# contained HERE — it can no longer poison the shared default `to_thread`
# executor that AsyncSafeDB writes ride on (audit 2026-08-22: the leaked-worker
# half of the calculator DoS). Threads can't be force-killed, so at worst two
# runaways exhaust this pool; the DB path stays clean.
_CALC_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="calc")

# Magnitude guard thresholds (DoS guards, not config knobs — Rule 2).
# _MATERIALIZE_LOG10: an exponent whose *value* exceeds 10^15 is a 16+ digit
#   number; materializing it (and then base**it) is where the `**` tower DoS
#   lives (`10**10**10**10` — the exponent alone would be a 10-billion-digit
#   int). 10**10**10 (exponent value 1e10) stays under this and resolves fast.
# _MAX_FACTORIAL_ARG: factorial(1e5) is ~0.1s; factorial(1e6) is ~5.5s and
#   climbs — cap the exact-combinatorial argument.
_MATERIALIZE_LOG10 = 15.0
_MAX_FACTORIAL_ARG = 100_000
_MAX_EXPRESSION_LEN = 500


def _log10_mag(expr) -> float:
    """Upper bound on log10(|expr|) as a float, computed structurally so it
    NEVER materializes a large integer. Returns math.inf when the magnitude
    blows past what we can safely reason about. Only the numeric constructs
    `parse_expr(evaluate=False)` yields for arithmetic input are handled
    precisely; symbols and transcendental calls (evalf'd at fixed precision,
    hence cheap regardless of argument) are treated as bounded."""
    from sympy import Add, Mul, Pow

    if expr.is_Number:
        try:
            v = abs(float(expr))
        except (OverflowError, ValueError, TypeError):
            return math.inf
        if v == 0.0:
            return -math.inf
        return math.log10(v)
    if isinstance(expr, Pow):
        base, exp = expr.as_base_exp()
        le = _log10_mag(exp)
        if le > _MATERIALIZE_LOG10:
            return math.inf          # exponent too large to materialize → blow-up
        try:
            ev = abs(float(exp))     # safe: |exp| <= 10^15, a ≤16-digit int
        except (OverflowError, ValueError, TypeError):
            return math.inf
        lb = _log10_mag(base)
        if lb == math.inf:
            return math.inf
        return ev * lb if lb > 0.0 else 0.0
    if isinstance(expr, Mul):
        total = 0.0
        for arg in expr.args:
            m = _log10_mag(arg)
            if m == math.inf:
                return math.inf
            total += m
        return total
    if isinstance(expr, Add):
        best = -math.inf
        for arg in expr.args:
            m = _log10_mag(arg)
            if m == math.inf:
                return math.inf
            best = max(best, m)
        return best + math.log10(max(len(expr.args), 1))
    return 0.0


# The eager blow-up functions. Unlike `**` (whose Pow defers under
# evaluate=False), these evaluate their argument at PARSE time even with
# evaluate=False — so `factorial(1000000)` builds a 5.5M-digit integer inside
# parse_expr before any guard could run. We therefore magnitude-check against a
# parse in which these names are inert placeholders (see `_safe_namespace`).
_FACT_NAMES = frozenset({
    "factorial", "factorial2", "subfactorial", "binomial",
    "RisingFactorial", "FallingFactorial", "gamma",
})


def _safe_namespace() -> dict:
    """Namespace mapping the eager blow-up functions to inert generic Functions
    so their arguments can be inspected without triggering evaluation."""
    from sympy import Function
    return {name: Function(name) for name in _FACT_NAMES}


def _expr_too_large(expr) -> str | None:
    """Reject, BEFORE real evaluation, expressions whose exact evaluation would
    build an astronomically large integer: power towers with an unmaterializable
    exponent, and factorial/combinatorial calls with a huge argument. Expects
    the placeholder parse from `_safe_namespace` (blow-up funcs are inert
    AppliedUndef nodes). Returns a short reason, or None if safe."""
    from sympy import Pow, preorder_traversal
    from sympy.core.function import AppliedUndef

    max_fact_log10 = math.log10(_MAX_FACTORIAL_ARG)
    for node in preorder_traversal(expr):
        if isinstance(node, Pow):
            _, exp = node.as_base_exp()
            if _log10_mag(exp) > _MATERIALIZE_LOG10:
                return "exponent too large"
        elif isinstance(node, AppliedUndef) and node.func.__name__ in _FACT_NAMES:
            for arg in node.args:
                if _log10_mag(arg) > max_fact_log10:
                    return "combinatorial argument too large"
    return None

# Block code-injection patterns before they reach SymPy's internal eval
_UNSAFE_RE = re.compile(
    r"__\w+__|(?<!\w)import\s|(?<!\w)eval\s*\(|(?<!\w)exec\s*\(|"
    r"(?<!\w)compile\s*\(|(?<!\w)globals\s*\(|(?<!\w)locals\s*\(|"
    r"(?<!\w)getattr\s*\(|(?<!\w)setattr\s*\(|(?<!\w)delattr\s*\(|"
    r"(?<!\w)open\s*\(|(?<!\w)input\s*\(|(?<!\w)breakpoint\s*\(|"
    r"(?<!\w)vars\s*\(|"
    r"\bos\.|\bsys\.|\bsubprocess\.|\bshutil\.|\bsocket\.",
    re.IGNORECASE,
)

# Math words SymPy legitimately understands. Any OTHER alphabetic token of 3+
# letters is natural language leaking into the expression — with implicit
# multiplication enabled, SymPy happily parses "Calculate 47*89" into
# C*a*l*c*u*l*a*t*e * 47*89 and returns algebra soup with success=True.
# Reject it up front with an error that tells the model what to do instead.
_MATH_WORDS = {
    "sqrt", "cbrt", "root", "abs", "sign", "exp", "log", "ln",
    "sin", "cos", "tan", "asin", "acos", "atan", "atan2",
    "sinh", "cosh", "tanh", "asinh", "acosh", "atanh",
    "pi", "oo", "inf", "infinity", "nan",
    "integrate", "diff", "solve", "limit", "summation", "product",
    "factorial", "binomial", "gamma", "floor", "ceiling", "mod",
    "min", "max", "rational", "simplify", "expand", "factor",
}

_WORD_RE = re.compile(r"[A-Za-z]{3,}")


class CalculatorTool(BaseTool):
    name = "calculator"
    description = (
        "Evaluate mathematical expressions using SymPy. Supports arithmetic, algebra, calculus, and symbolic math. "
        "Returns the expression and its evaluated result. "
        "Use for ANY calculation, even simple ones — never do mental math. "
        "Pass the ENTIRE expression in ONE call (e.g. '47*89+156', not '47*89' then adding 156 yourself). "
        "Do NOT use for string manipulation or non-math operations."
    )
    parameters = "expression: str"
    input_schema = {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "Mathematical expression to evaluate (e.g., '2**10', 'sqrt(144)', 'integrate(x**2, x)').",
            },
        },
        "required": ["expression"],
    }

    async def execute(self, *, expression: str = "", **kwargs) -> ToolResult:
        if not expression:
            return ToolResult(output="", success=False, error="No expression provided", error_category=ErrorCategory.VALIDATION)

        # Bound literal sizes so a pasted 10k-digit number can't blow up parsing.
        if len(expression) > _MAX_EXPRESSION_LEN:
            return ToolResult(
                output="", success=False,
                error=f"Expression too long (>{_MAX_EXPRESSION_LEN} chars).",
                error_category=ErrorCategory.VALIDATION,
            )

        # Input sanitization — reject anything that looks like code injection
        if _UNSAFE_RE.search(expression):
            return ToolResult(
                output="",
                success=False,
                error="Expression contains disallowed patterns. Use pure math only.",
                error_category=ErrorCategory.VALIDATION,
            )

        # Reject natural-language words before SymPy turns them into symbols
        stray = [w for w in _WORD_RE.findall(expression) if w.lower() not in _MATH_WORDS]
        if stray:
            return ToolResult(
                output="",
                success=False,
                error=(
                    f"Not a pure math expression (found words: {', '.join(stray[:3])}). "
                    "Pass only the math itself, e.g. '47*89+156'."
                ),
                error_category=ErrorCategory.VALIDATION,
            )

        try:
            from sympy.parsing.sympy_parser import (
                parse_expr,
                standard_transformations,
                implicit_multiplication_application,
            )

            transformations = standard_transformations + (implicit_multiplication_application,)
            # Parse WITHOUT evaluating: a power tower like 9**9**9 must NOT be
            # expanded into a multi-hundred-million-digit integer at parse time
            # (the DoS). evalf() then resolves it through mpmath's floating point
            # (compact + fast). The magnitude guard rejects the pathological
            # cases mpmath can't shortcut (unmaterializable exponents, huge
            # factorials) BEFORE any evaluation happens, and evalf runs on a
            # dedicated pool so a residual runaway can't poison DB writes.
            # Pass 1 — magnitude guard on a parse where factorial/gamma/etc. are
            # inert placeholders, so the check runs BEFORE those functions can
            # build a giant integer at parse time.
            guard_tree = parse_expr(
                expression, local_dict=_safe_namespace(),
                transformations=transformations, evaluate=False,
            )
            too_large = _expr_too_large(guard_tree)
            if too_large:
                return ToolResult(
                    output="", success=False,
                    error=f"Expression too large to evaluate safely ({too_large}). "
                          "Try a smaller magnitude.",
                    error_category=ErrorCategory.VALIDATION,
                )
            # Pass 2 — real parse (cleared by the guard) + deferred numeric eval.
            parsed = parse_expr(
                expression, local_dict={},
                transformations=transformations, evaluate=False,
            )
            loop = asyncio.get_running_loop()
            result = await asyncio.wait_for(
                loop.run_in_executor(_CALC_EXECUTOR, parsed.evalf),
                timeout=10.0,
            )

            # Format nicely
            output = f"{expression} = {result}"

            # If it's a real integer result, show without decimals.
            # NOTE: SymPy's Float.__eq__ against a Python int is STRUCTURAL
            # equality — Float(4339.0) == 4339 is False — so the old
            # `result == int(result)` branch never fired and every integer
            # shipped with a 15-digit decimal tail ("4339.00000000000").
            # The production 9B then mis-copied that tail into answers
            # (live audit 2026-06-10: answered 4329/4325 for 4339).
            # `.equals()` compares mathematically. The magnitude guard keeps
            # huge results (2**1000) in scientific notation instead of
            # printing 300 digits of false precision from a 15-digit Float.
            try:
                if result.is_real and abs(result) < 10**15 and result.equals(int(result)):
                    output = f"{expression} = {int(result)}"
            except (TypeError, ValueError, OverflowError):
                pass

            return ToolResult(output=output, success=True)

        except Exception as e:
            return ToolResult(
                output="",
                success=False,
                error=f"Math error: {e}. Check expression syntax.",
                error_category=ErrorCategory.VALIDATION,
            )
