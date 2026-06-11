"""Calculator tool — safe SymPy math evaluation."""

from __future__ import annotations

import asyncio
import logging
import re

from app.tools.base import BaseTool, ToolResult, ErrorCategory

logger = logging.getLogger(__name__)

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
            result = await asyncio.wait_for(
                asyncio.to_thread(lambda: parse_expr(expression, local_dict={}, transformations=transformations).evalf()),
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
