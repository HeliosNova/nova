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

            # If it's a real integer result, show without decimals
            if result.is_real and result == int(result):
                output = f"{expression} = {int(result)}"

            return ToolResult(output=output, success=True)

        except Exception as e:
            return ToolResult(
                output="",
                success=False,
                error=f"Math error: {e}. Check expression syntax.",
                error_category=ErrorCategory.VALIDATION,
            )
