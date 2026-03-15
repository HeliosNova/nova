"""Calculator tool — safe SymPy math evaluation."""

from __future__ import annotations

import logging
import re

from app.tools.base import BaseTool, ToolResult

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
    description = "Evaluate math expressions with SymPy. Use for ANY calculation, even simple ones."
    parameters = "expression: str"

    async def execute(self, *, expression: str = "", **kwargs) -> ToolResult:
        if not expression:
            return ToolResult(output="", success=False, error="No expression provided")

        # Input sanitization — reject anything that looks like code injection
        if _UNSAFE_RE.search(expression):
            return ToolResult(
                output="",
                success=False,
                error="Expression contains disallowed patterns. Use pure math only.",
            )

        try:
            from sympy.parsing.sympy_parser import (
                parse_expr,
                standard_transformations,
                implicit_multiplication_application,
            )

            transformations = standard_transformations + (implicit_multiplication_application,)
            result = parse_expr(expression, local_dict={}, transformations=transformations)
            evaluated = result.evalf()

            # Format nicely
            output = f"{expression} = {evaluated}"

            # If it's a real integer result, show without decimals
            if evaluated.is_real and evaluated == int(evaluated):
                output = f"{expression} = {int(evaluated)}"

            return ToolResult(output=output, success=True)

        except Exception as e:
            return ToolResult(
                output="",
                success=False,
                error=f"Math error: {e}. Check expression syntax.",
            )
