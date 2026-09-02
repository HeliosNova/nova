"""Deterministic verifiers for GRPO online-rollout scoring.

When GRPOTrainer samples N fresh rollouts per prompt at training time, each
rollout needs a reward. For Nova's RLVR signals these are deterministic:
  - `math_correct`: re-run the calculator on the response, compare answers
  - `json_valid`: try json.loads on the response
  - `code_passes_tests`: exec the response, check exit
  - `schema_match`: validate against a tool's args schema

This module exposes one entry point — `verify(signal_type, query, response, evidence)`
— that picks the right verifier and returns a reward in [0, 1]. Verifiers
are pure functions: no DB writes, no side effects, safe to call from a
training loop on every rollout.

For signal types that aren't replayable (e.g. `tool_correct` on web_search,
`claim_grounded` requiring evidence pool), the verifier returns None to
signal "use the precomputed reward from the dataset row" and the trainer
falls back to that.

Public surface:
    verify(signal_type, query, response, evidence=None) -> float | None
    is_replayable(signal_type) -> bool
"""

from __future__ import annotations

import ast
import json
import logging
import re
from typing import Callable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Signal-type → verifier dispatch
# ---------------------------------------------------------------------------

REPLAYABLE_TYPES = frozenset({
    "math_correct",
    "json_valid",
    "schema_match",
})

# Replayable in principle but expensive (`code_passes_tests` runs Python in a
# subprocess). Excluded from REPLAYABLE_TYPES by default to keep training
# loops fast; opt in via `replay_code=True` if the operator wants it.


def is_replayable(signal_type: str, *, allow_code: bool = False) -> bool:
    """True iff this signal type can be re-verified deterministically."""
    if signal_type == "code_passes_tests":
        return allow_code
    return signal_type in REPLAYABLE_TYPES


# ---------------------------------------------------------------------------
# math_correct
# ---------------------------------------------------------------------------

_NUMBER_RE = re.compile(
    r"-?\d+(?:[\.,]\d+)?(?:e[+\-]?\d+)?",
    re.IGNORECASE,
)
_EXPR_RE = re.compile(
    r"\b(\d+(?:[\.,]\d+)?\s*(?:[+\-*/^×÷]|\bplus\b|\bminus\b|\btimes\b|\bdivided by\b)\s*)+\d+(?:[\.,]\d+)?\b",
    re.IGNORECASE,
)


def _extract_numbers(text: str) -> list[float]:
    """Pull all numbers from text. Handles commas as thousand separators."""
    out = []
    for m in _NUMBER_RE.finditer(text or ""):
        s = m.group(0).replace(",", "")
        try:
            out.append(float(s))
        except ValueError:
            pass
    return out


def _safe_eval_arith(expr: str) -> float | None:
    """Evaluate an arithmetic-only expression. Returns None on any failure.

    Uses ast.parse + a constant-folder so we never exec(). Supports:
      + - * / ** unary minus, parenthesized sub-expressions, integer/float literals.
    """
    expr = expr.replace("×", "*").replace("÷", "/").replace("^", "**")
    expr = re.sub(r"\b(plus)\b", "+", expr, flags=re.IGNORECASE)
    expr = re.sub(r"\b(minus)\b", "-", expr, flags=re.IGNORECASE)
    expr = re.sub(r"\b(times)\b", "*", expr, flags=re.IGNORECASE)
    expr = re.sub(r"\bdivided\s+by\b", "/", expr, flags=re.IGNORECASE)
    expr = expr.replace(",", "")
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        return None

    def _walk(n):
        if isinstance(n, ast.Expression):
            return _walk(n.body)
        if isinstance(n, ast.Constant):
            if isinstance(n.value, (int, float)):
                return n.value
            return None
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub):
            v = _walk(n.operand)
            return -v if v is not None else None
        if isinstance(n, ast.BinOp):
            l = _walk(n.left)
            r = _walk(n.right)
            if l is None or r is None:
                return None
            if isinstance(n.op, ast.Add):
                return l + r
            if isinstance(n.op, ast.Sub):
                return l - r
            if isinstance(n.op, ast.Mult):
                return l * r
            if isinstance(n.op, ast.Div):
                return l / r if r != 0 else None
            if isinstance(n.op, ast.Pow):
                if abs(l) > 1e6 or abs(r) > 100:
                    return None  # avoid blow-up
                return l ** r
            return None
        return None

    try:
        return _walk(tree)
    except Exception:
        return None


def verify_math(query: str, response: str, *, tolerance: float = 0.01) -> float | None:
    """Reward for a math response. 1.0 if any number in response equals the
    evaluated query expression within tolerance, else 0.0. None if no
    arithmetic expression found in the query (signal is unverifiable here).
    """
    if not query or not response:
        return None
    m = _EXPR_RE.search(query)
    if not m:
        return None
    expected = _safe_eval_arith(m.group(0))
    if expected is None:
        return None
    nums = _extract_numbers(response)
    if not nums:
        return 0.0
    for n in nums:
        if abs(n - expected) <= max(tolerance, abs(expected) * tolerance):
            return 1.0
    return 0.0


# ---------------------------------------------------------------------------
# json_valid
# ---------------------------------------------------------------------------

def verify_json(query: str, response: str) -> float | None:
    """1.0 if the response (or its first {...} block) parses as JSON."""
    if not response:
        return 0.0
    s = response.strip()
    try:
        json.loads(s)
        return 1.0
    except Exception:
        pass
    m = re.search(r"\{.*\}", s, re.DOTALL)
    if not m:
        return 0.0
    try:
        json.loads(m.group(0))
        return 1.0
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# schema_match
# ---------------------------------------------------------------------------

def _balanced_json(text: str) -> dict | None:
    """Find the first balanced JSON object in `text` and parse it.

    Walks character by character tracking brace depth and string state so
    nested {...} (e.g. tool args) is handled correctly. Returns parsed dict
    or None.
    """
    if not text:
        return None
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        else:
            if c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[start : i + 1])
                        return obj if isinstance(obj, dict) else None
                    except Exception:
                        return None
    return None


def verify_schema(query: str, response: str, evidence: str | None = None) -> float | None:
    """Reward for tool-call schema match.

    Evidence (if provided) names the expected tool. Response should contain a
    tool-call JSON object {"tool": "<name>", "args": {...}}. We don't validate
    args here (no schema registry plugged in) — just that the shape is right
    and the tool name matches if evidence specifies one.
    """
    if not response:
        return 0.0
    obj = _balanced_json(response)
    if obj is None or "tool" not in obj or "args" not in obj:
        return 0.0
    if evidence and "tool=" in evidence:
        expected = evidence.split("tool=", 1)[1].split()[0].strip(",")
        if obj.get("tool") != expected:
            return 0.5  # right shape, wrong target
    return 1.0


# ---------------------------------------------------------------------------
# Top-level dispatch
# ---------------------------------------------------------------------------

_VERIFIERS: dict[str, Callable[..., float | None]] = {
    "math_correct": verify_math,
    "json_valid": verify_json,
    "schema_match": verify_schema,
}


def verify(
    signal_type: str,
    query: str,
    response: str,
    evidence: str | None = None,
) -> float | None:
    """Re-score a rollout deterministically. None if not replayable."""
    fn = _VERIFIERS.get(signal_type)
    if fn is None:
        return None
    try:
        if signal_type == "schema_match":
            return fn(query, response, evidence)
        return fn(query, response)
    except Exception as e:
        logger.warning("[grpo_verifier] %s failed: %s", signal_type, e)
        return None
