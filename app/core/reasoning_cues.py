"""Cheap regex cues that decide how much reasoning a query deserves.

These lived in app/core/agent_loop.py until 2026-10-10, when the agent loop
(plan/act/critique/synthesize, workspaces, delegation) was cut. brain.py had
imported them from there to gate three things on the generic think() path
that every query monitor and the eval harness run through:

- ``_is_math_logic_word_problem`` — the [MATH TOOL ROUTING] system block that
  forces the calculator for implicit arithmetic (model program item 2,
  2026-08-24);
- ``_is_hard_reasoning_query`` — the selective-thinking auto-gate (reasoning
  eval 0.714 -> 1.0 on 2026-08-24/25) and the best-of-N resampling gate;
- ``_is_deliberation_query`` — the lesson cap for design/walkthrough queries.

No LLM call, no I/O. Keep them that way.
"""
from __future__ import annotations

import re

_HARD_REASONING_RE = re.compile(
    r"\b(compare|analyze|analyse|explain|derive|prove|reason|"
    r"trade.?offs?|pros?\s+and\s+cons?|step.by.step|"
    r"why\s+(does|is|are|would|should)|how\s+(does|would|should)|"
    r"what.s\s+the\s+best|which\s+is\s+better|"
    r"design|architect|strategy|implications?)\b",
    re.IGNORECASE,
)

# Math / logic WORD PROBLEMS — queries that need careful System-2 reasoning but
# don't announce themselves with "analyze/derive". A live probe (2026-06-13)
# showed the 9B failing the classic bat-and-ball trap (it gave the intuitive
# wrong ball=$0.10) because nothing flagged it for extended thinking. These
# carry relational/constraint language; paired with >=2 numbers they signal a
# constraint-satisfaction problem where pattern-matching loses.
_MATH_LOGIC_RE = re.compile(
    r"\b(more than|less than|fewer than|greater than|twice|thrice|"
    r"three times|half of|double|triple|older than|younger than|"
    r"faster than|slower than|taller than|heavier than|"
    r"how many|how much|how long|how old|how far|how fast|"
    r"ratio|combined|altogether|apiece|"
    r"costs? .{0,25}\bmore\b|in total|doubles every)\b",
    re.IGNORECASE,
)
_NUM_RE = re.compile(r"\d+(?:[.,]\d+)?")


def _is_math_logic_word_problem(query: str) -> bool:
    """Constraint/relational language + at least two numbers => a word problem
    that benefits from step-by-step reasoning rather than pattern-matching."""
    if not query or len(query) < 20:
        return False
    return bool(_MATH_LOGIC_RE.search(query)) and len(_NUM_RE.findall(query)) >= 2


def _is_hard_reasoning_query(query: str) -> bool:
    """Heuristic: does this query benefit from extended (System-2) reasoning?

    Hard queries: comparative reasoning, derivation, design, multi-factor
    analysis, AND math/logic word problems (constraint satisfaction). Cheap
    regex — no LLM call.
    """
    if not query or len(query) < 20:
        return False
    return bool(_HARD_REASONING_RE.search(query)) or _is_math_logic_word_problem(query)


# Observed bimodal pattern across 6 eval runs of `deliberation_chain_of_reasoning`
# (0.85, 0.20, 0.80, 0.85, 0.20, 0.20): the 0.20 cases were answered at summary
# depth. brain caps retrieved lessons to the two highest-confidence ones for
# these queries so five short lessons cannot teach the model that depth.
_DELIBERATION_QUERY_RE = re.compile(
    r"(?i)\b("
    r"step.by.step|walk\s+me\s+through|"
    r"design\s+(?:a|an|the)\s+\w+|architect\s+(?:a|an|the)?\s*\w+|"
    r"how\s+would\s+you\s+(?:design|architect|build|approach|tackle|solve|implement)|"
    r"plan\s+for\s+a\s+\w+|trade.?offs?\s+between|"
    r"derive(?:\s+the)?|prove\s+that|show\s+me\s+how|"
    r"reason\s+through|chain.of.thought"
    r")\b"
)


def _is_deliberation_query(query: str) -> bool:
    """True iff the query asks for a structured multi-section walkthrough."""
    return bool(query and _DELIBERATION_QUERY_RE.search(query))
