"""A-HMAD style role-specialized debate.

When a draft answer is high-stakes (medical / legal / financial / strong
factual claim), single-agent reasoning has a known ceiling. Adaptive
Heterogeneous Multi-Agent Debate (Springer s44443-025-00353-3, 2026) shows
that *role-specialized* critics + a judge beats more parallel workers of
the same kind.

We run three critics serially (parallel costs as much VRAM but gives no
quality bump on a 24GB single-GPU local setup), then a judge consolidates:

  * Logic Critic    — reasoning steps, math, internal consistency
  * Fact Verifier   — unverified claims, KG/web grounding gaps
  * Strategy Reviewer — was the right tool used, scope correct, edge cases
  * Judge           — synthesizes: keep / amend / replace / hedge

Gated by `ENABLE_DEBATE` (default False — opt-in). The gate function
`should_debate(query, intent, draft)` decides per request; the brain calls
`run_debate(...)` only when it returns True. Sub-agents already running
inside structural decomposition skip debate entirely (would compound
latency and create depth-2 spirals).

Public surface:
    should_debate(query, intent, draft)         -> bool
    run_debate(query, draft, *, evidence, ...)  -> DebateResult
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from app.config import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Trigger heuristics
# ---------------------------------------------------------------------------

_HIGH_STAKES_DOMAINS = re.compile(
    r"\b(?:"
    r"medical|medication|dosage|diagnos\w*|symptom|treatment|prescription|"
    r"legal|lawsuit|sue|tax(?:es|able)?|contract law|jurisdiction|"
    r"financial advice|invest(?:ment)?|portfolio|retirement|annuity|"
    r"safety|hazard|toxic|lethal|overdose|side effects?|"
    r"security vulnerab|exploit|cve-?\d|0day|zero[- ]day"
    r")\b",
    re.IGNORECASE,
)

_STRONG_CLAIM_RE = re.compile(
    r"\b(?:"
    r"definitely|certainly|always|never|impossible|guaranteed|"
    r"proven|prove[ds]?|undisputed|fact is|the truth is"
    r")\b",
    re.IGNORECASE,
)

_NUMERIC_CLAIM_RE = re.compile(
    r"\b\d+(?:[\.,]\d+)?\s*(?:%|percent|dollars?|usd|years?|kg|mg|mph|km|hours?|days?)\b",
    re.IGNORECASE,
)


def is_enabled() -> bool:
    return bool(getattr(config, "ENABLE_DEBATE", False))


def should_debate(query: str, intent: str, draft: str) -> bool:
    """True when the request looks high-stakes enough for a debate.

    Cheap gate — pure regex/length checks. The actual debate is expensive
    so the gate must be conservative.
    """
    if not is_enabled():
        return False
    if intent != "general":
        return False
    if not query or not draft:
        return False
    # Don't debate tiny exchanges
    if len(query) < 30 or len(draft) < 200:
        return False
    # Don't debate when the model already hedged extensively
    hedges = sum(1 for marker in ("not sure", "may", "might", "perhaps", "i think") if marker in draft.lower())
    if hedges >= 3:
        return False
    if _HIGH_STAKES_DOMAINS.search(query) or _HIGH_STAKES_DOMAINS.search(draft):
        return True
    if _STRONG_CLAIM_RE.search(draft) and _NUMERIC_CLAIM_RE.search(draft):
        return True
    return False


# ---------------------------------------------------------------------------
# Critic + Judge prompts
# ---------------------------------------------------------------------------

_LOGIC_CRITIC_PROMPT = """You are a Logic Critic reviewing an answer for reasoning errors.

Find any of:
- arithmetic/math mistakes
- broken inferences (premise -> conclusion gaps)
- internal contradictions
- circular reasoning
- conflated concepts

QUERY:
{query}

ANSWER UNDER REVIEW:
{draft}

Respond with JSON:
{{"verdict": "ok"|"issues", "issues": ["...", "..."], "severity": "low"|"medium"|"high"}}
At most 3 issues. Be specific (quote the broken step) — empty list if no issues."""


_FACT_VERIFIER_PROMPT = """You are a Fact Verifier reviewing an answer for unsupported claims.

Find any of:
- specific numbers / dates / names not backed by the cited evidence
- "X is Y" claims that need a source and don't have one
- claims that contradict the provided evidence

QUERY:
{query}

EVIDENCE PROVIDED TO MODEL (KG facts, retrieved chunks, tool outputs):
{evidence}

ANSWER UNDER REVIEW:
{draft}

Respond with JSON:
{{"verdict": "ok"|"issues", "issues": ["...", "..."], "severity": "low"|"medium"|"high"}}
At most 3 issues. Quote the unsupported claim. Empty list if grounding is solid."""


_STRATEGY_REVIEWER_PROMPT = """You are a Strategy Reviewer checking that the answer addresses the right question.

Find any of:
- scope mismatch (answers a different question)
- missed obvious edge case
- wrong tool would have been better (search for fresh data, calculator for math)
- safety / harm concern not flagged

QUERY:
{query}

ANSWER UNDER REVIEW:
{draft}

Respond with JSON:
{{"verdict": "ok"|"issues", "issues": ["...", "..."], "severity": "low"|"medium"|"high"}}
At most 3 issues. Empty list if scope and approach are correct."""


_JUDGE_PROMPT = """You are the Judge consolidating three critics on a draft answer.

Each critic returned {{verdict, issues[], severity}}. Decide one of:
- KEEP: critics had no material issues; ship the draft as-is
- AMEND: small fixes; the rewrite preserves >70% of the draft
- REPLACE: substantive errors; provide a fresh answer from scratch
- HEDGE: the underlying question is uncertain; add an honest hedge to the draft

QUERY:
{query}

DRAFT:
{draft}

EVIDENCE:
{evidence}

CRITIC REPORTS:
- Logic Critic: {logic}
- Fact Verifier: {fact}
- Strategy Reviewer: {strategy}

Respond with JSON:
{{"action": "keep"|"amend"|"replace"|"hedge",
  "final_answer": "...",
  "rationale": "one sentence why"}}
If action == keep, set final_answer to the original draft verbatim."""


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class CriticReport:
    role: str
    verdict: str
    issues: list[str] = field(default_factory=list)
    severity: str = "low"
    raw: str = ""


@dataclass
class DebateResult:
    action: str            # 'keep' | 'amend' | 'replace' | 'hedge'
    final_answer: str
    rationale: str
    critics: list[CriticReport] = field(default_factory=list)
    debate_ms: int = 0


# ---------------------------------------------------------------------------
# Critic + Judge LLM calls
# ---------------------------------------------------------------------------

async def _run_critic(role: str, prompt: str, max_tokens: int = 350) -> CriticReport:
    """Run a single critic call. Returns CriticReport — never raises."""
    from app.core import llm

    try:
        raw = await asyncio.wait_for(
            llm.invoke_nothink(
                [{"role": "user", "content": prompt}],
                json_mode=True,
                json_prefix="{",
                max_tokens=max_tokens,
                temperature=0.2,
            ),
            timeout=max(float(config.INTERNAL_LLM_TIMEOUT), 60.0),
        )
    except Exception as e:
        logger.warning("[debate/%s] critic LLM failed: %s", role, e)
        return CriticReport(role=role, verdict="ok", raw="")

    if not raw:
        return CriticReport(role=role, verdict="ok", raw="")
    try:
        obj = json.loads(raw) if isinstance(raw, str) else raw
    except Exception:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        try:
            obj = json.loads(m.group(0)) if m else {}
        except Exception:
            obj = {}
    if not isinstance(obj, dict):
        return CriticReport(role=role, verdict="ok", raw=raw[:500])

    verdict = str(obj.get("verdict", "ok")).strip().lower()
    if verdict not in ("ok", "issues"):
        verdict = "ok"
    issues_raw = obj.get("issues") or []
    issues: list[str] = []
    if isinstance(issues_raw, list):
        for item in issues_raw[:3]:
            s = str(item).strip()
            if s:
                issues.append(s[:300])
    severity = str(obj.get("severity", "low")).strip().lower()
    if severity not in ("low", "medium", "high"):
        severity = "low"
    return CriticReport(
        role=role,
        verdict=verdict if issues else "ok",  # empty issues -> ok regardless
        issues=issues,
        severity=severity,
        raw=raw[:500],
    )


async def _run_judge(
    query: str,
    draft: str,
    evidence: str,
    critics: list[CriticReport],
) -> tuple[str, str, str]:
    """Run the judge LLM. Returns (action, final_answer, rationale)."""
    from app.core import llm

    def _summarize(c: CriticReport) -> str:
        if c.verdict == "ok" or not c.issues:
            return "ok (no issues)"
        return f"{c.severity}: " + " | ".join(c.issues)

    prompt = _JUDGE_PROMPT.format(
        query=query[:2000],
        draft=draft[:4000],
        evidence=(evidence or "[no evidence captured]")[:3000],
        logic=_summarize(critics[0]) if len(critics) > 0 else "missing",
        fact=_summarize(critics[1]) if len(critics) > 1 else "missing",
        strategy=_summarize(critics[2]) if len(critics) > 2 else "missing",
    )
    try:
        raw = await asyncio.wait_for(
            llm.invoke_nothink(
                [{"role": "user", "content": prompt}],
                json_mode=True,
                json_prefix="{",
                max_tokens=2200,
                temperature=0.3,
            ),
            timeout=max(float(config.INTERNAL_LLM_TIMEOUT) * 2, 120.0),
        )
    except Exception as e:
        logger.warning("[debate/judge] LLM failed: %s — keeping draft", e)
        return "keep", draft, "judge LLM failed"

    if not raw:
        return "keep", draft, "judge returned empty"
    try:
        obj = json.loads(raw) if isinstance(raw, str) else raw
    except Exception:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        try:
            obj = json.loads(m.group(0)) if m else {}
        except Exception:
            obj = {}
    if not isinstance(obj, dict):
        return "keep", draft, "judge JSON malformed"

    action = str(obj.get("action", "keep")).strip().lower()
    if action not in ("keep", "amend", "replace", "hedge"):
        action = "keep"
    final = obj.get("final_answer")
    if not isinstance(final, str) or not final.strip():
        final = draft
    rationale = str(obj.get("rationale", ""))[:400]
    return action, final.strip(), rationale


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

async def run_debate(
    query: str,
    draft: str,
    *,
    evidence: str = "",
    timeout_total: float | None = None,
    decomposed: bool = False,
) -> DebateResult:
    """Run logic / fact / strategy critics, then judge. Always returns a
    DebateResult — exceptions degrade to action='keep' so the calling brain
    keeps the original draft.

    If `decomposed=True` the draft came from multi-agent merge synthesis,
    which means the constituent sub-agents already validated reasoning and
    tool-strategy via their own reflexion passes. In that case we run ONLY
    the Fact Verifier (the one critic that needs fresh evidence against the
    merged claims) — cutting debate latency by ~2/3 without sacrificing the
    grounding check on the synthesis step itself.
    """
    import time

    if not is_enabled():
        return DebateResult(
            action="keep", final_answer=draft, rationale="debate disabled",
        )
    if not query or not draft:
        return DebateResult(
            action="keep", final_answer=draft, rationale="empty input",
        )

    start = time.monotonic()
    deadline = start + (timeout_total or 240.0)

    critic_set: tuple[tuple[str, str], ...] = (
        (("fact", _FACT_VERIFIER_PROMPT),)
        if decomposed
        else (
            ("logic", _LOGIC_CRITIC_PROMPT),
            ("fact", _FACT_VERIFIER_PROMPT),
            ("strategy", _STRATEGY_REVIEWER_PROMPT),
        )
    )

    critics: list[CriticReport] = []
    for role, template in critic_set:
        if time.monotonic() >= deadline:
            critics.append(CriticReport(role=role, verdict="ok", raw="(timed out)"))
            continue
        prompt = template.format(
            query=query[:2000],
            draft=draft[:4000],
            evidence=(evidence or "[none]")[:3000],
        )
        report = await _run_critic(role, prompt)
        critics.append(report)

    if all(c.verdict == "ok" for c in critics):
        elapsed = int((time.monotonic() - start) * 1000)
        return DebateResult(
            action="keep",
            final_answer=draft,
            rationale="all critics returned ok",
            critics=critics,
            debate_ms=elapsed,
        )

    if time.monotonic() >= deadline:
        elapsed = int((time.monotonic() - start) * 1000)
        return DebateResult(
            action="keep",
            final_answer=draft,
            rationale="judge skipped (timeout)",
            critics=critics,
            debate_ms=elapsed,
        )

    action, final, rationale = await _run_judge(query, draft, evidence, critics)
    elapsed = int((time.monotonic() - start) * 1000)
    logger.info(
        "[debate] action=%s critics=%s elapsed=%dms",
        action,
        ",".join(f"{c.role}={c.verdict}" for c in critics),
        elapsed,
    )
    return DebateResult(
        action=action,
        final_answer=final or draft,
        rationale=rationale,
        critics=critics,
        debate_ms=elapsed,
    )
