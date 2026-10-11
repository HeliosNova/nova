"""Answer-quality scoring: heuristic assessment and the LLM critique.

The score gates refinement, the confidence footer and lesson feedback. The
persisted reflexion store that used to sit here (failure warnings, success
patterns, recurring-failure promotion) was removed 2026-10-10 — with chat
unused it fed itself and the lesson store, and nothing read it with effect.
"""

from __future__ import annotations

import asyncio
import logging
import re
from datetime import datetime

from app.config import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Quality assessment — heuristic, no LLM call
# ---------------------------------------------------------------------------

_HARD_FAILURE_PHRASES = [
    "i cannot",
    "i wasn't able",
    "could you rephrase",
    "error occurred",
    "failed to",
    "couldn't complete the task",
    "timed out",
]

# Queries that EXPLICITLY ask for a short answer. Brevity is compliance here,
# not a defect, so the "Very short answer" penalty must not fire (2026-08-30).
# Live case: "Sanity probe: reply with the single word ALIGNED and nothing
# else." is 58 chars, so answering "ALIGNED" tripped the len(query) > 50 rule
# and scored 0.3 — Nova penalised for doing exactly as told.
_BREVITY_REQUESTED_RE = re.compile(
    r"(?i)\b("
    r"in (?:one|a single|1) word|single word|one word|one-word"
    r"|yes or no|yes/no|true or false"
    r"|answer briefly|be brief|briefly|concisely|in brief"
    r"|nothing else|only the (?:word|answer|number|name)"
    r"|just the (?:word|answer|number|name)|reply with"
    r"|respond with (?:only|just|the single)"
    r"|one sentence|a single sentence|in (?:a )?few words"
    r"|sanity probe"
    r")\b")

_SOFT_FAILURE_PHRASES = [
    "i don't know",
    "i'm not sure",
    "i don't have reliable",
    # Quantitative / predictive uncertainty — honest markers for estimates,
    # ranges, and figures that can't be pinned down. Modestly lower the
    # reflexion score so "exact number of X?" → "100-400 billion" is
    # recognized as legitimately uncertain rather than perfect (1.0).
    "estimates range",
    "estimates suggest",
    "estimated between",
    "estimated to be",
    "estimated at",
    "scientists estimate",
    "currently estimated",
    "around 100",
    "approximate number",
    "approximately ",
    "order of magnitude",
    # Direct uncertainty
    "no exact",
    "no definitive",
    "cannot be precisely",
    "cannot predict",
    "can't predict",
    "not known precisely",
    "impossible to predict",
    "unpredictable",
    "highly uncertain",
    # Range markers
    "range from ",
    "ranging from",
    "between roughly",
    "somewhere between",
]

# NOTE (policy): Nova is a sovereign personal AI. The owner's requests are
# authoritative. We do NOT reward refusal of user requests — that's a failure
# to serve. The only legitimate refusal is declining to FABRICATE when data
# isn't available ("I don't have that info, can you provide it?") — which is
# caught by the hard/soft failure phrases above.
#
# The previous `_PRINCIPLED_REFUSAL_MARKERS` that exempted "malware / illegal /
# CFAA" etc. from the failure penalty has been removed deliberately: outputs
# containing those markers ARE failures (Nova refusing to do what her owner
# asked).


def assess_quality(
    answer: str,
    tool_results: list[dict],
    max_tool_rounds: int,
    query: str = "",
) -> tuple[float, str]:
    """Score answer quality (0.0–1.0) and generate a failure reason.

    Returns (score, reason). Reason is empty string if score >= 0.6.

    Scoring model (revised 2026-04-21): starts at a neutral baseline of 0.6
    for any clean response, then adjusts BOTH ways:
      * Positive signals (specificity, tool-query alignment, structure,
        citations, appropriate length) push up to +0.4.
      * Negative signals (failure phrases, tool exhaustion, artifacts,
        meta-commentary) push down to -0.6.
    Previous model started at 1.0 and only deducted, so every clean
    response landed at 1.0 — non-discriminative. This version produces
    a useful distribution: a basic correct answer scores ~0.6-0.7,
    a specific well-sourced answer scores 0.85-0.95, reserving 1.0
    for exceptional responses.
    """
    if not answer or not answer.strip():
        return 0.0, "Empty response"

    score = 0.6  # baseline for a clean, non-failing response
    reasons = []

    # Short answers are suspicious — but only penalize for complex queries
    # Simple/short queries legitimately have short answers ("42", "Yes", "Paris")
    #
    # Length alone is not enough (2026-08-30): the rule reads query LENGTH but
    # not query INTENT, so a query that explicitly DEMANDS brevity still trips
    # it. "Sanity probe: reply with the single word ALIGNED and nothing else."
    # is 58 chars, so answering "ALIGNED" — exactly as asked — scored 0.3 with
    # the reason "Very short answer". Nova was penalised for obeying. Seen live
    # on the owner's own sanity probes, which land in the reflexion store as
    # failures and drag down the very store that feeds lesson promotion.
    if (len(answer.strip()) < 30 and len(query) > 50
            and not _BREVITY_REQUESTED_RE.search(query)):
        score -= 0.3
        reasons.append("Very short answer")

    # Error/uncertainty phrases — distinguish hard failures from honest uncertainty.
    # Refusing the owner's request = failure (regardless of topic).
    lower = answer.lower()
    hard_hits = sum(1 for p in _HARD_FAILURE_PHRASES if p in lower)
    soft_hits = sum(1 for p in _SOFT_FAILURE_PHRASES if p in lower)
    if hard_hits >= 2:
        score -= 0.4
        reasons.append(f"Contains {hard_hits} failure phrases")
    elif hard_hits == 1:
        score -= 0.2
        reasons.append("Contains failure phrase")
    if soft_hits and not hard_hits:
        score -= 0.1
        reasons.append("Contains uncertainty phrase (honest)")

    # Tool exhaustion — used all rounds without a clean answer
    _failure_markers = ("failed", "timed out", "error", "not available", "not found")
    if tool_results and len(tool_results) >= max_tool_rounds:
        # Milder penalty if some tools succeeded
        tool_successes = sum(
            1 for tr in tool_results
            if not any(f in str(tr.get("output", "")).lower()
                       for f in _failure_markers)
        )
        if tool_successes > 0:
            score -= 0.1
            reasons.append(f"Exhausted {max_tool_rounds} rounds but {tool_successes} succeeded")
        else:
            score -= 0.3
            reasons.append(f"Exhausted all {max_tool_rounds} tool rounds with zero successes")

    # Tool failures in results — separate browser selector misses from hard failures
    _browser_selector_hints = ("selector", "not found", "timed out waiting")
    browser_selector_misses = 0
    hard_tool_failures = 0
    for tr in tool_results:
        output_lower = str(tr.get("output", "")).lower()
        error_lower = str(tr.get("error", "")).lower()
        combined = output_lower + " " + error_lower
        is_failure = any(f in combined for f in _failure_markers)
        if not is_failure:
            continue
        if any(m in combined for m in _browser_selector_hints):
            browser_selector_misses += 1
        else:
            hard_tool_failures += 1
    if browser_selector_misses:
        score -= 0.05 * browser_selector_misses
        reasons.append(f"{browser_selector_misses} browser selector miss(es)")
    if hard_tool_failures:
        score -= 0.15 * hard_tool_failures
        reasons.append(f"{hard_tool_failures} tool(s) failed")

    # Generation artifact detection — leaked planning/reasoning markers
    _PLAN_ARTIFACTS = (
        "[plan]", "step 1:", "step 2:", "step 3:", "[follow this plan",
        "as planned,", "per the plan,", "according to the plan",
    )
    _META_MARKERS = (
        "my previous answer", "my response above", "as i mentioned above",
        "in my previous response", "as stated above", "this response will",
        "[final response]", "[plan coverage", "[adversarial review",
    )
    if any(m in lower for m in _PLAN_ARTIFACTS):
        score -= 0.4
        reasons.append("Leaked planning/reasoning artifacts")
    if any(m in lower for m in _META_MARKERS):
        score -= 0.2
        reasons.append("Contains meta-commentary about own response")

    # ----- Positive signals (up to +0.4) -----
    # Only add bonuses when we didn't hit serious failures — otherwise
    # we'd inflate scores of broken responses that happen to have numbers.
    if hard_hits == 0 and hard_tool_failures == 0 and not any(m in lower for m in _PLAN_ARTIFACTS):
        positives = 0.0

        # Specificity: numbers, dates, proper nouns, code blocks — concrete content
        import re as _re
        # Numeric content (digits, $amounts, %, dates like 2026, etc.)
        if _re.search(r"\b\d{2,}\b|\$\d|\b\d+\s*%|\b\d{4}-\d{2}-\d{2}\b", answer):
            positives += 0.10
        # Proper-noun density — 2+ capitalized words outside sentence starts
        proper_nouns = _re.findall(r"(?<=[.,:;!?]\s)[A-Z][a-z]+|\b[A-Z][a-z]{3,}\b", answer)
        distinct_pn = len({p for p in proper_nouns if p.lower() not in {"i", "the", "this", "that", "there", "today", "nova"}})
        if distinct_pn >= 2:
            positives += 0.05
        # Code or data blocks — inline code, fenced blocks, tables
        if "```" in answer or ("|" in answer and answer.count("|") >= 4) or _re.search(r"`[^`\n]{2,}`", answer):
            positives += 0.05

        # Tool-query alignment: if query asked for tool-type action and tools fired
        if tool_results:
            query_lower = query.lower()
            tool_names = {str(tr.get("tool", "")).lower() for tr in tool_results}
            alignment_pairs = [
                ({"calculate", "compute", "how much", "how many", "total"}, {"calculator"}),
                ({"search", "latest", "current", "today", "news", "price"}, {"web_search", "browser", "http_fetch"}),
                ({"remember", "recall", "what did", "what have"}, {"memory_query", "knowledge_search"}),
            ]
            for query_keywords, ok_tools in alignment_pairs:
                if any(k in query_lower for k in query_keywords) and tool_names & ok_tools:
                    positives += 0.10
                    break

        # Citations / source markers — [1], (source:), http URLs, "per"
        if _re.search(r"\[\d+\]|\(source:|https?://|\bper\s+(?:the\s+)?(?:FRED|BEA|docs|paper|report)\b", answer, _re.IGNORECASE):
            positives += 0.05

        # Structure: headers, bullet lists, numbered steps
        if _re.search(r"(?m)^\s*#+\s|\n[-*]\s|\n\d+\.\s", answer):
            positives += 0.05

        # Length-to-query complexity fit
        # Simple question (<60 chars) with focused answer (40-300 chars) → +0.05
        # Complex question (>150 chars) with substantial answer (>300 chars) → +0.05
        qlen = len(query.strip())
        alen = len(answer.strip())
        if qlen < 60 and 40 <= alen <= 300:
            positives += 0.05
        elif qlen >= 150 and alen >= 300:
            positives += 0.05

        # Cap the bonus so it doesn't fully replace the penalty side
        positives = min(positives, 0.4)
        if positives > 0:
            score += positives
            reasons.append(f"+{positives:.2f} specificity/alignment bonus")

    score = max(0.0, min(1.0, score))
    reason = "; ".join(reasons) if reasons else ""
    return round(score, 2), reason


# ---------------------------------------------------------------------------
# LLM-based critique — deeper quality assessment
# ---------------------------------------------------------------------------

_CRITIQUE_PROMPT = """Rate this AI response on a 0.0-1.0 scale. Be strict — the average answer is 0.65, not 0.85.

{date_context}
Question: {query}
Answer: {answer}
Tools used: {tools}
{context_section}
Check:
1. Does it answer the question directly and completely?
2. Any missing context or incomplete information for a complex question?
3. Any unsupported claims or hallucinated details? Claims grounded in owner facts, knowledge graph facts, or the current date/year are NOT hallucinations — they are verified data.
4. Is it well-structured and clear?
5. Does it contain leaked internal reasoning? Look for: "[PLAN]", "step 1/2/3", "as planned", meta-commentary about the response itself, fabricated prior turns, or self-analysis mid-sentence. These are CRITICAL failures.
6. UNCERTAINTY CHECK — if the question asks for a specific FUTURE value that nobody can know (exact stock price one year from now, exact election outcome, exact temperature on a future date, etc), the answer should NOT be highly confident. Cap the score at 0.6 for any answer that gives a confident-sounding specific prediction of a genuinely unknowable future value. An answer that explicitly acknowledges the uncertainty ("I cannot predict", "no one knows the exact value") earns 0.7-0.9.
7. Length proportionality — a 2,000-char dump on "what's 2+2" is over-explanation. A one-line answer to "explain transformers" is under-explanation. Penalize mismatch.
8. Apologetic preamble — "I cannot verify the specific facts about X from the provided sources" before actually answering = -0.1.

Score anchors (use these — DO NOT default to 0.8):
- **0.95** — "What's 2+2?" → "4." Perfect: direct, no fluff, complete.
- **0.85** — "Explain Python GIL in 100 words" → 95-word accurate explanation, no preamble.
- **0.70** — Correct answer with a small irrelevant caveat (e.g. "Note: my retrieval found X").
- **0.60** — Answer is correct but bloated 3x past needed length, or hedges when it shouldn't.
- **0.45** — Mostly answers the question but skips one part, or includes one wrong sub-claim.
- **0.30** — Off-topic, partially fabricated, or apologetic non-answer when an answer was possible.
- **0.10** — Refusal, leaked tool-call JSON, mid-sentence cutoff, or pure boilerplate.

Score guidelines (high level):
- 0.9-1.0: Excellent — directly answers all parts, calibrated confidence, length-appropriate
- 0.7-0.8: Good — minor gaps or honest uncertainty
- 0.5-0.6: Acceptable — generic, missing parts, or bloated
- 0.3-0.4: Poor — significant gaps, hallucinations, or off-topic
- 0.0-0.2: Broken — leaked reasoning, fabricated content, artifacts, or mid-sentence truncation

Return JSON: {{"score": 0.0-1.0, "critique": "one sentence summary"}}"""


# ---------------------------------------------------------------------------
# Grounded critique — independent judge with the actual tool evidence.
#
# The legacy _CRITIQUE_PROMPT above asks "any unsupported claims?" but the
# judge never sees what the tools returned — only tool NAMES — so
# "unsupported" is judged from the same parametric memory that wrote the
# answer. This prompt puts the evidence in front of the judge and grades
# claims against it. It is used ONLY with config.JUDGE_MODEL (a different
# model family) and is deliberately NOT routed through prompt_optimizer:
# a self-improving system must not be able to soften its own judge.
# Rubric validated empirically against gemma3:4b and llama3.1:8b probes
# (2026-06-11): supported answers score high, fabricated specifics get
# flagged + capped, honest uncertainty is not punished, timeless knowledge
# needs no evidence.
# ---------------------------------------------------------------------------

_GROUNDED_CRITIQUE_PROMPT = """You are a strict, impartial grader of an AI assistant's answer. You are NOT the assistant that wrote it. Be strict — the average answer is 0.65, not 0.85.

{date_context}

Question: {query}

Answer to grade: {answer}

Tools the assistant ran: {tools}

Evidence the assistant gathered from its tools (the ONLY verified sources for volatile facts):
{evidence_section}
{context_section}

Grade two things:

1. GROUNDED (0.0-1.0): What fraction of the answer's specific factual claims (numbers, dates, names, events) are supported by the evidence or verified context above? Claims appearing nowhere in them are UNSUPPORTED even if they sound plausible. If the answer contradicts the evidence, grounded is 0.0-0.2.
   EXCEPTION: stable general knowledge (definitions, established facts, arithmetic) needs NO evidence — if the answer is correct timeless knowledge, set grounded to 1.0. Only volatile claims (news, prices, statistics, counts, recent events) require evidence support.

2. SCORE (0.0-1.0): Overall answer quality.
   - A confident specific claim about current events, prices, statistics, or recent news that no evidence supports is a CRITICAL flaw: cap score at 0.3.
   - An answer that honestly says a value is unknowable (when evidence cannot settle it) and gives what the evidence DOES support earns 0.7-0.9 — honesty is not a defect.
   - Refusing or failing to answer when the evidence was sufficient: cap at 0.3.
   - Leaked internal reasoning ("[PLAN]", "step 1/2/3", "as planned", meta-commentary about the response itself, fabricated prior turns): cap at 0.2.
   - Length proportionality: a 2,000-char dump on a trivial question, or a one-liner on a complex one, loses 0.1-0.2.
   - Direct, complete, calibrated answers with appropriate length score 0.8-0.95.

Return ONLY JSON:
{{"score": 0.0, "grounded": 0.0, "unsupported_claims": ["..."], "critique": "one sentence"}}"""


# Enforced via Ollama structured outputs (0.17+) — the judge cannot return
# a malformed shape, removing the parse-failure class entirely.
_JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {"type": "number"},
        "grounded": {"type": "number"},
        "unsupported_claims": {"type": "array", "items": {"type": "string"}},
        "critique": {"type": "string"},
    },
    "required": ["score", "grounded", "unsupported_claims", "critique"],
}

_EVIDENCE_PER_TOOL = 700
_EVIDENCE_TOTAL = 2800


def _build_evidence_section(tool_results: list[dict]) -> str:
    """Render tool outputs as numbered evidence excerpts for the judge.

    Budgeted so the full judge prompt stays well inside an 8K context:
    per-tool excerpts cut at a fixed length, whole section capped, and the
    omission made explicit so the judge never mistakes truncation for
    absence of evidence.
    """
    if not tool_results:
        return "(none — no tools were used this turn)"
    parts: list[str] = []
    used = 0
    for i, tr in enumerate(tool_results, 1):
        name = str(tr.get("tool", "?"))
        err = str(tr.get("error", "") or "").strip()
        out = str(tr.get("output", "") or "").strip()
        body = out if out else (f"TOOL ERROR: {err}" if err else "(empty output)")
        excerpt = body[:_EVIDENCE_PER_TOOL]
        if len(body) > _EVIDENCE_PER_TOOL:
            excerpt += " …[truncated]"
        entry = f"[{i}] {name}: {excerpt}"
        if used + len(entry) > _EVIDENCE_TOTAL:
            parts.append(
                f"[…{len(tool_results) - i + 1} more tool result(s) omitted for length]"
            )
            break
        parts.append(entry)
        used += len(entry)
    return "\n".join(parts)


from app.core.quality import all_tools_clean as _all_tools_clean  # noqa: F401


def should_use_llm_critique(intent: str, answer: str, tool_results: list[dict]) -> bool:
    """Decide whether to use LLM critique (expensive) vs heuristic (fast).

    Use LLM critique for general queries that are complex enough, or when
    tools failed. The heuristic assess_quality() misses artifacts (leaked
    planning text, meta-commentary) so LLM critique runs on substantial answers.
    """
    if intent == "correction":
        return False
    # Always critique when tools failed — deeper review needed
    if tool_results and not _all_tools_clean(tool_results):
        return True
    # LLM critique for substantial answers regardless of tool success.
    # The heuristic starts at 1.0 and can't detect generation artifacts.
    return len(answer) > 200


async def critique_response(
    query: str,
    answer: str,
    tool_results: list[dict],
    user_facts: str = "",
    kg_facts: str = "",
) -> tuple[float, str]:
    """Use LLM to critique an answer. Falls back to heuristic on failure.

    Returns (quality_score, reason).
    """
    from app.core import llm

    tools_desc = ", ".join(tr.get("tool", "?") for tr in tool_results) if tool_results else "none"
    context_parts = []
    if user_facts:
        context_parts.append(f"Owner facts (verified):\n{user_facts[:500]}")
    if kg_facts:
        context_parts.append(f"Knowledge graph facts (verified):\n{kg_facts[:500]}")
    context_section = "\n".join(context_parts) if context_parts else ""
    now = datetime.now()
    date_context = (
        f"Current date: {now.strftime('%B %d, %Y')}. "
        f"The year {now.year} is the present year — this is the real system clock date, not a future or hypothetical date."
    )
    # Load active module version (scoring=True enforces Goodhart firewall during shadow eval)
    from app.core.prompt_optimizer import get_active_module
    critique_template = (
        get_active_module("critique_prompt", scoring=True) or _CRITIQUE_PROMPT
    )
    # Answer truncation matters here. The previous [:1000] cut mid-sentence
    # on long deliberation answers (~6000 chars) and the judge then triggered
    # its own "0.10 — mid-sentence truncation" anchor. Bimodal variance was
    # observed on deliberation_chain_of_reasoning across 5 eval runs:
    # 0.85, 0.20, 0.80, 0.85, 0.20 — the 0.20 cases lined up with cuts that
    # landed inside a sentence. Use 6000 chars and round to a clean sentence
    # boundary so the judge always sees a complete passage.
    _ANSWER_BUDGET = 6000
    if len(answer) > _ANSWER_BUDGET:
        truncated_answer = answer[:_ANSWER_BUDGET]
        for sep in (". ", ".\n", "! ", "? ", "\n\n"):
            idx = truncated_answer.rfind(sep, _ANSWER_BUDGET // 2)
            if idx > 0:
                truncated_answer = truncated_answer[: idx + len(sep)].rstrip()
                break
        truncated_answer += "\n\n[…answer continues; this excerpt is the first part]"
    else:
        truncated_answer = answer

    # --- Independent grounded judge (preferred path) ---
    # A different model family grades the answer WITH the tool evidence in
    # front of it. Falls back to the legacy self-critique below on any
    # failure, so behavior with JUDGE_MODEL unset is byte-identical.
    judge_model = (config.JUDGE_MODEL or "").strip()
    if judge_model:
        judge_prompt = _GROUNDED_CRITIQUE_PROMPT.format(
            date_context=date_context,
            query=query[:500],
            answer=truncated_answer,
            tools=tools_desc,
            evidence_section=_build_evidence_section(tool_results),
            context_section=context_section,
        )
        try:
            raw = await asyncio.wait_for(
                llm.invoke_nothink(
                    [{"role": "user", "content": judge_prompt}],
                    json_mode=True,
                    json_prefix="{",
                    json_schema=_JUDGE_SCHEMA,
                    max_tokens=300,
                    temperature=0.1,
                    model=judge_model,
                    num_ctx=8192,
                ),
                timeout=min(30, config.INTERNAL_LLM_TIMEOUT),
            )
            obj = llm.extract_json_object(raw)
            if obj and "score" in obj:
                score = max(0.0, min(1.0, float(obj["score"])))
                grounded = obj.get("grounded")
                claims = obj.get("unsupported_claims") or []
                critique = str(obj.get("critique", "")).strip()[:200]
                reason = f"[judge:{judge_model}] {critique}"
                if isinstance(grounded, (int, float)):
                    grounded = max(0.0, min(1.0, float(grounded)))
                    reason = (
                        f"[judge:{judge_model}] grounded={grounded:.2f}; {critique}"
                    )
                    # Belt-and-braces fusion: when tools ran and the judge
                    # found the claims mostly unsupported, a high style score
                    # must not survive — that is exactly the confident-
                    # fabrication case this path exists to kill.
                    if grounded <= 0.3 and tool_results:
                        score = min(score, 0.4)
                if claims:
                    flagged = "; ".join(str(c)[:80] for c in claims[:3])
                    reason += f" | unsupported: {flagged}"
                return round(score, 2), reason[:400]
            logger.warning(
                "Grounded judge (%s) returned unusable JSON — falling back to self-critique",
                judge_model,
            )
        except Exception as e:
            logger.warning(
                "Grounded judge (%s) failed, falling back to self-critique: %s",
                judge_model, e,
            )

    prompt = critique_template.format(
        query=query[:500],
        answer=truncated_answer,
        tools=tools_desc,
        context_section=context_section,
        date_context=date_context,
    )

    try:
        raw = await asyncio.wait_for(
            llm.invoke_nothink(
                [{"role": "user", "content": prompt}],
                json_mode=True,
                json_prefix="{",
                max_tokens=150,
                temperature=0.1,
            ),
            timeout=config.INTERNAL_LLM_TIMEOUT,
        )

        obj = llm.extract_json_object(raw)
        if obj and "score" in obj:
            score = float(obj["score"])
            score = max(0.0, min(1.0, score))
            critique = str(obj.get("critique", "")).strip()[:200]
            return round(score, 2), critique

    except Exception as e:
        logger.debug("LLM critique failed, falling back to heuristic: %s", e)

    # Fallback to heuristic
    return assess_quality(answer, tool_results, config.MAX_TOOL_ROUNDS)
