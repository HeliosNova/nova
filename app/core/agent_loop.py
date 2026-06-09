"""Agent deliberation loop: reason → plan → act → observe → critique → revise.

This is Nova's thinking engine for non-trivial queries. Unlike `brain.think()`
which runs a single LLM call (with optional tool loop), this module:

  1. Plans: decompose the query into steps before answering.
  2. Acts per-step: reason about each step's goal, decide tool or direct answer.
  3. Observes: capture each step's output in a scratchpad (working memory).
  4. Critiques: ask the model whether the step actually accomplished its goal.
  5. Revises or advances: if critique fails, try a different approach; otherwise
     move to the next step.
  6. Self-consistency (optional): for hard reasoning steps, sample N chains and
     take the majority/best answer.
  7. Synthesizes: build the final answer from the scratchpad once the plan is
     complete (or aborted).

The whole point is to give a small local model the ability to deliberate, retry,
and compose — rather than one-shot the answer with a stuffed prompt.

This module is standalone — `brain.think()` is unchanged. Callers can route to
either path. A `/api/agent` endpoint exposes this directly.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from app.core import llm
from app.core.llm import extract_json_object

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Search query simplification
# ---------------------------------------------------------------------------

# Patterns that wrap an entity in a descriptive phrase. We strip the wrapper
# to return just the entity, since search engines (Wikipedia especially) rank
# entity queries above descriptive ones. Conservative — only strips clear
# patterns, leaves ambiguous queries alone.
# Words that describe the dimension being asked about, not the entity.
# When stripping these from a query leaves a recognizable entity, do so.
_DIMENSION_WORDS = {
    "area", "areas", "population", "populations", "gdp", "capital", "capitals",
    "landmass", "land area", "size", "land", "total area", "total population",
    "square", "kilometers", "kilometres", "km", "km2", "km^2", "miles", "mi",
    "current", "latest", "recent", "today's", "todays", "actual", "real",
    "what", "year", "when", "did", "was", "is", "the", "a", "an",
    "value", "amount", "number", "of", "for", "in", "to", "by",
    "started", "start", "began", "begin", "happen", "occur", "occurred",
    "founded", "established", "called",
}

# Quick proper-noun detector: capitalized word that's not a stop word at start.
_PROPER_NOUN_RE = re.compile(r"\b[A-Z][a-zA-Z]{2,}(?:\s+[A-Z][a-zA-Z]+)*\b")


def _simplify_search_query(q: str) -> str:
    """Rewrite descriptive queries to entity-form. Returns simplified query.

    Strategy:
      1. If the query starts with a clear descriptive prefix (area of X / when
         did X / current X), strip the prefix to the entity.
      2. Else, if the query contains proper nouns AND dimension words,
         extract just the proper noun phrase(s).
      3. Otherwise, leave the query alone.
    """
    q = q.strip().rstrip("?.").strip()

    # Phase 1: descriptive prefix patterns
    prefix_patterns = [
        (re.compile(r"^(?:the\s+)?(?:area|population|gdp|capital|landmass|size|land)\s+of\s+(.+?)(?:\s+in\s+\w+)?(?:\s*\(.*?\))?$", re.IGNORECASE), r"\1"),
        (re.compile(r"^(?:current|latest|recent|today's?)\s+(.+)$", re.IGNORECASE), r"\1"),
        (re.compile(r"^what year (?:did|was)\s+(?:the\s+)?(.+?)\s+(?:start|begin|happen|occur|founded|established)\b.*$", re.IGNORECASE), r"\1"),
        (re.compile(r"^when\s+(?:did|was)\s+(?:the\s+)?(.+?)(?:\s+happen|\s+occur|\s+founded|\s+established)?(?:\?|$)", re.IGNORECASE), r"\1"),
    ]
    for pattern, replacement in prefix_patterns:
        new = pattern.sub(replacement, q).strip()
        if new and new.lower() != q.lower():
            logger.info("query simplified (prefix): %r -> %r", q[:80], new[:80])
            return new

    # Phase 2: proper-noun extraction. If query has 3+ words, contains
    # dimension words, AND has proper nouns, return just the proper nouns.
    words = q.split()
    if len(words) >= 3:
        proper_nouns = _PROPER_NOUN_RE.findall(q)
        if proper_nouns:
            # Check if at least one dimension word is present (so this is a
            # descriptive-about-entity query, not a pure proper-noun phrase).
            lower_words = {w.lower().rstrip(",.") for w in words}
            if lower_words & _DIMENSION_WORDS:
                simplified = " ".join(proper_nouns)
                if simplified.lower() != q.lower():
                    logger.info("query simplified (proper nouns): %r -> %r", q[:80], simplified[:80])
                    return simplified

    return q


# ---------------------------------------------------------------------------
# Tree-of-thought heuristic
# ---------------------------------------------------------------------------

_HARD_REASONING_RE = re.compile(
    r"\b(compare|analyze|analyse|explain|derive|prove|reason|"
    r"trade.?offs?|pros?\s+and\s+cons?|step.by.step|"
    r"why\s+(does|is|are|would|should)|how\s+(does|would|should)|"
    r"what.s\s+the\s+best|which\s+is\s+better|"
    r"design|architect|strategy|implications?)\b",
    re.IGNORECASE,
)


def _is_hard_reasoning_query(query: str) -> bool:
    """Heuristic: does this query benefit from tree-of-thought sampling?

    Hard queries: comparative reasoning, derivation, design, multi-factor
    analysis. Cheap regex — no LLM call.
    """
    if not query or len(query) < 20:
        return False
    return bool(_HARD_REASONING_RE.search(query))


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

STEP_PENDING = "pending"
STEP_IN_PROGRESS = "in_progress"
STEP_DONE = "done"
STEP_FAILED = "failed"
STEP_REVISED = "revised"


@dataclass
class Step:
    id: int
    description: str           # one-line goal of this step (kept clean across revisions)
    needs: list[str] = field(default_factory=list)   # tools/info this step expects to use
    status: str = STEP_PENDING
    action: dict | None = None                       # what we did
    observation: str | None = None                   # result text
    critique: str | None = None                      # judge feedback
    attempts: int = 0
    revision_hint: str | None = None                 # judge's last suggestion when revising
    started_at: float | None = None
    completed_at: float | None = None


@dataclass
class Plan:
    goal: str
    steps: list[Step]
    cursor: int = 0
    aborted: bool = False
    abort_reason: str | None = None

    def next_step(self) -> Step | None:
        while self.cursor < len(self.steps):
            s = self.steps[self.cursor]
            if s.status in (STEP_PENDING, STEP_REVISED):
                return s
            self.cursor += 1
        return None

    def advance(self) -> None:
        if self.cursor < len(self.steps):
            self.steps[self.cursor].status = STEP_DONE
            self.cursor += 1

    def mark_failed(self, reason: str) -> None:
        if self.cursor < len(self.steps):
            self.steps[self.cursor].status = STEP_FAILED
        self.aborted = True
        self.abort_reason = reason

    def revise_current(self, suggestion: str) -> None:
        if self.cursor < len(self.steps):
            s = self.steps[self.cursor]
            s.status = STEP_REVISED
            s.attempts += 1
            # Stash suggestion separately so the reasoning prompt can surface it
            # WITHOUT polluting the step description (which is used as a search
            # query in some tool prompts).
            s.revision_hint = suggestion

    def completed_steps(self) -> list[Step]:
        return [s for s in self.steps if s.status == STEP_DONE]


@dataclass
class Critique:
    satisfied: bool
    fatal: bool = False             # if true, stop trying entirely
    suggestion: str = ""            # what to do differently
    reasoning: str = ""             # judge's explanation


@dataclass
class AgentResult:
    query: str
    answer: str
    plan: Plan
    scratchpad_text: str
    iterations: int
    duration_seconds: float
    success: bool


class Scratchpad:
    """Working memory for an agent run. Bounded by char budget.

    The scratchpad holds:
      - The original query
      - The plan
      - Per-step actions, observations, critiques
      - Free-form findings the agent wants to remember across steps

    `render_for_step` returns ONLY what is relevant to the current step
    (most recent + plan-level summary), not the full scratchpad. This keeps
    per-step prompts focused.
    """

    def __init__(self, query: str, max_chars: int = 8000):
        self.query = query
        self.max_chars = max_chars
        self.findings: dict[str, str] = {}

    def render_for_step(
        self,
        plan: Plan,
        step: Step,
        *,
        keep_step_ids: set[int] | None = None,
        keep_finding_keys: set[str] | None = None,
    ) -> str:
        """Render scratchpad scoped to this step's relevant context.

        Includes: goal, prior step summaries (1 line each), key findings,
        and the current step's prior attempts (if revising).

        Optional `keep_step_ids` / `keep_finding_keys` come from MAD-MM
        memory masking (see `_mask_prior_observations`). When provided,
        only items in the sets are included. When `None` (default), all
        items are kept — original behavior.
        """
        lines: list[str] = [f"GOAL: {plan.goal}"]

        # Prior step summaries — just the essentials
        prior = [s for s in plan.steps if s.status == STEP_DONE]
        if prior:
            # Window first, THEN apply mask — keeps "last 5" semantics intact
            window = prior[-5:]
            if keep_step_ids is not None:
                window = [s for s in window if s.id in keep_step_ids]
            if window:
                lines.append("\nPRIOR STEPS:")
                for s in window:
                    obs = (s.observation or "")[:200]
                    lines.append(f"  [step {s.id}] {s.description[:80]} -> {obs}")

        # Findings (free-form)
        if self.findings:
            items = list(self.findings.items())[:10]
            if keep_finding_keys is not None:
                items = [(k, v) for k, v in items if k in keep_finding_keys]
            if items:
                lines.append("\nFINDINGS:")
                for k, v in items:
                    lines.append(f"  - {k}: {str(v)[:150]}")

        # Current step attempts (if revising) — never masked: this is the
        # signal the loop is in retry mode and needs to know what was tried.
        if step.attempts > 0 and step.observation:
            lines.append(
                f"\nPREVIOUS ATTEMPT FOR THIS STEP (attempt {step.attempts}):"
            )
            lines.append(f"  Did: {json.dumps(step.action)[:200]}")
            lines.append(f"  Got: {(step.observation or '')[:300]}")
            lines.append(f"  Critique: {(step.critique or '')[:200]}")

        text = "\n".join(lines)
        # Hard cap
        if len(text) > self.max_chars:
            text = text[: self.max_chars] + "\n[truncated]"
        return text

    def render_full(self, plan: Plan) -> str:
        """Render everything for final synthesis."""
        lines = [f"QUERY: {self.query}", f"\nPLAN ({len(plan.steps)} steps):"]
        for s in plan.steps:
            lines.append(f"  [{s.status}] {s.id}. {s.description[:120]}")
            if s.observation:
                lines.append(f"      -> {(s.observation)[:300]}")
        if self.findings:
            lines.append("\nFINDINGS:")
            for k, v in self.findings.items():
                lines.append(f"  - {k}: {str(v)[:200]}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

PLAN_PROMPT = """You are planning how to answer a user query. Decompose it into concrete checkable steps.

Available tools (use exact name in "needs"):
{tools}

USER QUERY: {query}

DECOMPOSITION RULES — follow strictly:
- COUNT the distinct asks in the query. Words like "and", "then", "find X and find Y", or any list of separate facts to retrieve = SEPARATE STEPS.
- If a step requires looking up an external fact, that fact is its own step (one search per fact).
- A SINGLE step is ONLY appropriate for: pure math you can do in your head, a single definition, a single well-known fact — NOT for lookups, NOT for comparisons.
- Each step must be SPECIFIC — "search web for current population of Japan" not "research populations".
- Synthesis happens automatically after the last step. Do NOT add a "summarize" step.

DECOMPOSITION SELF-CHECK — before outputting:
1. Count the distinct facts/values the query needs.
2. If that count is >= 2, your plan MUST have >= 2 steps (one per fact).
3. If the query asks to "compare" or "calculate from" two values, add a final step for the comparison/calculation AFTER the lookups.

Output JSON: {{"steps": [{{"description": "specific action", "needs": ["tool_name"]}}, ...]}}

Examples:
- "What is 2+2?" -> 1 step: [{{"description":"Compute 2+2","needs":[]}}]
- "Find area of Texas and area of California, then which is larger" -> 3 steps:
   1. {{"description":"Search web for area of Texas in square km","needs":["web_search"]}}
   2. {{"description":"Search web for area of California in square km","needs":["web_search"]}}
   3. {{"description":"Compare retrieved areas and state which is larger","needs":[]}}
- "Compare population of Japan vs Korea, then density given areas X and Y" -> 3+ steps (one per population, one per density).

JSON:"""


REASON_ACT_PROMPT = """You are executing one step of a plan to answer the user's query. Decide either:
  (a) Call a tool: output {{"tool": "tool_name", "args": {{...}}}}
  (b) Direct reasoning: output {{"answer": "your reasoning result"}}

CONTEXT (only what's relevant for this step):
{scratchpad}

CURRENT STEP: {step_description}{revision_block}

Available tools and their schemas:
{tools_schema}

Output the JSON action. Nothing else.

SEARCH QUERY GUIDANCE (when calling web_search / knowledge_search):
- For facts about a SPECIFIC entity (state, country, person, company, product), query JUST the entity name. Wikipedia and search engines have the rest in the article: "Texas" not "area of Texas in square km", "Tesla Model Y" not "current price of Tesla Model Y".
- For events/dates, query the event name: "French Revolution" not "what year did the French Revolution start".
- For general topics, use 2-4 keyword phrases, not full sentences.
- AVOID query words like "current", "latest", "recent" — they bias rankings toward news articles.
- The fact extraction step will pull the specific number/date/name out of the article — your job is just to land on the RIGHT article.

When this step needs a value from a PRIOR step, look at the FINDINGS section of the context and reference the EXACT value by its key. Do NOT invent or re-derive.

JSON:"""


EXTRACT_FACTS_PROMPT = """Pull every concrete fact from this text into a structured findings dict. Be aggressive — pull EVERYTHING specific.

STEP GOAL: {step_description}
TEXT TO MINE: {observation}

Look for:
- Numbers with units: "695,660 km2", "31.7 million residents", "$67,500", "2.5%", "100,000 req/s"
- Numbers in parentheses or inline: "(123 million)", "as of 2025: 51.6M"
- Dates and years: "1789", "April 2025", "founded 1789"
- Names of entities, cities, people, companies, algorithms, libraries
- Statistics in any form
- Design choices and architectural decisions: "uses Redis for shared state",
  "token-bucket per user-key", "p99 < 50ms target"
- Key constraints driving the design: "must sustain 100k req/sec"

Output JSON: {{"facts": {{"snake_case_key": "value with unit", ...}}}}

Rules:
- Use snake_case keys descriptive of WHAT the value is (e.g. "texas_area_km2" not "v1", "rate_limit_algorithm" not "alg").
- Pull AT LEAST one fact if the text has any specific information (numbers, names, design choices) — empty {{}} only if there is genuinely nothing to extract (pure error text).
- Keep values SHORT: the number with its unit, a date, a single name, or a single design choice. Not a sentence.

EXAMPLES (study these):
- TEXT: "Texas covers 268,596 square miles (695,660 km2) with a population of over 31.7 million in 2025."
  -> {{"facts": {{"texas_area_sq_mi": "268596", "texas_area_km2": "695660", "texas_population_2025": "31.7 million"}}}}
- TEXT: "The French Revolution of 1848 (Révolution française de 1848), also known as the February Revolution, was the second French revolution."
  -> {{"facts": {{"french_revolution_1848_year": "1848", "french_revolution_1848_aka": "February Revolution"}}}}
- TEXT: "[1] California State\n    https://example.com/ca\n    California has 423,970 km2 of total area..."
  -> {{"facts": {{"california_area_km2": "423970"}}}}
- TEXT: "[tool error: connection refused]"
  -> {{"facts": {{}}}}

JSON:"""


CRITIQUE_PROMPT = """Judge whether this step's goal is met. Be strict on factual misses, but don't reject correct data hidden in parenthetical / inline form.

STEP GOAL: {step_description}
ACTION TAKEN: {action}
OBSERVATION: {observation}
EXTRACTED FACTS (parsed from observation by the fact-extractor): {findings}

Decision rules:
1. If EXTRACTED FACTS contains a value that answers the step's goal → satisfied=true. (Example: goal "find area of Texas in km^2" + findings has "texas_area_km2: 695660" → satisfied=true.)
2. If the observation is from an authoritative source (Wikipedia, government data, etc.) AND the article topic matches the goal, even if the specific value is buried/parenthetical → satisfied=true. The fact-extractor will pull it.
3. If the observation is an error string ("[tool error: ...]", "[tool not found]") → satisfied=false. fatal=true if tool genuinely unavailable.
4. If the observation is from an unrelated topic (asked about Texas, got Darwin, California town) → satisfied=false; suggest more specific query.
5. If the goal asks for a specific number/date/name AND extracted facts have nothing relevant AND observation has no matching content → satisfied=false.

Output JSON:
{{
  "satisfied": true|false,
  "fatal": true|false,
  "suggestion": "if not satisfied, what specifically to do differently (1 sentence, concrete)",
  "reasoning": "1 sentence why"
}}

JSON:"""


SYNTHESIZE_PROMPT = """You are formatting a final answer for the user. The reasoning is already done — present it cleanly.

USER QUERY: {query}

WORK ALREADY COMPLETED:
{scratchpad}

Output ONLY the final answer. Do NOT:
- Repeat or re-derive the steps
- Question the work that was done
- Add disclaimers, hedging, or "let me think"
- Wrap in commentary like "based on the above..."

NEVER reference the internal plan or scaffolding to the user. The user does
not know about steps, search logs, retrieval attempts, scratchpads, or any
pipeline metadata. Treat all of that as your private working notes:
- DO NOT write phrases like "step 3 of 5", "step 26/174", "scenario X/Y"
- DO NOT say "the analysis plan", "the completed plan", "based on the plan"
- DO NOT say "search logs", "retrieval attempts", "failed search results"
  (you may say "I couldn't find a current figure for X" instead)
- DO NOT say "marked as requiring live configuration", "not determined here",
  "as indicated by the search logs"
- DO NOT cite scratchpad keys, findings keys, or step IDs

CRITICAL — for any step marked [failed]:
- DO NOT fabricate the missing fact
- DO NOT use a "common knowledge" guess as if it were the missing data
- Plainly state what could not be determined and why (e.g. "couldn't verify Japan's current population — search didn't return a usable figure")
- Skip parts of the answer that depend on the failed data, or mark them clearly as estimated/unverified

ANSWER:"""


SELF_CONSISTENCY_PROMPT = """{base_prompt}

You will be sampled multiple times for this step. Reason carefully and pick the most defensible action."""


# ---------------------------------------------------------------------------
# Deliberation-class query detector (used to suppress single-step bypass)
# ---------------------------------------------------------------------------
#
# Some queries land at AgentLoop with a 1-step plan because the planner can't
# decompose them into atomic facts (no lookups needed). Examples: "design X
# step by step", "walk me through Y", "explain how Z works in detail",
# "architect a Q for N". The model produces a first-pass answer in a single
# REASON_ACT step, the synthesis bypass returns it directly without going
# through SYNTHESIZE_PROMPT, and the result is a thin 2k-char answer when a
# proper 5-section response was warranted.
#
# Observed bimodal pattern across 6 eval runs of `deliberation_chain_of_reasoning`:
#   0.85, 0.20, 0.80, 0.85, 0.20, 0.20 — the 0.20 cases all matched the
#   short-circuit path (30-50s vs 124-255s for the synthesis path).
#
# When the query matches this pattern, we force synthesis even for 1-step
# plans so the model is asked to expand into proper structured output.

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
    """True iff the query needs structured multi-section output and should
    not be short-circuited by the single-step synthesis bypass.
    """
    return bool(query and _DELIBERATION_QUERY_RE.search(query))


# ---------------------------------------------------------------------------
# Synthesis output sanitizer — strips scaffolding leaks from final answers
# ---------------------------------------------------------------------------
#
# Even with a tightened SYNTHESIZE_PROMPT, the model occasionally leaks
# pipeline metadata ("step 26/174", "as indicated by failed retrieval
# attempts in the search logs", "marked as requiring live configuration").
# These are harmless internally but bleed credibility in user-facing answers.
# We strip them in two passes:
#   1. Sentence-level deletion for sentences that are MOSTLY scaffolding —
#      i.e. they primarily reference the plan/logs and would be lost if
#      removed (no claim survives without the scaffold reference).
#   2. Phrase-level redaction for shorter scaffolding mentions embedded in
#      otherwise-good sentences. Replace with empty string and clean spacing.
#
# Conservative: regex matches must be specific enough not to hit normal
# user-facing references to "search results", "the plan", etc.

import re as _re_sanitize

_SCAFFOLD_SENTENCE_PATTERNS = [
    # Step / scenario IDs: "26/174 scenario", "step 3 of 5 scenario"
    _re_sanitize.compile(
        r"[^.!?\n]*\b(?:step\s+)?\d+/\d+\s+scenario[^.!?\n]*[.!?]",
        _re_sanitize.IGNORECASE,
    ),
    # "as indicated by ... search logs"
    _re_sanitize.compile(
        r"[^.!?\n]*\bas\s+indicated\s+by[^.!?\n]*\bsearch\s+log[^.!?\n]*[.!?]",
        _re_sanitize.IGNORECASE,
    ),
    # "based on (the |your |the completed )?(analysis|deliberation) plan"
    _re_sanitize.compile(
        r"[^.!?\n]*\bbased\s+on[^.!?\n]*\b(?:analysis|deliberation|completed)\s+plan[^.!?\n]*[.!?]",
        _re_sanitize.IGNORECASE,
    ),
    # "marked as requiring live configuration"
    _re_sanitize.compile(
        r"[^.!?\n]*\bmarked\s+as\s+requiring\s+live\s+configuration[^.!?\n]*[.!?]",
        _re_sanitize.IGNORECASE,
    ),
    # "from <the|our|your|available|provided> search logs"
    _re_sanitize.compile(
        r"[^.!?\n]*\b(?:in|from)\s+(?:the|our|your|available|provided)\s+"
        r"search\s+logs?\b[^.!?\n]*[.!?]",
        _re_sanitize.IGNORECASE,
    ),
    # Standalone "search logs" reference inside a parenthetical note
    _re_sanitize.compile(
        r"\*?\(\s*Note[^)]*\bsearch\s+logs?\b[^)]*\)\*?",
        _re_sanitize.IGNORECASE,
    ),
    # Standalone "not explicitly verified" with reference to search/retrieval
    _re_sanitize.compile(
        r"[^.!?\n]*\bnot\s+(?:explicitly\s+)?verified\s+in\s+(?:the\s+)?"
        r"(?:available|provided)\s+search[^.!?\n]*[.!?]",
        _re_sanitize.IGNORECASE,
    ),
    # "failed retrieval attempts on" (followed by topic)
    _re_sanitize.compile(
        r"[^.!?\n]*\bfailed\s+retrieval\s+attempts\s+(?:on|for)\b[^.!?\n]*[.!?]",
        _re_sanitize.IGNORECASE,
    ),
]

_SCAFFOLD_PHRASE_PATTERNS = [
    # Inline parenthetical refs: "(step 26/174)", "(based on the analysis plan)"
    _re_sanitize.compile(
        r"\s*\((?:step\s+)?\d+/\d+(?:\s+scenario)?\)",
        _re_sanitize.IGNORECASE,
    ),
    _re_sanitize.compile(
        r"\s*\(based\s+on\s+(?:the\s+)?(?:analysis|deliberation|completed)\s+plan[^)]*\)",
        _re_sanitize.IGNORECASE,
    ),
    # ", as indicated by ... search logs"
    _re_sanitize.compile(
        r",\s*as\s+indicated\s+by[^,.!?\n]*search\s+log[^,.!?\n]*",
        _re_sanitize.IGNORECASE,
    ),
    # "in the provided search results" / "in the search results"
    _re_sanitize.compile(
        r"\s*(?:in|from)\s+(?:the|your)\s+(?:provided\s+)?search\s+results?",
        _re_sanitize.IGNORECASE,
    ),
    # "your completed analysis plan" / "the completed plan"
    _re_sanitize.compile(
        r"\s*(?:your|the)\s+(?:completed\s+)?(?:analysis|deliberation)\s+plan",
        _re_sanitize.IGNORECASE,
    ),
    # Trailing "(not specified here)" / "(not determined in...)"
    _re_sanitize.compile(
        r"\s*\(not\s+(?:specified|determined)\s+(?:in|here)[^)]*\)",
        _re_sanitize.IGNORECASE,
    ),
]


def sanitize_synthesis(text: str) -> tuple[str, int]:
    """Strip scaffolding leaks from a synthesized deliberation answer.

    Returns (cleaned, n_changes). Phrase-level redaction runs FIRST so that
    parentheticals like "(based on the completed analysis plan)" inside
    otherwise-good sentences get redacted instead of triggering whole-
    sentence deletion. Then sentence-level patterns drop sentences that ARE
    primarily scaffolding (no usable claim survives if removed).
    """
    if not text:
        return text, 0
    out = text
    changes = 0

    # Pass 1: phrase-level redaction (runs FIRST — see docstring)
    for pat in _SCAFFOLD_PHRASE_PATTERNS:
        new = pat.sub("", out)
        if new != out:
            changes += 1
            out = new

    # Pass 2: sentence-level deletion for scaffold-dominant sentences
    for pat in _SCAFFOLD_SENTENCE_PATTERNS:
        new = pat.sub("", out)
        if new != out:
            changes += 1
            out = new

    # Cleanup: collapse double-spaces and orphaned sentence joiners
    out = _re_sanitize.sub(r"  +", " ", out)
    out = _re_sanitize.sub(r"\.\s*\.+", ".", out)
    out = _re_sanitize.sub(r",\s*,", ",", out)
    # Trim spaces before punctuation
    out = _re_sanitize.sub(r"\s+([.,!?])", r"\1", out)
    # Trailing/leading whitespace per line
    out = "\n".join(line.rstrip() for line in out.splitlines())
    return out.strip(), changes


# ---------------------------------------------------------------------------
# MAD-MM subjective memory masking (ICLR 2026, arxiv 2603.20215)
# ---------------------------------------------------------------------------
#
# When a step retries, the original attempt got misled — often by an irrelevant
# or wrong prior step / finding bleeding into the scratchpad. MAD-MM's subjective
# masking pass asks the LLM, per prior item, whether it's actually useful for
# the current goal. We batch all items into ONE LLM call (vs MAD-MM's per-item
# loop) to keep latency tolerable on local hardware — one ~2-5s call vs
# 10×2-5s. Conservative on parse failure (keep everything).

_MASK_PROMPT = """You will be shown several memory items from prior reasoning steps.
For the CURRENT STEP below, decide which items are useful and which should be ignored.

Useful = directly informs the current step (a value to reuse, a constraint to honor, a fact to cite).
Ignore = irrelevant, contradicts the goal, or led the prior attempt astray.

CURRENT STEP: {step_description}

OVERALL GOAL: {goal}

ITEMS (id -> content):
{items}

Respond with one JSON object mapping each id to "yes" (useful) or "no" (ignore):
{{"id_1": "yes"|"no", "id_2": "yes"|"no", ...}}
Only the JSON. Default to "yes" when uncertain — drop only items you're confident are unhelpful."""


async def _mask_prior_observations(
    goal: str,
    step_description: str,
    prior_step_items: list[tuple[int, str]],
    finding_items: list[tuple[str, str]],
    *,
    timeout: float = 90.0,
) -> tuple[set[int], set[str], dict[str, str]]:
    """Subjective MAD-MM mask on prior step observations + scratchpad findings.

    Args:
        goal: the plan-level goal text.
        step_description: the current step's description (what the LLM is about
            to attempt).
        prior_step_items: list of (step_id, "description -> observation") pairs
            already trimmed to the relevant window.
        finding_items: list of (key, "value") pairs from the scratchpad.
        timeout: per-call ceiling.

    Returns:
        (keep_step_ids, keep_finding_keys, decisions_log).
        On any failure (LLM error, parse error, empty items) returns the
        all-keep sets so the caller falls back to the unfiltered scratchpad.
    """
    if not prior_step_items and not finding_items:
        return set(), set(), {}

    all_step_ids = {sid for sid, _ in prior_step_items}
    all_finding_keys = {k for k, _ in finding_items}
    keep_all = (all_step_ids, all_finding_keys, {})

    # Build labelled item list. Use stable, parseable ids — the LLM keys its
    # JSON response by these.
    label_to_kind: dict[str, tuple[str, int | str]] = {}
    rendered: list[str] = []
    for sid, content in prior_step_items:
        label = f"step_{sid}"
        label_to_kind[label] = ("step", sid)
        rendered.append(f"  {label}: {content[:240]}")
    for k, content in finding_items:
        label = f"finding_{k}"
        label_to_kind[label] = ("finding", k)
        rendered.append(f"  {label}: {str(content)[:200]}")

    prompt = _MASK_PROMPT.format(
        goal=goal[:500],
        step_description=step_description[:400],
        items="\n".join(rendered),
    )

    # Optional stronger judge model — empty string falls through to default.
    judge_model: str | None = None
    try:
        from app.config import config as _cfg
        jm = (getattr(_cfg, "MAD_MM_JUDGE_MODEL", "") or "").strip()
        if jm:
            judge_model = jm
    except Exception:
        pass

    try:
        raw = await asyncio.wait_for(
            llm.invoke_nothink(
                [{"role": "user", "content": prompt}],
                json_mode=True,
                json_prefix="{",
                max_tokens=400,
                temperature=0.1,
                model=judge_model,
            ),
            timeout=timeout,
        )
    except Exception as e:
        logger.warning("[mad-mm] mask LLM failed: %s — keeping all", e)
        return keep_all

    if not raw:
        return keep_all

    try:
        obj = json.loads(raw) if isinstance(raw, str) else raw
    except Exception:
        obj = extract_json_object(raw) or {}
    if not isinstance(obj, dict):
        return keep_all

    keep_step_ids: set[int] = set()
    keep_finding_keys: set[str] = set()
    decisions: dict[str, str] = {}
    for label, (kind, key) in label_to_kind.items():
        verdict_raw = obj.get(label, "yes")
        verdict = str(verdict_raw).strip().lower()
        if verdict not in ("yes", "no"):
            verdict = "yes"  # conservative on ambiguous output
        decisions[label] = verdict
        if verdict == "yes":
            if kind == "step":
                keep_step_ids.add(key)  # type: ignore[arg-type]
            else:
                keep_finding_keys.add(key)  # type: ignore[arg-type]

    # Guard against an all-drop pathological response — the LLM may collapse
    # under prompt confusion and label everything "no". Falling back to keep-all
    # is safer than rendering an empty scratchpad.
    if not keep_step_ids and not keep_finding_keys and (all_step_ids or all_finding_keys):
        logger.warning("[mad-mm] mask collapsed to empty — keeping all instead")
        return keep_all

    n_dropped = (len(all_step_ids) - len(keep_step_ids)) + (
        len(all_finding_keys) - len(keep_finding_keys)
    )
    if n_dropped:
        logger.info("[mad-mm] masked %d/%d prior items for step",
                    n_dropped, len(label_to_kind))
    return keep_step_ids, keep_finding_keys, decisions


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------


class AgentLoop:
    """Deliberation loop. Stateless across calls; one solve() = one run."""

    def __init__(self, tools: Any | None = None):
        # tools is a ToolRegistry (or None — pure reasoning mode)
        self.tools = tools

    async def solve(
        self,
        query: str,
        *,
        max_iterations: int = 10,
        sample_n: int = 1,
        on_event: Any | None = None,  # callback(event_type, payload) for streaming
    ) -> AgentResult:
        """Run the deliberation loop on a query. Returns final answer + trace."""
        t0 = time.monotonic()
        scratchpad = Scratchpad(query)

        # Tree-of-thought: auto-bump sample_n for queries that look like hard
        # reasoning (compare/analyze/explain/derive/why-style). Caller-provided
        # sample_n>1 wins; we only bump from the default of 1.
        if sample_n <= 1:
            try:
                from app.config import config as _cfg
                if _cfg.ENABLE_TREE_OF_THOUGHT and _is_hard_reasoning_query(query):
                    sample_n = max(2, _cfg.TOT_SAMPLE_N)
                    logger.info("Tree-of-thought: bumped sample_n=%d for hard query: %r",
                                sample_n, query[:80])
            except Exception as e:
                logger.debug("ToT bump check failed: %s", e)

        # Hydrate from persistent workspace if a prior run on this query
        # signature exists. Prior findings prefixed with "prior:" so the
        # planner can see them but knows they're carry-over context, not
        # this-run observations.
        workspace_entry = None
        try:
            from app.database import get_db
            from app.core.agent_workspace import load_workspace, hydrate_scratchpad
            workspace_entry = load_workspace(get_db(), query)
            if workspace_entry:
                added = hydrate_scratchpad(scratchpad, workspace_entry)
                if added:
                    await self._emit(
                        on_event,
                        "workspace_hydrated",
                        {
                            "prior_runs": workspace_entry.run_count,
                            "findings_carried": added,
                            "prior_success_rate": (
                                workspace_entry.success_count
                                / max(workspace_entry.run_count, 1)
                            ),
                        },
                    )
        except Exception as e:
            logger.warning("[AgentLoop] workspace load failed: %s", e)

        plan = await self._plan(query)
        await self._emit(on_event, "plan", {"steps": [s.description for s in plan.steps]})

        iterations = 0
        for _ in range(max_iterations):
            step = plan.next_step()
            if step is None:
                break
            iterations += 1
            step.status = STEP_IN_PROGRESS
            step.started_at = time.monotonic()
            await self._emit(on_event, "step_start", {"id": step.id, "description": step.description})

            # MAD-MM subjective memory masking — opt-in. Only fire when the
            # step is being RETRIED (the original attempt was misled) AND
            # there are enough rendered items in the scratchpad to meaningfully
            # change the prompt. Counts both prior DONE steps and free-form
            # findings — findings get rendered regardless of step status, so
            # they're what most often need masking after a retry. Adds one
            # batched LLM call. See _mask_prior_observations.
            mask_step_ids: set[int] | None = None
            mask_finding_keys: set[str] | None = None
            try:
                from app.config import config as _cfg
                if (
                    getattr(_cfg, "ENABLE_MAD_MM_MASKING", False)
                    and step.attempts > 0
                ):
                    prior_done = [s for s in plan.steps if s.status == STEP_DONE]
                    finding_count = len(scratchpad.findings)
                    n_renderable = len(prior_done) + finding_count
                    min_prior = int(getattr(_cfg, "MAD_MM_MIN_PRIOR_STEPS", 3))
                    if n_renderable >= min_prior:
                        prior_items = [
                            (s.id, f"{s.description[:80]} -> {(s.observation or '')[:200]}")
                            for s in prior_done[-5:]
                            if s.observation
                        ]
                        finding_items = [
                            (k, str(v)) for k, v in list(scratchpad.findings.items())[:10]
                        ]
                        mask_step_ids, mask_finding_keys, _decisions = await _mask_prior_observations(
                            goal=plan.goal,
                            step_description=step.description,
                            prior_step_items=prior_items,
                            finding_items=finding_items,
                        )
                        n_dropped = (
                            (len([s.id for s in prior_done[-5:] if s.observation]) - len(mask_step_ids))
                            + (min(finding_count, 10) - len(mask_finding_keys))
                        )
                        logger.info(
                            "[mad-mm] step %d retry attempt=%d: masked %d/%d items "
                            "(kept %d steps, %d findings)",
                            step.id, step.attempts, n_dropped, n_renderable,
                            len(mask_step_ids), len(mask_finding_keys),
                        )
                        await self._emit(on_event, "mad_mm_mask", {
                            "step_id": step.id,
                            "attempts": step.attempts,
                            "kept_steps": sorted(mask_step_ids),
                            "kept_findings": sorted(mask_finding_keys),
                        })
            except Exception as e:
                logger.warning("[mad-mm] mask wrapper failed: %s — proceeding unmasked", e)
                mask_step_ids = None
                mask_finding_keys = None

            ctx = scratchpad.render_for_step(
                plan, step,
                keep_step_ids=mask_step_ids,
                keep_finding_keys=mask_finding_keys,
            )
            try:
                action = await self._reason_act(step, ctx, sample_n=sample_n)
            except Exception as e:
                logger.exception("reason_act failed for step %d", step.id)
                action = {"answer": f"[reasoning failed: {e}]"}

            observation = await self._execute_action(action)
            step.action = action
            step.observation = observation
            await self._emit(on_event, "observation", {"id": step.id, "observation": observation[:300]})

            # Extract concrete facts so later steps can use them by name. This
            # is the "working memory" that makes multi-step plans actually
            # share data — without it, step N just re-derives or hallucinates
            # values from step N-1's prose observation.
            extracted: dict[str, str] = {}
            try:
                extracted = await self._extract_facts(step, observation)
                if extracted:
                    scratchpad.findings.update(extracted)
                    await self._emit(on_event, "findings", {"id": step.id, "added": list(extracted.keys())})
            except Exception as e:
                logger.warning("fact extraction failed for step %d: %s", step.id, e)

            critique = await self._critique(step, action, observation, extracted)
            step.critique = critique.reasoning
            await self._emit(on_event, "critique", {"id": step.id, "satisfied": critique.satisfied, "reason": critique.reasoning})

            step.completed_at = time.monotonic()

            if critique.satisfied:
                plan.advance()
            elif critique.fatal:
                plan.mark_failed(critique.suggestion or critique.reasoning)
                break
            elif step.attempts >= 2:
                # Exhausted retries on this step. Before marching on we ask
                # "is the PLAN wrong, not just this step?" — if the failure
                # is structural (every step has been failing), revise the
                # plan from the current cursor forward instead of pressing on
                # with what's clearly broken.
                step.status = STEP_FAILED
                plan.cursor += 1
                failed_so_far = sum(1 for s in plan.steps if s.status == STEP_FAILED)
                done_so_far = sum(1 for s in plan.steps if s.status == STEP_DONE)
                if failed_so_far >= 2 and done_so_far == 0:
                    # Pure failure run — bail and try a fresh plan
                    try:
                        await self._emit(
                            on_event, "plan_revision",
                            {"reason": "≥2 failed steps with 0 done — replanning"},
                        )
                        new_plan = await self._plan(query, prior_failures=[
                            (s.description, s.observation or "")
                            for s in plan.steps if s.status == STEP_FAILED
                        ])
                        if new_plan and new_plan.steps:
                            plan = new_plan
                            await self._emit(
                                on_event, "plan",
                                {"steps": [s.description for s in plan.steps], "revised": True},
                            )
                    except Exception as e:
                        logger.warning("plan revision failed: %s", e)
            else:
                plan.revise_current(critique.suggestion)

        answer = await self._synthesize(scratchpad, plan)
        duration = time.monotonic() - t0
        success = (
            not plan.aborted
            and any(s.status == STEP_DONE for s in plan.steps)
        )
        result = AgentResult(
            query=query,
            answer=answer,
            plan=plan,
            scratchpad_text=scratchpad.render_full(plan),
            iterations=iterations,
            duration_seconds=duration,
            success=success,
        )

        # Real learning hooks — fire-and-forget so they don't block the response
        try:
            await self._learn_from_run(result, scratchpad)
        except Exception as e:
            logger.warning("learn_from_run failed: %s", e)

        # Persist scratchpad findings to the agent workspace so future runs
        # of similar queries inherit prior progress. Strip "prior:" keys —
        # those came from a previous run and saving them again would inflate.
        try:
            from app.database import get_db
            from app.core.agent_workspace import save_workspace
            save_findings = {
                k: v for k, v in scratchpad.findings.items()
                if not k.startswith("prior:")
            }
            if save_findings or answer:
                save_workspace(
                    get_db(),
                    query=query,
                    findings=save_findings,
                    answer=answer,
                    success=success,
                )
        except Exception as e:
            logger.warning("[AgentLoop] workspace save failed: %s", e)

        return result

    async def _learn_from_run(self, result: AgentResult, scratchpad: Scratchpad) -> None:
        """Update Nova's persistent memory from this run.

        Three hooks:
          1. **Skill extraction**: successful multi-step runs with a coherent
             tool sequence get logged as auto_tool_candidates so the existing
             skill-extraction infra can promote them.
          2. **Capability gap**: failed steps log a capability_gap row so the
             daily Capability Review monitor sees them.
          3. **Curiosity queue**: failed-search topics go into curiosity_queue
             for proactive research on the next maintenance cycle.
        """
        from app.database import get_db
        db = get_db()
        completed = [s for s in result.plan.steps if s.status == STEP_DONE]
        failed = [s for s in result.plan.steps if s.status == STEP_FAILED]

        # 1. Skill candidate from successful tool sequence
        if result.success and len(completed) >= 2:
            tool_seq = []
            for s in completed:
                if s.action and "tool" in s.action:
                    tool_seq.append(s.action["tool"])
            if tool_seq:
                seq_key = "|".join(sorted(set(tool_seq)))
                try:
                    db.execute(
                        "INSERT INTO auto_tool_candidates "
                        "(query, tool_sequence, sequence_key, created_at, triggered) "
                        "VALUES (?, ?, ?, CURRENT_TIMESTAMP, 0)",
                        (result.query[:500], json.dumps(tool_seq), seq_key),
                    )
                    logger.info("agent_loop: logged skill candidate seq=%s for %r", seq_key, result.query[:60])
                except Exception as e:
                    logger.warning("skill candidate insert failed: %s", e)

        # 2. Capability gap rows from failed steps
        for s in failed:
            try:
                db.execute(
                    "INSERT INTO capability_gaps "
                    "(query, reason, tools_tried, quality_score, reviewed, created_at) "
                    "VALUES (?, ?, ?, 0.0, 0, CURRENT_TIMESTAMP)",
                    (
                        s.description[:500],
                        f"agent step failed after {s.attempts} attempts: {s.critique or 'no critique'}"[:500],
                        json.dumps([s.action.get("tool", "")] if s.action else []),
                    ),
                )
            except Exception as e:
                logger.warning("capability_gap insert failed: %s", e)

        # 3. Curiosity queue from failed search-style steps — Nova will research
        # these on the next maintenance cycle and integrate findings as KG facts.
        for s in failed:
            if s.action and s.action.get("tool") in {"web_search", "knowledge_search", "browser"}:
                topic = s.description[:200]
                try:
                    # Avoid duplicates by checking existing pending items
                    existing = db.fetchone(
                        "SELECT id FROM curiosity_queue "
                        "WHERE topic = ? AND status = 'pending' LIMIT 1",
                        (topic,),
                    )
                    if not existing:
                        db.execute(
                            "INSERT INTO curiosity_queue "
                            "(topic, source, urgency, status, attempts, created_at) "
                            "VALUES (?, 'agent_failure', 0.7, 'pending', 0, CURRENT_TIMESTAMP)",
                            (topic,),
                        )
                        logger.info("agent_loop: queued curiosity research: %s", topic[:80])
                except Exception as e:
                    logger.warning("curiosity_queue insert failed: %s", e)

    # -----------------------------------------------------------------------
    # Internal stages
    # -----------------------------------------------------------------------

    async def _plan(self, query: str, *, prior_failures: list[tuple[str, str]] | None = None) -> Plan:
        """Decompose the query into a list of steps.

        If `prior_failures` is provided, prepend a "DO NOT TRY THESE — they
        already failed" section so the new plan picks a different angle.
        """
        tools_text = self._tools_brief()
        failure_block = ""
        if prior_failures:
            lines = []
            for desc, obs in prior_failures[:5]:
                lines.append(f"  - tried: {desc[:120]} → {(obs or '')[:200]}")
            failure_block = (
                "\n\nPRIOR ATTEMPTS THAT FAILED — DO NOT REPEAT, pick a different angle:\n"
                + "\n".join(lines)
                + "\n"
            )
        prompt = PLAN_PROMPT.format(query=query, tools=tools_text) + failure_block
        resp = await llm.invoke_nothink([{"role": "user", "content": prompt}], max_tokens=600, json_mode=True)
        try:
            obj = json.loads(resp) if resp.strip().startswith("{") else extract_json_object(resp)
            steps_data = obj.get("steps", []) if isinstance(obj, dict) else []
        except Exception:
            steps_data = []
        if not steps_data:
            # Fallback: treat the query as a single-step reasoning task
            steps_data = [{"description": query, "needs": []}]
        steps = [
            Step(id=i, description=str(s.get("description", "")), needs=list(s.get("needs", [])))
            for i, s in enumerate(steps_data)
            if s.get("description")
        ]
        return Plan(goal=query, steps=steps)

    async def _reason_act(self, step: Step, scratchpad_text: str, *, sample_n: int = 1) -> dict:
        """Decide what action to take for this step."""
        tools_schema = self._tools_schema()
        revision_block = ""
        if step.revision_hint:
            revision_block = f"\n\nPREVIOUS ATTEMPT FAILED. JUDGE'S SUGGESTION: {step.revision_hint}\nApply this suggestion when choosing your next action."
        prompt = REASON_ACT_PROMPT.format(
            scratchpad=scratchpad_text,
            step_description=step.description,
            revision_block=revision_block,
            tools_schema=tools_schema,
        )

        if sample_n > 1:
            actions: list[dict] = []
            for _ in range(sample_n):
                resp = await llm.invoke_nothink([{"role": "user", "content": prompt}], max_tokens=400, json_mode=True, temperature=0.7)
                try:
                    obj = json.loads(resp) if resp.strip().startswith("{") else extract_json_object(resp)
                    if isinstance(obj, dict):
                        actions.append(obj)
                except Exception:
                    continue
            if not actions:
                return {"answer": "[no valid action sampled]"}
            return self._most_consistent(actions)

        resp = await llm.invoke_nothink([{"role": "user", "content": prompt}], max_tokens=400, json_mode=True, temperature=0.2)
        try:
            obj = json.loads(resp) if resp.strip().startswith("{") else extract_json_object(resp)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
        return {"answer": resp[:500]}

    async def _execute_action(self, action: dict) -> str:
        """Execute the chosen action and return observation text."""
        if not isinstance(action, dict):
            return f"[invalid action: {action!r}]"

        if "tool" in action:
            if self.tools is None:
                return "[no tools registered]"
            tool_name = action["tool"]
            args = action.get("args", {}) or {}

            # Query rewrite: for search-style tools, strip "area of X" /
            # "population of X" / "X price" / "current Y" descriptive phrasing
            # down to the entity name. Wikipedia and most engines rank entity
            # queries higher than descriptive natural-language ones.
            if tool_name in {"web_search", "knowledge_search"} and "query" in args:
                args["query"] = _simplify_search_query(str(args["query"]))

            # Tool creation on demand: if the requested tool doesn't exist,
            # try to write one. Limited to pure-Python tools (no network) due
            # to the safety sandbox; useful for math/string/parsing tools the
            # agent identifies it needs.
            if hasattr(self.tools, "_tools") and tool_name not in self.tools._tools:
                created = await self._try_create_tool(tool_name, args)
                if created:
                    logger.info("agent_loop: created tool on demand: %s", tool_name)
                else:
                    return f"[tool not found and could not be created: {tool_name}]"

            try:
                if hasattr(self.tools, "execute"):
                    return (await self.tools.execute(tool_name, args))[:2000]
                # Fallback: dict-of-tools
                tool = self.tools.get(tool_name) if hasattr(self.tools, "get") else None
                if tool is None:
                    return f"[tool not found: {tool_name}]"
                result = await tool.execute(**args) if asyncio.iscoroutinefunction(getattr(tool, "execute", None)) else tool.execute(**args)
                if hasattr(result, "data"):
                    return str(result.data)[:2000]
                return str(result)[:2000]
            except Exception as e:
                return f"[tool error: {e}]"

        if "answer" in action:
            return str(action["answer"])[:2000]

        return f"[unknown action shape: {list(action.keys())}]"

    async def _extract_facts(self, step: Step, observation: str) -> dict[str, str]:
        """Pull concrete facts out of an observation into a findings dict.

        Skips for short/error observations and for steps with no real data.
        Returns {} on parse failure rather than raising.
        """
        # Skip error markers like "[tool error: ...]", "[tool not found]" but
        # NOT search results which begin with "[1] Title\n...". The latter
        # have content beyond the bracket on the same line.
        if not observation or len(observation) < 20:
            return {}
        first_line = observation.lstrip().splitlines()[0] if observation.strip() else ""
        if first_line.startswith("[") and " " not in first_line[:30]:
            # Looks like an error tag (no space in first 30 chars after bracket)
            return {}
        prompt = EXTRACT_FACTS_PROMPT.format(
            step_description=step.description[:300],
            observation=observation[:1500],
        )
        async def _try_extract(temp: float, more_aggressive: bool = False) -> dict[str, str]:
            p = prompt
            if more_aggressive:
                # Domain-neutral retry: the original prompt was geo-biased
                # ("km2, sq mi, million") and returned empty for deliberation
                # answers about algorithms / system design / abstract concepts.
                # Now nudge the model toward conceptual facts as well.
                p += (
                    "\n\nIMPORTANT: Pull AT LEAST 2-3 facts. Facts can be:"
                    "\n- numeric (counts, sizes, durations, percentages, dates)"
                    "\n- named entities (algorithms, libraries, services, people, places)"
                    "\n- design choices ('uses Redis for shared state', 'token-bucket per user')"
                    "\n- key constraints ('must sustain 100k req/sec', 'p99 < 50ms')"
                    "\nReturn empty {{}} ONLY if the text is pure error output."
                )
            try:
                resp = await llm.invoke_nothink(
                    [{"role": "user", "content": p}],
                    max_tokens=400,
                    json_mode=True,
                    temperature=temp,
                )
                obj = json.loads(resp) if resp.strip().startswith("{") else extract_json_object(resp)
                if isinstance(obj, dict):
                    facts = obj.get("facts") or {}
                    if isinstance(facts, dict):
                        return {
                            str(k): str(v)[:200]
                            for k, v in facts.items()
                            if v is not None and str(v).strip()
                        }
            except Exception as e:
                logger.warning("extract_facts parse: %s", e)
            return {}

        # First pass at temp 0.0
        result = await _try_extract(0.0)
        # Retry once if empty AND observation is long enough to plausibly have facts
        if not result and len(observation) > 100:
            logger.info("extract_facts: empty on first pass, retrying with stronger prompt")
            result = await _try_extract(0.3, more_aggressive=True)
        return result

    async def _critique(self, step: Step, action: dict, observation: str, findings: dict | None = None) -> Critique:
        """Judge whether the step accomplished its goal."""
        findings_str = json.dumps(findings or {}, indent=2)[:600] if findings else "{}"
        prompt = CRITIQUE_PROMPT.format(
            step_description=step.description,
            action=json.dumps(action)[:400],
            observation=observation[:1500],
            findings=findings_str,
        )
        try:
            resp = await llm.invoke_nothink([{"role": "user", "content": prompt}], max_tokens=200, json_mode=True, temperature=0.1)
            obj = json.loads(resp) if resp.strip().startswith("{") else extract_json_object(resp)
            if isinstance(obj, dict):
                return Critique(
                    satisfied=bool(obj.get("satisfied", False)),
                    fatal=bool(obj.get("fatal", False)),
                    suggestion=str(obj.get("suggestion", "")),
                    reasoning=str(obj.get("reasoning", "")),
                )
        except Exception as e:
            logger.warning("critique parse failed: %s", e)
        # Default: assume satisfied if observation is non-empty and not an error
        looks_ok = observation and not observation.startswith("[") and len(observation) > 5
        return Critique(satisfied=bool(looks_ok), reasoning="(critique parse failed; default heuristic)")

    async def _synthesize(self, scratchpad: Scratchpad, plan: Plan) -> str:
        """Build the final answer from the completed work.

        Shortcut: if the plan has exactly one completed step and it produced a
        usable observation, return that directly — no synthesis needed.
        Synthesis tends to over-edit short answers (e.g. "4" -> rambling 200
        words ending in the wrong number).
        """
        completed = [s for s in plan.steps if s.status == STEP_DONE]
        # Single-step plans: bypass synthesis entirely — UNLESS the query is
        # a deliberation-class query (design/walk-through/derive) where the
        # caller expects a structured multi-section answer. The synthesis
        # prompt enforces that structure; the bypass returns the model's
        # first-pass observation as-is, which is often shallow.
        # Bimodal regression observed on `deliberation_chain_of_reasoning`:
        # 0.85 with synthesis vs 0.20 with bypass.
        if len(completed) == 1 and len(plan.steps) == 1:
            obs = (completed[0].observation or "").strip()
            if obs and not obs.startswith("["):
                if _is_deliberation_query(scratchpad.query):
                    logger.info(
                        "[synthesis] single-step plan but query is deliberation-class — "
                        "running synthesis instead of bypass"
                    )
                else:
                    return obs

        prompt = SYNTHESIZE_PROMPT.format(
            query=scratchpad.query,
            scratchpad=scratchpad.render_full(plan),
        )
        try:
            raw = await llm.invoke_nothink(
                [{"role": "user", "content": prompt}],
                max_tokens=1200,
                temperature=0.1,
            )
        except Exception as e:
            return f"[synthesis failed: {e}]\n\n" + scratchpad.render_full(plan)

        # Defense-in-depth: even with the tightened prompt, the model
        # occasionally leaks scaffold metadata. Strip it here.
        cleaned, n_redactions = sanitize_synthesis(raw or "")
        if n_redactions > 0:
            logger.info(
                "[synthesis] sanitized %d scaffold-leak pattern(s) from final answer",
                n_redactions,
            )
        return cleaned

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def _iter_tools(self):
        """Yield (name, tool) for either a ToolRegistry (._tools dict) or a plain dict."""
        if self.tools is None:
            return
        registry = getattr(self.tools, "_tools", None)
        if registry is not None:
            for name, tool in registry.items():
                yield name, tool
            return
        try:
            for name in self.tools.keys():
                yield name, self.tools.get(name) if hasattr(self.tools, "get") else None
        except Exception:
            return

    def _tools_brief(self) -> str:
        """One-line tool list for planning."""
        if self.tools is None:
            return "(no tools available — reasoning only)"
        items = [f"  - {name}" for name, _ in self._iter_tools()]
        return "\n".join(items) if items else "(no tools)"

    def _tools_schema(self) -> str:
        """Compact tool schema for action prompts."""
        if self.tools is None:
            return "(none)"
        lines = []
        for name, tool in self._iter_tools():
            desc = getattr(tool, "description", "") if tool else ""
            lines.append(f"  {name}: {desc[:80]}")
        return "\n".join(lines) if lines else "(none)"

    def _most_consistent(self, actions: list[dict]) -> dict:
        """Pick the most-repeated action shape from N samples (self-consistency).

        Equality on (action_type, normalized args / answer text). For ties,
        return the first.
        """
        if len(actions) == 1:
            return actions[0]
        keys: list[tuple] = []
        for a in actions:
            if "tool" in a:
                args_key = json.dumps(a.get("args", {}), sort_keys=True)[:200]
                keys.append(("tool", a["tool"], args_key))
            elif "answer" in a:
                # Normalize whitespace for comparison
                normalized = re.sub(r"\s+", " ", str(a["answer"])).strip()[:200]
                keys.append(("answer", normalized))
            else:
                keys.append(("unknown", json.dumps(a, sort_keys=True)[:200]))
        counts = Counter(keys)
        winner_key = counts.most_common(1)[0][0]
        for a, k in zip(actions, keys):
            if k == winner_key:
                return a
        return actions[0]

    async def _try_create_tool(self, tool_name: str, args: dict) -> bool:
        """Generate, validate, and register a custom Python tool on demand.

        When the agent's planner asks for a tool that doesn't exist, this
        invokes the LLM to draft a pure-Python function, runs it through the
        safety sandbox, registers it via CustomToolStore, and hot-loads it
        into the live registry.

        Constraints (per the existing custom_tools safety sandbox):
          - no network (httpx, requests, urllib are blocked)
          - no os/subprocess/eval/exec
          - no file I/O outside /tmp
          - pure logic only (math, string, parsing, transforms)

        Returns True on success, False otherwise.
        """
        from app.core.custom_tools import CustomToolStore, DynamicTool
        from app.core.brain import get_services

        try:
            svc = get_services()
            store = svc.custom_tools
            if store is None:
                logger.info("tool create skipped: CustomToolStore not initialized")
                return False
        except Exception as e:
            logger.info("tool create skipped: services not ready (%s)", e)
            return False

        # Draft the tool. Show the requested name + the args the agent wants
        # to pass — that's the contract the function must satisfy.
        param_keys = list(args.keys()) if isinstance(args, dict) else []
        draft_prompt = f"""Write a Python function for a NEW tool that doesn't exist yet.

REQUESTED TOOL NAME: {tool_name}
ARGS THE AGENT WILL PASS: {param_keys}

Requirements:
1. Function MUST be named `run` and accept ONLY the listed args as keyword arguments.
2. Function MUST return a string (use str(result) on the final value).
3. Use any imports you need — httpx, urllib, os, subprocess, json, re, math, etc. The owner has full system access; tools inherit it.
4. Handle errors gracefully — return a clear string explanation rather than raising on expected failures.
5. Be efficient: time-bound network/disk operations (timeouts, max iterations).

Output JSON:
{{"description": "what it does, 1 sentence", "parameters": [{{"name":"x","type":"int","description":"..."}}], "code": "def run(x: int) -> str:\\n    return str(...)"}}

Only refuse if the request is genuinely impossible (e.g. requires a model the system doesn't have access to). To refuse, output: {{"refuse": "reason"}}

JSON:"""
        try:
            resp = await llm.invoke_nothink(
                [{"role": "user", "content": draft_prompt}],
                max_tokens=600,
                json_mode=True,
                temperature=0.2,
            )
            obj = json.loads(resp) if resp.strip().startswith("{") else extract_json_object(resp)
            if not isinstance(obj, dict):
                return False
            if obj.get("refuse"):
                logger.info("tool create refused for %s: %s", tool_name, obj["refuse"])
                return False
            description = obj.get("description", "")
            parameters = obj.get("parameters", [])
            code = obj.get("code", "")
            if not description or not code:
                return False
        except Exception as e:
            logger.warning("tool draft parse failed: %s", e)
            return False

        try:
            tool_id = store.create_tool(
                name=tool_name,
                description=description,
                parameters=parameters,
                code=code,
            )
            if tool_id == -1:
                # CustomToolStore logged the rejection reason already
                return False
            record = store.get_tool(tool_name)
            if record is None:
                return False
            self.tools.register(DynamicTool(record, store))
            return True
        except Exception as e:
            logger.warning("tool register failed: %s", e)
            return False

    async def _emit(self, callback: Any, event_type: str, payload: dict) -> None:
        if callback is None:
            return
        try:
            res = callback(event_type, payload)
            if asyncio.iscoroutine(res):
                await res
        except Exception:
            logger.exception("agent event callback failed")
