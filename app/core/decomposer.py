"""Multi-agent decomposition heuristic — decides whether and how to split a query.

No LLM calls anywhere in this module.  All signals are pure regex/heuristic,
the same philosophy as should_plan() in planning.py.

Public API:
  should_decompose(query, intent, was_planned, ephemeral) -> bool
  decompose_query(query, intent, was_planned, plan, conv_id) -> DecompositionPlan | None
"""

from __future__ import annotations

import logging
import re
import uuid

from app.config import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Signal patterns  (compiled once at module load)
# ---------------------------------------------------------------------------

# +2: Explicit parallelism — "compare X and Y", "versus", "side by side", etc.
_PARALLEL_MARKERS = re.compile(
    r"\b(?:compare|comparing|comparison|contrast(?:ing)?|"
    r"both\b.{1,40}\band\b|simultaneously|in\s+parallel|"
    r"at\s+the\s+same\s+time|side[\s-]*by[\s-]*side|"
    r"versus|vs\.?)\b",
    re.IGNORECASE | re.DOTALL,
)

# +2: Explicit delegation words
_DELEGATION_WORDS = re.compile(
    r"\b(?:assign|split\s+(?:the\s+)?task|parallel\s+agents?|"
    r"multiple\s+agents?|break\s+(?:this\s+)?(?:down|up)\s+into|"
    r"run\s+(?:them\s+)?in\s+parallel)\b",
    re.IGNORECASE,
)

# +1: Multiple question marks  (simple: at least two '?' in the query)
_MULTI_QUESTION = re.compile(r"\?.*\?", re.DOTALL)

# +1: Multiple tool-type keywords (search, calculate, fetch…) when query > 200 chars
_TOOL_SIGNALS = re.compile(
    r"\b(?:search|find|look\s+up|fetch|latest|current|news|price|"
    r"calculate|compute|how\s+much|total|convert|compare|check|retrieve)\b",
    re.IGNORECASE,
)

# Proper-noun candidates for entity detection (+1 when ≥ 3 unique entities)
_PROPER_NOUN = re.compile(r"\b[A-Z][a-zA-Z]{1,}\b")
_STOPWORDS = frozenset({
    "I", "A", "An", "The", "In", "On", "At", "By", "For", "Of", "To",
    "And", "Or", "But", "Not", "Is", "Are", "Was", "Were", "Be", "Been",
    "What", "How", "Why", "When", "Where", "Which", "Who", "That", "This",
    "These", "Those", "With", "From", "Into", "About", "Over", "Under",
    "After", "Before", "Between", "Across", "During", "While", "Since",
    "My", "Your", "His", "Her", "Their", "Our", "Its",
    "Search", "Find", "Get", "Tell", "Show", "Give", "Calculate", "Do",
    "Please", "Can", "Could", "Would", "Should", "Will", "Let", "Make",
    # Imperative verbs that typically start multi-entity queries — these are
    # not entities themselves and must not become sub-agent roles.
    "Compare", "Contrast", "List", "Explain", "Describe", "Summarize",
    "Analyze", "Research", "Review", "Evaluate", "Rank", "Rate",
})

# Sequential-strategy markers (if detected, prefer sequential over parallel).
# Broadened to catch "first X ... then Y" with any verbs between, plus the
# specific short-form markers we originally tracked.
_SEQUENTIAL_MARKERS = re.compile(
    r"\bfirst\b[\s\S]{1,200}?\bthen\b"                       # "first ... then ..."
    r"|\bstep\s+by\s+step\b"
    r"|\bin\s+(?:that\s+)?order\b"
    r"|\bafter\s+(?:that|finding|searching)\b"
    r"|\bbased\s+on\s+(?:(?:the|that|this|those|these)\s+)?(?:result|output|answer|finding|data)\b"
    r"|\busing\s+the\s+(?:result|output|data|finding)s?\s+(?:from|of)\b"
    r"|\bone\s+by\s+one\b",
    re.IGNORECASE,
)

# Patterns for splitting a compound "compare X <connector> Y" query into two
# sub-queries. Original (2026-03-23) only matched "and"; broadened 2026-05-13
# to also catch "to / with / versus / vs" so "Compare Tokyo to Osaka weather"
# now scores the +2 that pushes it over the trigger threshold. The eval
# `multi_agent_no_decompose` task is the safety net against over-firing.
_COMPARE_AND = re.compile(
    r"(?:compare|contrast|difference\s+between|differences?\s+between)\s+"
    r"(.+?)\s+(?:and|to|with|versus|vs\.?)\s+(.+?)(?:\s*[,;:]|\s+(?:in|for|over|during|by|from|as)\b|$)",
    re.IGNORECASE,
)

# "what is X and Y" / "tell me about X and Y"
_X_AND_Y = re.compile(
    r"(?:what\s+(?:is|are)|tell\s+me\s+about|explain|describe|give\s+me)\s+"
    r"(.+?)\s+and\s+(.+?)(?:\?|$)",
    re.IGNORECASE,
)


# Definitional "what is the difference between X and Y" — short queries that
# look like compares but are textbook one-pass definitions. Decomposing these
# spawns redundant sub-agents that each do a basic lookup; net effect is a
# 120s+ timeout for what should be a 10s answer.
_DEFINITIONAL_COMPARE_RE = re.compile(
    r"^\s*(?:what\s+(?:is|are)|how\s+(?:does|do))\s+the\s+"
    r"(?:difference|distinction|contrast|relationship)\s+between\s+",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def should_decompose(
    query: str,
    intent: str,
    was_planned: bool,
    ephemeral: bool,  # kept for API clarity; not a hard gate — depth guards it
) -> bool:
    """Decide whether structural multi-agent decomposition should fire.

    All checks are pure heuristic — no LLM call.  Returns True only when
    ALL gates pass AND signal score >= MULTI_AGENT_TRIGGER_THRESHOLD.
    """
    from app.core.agent_spawner import _structural_depth

    # Gate 1: Feature flag
    if not config.ENABLE_MULTI_AGENT:
        return False

    # Gate 2: Recursion depth cap — sub-agents can spawn sub-sub-agents up to MAX_STRUCTURAL_DEPTH.
    # Default depth=2 allows compare-of-compares and multi-step research to recurse.
    if _structural_depth.get() >= config.MAX_STRUCTURAL_DEPTH:
        return False

    # Gate 3: Trivial intents that cannot benefit from decomposition
    if intent in ("greeting", "correction"):
        return False

    # Gate 3b: Definitional "what is the difference between X and Y" queries.
    # These look like compares but are answerable in one pass — decomposing them
    # spawns 2 sub-agents that each do a definition lookup → 120s+ timeout for
    # what should be a 10s answer. Skip decomposition on short definitional
    # comparisons.
    if _DEFINITIONAL_COMPARE_RE.search(query) and len(query) < 200:
        logger.debug("Decomposition skipped: definitional comparison short query: %r", query[:80])
        return False

    score = _score_signals(query, was_planned)
    if score < config.MULTI_AGENT_TRIGGER_THRESHOLD:
        return False

    logger.debug(
        "Decomposition gate: score=%d >= threshold=%d for query=%r",
        score, config.MULTI_AGENT_TRIGGER_THRESHOLD, query[:80],
    )
    return True


def decompose_query(
    query: str,
    intent: str,
    was_planned: bool,
    plan: dict | None,
    conversation_id: str,
) -> "DecompositionPlan | None":
    """Build a DecompositionPlan from a query.

    No LLM call — purely heuristic.
    Returns None if extraction cannot produce at least 2 meaningful tasks
    (decomposition will be skipped and the normal path used instead).
    """
    from app.core.agent_spawner import DecompositionPlan

    strategy = _pick_strategy(query)
    tasks = _extract_tasks(query, strategy, conversation_id)

    if not tasks or len(tasks) < 2:
        logger.debug("Decomposition produced <2 tasks for query=%r — skipping", query[:80])
        return None

    # Cap at MAX_AGENT_COUNT
    tasks = tasks[: config.MAX_AGENT_COUNT]

    merge_instruction = _build_merge_instruction(query, tasks, strategy)

    return DecompositionPlan(
        strategy=strategy,
        tasks=tasks,
        merge_instruction=merge_instruction,
        max_parallel=min(len(tasks), config.MAX_PARALLEL_AGENTS),
    )


# ---------------------------------------------------------------------------
# Signal scoring
# ---------------------------------------------------------------------------

def _score_signals(query: str, was_planned: bool) -> int:
    """Compute the decomposition signal score (0 = definitely don't decompose)."""
    score = 0

    if _PARALLEL_MARKERS.search(query):
        score += 2

    # "Compare X and Y" / "difference between X and Y" — a structural compare
    # pattern that's a near-certain decomposable. Adds +2 on top of the
    # generic parallel marker so total reaches the threshold for these
    # canonical queries.
    if _COMPARE_AND.search(query) or _X_AND_Y.search(query):
        score += 2

    if _DELEGATION_WORDS.search(query):
        score += 2

    # Sequential markers imply a multi-step workflow ("first X, then Y") —
    # decompose into ordered sub-tasks. Score +2 to match parallel markers.
    if _SEQUENTIAL_MARKERS.search(query):
        score += 2

    # Explicit user-authored numbered/lettered groups: "(1) X (2) Y (3) Z" or
    # "(A) ... (B) ..." — structural signal but only meaningful when combined
    # with other complexity. Pure simple-lookup multi-questions get handled
    # better by parallel tool calls in a single brain.think() than by
    # spawning sub-agents (which compete for GPU and confuse synthesis).
    # Conservative: +1 per group beyond the first, capped at +2 total.
    group_markers = _NUMBERED_GROUP_RE.findall(query)
    n_groups = len(group_markers)
    if n_groups >= 2:
        score += min(2, n_groups - 1)

    # ≥ 3 distinct proper-noun candidates
    nouns = {w for w in _PROPER_NOUN.findall(query) if w not in _STOPWORDS}
    if len(nouns) >= 3:
        score += 1

    if _MULTI_QUESTION.search(query):
        score += 1

    if len(query) > 80:
        tool_count = len(_TOOL_SIGNALS.findall(query))
        if tool_count >= 2:
            score += 1

    if was_planned:
        score += 1

    return score


# ---------------------------------------------------------------------------
# Strategy selection
# ---------------------------------------------------------------------------

def _pick_strategy(query: str) -> str:
    if _SEQUENTIAL_MARKERS.search(query):
        return "sequential"
    # map-reduce would require a reduce task explicitly; keep parallel as default
    return "parallel"


# ---------------------------------------------------------------------------
# Task extraction
# ---------------------------------------------------------------------------

def _extract_tasks(
    query: str,
    strategy: str,
    conversation_id: str,
) -> "list[AgentTask] | None":
    """Try extraction methods in priority order, return first success.

    Numbered/lettered group split runs FIRST when the query has explicit
    structure like "(1) ... (2) ..." or "(A) ... (B) ..." — these are
    user-authored groupings and should be respected as primary boundaries
    even when entity-split would otherwise win on noun count.

    Entity-split runs second when the query has 3+ proper nouns — otherwise
    compare/and-split truncates multi-entity lists into a broken 2-agent plan.
    """
    # Honor user-authored grouping FIRST (compound compares, recursive cases)
    grouped = _try_grouped_split(query, conversation_id)
    if grouped:
        return grouped

    entity_result = _try_entity_split(query, conversation_id)
    if entity_result and len(entity_result) >= 3:
        return entity_result
    return (
        _try_compare_split(query, conversation_id)
        or _try_and_split(query, conversation_id)
        or entity_result
        or _try_fallback_split(query, conversation_id)
    )


# Patterns for explicit user grouping that should override entity-split.
# Matches "(1)" "(2)" or "(A)" "(B)" or "(i)" "(ii)" markers.
_NUMBERED_GROUP_RE = re.compile(
    r"\(\s*(?:\d+|[a-iA-I]|[ivxIVX]+)\s*\)",
)


def _try_grouped_split(
    query: str, conv_id: str
) -> "list[AgentTask] | None":
    """Split on explicit user-authored numbered/lettered groups.

    Examples that match:
      "Compare (1) X vs Y; (2) A vs B" -> 2 sub-tasks
      "For each: (A) X (B) Y (C) Z"     -> 3 sub-tasks

    Each segment becomes its own sub-agent. Recursive: each sub-agent's query
    contains the original framing, so it can itself decompose if it's a
    compare/and structure under MAX_STRUCTURAL_DEPTH.
    """
    markers = list(_NUMBERED_GROUP_RE.finditer(query))
    if len(markers) < 2:
        return None
    # Cap segments to MAX_AGENT_COUNT
    if len(markers) > config.MAX_AGENT_COUNT:
        markers = markers[: config.MAX_AGENT_COUNT]
    # Build segments from marker positions to next marker (or end)
    segments: list[str] = []
    for i, m in enumerate(markers):
        start = m.end()
        end = markers[i + 1].start() if i + 1 < len(markers) else len(query)
        seg = query[start:end].strip(" :;.,-")
        if len(seg) >= 10:
            segments.append(seg)
    if len(segments) < 2:
        return None

    t = config.AGENT_TASK_TIMEOUT
    tasks = []
    for i, seg in enumerate(segments):
        tasks.append(_make_task(
            str(uuid.uuid4()), conv_id,
            f"group-{i+1}-researcher",
            seg,  # the segment becomes the sub-agent's query — may itself decompose
            f"Address only group ({i+1}) of the original request: {seg[:80]}",
            t, [f"group-{i+1}"],
        ))
    return tasks


def _make_task(
    task_id: str,
    parent_id: str,
    role: str,
    query: str,
    focus: str,
    timeout: int,
    tags: list[str],
) -> "AgentTask":
    from app.core.agent_spawner import AgentTask
    return AgentTask(
        task_id=task_id,
        parent_run_id=parent_id,
        role=role,
        query=query,
        focus=focus,
        context_facts=[],
        context_lessons=[],
        shared_findings={},
        depth=1,
        timeout=timeout,
        tags=tags,
    )


def _try_compare_split(
    query: str, conv_id: str
) -> "list[AgentTask] | None":
    """'compare X and Y' → two focused sub-queries."""
    m = _COMPARE_AND.search(query)
    if not m:
        return None
    subj_a = m.group(1).strip()[:40]
    subj_b = m.group(2).strip()[:40]
    if not subj_a or not subj_b or subj_a.lower() == subj_b.lower():
        return None
    t = config.AGENT_TASK_TIMEOUT
    return [
        _make_task(str(uuid.uuid4()), conv_id,
                   f"{subj_a.lower().replace(' ','-')}-researcher",
                   f"Research {subj_a}: {query}",
                   f"Focus only on {subj_a}.", t, ["parallel-1"]),
        _make_task(str(uuid.uuid4()), conv_id,
                   f"{subj_b.lower().replace(' ','-')}-researcher",
                   f"Research {subj_b}: {query}",
                   f"Focus only on {subj_b}.", t, ["parallel-2"]),
    ]


def _try_and_split(
    query: str, conv_id: str
) -> "list[AgentTask] | None":
    """'what is X and Y' → two focused sub-queries."""
    m = _X_AND_Y.search(query)
    if not m:
        return None
    subj_a = m.group(1).strip()[:40]
    subj_b = m.group(2).strip()[:40]
    if not subj_a or not subj_b or subj_a.lower() == subj_b.lower():
        return None
    t = config.AGENT_TASK_TIMEOUT
    return [
        _make_task(str(uuid.uuid4()), conv_id,
                   f"{subj_a.lower().replace(' ','-')}-researcher",
                   f"Answer specifically for {subj_a}: {query}",
                   f"Focus only on {subj_a}.", t, ["parallel-1"]),
        _make_task(str(uuid.uuid4()), conv_id,
                   f"{subj_b.lower().replace(' ','-')}-researcher",
                   f"Answer specifically for {subj_b}: {query}",
                   f"Focus only on {subj_b}.", t, ["parallel-2"]),
    ]


def _try_entity_split(
    query: str, conv_id: str
) -> "list[AgentTask] | None":
    """3+ unique proper nouns → one task per noun (up to MAX_AGENT_COUNT)."""
    seen: set[str] = set()
    entities: list[str] = []
    for w in _PROPER_NOUN.findall(query):
        if w in _STOPWORDS or len(w) <= 2:
            continue
        wl = w.lower()
        if wl not in seen:
            seen.add(wl)
            entities.append(w)
    if len(entities) < 3:
        return None

    entities = entities[: config.MAX_AGENT_COUNT]
    t = config.AGENT_TASK_TIMEOUT
    tasks = []
    for i, entity in enumerate(entities):
        tasks.append(_make_task(
            str(uuid.uuid4()), conv_id,
            f"{entity.lower().replace(' ','-')}-researcher",
            f"Answer specifically for {entity}: {query}",
            f"Focus only on {entity}.", t, [f"parallel-{i+1}"],
        ))
    return tasks


def _try_fallback_split(
    query: str, conv_id: str
) -> "list[AgentTask] | None":
    """Split on 'and' conjunctions as a last resort; only if both halves are ≥ 15 chars."""
    parts = re.split(r"\s+and\s+", query, maxsplit=2, flags=re.IGNORECASE)
    if len(parts) < 2:
        return None
    parts = [p.strip() for p in parts if len(p.strip()) >= 15]
    if len(parts) < 2:
        return None

    t = config.AGENT_TASK_TIMEOUT
    tasks = []
    for i, part in enumerate(parts[: config.MAX_AGENT_COUNT]):
        tasks.append(_make_task(
            str(uuid.uuid4()), conv_id, f"researcher-{i+1}",
            part, "Answer this sub-question.", t, [f"parallel-{i+1}"],
        ))
    return tasks


# ---------------------------------------------------------------------------
# Merge instruction builder
# ---------------------------------------------------------------------------

def _build_merge_instruction(
    query: str,
    tasks: "list[AgentTask]",
    strategy: str,
) -> str:
    from app.core.prompt_optimizer import get_active_module
    roles = ", ".join(t.role for t in tasks)
    if strategy == "sequential":
        template = (
            get_active_module("merge_instruction_sequential")
            or "Synthesize the sequentially gathered findings ({roles}) into a "
               "complete, coherent answer. Original question: {query}"
        )
        return template.format(roles=roles, query=query)
    elif strategy == "map-reduce":
        map_roles = ", ".join(t.role for t in tasks[:-1])
        return (
            f"The map agents ({map_roles}) researched sub-topics. "
            f"Synthesize their findings into a comprehensive answer. "
            f"Original question: {query}"
        )
    else:  # parallel
        template = (
            get_active_module("merge_instruction_parallel")
            or "Synthesize the parallel research findings ({roles}) into a "
               "single, direct, coherent answer. Original question: {query}"
        )
        return template.format(roles=roles, query=query)
