"""Reflexion store — learn from failures automatically.

Lessons capture explicit user corrections ("actually, X not Y").
Reflexions capture silent failures: bad tool choices, hallucinations,
exhausted tool loops, low-quality answers. They're detected automatically
after each response, stored, and retrieved on similar future queries to
prevent repeating the same mistake patterns.

SQLite-only, keyword retrieval (same pattern as lessons + KG).
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass
from datetime import datetime

from app.config import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class Reflexion:
    id: int
    task_summary: str
    outcome: str          # "success" or "failure"
    reflection: str       # What went wrong / what to do differently
    quality_score: float  # 0.0–1.0
    tools_used: str       # Comma-separated tool names
    revision_count: int   # How many tool rounds were needed
    created_at: str


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
    if len(answer.strip()) < 30 and len(query) > 50:
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
                ({"calculate", "compute", "how much", "how many", "total"}, {"calculator", "code_exec"}),
                ({"search", "latest", "current", "today", "news", "price"}, {"web_search", "browser", "http_fetch"}),
                ({"remember", "recall", "what did", "what have"}, {"memory_query", "knowledge_search"}),
                ({"code", "script", "function", "print", "run"}, {"code_exec", "shell_exec"}),
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


# ---------------------------------------------------------------------------
# Stop words and normalization — shared via text_utils
# ---------------------------------------------------------------------------

from app.core.text_utils import STOP_WORDS as _STOP_WORDS  # noqa: E402
from app.core.text_utils import normalize_words as _base_normalize  # noqa: E402


def _normalize_words(text: str) -> set[str]:
    """Lowercase, strip punctuation, split into word set (min length 2)."""
    return _base_normalize(text, min_length=2)


# ---------------------------------------------------------------------------
# ReflexionStore
# ---------------------------------------------------------------------------

class ReflexionStore:
    """Stores and retrieves reflexions — experiential learning from failures.

    Uses hybrid keyword + vector (ChromaDB) search with RRF fusion.
    """

    _SCHEMA = """
    CREATE TABLE IF NOT EXISTS reflexions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        task_summary TEXT NOT NULL,
        outcome TEXT NOT NULL DEFAULT 'failure',
        reflection TEXT NOT NULL,
        quality_score REAL DEFAULT 0.5,
        tools_used TEXT DEFAULT '',
        revision_count INTEGER DEFAULT 0,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    CREATE INDEX IF NOT EXISTS idx_reflexions_outcome ON reflexions(outcome);
    CREATE INDEX IF NOT EXISTS idx_reflexions_quality ON reflexions(quality_score);
    """

    @property
    def MAX_REFLEXIONS(self) -> int:
        return config.MAX_REFLEXIONS

    def __init__(self, db):
        self._db = db
        self._collection = None
        for stmt in self._SCHEMA.strip().split(";"):
            stmt = stmt.strip()
            if stmt:
                self._db.execute(stmt)
        # Closure tracking: success-pattern A/B columns added via migration
        # so existing reflexions tables get them on first run.
        for col, typedef in [
            ("times_injected", "INTEGER DEFAULT 0"),
            ("quality_when_used_sum", "REAL DEFAULT 0.0"),
            ("quality_when_used_count", "INTEGER DEFAULT 0"),
        ]:
            try:
                self._db.execute(f"ALTER TABLE reflexions ADD COLUMN {col} {typedef}")
            except Exception:
                pass

    # --- ChromaDB vector collection ---

    def _get_collection(self):
        """Lazy-init ChromaDB collection for semantic reflexion search."""
        if self._collection is None:
            try:
                import chromadb
                from .embedding import open_collection
                client = chromadb.PersistentClient(path=config.CHROMADB_PATH)
                self._collection = open_collection(
                    client, "reflexions", reindex=self._backfill_collection,
                )
            except Exception as e:
                logger.warning("Failed to init reflexions ChromaDB collection: %s", e)
                return None
        return self._collection

    def _backfill_collection(self, collection) -> int:
        """Populate `collection` from the reflexions table, unconditionally.
        Shared by reindex_reflexions (guarded) and the embedder-rebuild path."""
        all_rows = self._db.fetchall("SELECT * FROM reflexions LIMIT 200")
        if not all_rows:
            return 0
        ids, documents, metadatas = [], [], []
        for row in all_rows:
            searchable = f"{row['task_summary']} {row['reflection']}".strip()
            if not searchable:
                continue
            ids.append(str(row["id"]))
            documents.append(searchable)
            metadatas.append({"outcome": row["outcome"], "quality_score": row["quality_score"]})
        if ids:
            collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
            logger.info("Reindexed %d reflexions into ChromaDB", len(ids))
        return len(ids)

    def reindex_reflexions(self) -> int:
        """One-time backfill of existing reflexions into ChromaDB. Returns count indexed."""
        collection = self._get_collection()
        if collection is None:
            return 0
        if collection.count() > 0:
            logger.info("Reflexions collection already has %d entries, skipping reindex", collection.count())
            return 0
        return self._backfill_collection(collection)

    def _add_to_vector(self, reflexion_id: int, task_summary: str, reflection: str, outcome: str, quality_score: float) -> None:
        """Add a single reflexion to the vector collection."""
        collection = self._get_collection()
        if collection is None:
            return
        try:
            searchable = f"{task_summary} {reflection}".strip()
            collection.add(
                ids=[str(reflexion_id)],
                documents=[searchable],
                metadatas=[{"outcome": outcome, "quality_score": quality_score}],
            )
        except Exception as e:
            logger.warning("Failed to add reflexion #%d to ChromaDB: %s", reflexion_id, e)

    def _remove_from_vector(self, reflexion_ids: list[int]) -> None:
        """Remove reflexions from vector collection."""
        collection = self._get_collection()
        if collection is None or not reflexion_ids:
            return
        try:
            collection.delete(ids=[str(rid) for rid in reflexion_ids])
        except Exception as e:
            logger.warning("Failed to remove reflexions from ChromaDB: %s", e)

    def store(
        self,
        task_summary: str,
        outcome: str,
        reflection: str,
        quality_score: float = 0.5,
        tools_used: list[str] | None = None,
        revision_count: int = 0,
        is_eval: bool = False,
    ) -> int:
        """Store a reflexion. Returns the reflexion ID.

        is_eval=True marks the reflexion as eval-harness-derived; dream's
        promote_reflexions step skips these so they never become lessons
        that re-contaminate retrieval. Tagging is_eval is the permanent
        fix for the deliberation bimodal pattern.
        """
        task_summary = task_summary.strip()[:500]
        reflection = reflection.strip()[:1000]
        outcome = outcome.strip().lower()
        if outcome not in ("success", "failure"):
            outcome = "failure"
        tools_str = ",".join(tools_used) if tools_used else ""

        # Dedup: don't store near-identical reflexions
        if self._is_duplicate(task_summary, reflection):
            return -1

        cursor = self._db.execute(
            "INSERT INTO reflexions "
            "(task_summary, outcome, reflection, quality_score, tools_used, revision_count, is_eval) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (task_summary, outcome, reflection, quality_score, tools_str, revision_count,
             1 if is_eval else 0),
        )

        reflexion_id = cursor.lastrowid
        self._add_to_vector(reflexion_id, task_summary, reflection, outcome, quality_score)
        self._prune()
        logger.info(
            "Stored reflexion #%d: outcome=%s, quality=%.2f, is_eval=%s",
            reflexion_id, outcome, quality_score, is_eval,
        )
        return reflexion_id

    @staticmethod
    def _rrf_fuse(keyword_ids: list[int], vector_ids: list[int], k: int = 60) -> list[int]:
        """Reciprocal Rank Fusion of two ranked ID lists."""
        scores: dict[int, float] = {}
        for rank, rid in enumerate(keyword_ids):
            scores[rid] = scores.get(rid, 0.0) + 1.0 / (k + rank + 1)
        for rank, rid in enumerate(vector_ids):
            scores[rid] = scores.get(rid, 0.0) + 1.0 / (k + rank + 1)
        return sorted(scores, key=lambda x: scores[x], reverse=True)

    def get_relevant(self, query: str, limit: int = 3, failures_only: bool = True, successes_only: bool = False) -> list[Reflexion]:
        """Get reflexions relevant to a query by hybrid keyword + vector search.

        By default returns only failures (useful as warnings).  Set
        failures_only=False to also retrieve success patterns, or
        successes_only=True to retrieve only successes.
        """
        outcome_filter = None
        if successes_only:
            outcome_filter = "success"
        elif failures_only:
            outcome_filter = "failure"

        # --- Keyword search ---
        # Successes: sort DESC (best first). Failures: sort ASC (worst first).
        sort_order = "DESC" if successes_only else "ASC"
        if outcome_filter:
            rows = self._db.fetchall(
                f"SELECT * FROM reflexions WHERE outcome = ? ORDER BY quality_score {sort_order} LIMIT 200",
                (outcome_filter,),
            )
        else:
            rows = self._db.fetchall(
                f"SELECT * FROM reflexions ORDER BY quality_score {sort_order} LIMIT 200"
            )
        rows_by_id = {row["id"]: row for row in rows}

        query_words = _normalize_words(query)
        keyword_scored: list[tuple[int, int]] = []
        if query_words:
            for row in rows:
                task_words = _normalize_words(row["task_summary"])
                reflection_words = _normalize_words(row["reflection"])
                all_words = task_words | reflection_words
                overlap = len(query_words & all_words)
                if overlap >= 2:
                    keyword_scored.append((overlap, row["id"]))
            keyword_scored.sort(key=lambda x: -x[0])
        keyword_ids = [rid for _, rid in keyword_scored[:limit * 3]]

        # --- Vector search via ChromaDB ---
        vector_ids: list[int] = []
        collection = self._get_collection()
        if collection and collection.count() > 0:
            try:
                where = {"outcome": outcome_filter} if outcome_filter else None
                results = collection.query(
                    query_texts=[query],
                    n_results=min(limit * 3, collection.count()),
                    where=where,
                )
                if results and results.get("ids"):
                    vector_ids = [int(rid) for rid in results["ids"][0]]
            except Exception as e:
                logger.debug("Reflexion vector search failed: %s", e)

        # --- RRF fusion ---
        if keyword_ids or vector_ids:
            fused_ids = self._rrf_fuse(keyword_ids, vector_ids)
        else:
            return []

        # Fetch any vector-only results not in our rows_by_id
        missing = [rid for rid in fused_ids if rid not in rows_by_id]
        if missing:
            placeholders = ",".join("?" * len(missing))
            extra = self._db.fetchall(
                f"SELECT * FROM reflexions WHERE id IN ({placeholders})", tuple(missing)
            )
            for row in extra:
                rows_by_id[row["id"]] = row

        result_reflexions = []
        for rid in fused_ids:
            if rid not in rows_by_id:
                continue
            row = rows_by_id[rid]
            # Apply outcome filter for vector-only results
            if outcome_filter and row["outcome"] != outcome_filter:
                continue
            result_reflexions.append(Reflexion(
                id=row["id"],
                task_summary=row["task_summary"],
                outcome=row["outcome"],
                reflection=row["reflection"],
                quality_score=row["quality_score"],
                tools_used=row["tools_used"],
                revision_count=row["revision_count"],
                created_at=row["created_at"],
            ))
            if len(result_reflexions) >= limit:
                break

        return result_reflexions

    def get_success_patterns(self, query: str, limit: int = 2) -> list[Reflexion]:
        """Get relevant success reflexions for positive reinforcement.

        Side effect: increments times_injected on the returned rows so we can
        later correlate injection with downstream quality (A/B closure).
        """
        results = self.get_relevant(query, limit=limit, failures_only=False, successes_only=True)
        if results:
            ids = tuple(r.id for r in results if getattr(r, "id", None) is not None)
            if ids:
                try:
                    placeholders = ",".join("?" for _ in ids)
                    self._db.execute(
                        f"UPDATE reflexions SET times_injected = times_injected + 1 "
                        f"WHERE id IN ({placeholders})",
                        ids,
                    )
                except Exception as e:
                    logger.debug("times_injected update failed: %s", e)
        return results

    def record_post_quality_for_injected(
        self, injected_ids: list[int], quality_score: float
    ) -> None:
        """Record the post-response quality for success patterns that were injected.

        Builds the running average needed to filter out patterns whose injection
        correlates with low downstream quality.
        """
        if not injected_ids or quality_score is None:
            return
        try:
            placeholders = ",".join("?" for _ in injected_ids)
            self._db.execute(
                f"UPDATE reflexions SET "
                f"  quality_when_used_sum = quality_when_used_sum + ?, "
                f"  quality_when_used_count = quality_when_used_count + 1 "
                f"WHERE id IN ({placeholders})",
                (float(quality_score), *injected_ids),
            )
        except Exception as e:
            logger.debug("record_post_quality_for_injected failed: %s", e)

    def get_useless_success_patterns(self, min_uses: int = 5,
                                       max_avg_quality: float = 0.5) -> list[int]:
        """Return ids of success patterns whose injection correlates with low
        downstream quality (>=min_uses observations, avg quality < threshold).
        Caller can demote/delete these.
        """
        try:
            rows = self._db.fetchall(
                "SELECT id, quality_when_used_sum, quality_when_used_count "
                "FROM reflexions "
                "WHERE outcome = 'success' AND quality_when_used_count >= ? "
                "AND (quality_when_used_sum / quality_when_used_count) < ?",
                (int(min_uses), float(max_avg_quality)),
            )
            return [int(r["id"]) for r in rows]
        except Exception as e:
            logger.debug("get_useless_success_patterns failed: %s", e)
            return []

    @staticmethod
    def format_success_patterns(reflexions: list[Reflexion]) -> str:
        """Format success reflexions as a prompt-ready block."""
        if not reflexions:
            return ""
        lines = []
        for r in reflexions:
            tools = f" (tools: {r.tools_used})" if r.tools_used else ""
            lines.append(f"- {r.task_summary[:100]}{tools}: {r.reflection}")
        return "\n".join(lines)

    @staticmethod
    def format_for_prompt(reflexions: list[Reflexion]) -> str:
        """Format reflexions as a prompt-ready warning block."""
        if not reflexions:
            return ""
        lines = []
        for r in reflexions:
            tools = f" (tools: {r.tools_used})" if r.tools_used else ""
            label = "Previous failure" if r.outcome == "failure" else "Previous success"
            lines.append(f"- {label}{tools}: {r.reflection}")
        return "\n".join(lines)

    def get_recent(self, limit: int = 50) -> list[Reflexion]:
        """Get most recent reflexions."""
        rows = self._db.fetchall(
            "SELECT id, task_summary, outcome, reflection, quality_score, "
            "tools_used, revision_count, created_at "
            "FROM reflexions ORDER BY created_at DESC LIMIT ?",
            (limit,),
        )
        return [
            Reflexion(
                id=r["id"],
                task_summary=r["task_summary"],
                outcome=r["outcome"],
                reflection=r["reflection"],
                quality_score=r["quality_score"],
                tools_used=r["tools_used"],
                revision_count=r["revision_count"],
                created_at=r["created_at"],
            )
            for r in rows
        ]

    def get_stats(self) -> dict:
        """Return reflexion statistics."""
        total = self._db.fetchone("SELECT COUNT(*) AS c FROM reflexions")["c"]
        failures = self._db.fetchone(
            "SELECT COUNT(*) AS c FROM reflexions WHERE outcome = 'failure'"
        )["c"]
        avg_quality = self._db.fetchone(
            "SELECT AVG(quality_score) AS avg FROM reflexions"
        )
        return {
            "total_reflexions": total,
            "failures": failures,
            "successes": total - failures,
            "avg_quality": round(avg_quality["avg"], 2) if avg_quality["avg"] else 0.0,
        }

    def decay_stale(self, days: int | None = None, decay_amount: float | None = None) -> int:
        """Lower quality score on old reflexions so they fade out. Returns count affected."""
        if days is None:
            days = config.REFLEXION_DECAY_DAYS
        if decay_amount is None:
            decay_amount = config.REFLEXION_DECAY_AMOUNT
        # Decay failures at full rate
        cursor_fail = self._db.execute(
            "UPDATE reflexions SET quality_score = MAX(0.0, quality_score - ?) "
            "WHERE created_at < datetime('now', ?) AND outcome = 'failure'",
            (decay_amount, f"-{days} days"),
        )
        # Decay successes at half rate (they remain useful longer)
        cursor_success = self._db.execute(
            "UPDATE reflexions SET quality_score = MAX(0.0, quality_score - ?) "
            "WHERE created_at < datetime('now', ?) AND outcome = 'success'",
            (decay_amount / 2, f"-{days} days"),
        )
        return cursor_fail.rowcount + cursor_success.rowcount

    def _is_duplicate(self, task_summary: str, reflection: str) -> bool:
        """Check if a near-identical reflexion already exists (Jaccard >= 0.6).

        Threshold lowered from 0.8 to 0.6 to catch more near-duplicate
        reflexions that are phrased slightly differently but convey the same lesson.
        """
        new_words = _normalize_words(task_summary + " " + reflection) - _STOP_WORDS
        if len(new_words) < 3:
            return False

        recent = self._db.fetchall(
            "SELECT task_summary, reflection FROM reflexions ORDER BY created_at DESC LIMIT 20"
        )
        for row in recent:
            existing_words = _normalize_words(
                row["task_summary"] + " " + row["reflection"]
            ) - _STOP_WORDS
            if not existing_words:
                continue
            overlap = len(new_words & existing_words)
            union = len(new_words | existing_words)
            if union > 0 and overlap / union >= 0.6:
                return True
        return False

    def find_recurring_failures(self, task_summary: str, threshold: float = 0.4, min_count: int = 3) -> list[Reflexion]:
        """Find similar past failures using Jaccard similarity on task_summary."""
        new_words = _normalize_words(task_summary) - _STOP_WORDS
        if len(new_words) < 2:
            return []

        failures = self._db.fetchall(
            "SELECT * FROM reflexions WHERE outcome = 'failure' ORDER BY created_at DESC LIMIT 100"
        )

        similar = []
        for row in failures:
            existing_words = _normalize_words(row["task_summary"]) - _STOP_WORDS
            if not existing_words:
                continue
            overlap = len(new_words & existing_words)
            union = len(new_words | existing_words)
            if union > 0 and overlap / union >= threshold:
                similar.append(Reflexion(
                    id=row["id"],
                    task_summary=row["task_summary"],
                    outcome=row["outcome"],
                    reflection=row["reflection"],
                    quality_score=row["quality_score"],
                    tools_used=row["tools_used"],
                    revision_count=row["revision_count"],
                    created_at=row["created_at"],
                ))

        return similar if len(similar) >= min_count else []

    def _prune(self) -> None:
        """If reflexions exceed MAX_REFLEXIONS, delete oldest low-quality ones."""
        count = self._db.fetchone("SELECT COUNT(*) AS c FROM reflexions")["c"]
        if count <= self.MAX_REFLEXIONS:
            return
        excess = count - self.MAX_REFLEXIONS
        # Collect IDs before deleting so we can sync ChromaDB
        pruned = self._db.fetchall(
            "SELECT id FROM reflexions ORDER BY quality_score ASC, created_at ASC LIMIT ?",
            (excess,),
        )
        pruned_ids = [row["id"] for row in pruned]
        if pruned_ids:
            placeholders = ",".join("?" for _ in pruned_ids)
            with self._db.transaction() as tx:
                tx.execute(
                    f"DELETE FROM reflexions WHERE id IN ({placeholders})",
                    tuple(pruned_ids),
                )
            self._remove_from_vector(pruned_ids)
        logger.info("Pruned %d reflexions (over %d limit)", excess, self.MAX_REFLEXIONS)


# ---------------------------------------------------------------------------
# Recurring failure promotion — auto-create lessons from repeated failures
# ---------------------------------------------------------------------------

async def check_recurring_failures(task_summary: str, learning_engine) -> None:
    """Check if similar failures have recurred 3+ times and auto-promote to a lesson.

    Uses the ReflexionStore from the active services to find patterns.
    """
    from app.core.brain import get_services
    from app.core import llm as llm_mod

    svc = get_services()
    if not svc.reflexions:
        return

    similar = svc.reflexions.find_recurring_failures(task_summary)
    if not similar:
        return

    # Collect the failure reflections for LLM synthesis
    reflections = [r.reflection for r in similar[:5]]
    task_summaries = [r.task_summary for r in similar[:3]]

    prompt = (
        "These failures keep recurring:\n\n"
        + "\n".join(f"- Task: {ts}\n  Failure: {ref}" for ts, ref in zip(task_summaries, reflections))
        + "\n\nWrite a concise lesson (1-2 sentences) that would prevent this failure pattern. "
        "Format as JSON: {\"topic\": \"brief topic\", \"lesson\": \"what to do differently\"}"
    )

    try:
        raw = await asyncio.wait_for(
            llm_mod.invoke_nothink(
                [{"role": "user", "content": prompt}],
                json_mode=True,
                json_prefix="{",
                max_tokens=200,
                temperature=0.2,
            ),
            timeout=config.INTERNAL_LLM_TIMEOUT,
        )
        obj = llm_mod.extract_json_object(raw)
        if not obj or "lesson" not in obj:
            return

        lesson_id = learning_engine.add_knowledge_lesson(
            topic=obj.get("topic", task_summary[:100]),
            correct_answer=obj["lesson"],
            lesson_text=f"Auto-lesson: {obj['lesson']}",
            context=task_summaries[0] if task_summaries else "",
        )
        logger.info(
            "Recurring failure promoted to lesson #%d: '%s' (from %d failures)",
            lesson_id, obj.get("topic", "")[:60], len(similar),
        )
    except Exception as e:
        logger.debug("Recurring failure promotion failed: %s", e)
