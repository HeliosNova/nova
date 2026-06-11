"""Dream Consolidation Engine — memory consolidation during idle time.

Inspired by Claude Code's autoDream (KAIROS feature).
4-phase pipeline:
  Phase 1: ORIENT  — read-only inventory of all memory stores
  Phase 2: GATHER  — scan for consolidation targets (stale, overlapping, broken)
  Phase 3: CONSOLIDATE — cross-system dedup, contradiction resolution, promotions, DPO mining
  Phase 4: PRUNE & REPORT — delete low-value items, generate digest
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class DreamInventory:
    """Phase 1 output: snapshot of all memory stores."""
    lesson_count: int = 0
    fact_count: int = 0
    kg_fact_count: int = 0
    reflexion_count: int = 0
    curiosity_pending: int = 0
    skill_count: int = 0
    conversation_count_24h: int = 0
    last_dream_at: str | None = None
    # Capacity pressure
    kg_near_limit: bool = False
    reflexions_near_limit: bool = False


@dataclass
class GatherSignals:
    """Phase 2 output: items flagged for consolidation."""
    stale_facts: list[dict] = field(default_factory=list)
    low_quality_reflexions: list[dict] = field(default_factory=list)
    high_quality_unpromoted: list[dict] = field(default_factory=list)
    recurring_failure_clusters: list[dict] = field(default_factory=list)
    oscillating_kg_facts: list[dict] = field(default_factory=list)
    kg_chains_to_compact: list[dict] = field(default_factory=list)
    failed_curiosity: list[dict] = field(default_factory=list)
    weak_skills: list[dict] = field(default_factory=list)
    lesson_contradictions: list[dict] = field(default_factory=list)
    quality_extremes: list[dict] = field(default_factory=list)  # best/worst for DPO


@dataclass
class ConsolidationResult:
    """Phase 3+4 output: what was done."""
    overlaps_merged: int = 0
    contradictions_resolved: int = 0
    reflexions_promoted: int = 0
    reflexions_pruned: int = 0
    lessons_pruned: int = 0
    kg_chains_compacted: int = 0
    kg_facts_archived: int = 0
    skills_disabled: int = 0
    curiosity_dismissed: int = 0
    curiosity_reset: int = 0
    dpo_pairs_generated: int = 0
    facts_refreshed: int = 0
    procedural_clusters_consolidated: int = 0
    procedural_lessons_subsumed: int = 0
    errors: list[str] = field(default_factory=list)
    # Two-phase timing (task #30, SCM/SleepGate split). 0.0 when the legacy
    # single-phase `consolidate()` is used; populated when run via the new
    # `consolidate_nrem` + `consolidate_rem` path.
    nrem_seconds: float = 0.0
    rem_seconds: float = 0.0
    nrem_completed: bool = False
    rem_completed: bool = False


# ---------------------------------------------------------------------------
# DreamConsolidator
# ---------------------------------------------------------------------------

class DreamConsolidator:
    """Runs the 4-phase dream consolidation pipeline."""

    def __init__(self, db):
        self._db = db  # AsyncSafeDB or SafeDB

    # ── Phase 1: ORIENT ──────────────────────────────────────────────────

    async def orient(self) -> DreamInventory:
        """Read-only inventory of all memory stores."""
        from app.core.brain import get_services
        from app.config import config

        svc = get_services()
        inv = DreamInventory()

        # Lessons
        row = await self._db.fetchone("SELECT COUNT(*) as c FROM lessons")
        inv.lesson_count = row["c"] if row else 0

        # User facts
        row = await self._db.fetchone("SELECT COUNT(*) as c FROM user_facts")
        inv.fact_count = row["c"] if row else 0

        # KG facts (current only)
        row = await self._db.fetchone("SELECT COUNT(*) as c FROM kg_facts WHERE valid_to IS NULL")
        inv.kg_fact_count = row["c"] if row else 0
        inv.kg_near_limit = inv.kg_fact_count > (config.MAX_KG_FACTS * 0.85)

        # Reflexions
        row = await self._db.fetchone("SELECT COUNT(*) as c FROM reflexions")
        inv.reflexion_count = row["c"] if row else 0
        inv.reflexions_near_limit = inv.reflexion_count > 170  # MAX_REFLEXIONS=200

        # Curiosity pending
        row = await self._db.fetchone("SELECT COUNT(*) as c FROM curiosity_queue WHERE status='pending'")
        inv.curiosity_pending = row["c"] if row else 0

        # Skills
        row = await self._db.fetchone("SELECT COUNT(*) as c FROM skills WHERE enabled=1")
        inv.skill_count = row["c"] if row else 0

        # Recent conversations
        cutoff = (datetime.utcnow() - timedelta(hours=24)).isoformat()
        row = await self._db.fetchone(
            "SELECT COUNT(DISTINCT conversation_id) as c FROM messages WHERE created_at > ?",
            (cutoff,),
        )
        inv.conversation_count_24h = row["c"] if row else 0

        # Last dream timestamp
        row = await self._db.fetchone(
            "SELECT value FROM system_state WHERE key='last_dream_at'"
        )
        inv.last_dream_at = row["value"] if row else None

        logger.info(
            "[Dream] Orient: %d lessons, %d facts, %d KG, %d reflexions, %d curiosity, %d skills, %d convos(24h)",
            inv.lesson_count, inv.fact_count, inv.kg_fact_count,
            inv.reflexion_count, inv.curiosity_pending, inv.skill_count,
            inv.conversation_count_24h,
        )
        return inv

    # ── Phase 2: GATHER SIGNAL ───────────────────────────────────────────

    async def gather(self, inventory: DreamInventory) -> GatherSignals:
        """Scan all stores for consolidation targets."""
        signals = GatherSignals()
        since = inventory.last_dream_at or (datetime.utcnow() - timedelta(days=7)).isoformat()

        # 1. Stale user facts (not accessed in 60+ days, but respect source authority)
        cutoff_60d = (datetime.utcnow() - timedelta(days=60)).isoformat()
        rows = await self._db.fetchall(
            "SELECT id, key, value, source, confidence, last_accessed_at, access_count "
            "FROM user_facts WHERE last_accessed_at < ? OR last_accessed_at IS NULL",
            (cutoff_60d,),
        )
        signals.stale_facts = [dict(r) for r in rows]

        # 2. Low-quality reflexions (candidates for pruning)
        rows = await self._db.fetchall(
            "SELECT id, task_summary, outcome, quality_score, created_at "
            "FROM reflexions WHERE quality_score < 0.2 ORDER BY quality_score ASC LIMIT 50"
        )
        signals.low_quality_reflexions = [dict(r) for r in rows]

        # 3. High-quality reflexions never promoted to lessons.
        # Exclude is_eval=1: eval-harness reflexions are not real-world
        # successes — they're synthetic reproductions and promoting them
        # creates a self-feedback loop where each eval run accumulates more
        # near-duplicate lessons for the same query. That contamination was
        # the root cause of the deliberation_chain_of_reasoning bimodal
        # variance pattern (0.20 ↔ 0.85+ across runs).
        rows = await self._db.fetchall(
            "SELECT id, task_summary, outcome, reflection, quality_score "
            "FROM reflexions WHERE quality_score >= 0.9 AND outcome = 'success' "
            "AND COALESCE(is_eval, 0) = 0 "
            "ORDER BY quality_score DESC LIMIT 20"
        )
        signals.high_quality_unpromoted = [dict(r) for r in rows]

        # 4. Oscillating KG facts (superseded 3+ times on same subject+predicate)
        rows = await self._db.fetchall(
            "SELECT subject, predicate, COUNT(*) as chain_len "
            "FROM kg_facts WHERE superseded_by IS NOT NULL "
            "GROUP BY subject, predicate HAVING COUNT(*) >= 3 "
            "ORDER BY chain_len DESC LIMIT 20"
        )
        signals.oscillating_kg_facts = [dict(r) for r in rows]

        # 5. KG chains to compact (superseded facts older than 60 days)
        cutoff_kg = (datetime.utcnow() - timedelta(days=60)).isoformat()
        rows = await self._db.fetchall(
            "SELECT id, subject, predicate, object, superseded_by, created_at "
            "FROM kg_facts WHERE superseded_by IS NOT NULL AND valid_to < ? "
            "ORDER BY subject, predicate, created_at",
            (cutoff_kg,),
        )
        signals.kg_chains_to_compact = [dict(r) for r in rows]

        # 6. Failed curiosity (exhausted all attempts)
        rows = await self._db.fetchall(
            "SELECT id, topic, source, urgency, attempts "
            "FROM curiosity_queue WHERE status='failed' "
            "ORDER BY urgency DESC LIMIT 20"
        )
        signals.failed_curiosity = [dict(r) for r in rows]

        # 7. Weak skills (low success rate for 30+ days)
        cutoff_30d = (datetime.utcnow() - timedelta(days=30)).isoformat()
        rows = await self._db.fetchall(
            "SELECT id, name, trigger_pattern, success_rate, times_used, created_at "
            "FROM skills WHERE enabled=1 AND success_rate < 0.2 AND times_used >= 3 "
            "AND created_at < ?",
            (cutoff_30d,),
        )
        signals.weak_skills = [dict(r) for r in rows]

        # 8. Lesson contradictions (same topic, different correct_answer)
        rows = await self._db.fetchall(
            "SELECT a.id as id_a, b.id as id_b, a.topic, "
            "a.correct_answer as answer_a, b.correct_answer as answer_b, "
            "a.confidence as conf_a, b.confidence as conf_b "
            "FROM lessons a JOIN lessons b ON a.topic = b.topic AND a.id < b.id "
            "LIMIT 20"
        )
        signals.lesson_contradictions = [dict(r) for r in rows]

        # 9. Quality extremes for DPO mining (best + worst from recent conversations)
        rows = await self._db.fetchall(
            "SELECT task_summary, reflection, quality_score, outcome "
            "FROM reflexions WHERE created_at > ? "
            "AND (quality_score >= 0.85 OR quality_score <= 0.3) "
            "ORDER BY quality_score DESC",
            (since,),
        )
        signals.quality_extremes = [dict(r) for r in rows]

        total = (
            len(signals.stale_facts) + len(signals.low_quality_reflexions) +
            len(signals.high_quality_unpromoted) + len(signals.oscillating_kg_facts) +
            len(signals.kg_chains_to_compact) + len(signals.failed_curiosity) +
            len(signals.weak_skills) + len(signals.lesson_contradictions) +
            len(signals.quality_extremes)
        )
        logger.info(
            "[Dream] Gather: %d targets (stale_facts=%d, low_reflexions=%d, "
            "unpromoted=%d, osc_kg=%d, kg_chains=%d, failed_curiosity=%d, "
            "weak_skills=%d, contradictions=%d, dpo_candidates=%d)",
            total, len(signals.stale_facts), len(signals.low_quality_reflexions),
            len(signals.high_quality_unpromoted), len(signals.oscillating_kg_facts),
            len(signals.kg_chains_to_compact), len(signals.failed_curiosity),
            len(signals.weak_skills), len(signals.lesson_contradictions),
            len(signals.quality_extremes),
        )
        return signals

    # ── Phase 3: CONSOLIDATE ─────────────────────────────────────────────

    # ── Phase 3a: NREM (structural / deterministic consolidation) ──────
    #
    # The "filing" pass — non-LLM operations that prune, compact, and
    # disable. Fast (< 5s typical), can't fail catastrophically, doesn't
    # depend on LLM availability. Inspired by SCM (arxiv 2604.20943)
    # NREM phase + SleepGate (arxiv 2603.14517) forgetting gate.
    async def consolidate_nrem(self, signals: GatherSignals, result: ConsolidationResult, svc) -> None:
        """Structural/deterministic substeps. Idempotent on partial failure."""
        # 3a. Prune low-quality reflexions
        await self._prune_reflexions(signals, result)
        # 3b. Compact KG chains
        await self._compact_kg_chains(signals, result)
        # 3c. Disable broken skills
        await self._disable_weak_skills(signals, result)
        # 3d. Handle failed curiosity
        await self._handle_failed_curiosity(signals, result)
        # 3e. Refresh critical stale facts (don't prune user-stated facts)
        await self._refresh_stale_facts(signals, result)
        # 3h. Mine DPO pairs from quality extremes (deterministic extraction,
        # no LLM call in the mining step itself).
        await self._mine_dpo_pairs(signals, result)
        result.nrem_completed = True

    # ── Phase 3b: REM (integrative / LLM-driven consolidation) ─────────
    #
    # The "abstractive" pass — LLM-driven operations that promote, resolve,
    # and generalize. Slow (10-60s typical), depends on LLM availability.
    # SCM REM phase: combines pruned memories into novel patterns.
    async def consolidate_rem(self, signals: GatherSignals, result: ConsolidationResult, svc) -> None:
        """LLM-driven integrative substeps. Failures here must not roll back NREM."""
        # 3f. Promote high-quality reflexions to lessons (LLM)
        await self._promote_reflexions(signals, result, svc)
        # 3g. Resolve lesson contradictions (LLM)
        await self._resolve_contradictions(signals, result, svc)
        # 3i. Procedural memory consolidation — cluster similar lessons,
        # generalize via LLM, prune subsumed members.
        from app.config import config as _cfg
        if getattr(_cfg, "ENABLE_PROCEDURAL_CONSOLIDATION", True):
            await self._consolidate_procedural_memory(result, svc)
        result.rem_completed = True

    async def consolidate(self, signals: GatherSignals) -> ConsolidationResult:
        """Execute consolidation actions. LLM-assisted where needed.

        Runs under MAINTENANCE isolation — only memory/knowledge/calculator tools allowed.

        Back-compat wrapper around the two-phase split (consolidate_nrem +
        consolidate_rem). When two-phase dream is enabled at the `run()`
        level the phases are dispatched separately with their own time
        budgets and failure isolation.
        """
        from app.core.brain import get_services
        from app.core.access_tiers import set_tool_whitelist, MAINTENANCE_TOOLS

        svc = get_services()
        result = ConsolidationResult()

        # Apply tool isolation — consolidation should only read/write memory, not execute tools
        set_tool_whitelist(MAINTENANCE_TOOLS)
        try:
            await self.consolidate_nrem(signals, result, svc)
            await self.consolidate_rem(signals, result, svc)
        finally:
            set_tool_whitelist(None)

        logger.info(
            "[Dream] Consolidate: merged=%d, contradictions=%d, promoted=%d, "
            "pruned_reflexions=%d, pruned_lessons=%d, kg_compacted=%d, "
            "skills_disabled=%d, dpo_pairs=%d, proc_clusters=%d, "
            "proc_subsumed=%d, errors=%d",
            result.overlaps_merged, result.contradictions_resolved,
            result.reflexions_promoted, result.reflexions_pruned,
            result.lessons_pruned, result.kg_chains_compacted,
            result.skills_disabled, result.dpo_pairs_generated,
            result.procedural_clusters_consolidated,
            result.procedural_lessons_subsumed,
            len(result.errors),
        )
        return result

    async def _prune_reflexions(self, signals: GatherSignals, result: ConsolidationResult):
        """Delete reflexions with quality < 0.2."""
        ids = [r["id"] for r in signals.low_quality_reflexions]
        if not ids:
            return
        placeholders = ",".join("?" for _ in ids)
        try:
            await self._db.execute(
                f"DELETE FROM reflexions WHERE id IN ({placeholders})", tuple(ids)
            )
            result.reflexions_pruned = len(ids)
        except Exception as e:
            result.errors.append(f"reflexion prune: {e}")
            logger.warning("[Dream] Reflexion prune failed: %s", e)

    async def _compact_kg_chains(self, signals: GatherSignals, result: ConsolidationResult):
        """Remove intermediate superseded KG facts, keep first and last."""
        if not signals.kg_chains_to_compact:
            return

        # Group by (subject, predicate)
        chains: dict[tuple, list] = {}
        for fact in signals.kg_chains_to_compact:
            key = (fact["subject"], fact["predicate"])
            chains.setdefault(key, []).append(fact)

        ids_to_delete = []
        for key, facts in chains.items():
            if len(facts) <= 2:
                continue
            # Keep first (origin) and last (most recent superseded), delete middle
            sorted_facts = sorted(facts, key=lambda f: f["created_at"])
            middle = sorted_facts[1:-1]
            ids_to_delete.extend(f["id"] for f in middle)

        if ids_to_delete:
            placeholders = ",".join("?" for _ in ids_to_delete)
            try:
                await self._db.execute(
                    f"DELETE FROM kg_facts WHERE id IN ({placeholders})",
                    tuple(ids_to_delete),
                )
                result.kg_chains_compacted = len(ids_to_delete)
            except Exception as e:
                result.errors.append(f"KG compact: {e}")
                logger.warning("[Dream] KG compact failed: %s", e)

    async def _disable_weak_skills(self, signals: GatherSignals, result: ConsolidationResult):
        """Disable skills with < 0.2 success rate and 3+ uses."""
        for skill in signals.weak_skills:
            try:
                await self._db.execute(
                    "UPDATE skills SET enabled=0 WHERE id=?", (skill["id"],)
                )
                result.skills_disabled += 1
                logger.info("[Dream] Disabled weak skill: %s (rate=%.2f)", skill["name"], skill["success_rate"])
            except Exception as e:
                result.errors.append(f"disable skill {skill['name']}: {e}")

    async def _handle_failed_curiosity(self, signals: GatherSignals, result: ConsolidationResult):
        """Dismiss or reset failed curiosity items based on failure reason."""
        for item in signals.failed_curiosity:
            topic = item.get("topic", "")
            # Subjective/unanswerable questions → dismiss
            subjective_signals = ["best ", "favorite ", "preferred ", "opinion "]
            if any(topic.lower().startswith(s) for s in subjective_signals):
                try:
                    await self._db.execute(
                        "UPDATE curiosity_queue SET status='dismissed' WHERE id=?",
                        (item["id"],),
                    )
                    result.curiosity_dismissed += 1
                except Exception as e:
                    result.errors.append(f"dismiss curiosity: {e}")
            else:
                # Factual questions with transient failures → reset for retry
                try:
                    await self._db.execute(
                        "UPDATE curiosity_queue SET status='pending', attempts=0 WHERE id=?",
                        (item["id"],),
                    )
                    result.curiosity_reset += 1
                except Exception as e:
                    result.errors.append(f"reset curiosity: {e}")

    async def _refresh_stale_facts(self, signals: GatherSignals, result: ConsolidationResult):
        """Refresh access timestamps for system-critical facts. Never prune user-stated facts."""
        critical_keys = {"timezone", "email", "phone", "name", "location", "language"}
        now = datetime.utcnow().isoformat()
        for fact in signals.stale_facts:
            key = fact.get("key", "").lower()
            source = fact.get("source", "")
            # System-critical facts: refresh access
            if key in critical_keys or any(k in key for k in critical_keys):
                try:
                    await self._db.execute(
                        "UPDATE user_facts SET last_accessed_at=? WHERE id=?",
                        (now, fact["id"]),
                    )
                    result.facts_refreshed += 1
                except Exception as e:
                    result.errors.append(f"refresh fact: {e}")

    async def _promote_reflexions(self, signals: GatherSignals, result: ConsolidationResult, svc):
        """Promote high-quality success reflexions to knowledge lessons via LLM."""
        if not signals.high_quality_unpromoted or not svc.learning:
            return

        import asyncio
        from app.core.llm import invoke_nothink, extract_json_object

        # Batch up to 5 per dream cycle to limit LLM cost
        candidates = signals.high_quality_unpromoted[:5]
        for ref in candidates:
            # Yield control between iterations — avoids burst-sending to Ollama
            await asyncio.sleep(0)
            try:
                # Truncate fields to prevent Ollama 400 (context overflow) on long reflexions
                task_summary = (ref.get("task_summary") or "")[:800]
                reflection = (ref.get("reflection") or "")[:1200]
                prompt = (
                    f"Extract a reusable lesson from this successful experience.\n\n"
                    f"Task: {task_summary}\n"
                    f"Reflection: {reflection}\n"
                    f"Quality: {ref['quality_score']}\n\n"
                    f"Return JSON: {{\"topic\": \"...\", \"lesson\": \"...\"}}\n"
                    f"Topic should be 3-5 words. Lesson should be one actionable sentence."
                )
                resp = await invoke_nothink(
                    [{"role": "user", "content": prompt}],
                    json_mode=True,
                    max_tokens=200,
                )
                if not resp:
                    continue
                data = extract_json_object(resp)
                if not data:
                    continue
                topic = str(data.get("topic", "")).strip()
                lesson = str(data.get("lesson", "")).strip()
                if topic and lesson and len(lesson) > 10:
                    svc.learning.add_knowledge_lesson(
                        topic=topic,
                        correct_answer=lesson,
                        lesson_text=f"Promoted from success reflexion (quality={ref['quality_score']})",
                        context="dream_consolidation",
                        confidence=0.7,
                    )
                    result.reflexions_promoted += 1
            except Exception as e:
                result.errors.append(f"promote reflexion: {e}")
                logger.warning("[Dream] Reflexion promotion failed: %s", e)

    async def _resolve_contradictions(self, signals: GatherSignals, result: ConsolidationResult, svc):
        """Resolve lesson contradictions via LLM arbitration."""
        if not signals.lesson_contradictions:
            return

        import asyncio
        from app.core.llm import invoke_nothink, extract_json_object

        # Batch up to 3 per dream cycle
        for pair in signals.lesson_contradictions[:3]:
            # Yield control between iterations — avoids burst-sending to Ollama
            await asyncio.sleep(0)
            try:
                # Truncate fields to prevent Ollama 400 (context overflow) on long lessons
                topic = (pair.get("topic") or "")[:200]
                answer_a = (pair.get("answer_a") or "")[:600]
                answer_b = (pair.get("answer_b") or "")[:600]
                prompt = (
                    f"Two lessons have the same topic but different answers. Are they contradictory "
                    f"or is one a specific exception to the other?\n\n"
                    f"Topic: {topic}\n"
                    f"Lesson A (confidence {pair['conf_a']:.2f}): {answer_a}\n"
                    f"Lesson B (confidence {pair['conf_b']:.2f}): {answer_b}\n\n"
                    f"Return JSON: {{\"contradictory\": true/false, \"keep\": \"A\" or \"B\" or \"both\", "
                    f"\"reason\": \"...\"}}"
                )
                resp = await invoke_nothink(
                    [{"role": "user", "content": prompt}],
                    json_mode=True,
                    max_tokens=200,
                )
                if not resp:
                    continue
                data = extract_json_object(resp)
                if not data:
                    continue
                if data.get("contradictory") and data.get("keep") in ("A", "B"):
                    loser_id = pair["id_b"] if data["keep"] == "A" else pair["id_a"]
                    # Lower confidence of the losing lesson
                    await self._db.execute(
                        "UPDATE lessons SET confidence = MAX(0.1, confidence - 0.3) WHERE id=?",
                        (loser_id,),
                    )
                    result.contradictions_resolved += 1
                    logger.info(
                        "[Dream] Resolved contradiction on '%s': keep %s (%s)",
                        pair["topic"], data["keep"], data.get("reason", ""),
                    )
            except Exception as e:
                result.errors.append(f"resolve contradiction: {e}")
                logger.warning("[Dream] Contradiction resolution failed: %s", e)

    async def _consolidate_procedural_memory(
        self,
        result: ConsolidationResult,
        svc,
    ) -> None:
        """Cluster similar lessons, generalize each cluster, prune subsumed members.

        Procedural memory consolidation (Liu et al. 2025; SimpleMem 2026):
        when 3+ lessons share a topic and an answer-shape, a single canonical
        lesson with broader applicability is better than N near-duplicates.
        We:
          1. Group lessons by jaccard(topic_tokens) >= 0.6 AND
             jaccard(answer_tokens) >= 0.5 — only true near-duplicates qualify.
          2. For each cluster of >=3 not seen since `last_consolidated_at`,
             ask the LLM to write a single generalized lesson.
          3. Insert the generalized lesson with confidence = max(member confidence)
             and provenance = 'procedural_consolidation:N'.
          4. Lower-confidence the original members so they fall out of the
             retrieval head (keep them for audit, not for retrieval).
          5. Persist the cluster signature so we don't reconsolidate next cycle.

        Cap: 3 clusters per dream cycle (each calls LLM once).
        """
        from app.core.llm import invoke_nothink, extract_json_object

        # Pull recent / re-touched lessons that haven't already been
        # demoted by a prior consolidation
        try:
            rows = await self._db.fetchall(
                "SELECT id, topic, correct_answer, lesson_text, confidence, "
                "times_helpful "
                "FROM lessons "
                "WHERE confidence >= 0.5 "
                "AND lesson_text NOT LIKE 'Procedural-consolidation:%' "
                "ORDER BY times_helpful DESC LIMIT 200"
            )
        except Exception as e:
            result.errors.append(f"procedural fetch: {e}")
            return
        if not rows or len(rows) < 3:
            return

        # Token + crude stemming so 'limiter' / 'limiters' / 'limiting' all map
        # to the same root. Without this, near-duplicate lessons (the
        # rate-limiter cluster observed across 7 eval runs) escape clustering
        # because their topic strings differ only in -ing / -er / -s suffixes.
        _SUFFIX_RE = re.compile(r"(?:ing|ings|ers|er|ies|ied|ed|es|s|ly)$")
        def _stem(t: str) -> str:
            if len(t) >= 6:
                stripped = _SUFFIX_RE.sub("", t)
                if len(stripped) >= 3:
                    return stripped
            return t

        def _tokens(s: str) -> set[str]:
            import re as _re
            raw = _re.findall(r"\b[a-z][a-z0-9]{2,}\b", (s or "").lower())
            return {_stem(t) for t in raw if len(t) >= 3}

        def _jaccard(a: set[str], b: set[str]) -> float:
            if not a or not b:
                return 0.0
            inter = len(a & b)
            union = len(a | b)
            return inter / union if union else 0.0

        # Pre-compute tokens once
        items = []
        for r in rows:
            topic_tokens = _tokens(r["topic"] or "")
            answer_tokens = _tokens((r["correct_answer"] or "")[:400])
            if len(topic_tokens) < 2 or len(answer_tokens) < 3:
                continue
            items.append({
                "id": r["id"],
                "topic": r["topic"] or "",
                "correct_answer": r["correct_answer"] or "",
                "confidence": float(r["confidence"] or 0.0),
                "times_helpful": int(r["times_helpful"] or 0),
                "topic_tokens": topic_tokens,
                "answer_tokens": answer_tokens,
            })
        if len(items) < 3:
            return

        # Greedy cluster — for each item, find unseen neighbors within thresholds
        seen: set[int] = set()
        clusters: list[list[dict]] = []
        for i, item in enumerate(items):
            if item["id"] in seen:
                continue
            cluster = [item]
            seen.add(item["id"])
            for j in range(i + 1, len(items)):
                cand = items[j]
                if cand["id"] in seen:
                    continue
                topic_jac = _jaccard(item["topic_tokens"], cand["topic_tokens"])
                ans_jac = _jaccard(item["answer_tokens"], cand["answer_tokens"])
                # Topic-primary clustering: same topic, near-any wording.
                # Original (0.6/0.5) and intermediate (0.45/0.4) both failed
                # on the rate-limiter cluster — 9 lessons with topic_jac
                # 0.45-1.0 (clearly the same topic) but answer_jac 0.04-0.24
                # because each conversation extracted differently-worded
                # conclusions. The answer threshold of 0.4 was wrong: it
                # rejects diverse-wording lessons with the same intent.
                # Now: strong topic match (0.50+) is enough; we still keep a
                # tiny answer floor (0.10) to reject conflicting-intent
                # lessons that happen to share topic words.
                if topic_jac >= 0.50 and ans_jac >= 0.10:
                    cluster.append(cand)
                    seen.add(cand["id"])
            if len(cluster) >= 3:
                clusters.append(cluster)

        if not clusters:
            return

        # Deduplicate against prior consolidations
        consolidated_count = 0
        subsumed_count = 0
        for cluster in clusters[:3]:  # cap per cycle
            await asyncio.sleep(0)
            cluster_ids = sorted(c["id"] for c in cluster)
            cluster_key = "ids:" + ",".join(str(i) for i in cluster_ids)
            try:
                existing = await self._db.fetchone(
                    "SELECT id FROM procedural_clusters WHERE cluster_key = ? "
                    "AND last_consolidated_at > datetime('now', '-7 days')",
                    (cluster_key,),
                )
                if existing:
                    continue  # consolidated this exact cluster recently
            except Exception:
                pass

            # Build the LLM prompt
            members_text = "\n\n".join(
                f"Lesson {idx+1} (confidence={m['confidence']:.2f}, "
                f"helpful={m['times_helpful']}):\n"
                f"  Topic: {m['topic'][:160]}\n"
                f"  Correct answer: {m['correct_answer'][:300]}"
                for idx, m in enumerate(cluster[:6])
            )
            prompt = (
                "You're consolidating multiple lessons into one general procedure. "
                "Write a single canonical lesson that captures the shared rule.\n\n"
                "Rules:\n"
                "- Topic: 3-7 words covering all members\n"
                "- Correct answer: one actionable sentence; cover the union, not\n"
                "  any one member's narrow case\n"
                "- Don't invent a rule the members don't agree on\n\n"
                f"MEMBERS:\n{members_text}\n\n"
                "Return JSON: {\"topic\": \"...\", \"correct_answer\": \"...\", "
                "\"rationale\": \"why these belong together\"}"
            )
            try:
                resp = await invoke_nothink(
                    [{"role": "user", "content": prompt}],
                    json_mode=True,
                    max_tokens=300,
                    temperature=0.3,
                )
            except Exception as e:
                result.errors.append(f"procedural LLM: {e}")
                continue
            if not resp:
                continue
            data = extract_json_object(resp)
            if not data:
                continue
            new_topic = str(data.get("topic", "")).strip()
            new_answer = str(data.get("correct_answer", "")).strip()
            if not new_topic or len(new_answer) < 12:
                continue

            new_confidence = max(m["confidence"] for m in cluster)
            try:
                # Insert the canonical lesson
                cursor = await self._db.execute(
                    "INSERT INTO lessons "
                    "(topic, correct_answer, lesson_text, context, confidence) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (
                        new_topic[:200],
                        new_answer[:1000],
                        f"Procedural-consolidation: merged {len(cluster)} lessons",
                        "procedural_consolidation",
                        min(0.95, new_confidence),
                    ),
                )
                new_lesson_id = cursor.lastrowid
                # Demote members so retrieval prefers the canonical lesson
                placeholders = ",".join("?" for _ in cluster_ids)
                await self._db.execute(
                    f"UPDATE lessons SET confidence = MAX(0.1, confidence - 0.4) "
                    f"WHERE id IN ({placeholders})",
                    tuple(cluster_ids),
                )
                # Persist cluster signature
                await self._db.execute(
                    "INSERT INTO procedural_clusters "
                    "(cluster_key, member_lesson_ids, canonical_lesson_id, "
                    "member_count) VALUES (?, ?, ?, ?)",
                    (
                        cluster_key,
                        json.dumps(cluster_ids),
                        new_lesson_id,
                        len(cluster),
                    ),
                )
                consolidated_count += 1
                subsumed_count += len(cluster)
                logger.info(
                    "[Dream] Procedural consolidation: %d lessons -> id=%s "
                    "(topic=%r)",
                    len(cluster), new_lesson_id, new_topic[:60],
                )
            except Exception as e:
                result.errors.append(f"procedural insert: {e}")
                logger.warning("[Dream] Procedural consolidation insert failed: %s", e)

        result.procedural_clusters_consolidated = consolidated_count
        result.procedural_lessons_subsumed = subsumed_count

    async def _mine_dpo_pairs(self, signals: GatherSignals, result: ConsolidationResult):
        """Generate DPO training pairs from quality extremes."""
        if not signals.quality_extremes:
            return

        from app.config import config

        successes = [r for r in signals.quality_extremes if r["quality_score"] >= 0.85]
        failures = [r for r in signals.quality_extremes if r["quality_score"] <= 0.3]

        if not successes or not failures:
            return

        # Pair successes with failures on similar topics (simple word overlap)
        pairs_written = 0
        training_path = config.TRAINING_DATA_PATH
        try:
            with open(training_path, "a", encoding="utf-8") as f:
                for success in successes[:5]:
                    s_words = set(success["task_summary"].lower().split())
                    best_match = None
                    best_overlap = 0
                    for failure in failures:
                        f_words = set(failure["task_summary"].lower().split())
                        overlap = len(s_words & f_words) / max(len(s_words | f_words), 1)
                        if overlap > best_overlap:
                            best_overlap = overlap
                            best_match = failure
                    if best_match and best_overlap > 0.2:
                        pair = {
                            "query": success["task_summary"],
                            "chosen": success["reflection"],
                            "rejected": best_match["reflection"],
                            "timestamp": datetime.utcnow().isoformat(),
                            "source": "dream_consolidation",
                        }
                        f.write(json.dumps(pair, ensure_ascii=False) + "\n")
                        pairs_written += 1
            result.dpo_pairs_generated = pairs_written
        except Exception as e:
            result.errors.append(f"DPO mining: {e}")
            logger.warning("[Dream] DPO mining failed: %s", e)

    # ── Phase 4: REPORT ──────────────────────────────────────────────────

    async def report(self, inventory: DreamInventory, result: ConsolidationResult) -> str:
        """Record dream completion and generate digest."""
        now = datetime.utcnow().isoformat()

        # Update last_dream_at in system_state
        await self._db.execute(
            "INSERT OR REPLACE INTO system_state (key, value, updated_at) VALUES (?, ?, ?)",
            ("last_dream_at", now, now),
        )

        # Write observation to daemon_log
        digest = self._format_digest(inventory, result)
        await self._db.execute(
            "INSERT INTO daemon_log (category, content, source) VALUES (?, ?, ?)",
            ("dream", digest, "dream_consolidator"),
        )

        # Prune old daemon log entries (keep 7 days)
        cutoff = (datetime.utcnow() - timedelta(days=7)).isoformat()
        await self._db.execute(
            "DELETE FROM daemon_log WHERE created_at < ?", (cutoff,)
        )

        # Trust decay — trust must be maintained through use (Sovereign-OS pattern)
        try:
            from app.core.trust import TrustManager
            from app.database import get_db
            trust_mgr = TrustManager(get_db())
            new_score = trust_mgr.decay()
            logger.info("[Dream] Trust decay applied: score now %.0f", new_score)
        except Exception as e:
            logger.warning("[Dream] Trust decay failed: %s", e)

        logger.info("[Dream] Cycle complete: %s", digest)
        return digest

    def _format_digest(self, inv: DreamInventory, result: ConsolidationResult) -> str:
        """Format a concise dream digest."""
        parts = []
        if result.reflexions_pruned:
            parts.append(f"{result.reflexions_pruned} reflexions pruned")
        if result.reflexions_promoted:
            parts.append(f"{result.reflexions_promoted} reflexions → lessons")
        if result.contradictions_resolved:
            parts.append(f"{result.contradictions_resolved} contradictions resolved")
        if result.kg_chains_compacted:
            parts.append(f"{result.kg_chains_compacted} KG facts compacted")
        if result.skills_disabled:
            parts.append(f"{result.skills_disabled} skills disabled")
        if result.curiosity_dismissed:
            parts.append(f"{result.curiosity_dismissed} curiosity dismissed")
        if result.curiosity_reset:
            parts.append(f"{result.curiosity_reset} curiosity reset for retry")
        if result.dpo_pairs_generated:
            parts.append(f"{result.dpo_pairs_generated} DPO pairs mined")
        if result.facts_refreshed:
            parts.append(f"{result.facts_refreshed} facts refreshed")
        if result.errors:
            # Surface up to 2 distinct error messages so recurring failures become visible
            # instead of hiding behind a "N errors" summary.
            uniq = list(dict.fromkeys(result.errors))[:2]
            sample = "; ".join(e[:160] for e in uniq)
            parts.append(f"{len(result.errors)} errors [{sample}]")

        actions = ", ".join(parts) if parts else "no actions needed"
        return (
            f"Dream cycle complete. "
            f"Memory: {inv.lesson_count} lessons, {inv.kg_fact_count} KG, "
            f"{inv.reflexion_count} reflexions, {inv.fact_count} facts. "
            f"Actions: {actions}."
        )

    # ── Full pipeline ────────────────────────────────────────────────────

    async def run(self) -> str:
        """Execute full 4-phase dream consolidation with per-phase time budgets.

        Enhanced with KAIROS pattern: time-budgeted phases + principle distillation.
        """
        import asyncio
        logger.info("[Dream] Starting consolidation cycle...")
        PHASE_TIMEOUT = 15  # seconds per phase

        # Phase 1: Orient
        try:
            inventory = await asyncio.wait_for(self.orient(), timeout=PHASE_TIMEOUT)
        except asyncio.TimeoutError:
            logger.warning("[Dream] Phase 1 (Orient) timed out after %ds", PHASE_TIMEOUT)
            inventory = DreamInventory()

        # Phase 2: Gather
        try:
            signals = await asyncio.wait_for(self.gather(inventory), timeout=PHASE_TIMEOUT)
        except asyncio.TimeoutError:
            logger.warning("[Dream] Phase 2 (Gather) timed out after %ds", PHASE_TIMEOUT)
            signals = GatherSignals()

        # Phase 3: Consolidate. Two-phase split (SCM/SleepGate, task #30)
        # gives NREM and REM independent timeouts and isolates REM failures.
        from app.config import config as _cfg
        if getattr(_cfg, "ENABLE_TWO_PHASE_DREAM", False):
            from app.core.brain import get_services as _get_services
            from app.core.access_tiers import (
                set_tool_whitelist as _set_tw,
                MAINTENANCE_TOOLS as _MT,
            )
            import time as _time

            result = ConsolidationResult()
            svc = _get_services()
            _set_tw(_MT)
            try:
                # Phase 3a: NREM — structural/deterministic
                t0 = _time.monotonic()
                try:
                    await asyncio.wait_for(
                        self.consolidate_nrem(signals, result, svc),
                        timeout=PHASE_TIMEOUT,
                    )
                except asyncio.TimeoutError:
                    logger.warning("[Dream] Phase 3a (NREM) timed out after %ds", PHASE_TIMEOUT)
                    result.errors.append("nrem_timeout")
                except Exception as e:
                    logger.warning("[Dream] Phase 3a (NREM) failed: %s", e)
                    result.errors.append(f"nrem_failed: {e}")
                result.nrem_seconds = _time.monotonic() - t0
                if result.nrem_completed:
                    logger.info(
                        "[Dream] Phase 3a (NREM) completed in %.2fs — "
                        "pruned=%d compacted=%d disabled=%d curio_reset=%d refreshed=%d dpo=%d",
                        result.nrem_seconds,
                        result.reflexions_pruned,
                        result.kg_chains_compacted,
                        result.skills_disabled,
                        result.curiosity_reset + result.curiosity_dismissed,
                        result.facts_refreshed,
                        result.dpo_pairs_generated,
                    )

                # Phase 3b: REM — integrative/LLM-driven. Its failure does NOT
                # roll back NREM's committed structural work.
                rem_budget = float(getattr(_cfg, "DREAM_REM_TIMEOUT_SECONDS", 60.0))
                t1 = _time.monotonic()
                try:
                    await asyncio.wait_for(
                        self.consolidate_rem(signals, result, svc),
                        timeout=rem_budget,
                    )
                except asyncio.TimeoutError:
                    logger.warning("[Dream] Phase 3b (REM) timed out after %.0fs — NREM results kept", rem_budget)
                    result.errors.append("rem_timeout")
                except Exception as e:
                    logger.warning("[Dream] Phase 3b (REM) failed: %s — NREM results kept", e)
                    result.errors.append(f"rem_failed: {e}")
                result.rem_seconds = _time.monotonic() - t1
                if result.rem_completed:
                    logger.info(
                        "[Dream] Phase 3b (REM) completed in %.2fs — "
                        "promoted=%d resolved=%d proc_clusters=%d proc_subsumed=%d",
                        result.rem_seconds,
                        result.reflexions_promoted,
                        result.contradictions_resolved,
                        result.procedural_clusters_consolidated,
                        result.procedural_lessons_subsumed,
                    )
            finally:
                _set_tw(None)
        else:
            try:
                result = await asyncio.wait_for(self.consolidate(signals), timeout=PHASE_TIMEOUT * 2)
            except asyncio.TimeoutError:
                logger.warning("[Dream] Phase 3 (Consolidate) timed out after %ds", PHASE_TIMEOUT * 2)
                result = ConsolidationResult()

        # Phase 3b: Principle distillation (EvolveR pattern)
        try:
            from app.core.brain import get_services
            svc = get_services()
            if svc.reflexions and svc.learning:
                distilled = await asyncio.wait_for(
                    svc.reflexions.distill_principles(learning_engine=svc.learning),
                    timeout=PHASE_TIMEOUT * 2,
                )
                if distilled:
                    logger.info("[Dream] Distilled %d principles from reflexion patterns", distilled)
        except (asyncio.TimeoutError, Exception) as e:
            logger.debug("[Dream] Principle distillation skipped: %s", e)

        # Phase 4: Report
        try:
            digest = await asyncio.wait_for(self.report(inventory, result), timeout=PHASE_TIMEOUT)
        except asyncio.TimeoutError:
            logger.warning("[Dream] Phase 4 (Report) timed out after %ds", PHASE_TIMEOUT)
            digest = "[Dream consolidation completed with timeouts]"

        return digest
