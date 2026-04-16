"""Skills — learned multi-step procedures from corrections.

When Nova gets corrected on a multi-step task, the correction handler can
extract a skill: trigger pattern + tool sequence + answer template.
Next time a matching query arrives, Nova follows the learned procedure.

Matching is two-stage:
  1. Regex — fast, exact, priority-sorted by pattern specificity.
  2. Semantic — ChromaDB cosine similarity fallback when regex misses.

Skills can be composed: a skill with composed_of = [id1, id2] expands
those sub-skills' steps first, then appends its own. This allows
multi-step patterns to be built from simpler learned primitives.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import threading
from dataclasses import dataclass, field

from app.config import config
from app.core import llm
from app.database import get_db

logger = logging.getLogger(__name__)


@dataclass
class Skill:
    id: int
    name: str
    trigger_pattern: str
    steps: list[dict]           # [{"tool": "...", "args_template": {...}, "output_key": "..."}]
    answer_template: str | None  # Template with {output_key} placeholders
    learned_from: int | None
    times_used: int
    success_rate: float
    enabled: bool
    created_at: str | None
    # Quality & lifecycle fields (added in migration 9)
    last_used_at: str | None = None
    consecutive_failures: int = 0
    source: str = "correction"   # "correction" | "auto" | "manual"
    composed_of: list[int] = field(default_factory=list)  # ordered sub-skill IDs


def _pattern_specificity(pattern: str) -> tuple[int, int, int]:
    """Return a specificity score for a regex pattern.

    Returns (length, -wildcard_count, -alternation_count) so that longer
    patterns with fewer wildcards/quantifiers and fewer alternations sort
    higher (more specific = higher priority). More `|` alternations = less
    specific trigger, so they receive a negative penalty.
    """
    wildcard_count = len(re.findall(r'[.*+?]|\{[\d,]+\}', pattern))
    alternation_count = pattern.count("|")
    return (len(pattern), -wildcard_count, -alternation_count)


class SkillStore:
    """Store, match, and execute learned skills."""

    def __init__(self, db=None):
        self._db = db or get_db()
        self._chroma_collection = None
        self._chroma_client = None
        self._chroma_lock = threading.Lock()

    # ------------------------------------------------------------------
    # ChromaDB skill collection — lazy init
    # ------------------------------------------------------------------

    def _get_skill_collection(self):
        """Lazy-init ChromaDB collection for semantic skill lookup."""
        if self._chroma_collection is not None:
            return self._chroma_collection
        with self._chroma_lock:
            if self._chroma_collection is None:
                import chromadb
                self._chroma_client = chromadb.PersistentClient(path=config.CHROMADB_PATH)
                self._chroma_collection = self._chroma_client.get_or_create_collection(
                    name="skills",
                    metadata={"hnsw:space": "cosine"},
                )
        return self._chroma_collection

    # Keep backward-compat alias used by HEAD's _index_skill / refine_skill / toggle_skill
    def _get_skills_collection(self):
        """Alias for _get_skill_collection (backward compat)."""
        if not config.ENABLE_SEMANTIC_SKILL_MATCHING:
            return None
        try:
            return self._get_skill_collection()
        except Exception:
            return None

    def _embed_skill(self, skill_id: int, name: str, trigger_pattern: str) -> None:
        """Add or update a skill's embedding in ChromaDB."""
        if not config.ENABLE_SEMANTIC_SKILL_MATCHING:
            return
        try:
            collection = self._get_skill_collection()
            doc_id = f"skill_{skill_id}"
            embed_text = f"{name}: {trigger_pattern}"
            # Upsert: delete old entry if present, then add
            try:
                collection.delete(ids=[doc_id])
            except Exception:
                pass
            collection.add(
                ids=[doc_id],
                documents=[embed_text],
                metadatas=[{"skill_id": str(skill_id), "name": name}],
            )
        except Exception as e:
            logger.warning("Failed to embed skill #%d '%s': %s", skill_id, name, e)

    def _find_semantic_duplicate(self, name: str, trigger_pattern: str) -> int | None:
        """Return an existing skill ID if trigger_pattern is semantically too close.

        Embeds the candidate trigger pattern and queries ChromaDB.  If the nearest
        existing skill has similarity >= SKILL_SEMANTIC_THRESHOLD, the candidate
        would answer the same queries as that skill and is therefore a duplicate.

        Returns the duplicate skill's ID, or None if no duplicate found
        (or if semantic matching is disabled / collection is empty).

        Called from create_skill() before the INSERT so near-duplicate skills
        cannot accumulate in the corpus.
        """
        if not config.ENABLE_SEMANTIC_SKILL_MATCHING:
            return None
        try:
            collection = self._get_skill_collection()
            if collection.count() == 0:
                return None
            embed_text = f"{name}: {trigger_pattern}"
            results = collection.query(
                query_texts=[embed_text],
                n_results=1,
                include=["distances", "metadatas"],
            )
            if not results["ids"] or not results["ids"][0]:
                return None
            distance = results["distances"][0][0]
            # ChromaDB cosine distance: 0 = identical, 2 = opposite.
            similarity = 1.0 - (distance / 2.0)
            if similarity < config.SKILL_SEMANTIC_THRESHOLD:
                return None
            metadata = results["metadatas"][0][0]
            existing_id = int(metadata["skill_id"])
            existing_name = metadata.get("name", f"skill_{existing_id}")
            # Exclude self-updates -- same-name skills are handled by the name-dedup
            # path that runs before this check; guard against edge-case order issues.
            existing_row = self._db.fetchone(
                "SELECT name FROM skills WHERE id = ?", (existing_id,)
            )
            if existing_row and existing_row["name"].lower() == name.lower():
                return None
            logger.info(
                "Semantic dedup: '%s' is %.3f similar to existing #%d '%s' "
                "(threshold %.2f)",
                name, similarity, existing_id, existing_name,
                config.SKILL_SEMANTIC_THRESHOLD,
            )
            return existing_id
        except Exception as e:
            logger.debug("Semantic dedup check failed (non-critical): %s", e)
            return None

    def _unembed_skill(self, skill_id: int) -> None:
        """Remove a skill's embedding from ChromaDB."""
        if not config.ENABLE_SEMANTIC_SKILL_MATCHING:
            return
        try:
            collection = self._get_skill_collection()
            collection.delete(ids=[f"skill_{skill_id}"])
        except Exception as e:
            logger.debug("Failed to unembed skill #%d: %s", skill_id, e)

    def sync_embeddings(self) -> int:
        """Sync all enabled DB skills into ChromaDB. Safe to call at startup.

        Returns the number of skills newly embedded.
        """
        if not config.ENABLE_SEMANTIC_SKILL_MATCHING:
            return 0
        try:
            collection = self._get_skill_collection()
        except Exception as e:
            logger.warning("Skill embedding sync skipped -- ChromaDB unavailable: %s", e)
            return 0

        rows = self._db.fetchall("SELECT id, name, trigger_pattern FROM skills WHERE enabled = 1")
        synced = 0
        for row in rows:
            doc_id = f"skill_{row['id']}"
            try:
                existing = collection.get(ids=[doc_id], include=[])
                if not existing["ids"]:
                    self._embed_skill(row["id"], row["name"], row["trigger_pattern"])
                    synced += 1
            except Exception as e:
                logger.debug("Skill sync error for #%d: %s", row["id"], e)
        if synced:
            logger.info("Synced %d skill embedding(s) to ChromaDB", synced)
        return synced

    # ------------------------------------------------------------------
    # Matching
    # ------------------------------------------------------------------

    def get_matching_skill(self, query: str) -> Skill | None:
        """Find the best matching skill for a query.

        Stage 1 -- Regex: checks all enabled skills, returns highest-specificity
        match (longer pattern = more specific = higher priority).

        Stage 2 -- Semantic: ChromaDB cosine similarity fallback when no regex
        match is found and ENABLE_SEMANTIC_SKILL_MATCHING is true.
        Uses SKILL_SEMANTIC_THRESHOLD to avoid false positives.
        """
        regex_hit = self._regex_match(query)
        if regex_hit:
            return regex_hit
        if config.ENABLE_SEMANTIC_SKILL_MATCHING:
            return self._semantic_match(query)
        return None

    def _regex_match(self, query: str) -> Skill | None:
        """Regex-based skill lookup (original implementation)."""
        rows = self._db.fetchall(
            "SELECT * FROM skills WHERE enabled = 1 ORDER BY times_used DESC, success_rate DESC, id ASC LIMIT ?",
            (config.MAX_SKILLS_CHECK,),
        )

        matches = []
        for row in rows:
            try:
                pattern = row["trigger_pattern"]
                # Check for ReDoS-prone patterns before running
                if _is_redos_risk(pattern):
                    logger.warning("Skill trigger pattern rejected (ReDoS risk): %s", pattern)
                    self._db.execute("UPDATE skills SET enabled = 0 WHERE id = ?", (row["id"],))
                    continue
                if re.search(pattern, query, re.IGNORECASE):
                    matches.append(row)
            except re.error:
                logger.warning("Invalid skill trigger pattern: %s", row["trigger_pattern"])
                continue

        if matches:
            matches.sort(
                key=lambda r: _pattern_specificity(r["trigger_pattern"]),
                reverse=True,
            )
            return self._row_to_skill(matches[0])

        # Semantic fallback — finds skills whose natural-language descriptions
        # are close to the query even when regex doesn't match exactly.
        return self._semantic_match(query)

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def _semantic_match(self, query: str) -> Skill | None:
        """Embedding-similarity skill lookup (fallback when regex misses).

        Returns the best skill whose embedding similarity ≥ SKILL_SEMANTIC_THRESHOLD,
        or None if no skill clears the bar.
        """
        try:
            collection = self._get_skill_collection()
            if collection.count() == 0:
                return None
            results = collection.query(
                query_texts=[query],
                n_results=1,
                include=["distances", "metadatas"],
            )
            if not results["ids"] or not results["ids"][0]:
                return None
            distance = results["distances"][0][0]
            # ChromaDB cosine distance: 0=identical, 2=opposite → similarity = 1 − distance/2
            similarity = 1.0 - (distance / 2.0)
            if similarity < config.SKILL_SEMANTIC_THRESHOLD:
                logger.debug(
                    "Semantic skill match below threshold: sim=%.3f threshold=%.3f",
                    similarity, config.SKILL_SEMANTIC_THRESHOLD,
                )
                return None
            skill_id = int(results["metadatas"][0][0]["skill_id"])
            skill = self.get_skill(skill_id)
            if skill and skill.enabled:
                logger.info(
                    "Semantic skill match: '%s' (id=%d, sim=%.3f)",
                    skill.name, skill_id, similarity,
                )
                return skill
        except Exception as e:
            logger.warning("Semantic skill lookup failed: %s", e)
        return None

    def create_skill(
        self,
        name: str,
        trigger_pattern: str,
        steps: list[dict],
        answer_template: str | None = None,
        learned_from: int | None = None,
        initial_success_rate: float = 0.7,
        source: str = "correction",
        composed_of: list[int] | None = None,
    ) -> int | None:
        """Create a new skill. Returns skill ID, or None if rejected by guards."""
        composed_of = composed_of or []

        # Guard: reject ReDoS-prone patterns
        if _is_redos_risk(trigger_pattern):
            logger.warning("Skill '%s' rejected: ReDoS risk (%s)", name, trigger_pattern)
            return None

        # Guard: reject overly broad trigger patterns
        if _is_too_broad(trigger_pattern):
            logger.warning("Skill '%s' rejected: trigger too broad (%s)", name, trigger_pattern)
            return None

        # Guard: reject skills whose args_template references undefined placeholders.
        # This catches correction-path skills that bypass the auto_skills pre-check.
        if _has_capture_group_mismatch(trigger_pattern, steps, answer_template):
            logger.warning(
                "Skill '%s' rejected: args_template references undefined placeholder "
                "(add (?P<name>…) groups or use {query}/{output_key})",
                name,
            )
            return None

        # Guard: semantic dedup — reject if trigger embeds too close to an existing skill.
        # Runs only when ChromaDB is initialised (i.e., at least one skill already exists).
        dup_id = self._find_semantic_duplicate(name, trigger_pattern)
        if dup_id is not None:
            logger.warning(
                "Skill '%s' rejected: semantically too similar to existing skill #%d "
                "(similarity ≥ %.2f). Narrow the trigger or merge with the existing skill.",
                name, dup_id, config.SKILL_SEMANTIC_THRESHOLD,
            )
            return None

        # Deduplication — if same trigger pattern exists, boost confidence
        existing = self._db.fetchone(
            "SELECT id, success_rate FROM skills WHERE trigger_pattern = ?",
            (trigger_pattern,),
        )
        if existing:
            new_rate = min(1.0, existing["success_rate"] + 0.1)
            self._db.execute(
                "UPDATE skills SET success_rate = ?, enabled = 1 WHERE id = ?",
                (new_rate, existing["id"]),
            )
            logger.info(
                "Skill dedup: boosted #%d confidence to %.2f (trigger: %s)",
                existing["id"], new_rate, trigger_pattern,
            )
            return existing["id"]

        # Name-based dedup — if same name (case-insensitive) exists, update it
        existing_by_name = self._db.fetchone(
            "SELECT id, trigger_pattern FROM skills WHERE LOWER(name) = LOWER(?)",
            (name,),
        )
        if existing_by_name:
            old_trigger = existing_by_name.get("trigger_pattern", "<unknown>") or "<unknown>"
            import sqlite3 as _sqlite3
            try:
                self._db.execute(
                    "UPDATE skills SET trigger_pattern = ?, steps = ?, answer_template = ?, "
                    "source = ?, composed_of = ?, enabled = 1 WHERE id = ?",
                    (
                        trigger_pattern,
                        json.dumps(steps),
                        answer_template,
                        source,
                        json.dumps(composed_of),
                        existing_by_name["id"],
                    ),
                )
            except _sqlite3.OperationalError:
                self._db.execute(
                    "UPDATE skills SET trigger_pattern = ?, steps = ?, answer_template = ?, "
                    "enabled = 1 WHERE id = ?",
                    (trigger_pattern, json.dumps(steps), answer_template, existing_by_name["id"]),
                )
            logger.info(
                "Skill updated: #%d '%s' trigger changed from '%s' to '%s'",
                existing_by_name["id"], name, old_trigger[:60], trigger_pattern[:60],
            )
            self._embed_skill(existing_by_name["id"], name, trigger_pattern)
            return existing_by_name["id"]

        import sqlite3 as _sqlite3
        try:
            cursor = self._db.execute(
                """INSERT INTO skills
                   (name, trigger_pattern, steps, answer_template, learned_from,
                    success_rate, source, composed_of)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    name,
                    trigger_pattern,
                    json.dumps(steps),
                    answer_template,
                    learned_from,
                    initial_success_rate,
                    source,
                    json.dumps(composed_of),
                ),
            )
        except _sqlite3.OperationalError:
            cursor = self._db.execute(
                """INSERT INTO skills
                   (name, trigger_pattern, steps, answer_template, learned_from, success_rate)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    name,
                    trigger_pattern,
                    json.dumps(steps),
                    answer_template,
                    learned_from,
                    initial_success_rate,
                ),
            )
        skill_id = cursor.lastrowid
        logger.info("Created skill #%d: '%s' (trigger: %s)", skill_id, name, trigger_pattern)
        self._embed_skill(skill_id, name, trigger_pattern)

        return skill_id

    def record_use(self, skill_id: int, success: bool) -> None:
        """Record a skill execution. Updates times_used, success_rate, and quality counters.

        Fast demotion: 3 consecutive failures disables the skill immediately
        (in addition to the slow EMA-based auto-disable at success_rate < 0.3).
        """
        success_val = 1.0 if success else 0.0
        alpha = config.SKILL_EMA_ALPHA

        import sqlite3 as _sqlite3
        if success:
            try:
                self._db.execute(
                    "UPDATE skills SET "
                    "times_used = times_used + 1, "
                    "success_rate = ? * ? + (1 - ?) * success_rate, "
                    "consecutive_failures = 0, "
                    "last_used_at = CURRENT_TIMESTAMP "
                    "WHERE id = ?",
                    (alpha, success_val, alpha, skill_id),
                )
            except _sqlite3.OperationalError:
                self._db.execute(
                    "UPDATE skills SET times_used = times_used + 1, "
                    "success_rate = ? * ? + (1 - ?) * success_rate WHERE id = ?",
                    (alpha, success_val, alpha, skill_id),
                )
        else:
            try:
                self._db.execute(
                    "UPDATE skills SET "
                    "times_used = times_used + 1, "
                    "success_rate = ? * ? + (1 - ?) * success_rate, "
                    "consecutive_failures = consecutive_failures + 1, "
                    "last_used_at = CURRENT_TIMESTAMP "
                    "WHERE id = ?",
                    (alpha, success_val, alpha, skill_id),
                )
            except _sqlite3.OperationalError:
                self._db.execute(
                    "UPDATE skills SET times_used = times_used + 1, "
                    "success_rate = ? * ? + (1 - ?) * success_rate WHERE id = ?",
                    (alpha, success_val, alpha, skill_id),
                )

        # Check for auto-disable conditions (separate read after atomic update)
        try:
            row = self._db.fetchone(
                "SELECT name, times_used, success_rate, consecutive_failures FROM skills WHERE id = ?",
                (skill_id,),
            )
        except _sqlite3.OperationalError:
            row = self._db.fetchone(
                "SELECT name, times_used, success_rate FROM skills WHERE id = ?",
                (skill_id,),
            )
        if not row:
            return

        # Fast demotion: 3+ consecutive failures
        consec = row["consecutive_failures"] if "consecutive_failures" in row.keys() else 0
        if consec >= 3:
            self._db.execute("UPDATE skills SET enabled = 0 WHERE id = ?", (skill_id,))
            logger.warning(
                "Fast-disabled skill #%d '%s': %d consecutive failures",
                skill_id, row["name"], consec,
            )
            return

        # Slow demotion: EMA success_rate below threshold after 5+ uses
        if row["times_used"] >= 5 and row["success_rate"] < 0.3:
            self._db.execute("UPDATE skills SET enabled = 0 WHERE id = ?", (skill_id,))
            logger.warning(
                "Auto-disabled skill #%d '%s': success_rate=%.2f after %d uses",
                skill_id, row["name"], row["success_rate"], row["times_used"],
            )

    def get_skill(self, skill_id: int) -> Skill | None:
        """Get a skill by ID."""
        row = self._db.fetchone("SELECT * FROM skills WHERE id = ?", (skill_id,))
        return self._row_to_skill(row) if row else None

    def get_all_skills(self, limit: int = 50, include_disabled: bool = True) -> list[Skill]:
        """Get all skills, optionally filtering to enabled only."""
        if include_disabled:
            rows = self._db.fetchall(
                "SELECT * FROM skills ORDER BY times_used DESC, success_rate DESC, id ASC LIMIT ?",
                (limit,),
            )
        else:
            rows = self._db.fetchall(
                "SELECT * FROM skills WHERE enabled = 1 ORDER BY times_used DESC, success_rate DESC, id ASC LIMIT ?",
                (limit,),
            )
        return [self._row_to_skill(r) for r in rows]

    def get_active_skills(self) -> list[Skill]:
        """Get all enabled skills for prompt injection."""
        rows = self._db.fetchall(
            "SELECT * FROM skills WHERE enabled = 1 ORDER BY times_used DESC, success_rate DESC, id ASC LIMIT 50"
        )
        return [self._row_to_skill(r) for r in rows]

    def toggle_skill(self, skill_id: int, enabled: bool) -> bool:
        """Enable or disable a skill. Re-enabling resets stats for a fresh start."""
        if enabled:
            import sqlite3 as _sqlite3
            try:
                cursor = self._db.execute(
                    "UPDATE skills SET enabled = 1, times_used = 0, success_rate = 0.7, "
                    "consecutive_failures = 0 WHERE id = ?",
                    (skill_id,),
                )
            except _sqlite3.OperationalError:
                cursor = self._db.execute(
                    "UPDATE skills SET enabled = 1, times_used = 0, success_rate = 0.7 WHERE id = ?",
                    (skill_id,),
                )
            # Re-index in ChromaDB
            skill = self.get_skill(skill_id)
            if skill:
                self._embed_skill(skill_id, skill.name, skill.trigger_pattern)
        else:
            cursor = self._db.execute(
                "UPDATE skills SET enabled = 0 WHERE id = ?",
                (skill_id,),
            )
        return cursor.rowcount > 0

    def delete_skill(self, skill_id: int) -> bool:
        """Delete a skill and remove it from the vector index."""
        cursor = self._db.execute("DELETE FROM skills WHERE id = ?", (skill_id,))
        if cursor.rowcount > 0:
            self._unembed_skill(skill_id)
            return True
        return False

    # ------------------------------------------------------------------
    # Skill composition
    # ------------------------------------------------------------------

    def get_composed_steps(self, skill: Skill) -> list[dict]:
        """Expand a composed skill into its full ordered step list.

        Fetches each sub-skill by ID in order, concatenates their steps,
        then appends the composing skill's own steps last. Cycles and
        missing/disabled sub-skills are skipped silently.

        Returns skill.steps unchanged if composed_of is empty.
        """
        if not skill.composed_of:
            return skill.steps

        seen: set[int] = {skill.id}
        all_steps: list[dict] = []

        for sub_id in skill.composed_of[:10]:  # cap at 10 to prevent runaway chains
            if sub_id in seen:
                logger.debug("Skill composition cycle detected at #%d, skipping", sub_id)
                continue
            seen.add(sub_id)
            sub = self.get_skill(sub_id)
            if sub and sub.enabled:
                all_steps.extend(sub.steps)
            else:
                logger.debug("Composed sub-skill #%d unavailable, skipping", sub_id)

        # Append the composing skill's own steps last (may be empty for pure compositions)
        all_steps.extend(skill.steps)
        return all_steps or skill.steps

    # ------------------------------------------------------------------
    # Metrics & maintenance
    # ------------------------------------------------------------------

    def get_skill_stats(self) -> dict:
        """Return aggregate skill system metrics."""
        rows = self._db.fetchall("SELECT * FROM skills")
        if not rows:
            return {
                "total": 0, "enabled": 0, "disabled": 0,
                "total_uses": 0, "avg_success_rate": 0.0,
                "stale_count": 0, "top_skills": [],
                "by_source": {},
            }

        total = len(rows)
        enabled = sum(1 for r in rows if r["enabled"])
        total_uses = sum(r["times_used"] for r in rows)
        avg_rate = sum(r["success_rate"] for r in rows) / total

        # Source breakdown
        by_source: dict[str, int] = {}
        for r in rows:
            src = _safe_col(r, "source", "correction")
            by_source[src] = by_source.get(src, 0) + 1

        # Stale: not used in SKILL_STALE_DAYS days (NULL last_used_at = never used = stale)
        stale_row = self._db.fetchone(
            "SELECT COUNT(*) FROM skills WHERE "
            "last_used_at IS NULL OR last_used_at < datetime('now', ?)",
            (f"-{config.SKILL_STALE_DAYS} days",),
        )
        stale_count = stale_row[0] if stale_row else 0

        top = self._db.fetchall(
            "SELECT id, name, times_used, success_rate FROM skills WHERE enabled = 1 "
            "ORDER BY times_used DESC LIMIT 5"
        )

        return {
            "total": total,
            "enabled": enabled,
            "disabled": total - enabled,
            "total_uses": total_uses,
            "avg_success_rate": round(avg_rate, 3),
            "stale_count": stale_count,
            "by_source": by_source,
            "top_skills": [
                {
                    "id": r["id"],
                    "name": r["name"],
                    "uses": r["times_used"],
                    "success_rate": round(r["success_rate"], 3),
                }
                for r in top
            ],
        }

    def decay_stale_skills(self) -> int:
        """Reduce success_rate on skills unused for SKILL_STALE_DAYS days.

        Each decay pass reduces success_rate by 5% (multiplicative).
        Skills that cross below 0.3 after decay are auto-disabled.
        Returns number of skills decayed.
        """
        rows = self._db.fetchall(
            "SELECT id, name, success_rate FROM skills WHERE enabled = 1 AND "
            "(last_used_at IS NULL OR last_used_at < datetime('now', ?))",
            (f"-{config.SKILL_STALE_DAYS} days",),
        )
        if not rows:
            return 0

        decayed = 0
        for row in rows:
            new_rate = row["success_rate"] * 0.95
            if new_rate < 0.3:
                self._db.execute(
                    "UPDATE skills SET success_rate = ?, enabled = 0 WHERE id = ?",
                    (new_rate, row["id"]),
                )
                logger.info(
                    "Disabled stale skill #%d '%s': success_rate decayed to %.2f",
                    row["id"], row["name"], new_rate,
                )
            else:
                self._db.execute(
                    "UPDATE skills SET success_rate = ? WHERE id = ?",
                    (new_rate, row["id"]),
                )
            decayed += 1

        logger.info("Skill staleness decay: %d skills decayed", decayed)
        return decayed

    async def refine_skill(self, skill_id: int, failure_context: str) -> bool:
        """Attempt to refine a failing skill instead of just degrading it.

        LLM analyzes the failure and suggests: narrow the trigger regex,
        adjust the steps, or skip (not refinable). Returns True if refined.
        """
        skill = self.get_skill(skill_id)
        if not skill or not skill.enabled:
            return False

        prompt = (
            f"A learned skill is failing. Analyze and suggest a fix.\n\n"
            f"Skill name: {skill.name}\n"
            f"Trigger pattern: {skill.trigger_pattern}\n"
            f"Steps: {json.dumps(skill.steps)}\n"
            f"Failure context: {failure_context[:500]}\n\n"
            "Options:\n"
            "1. Narrow the trigger regex to be more specific\n"
            "2. Adjust the tool steps\n"
            "3. Skip — not refinable\n\n"
            "Return JSON: {\"action\": \"narrow\"|\"adjust\"|\"skip\", "
            "\"new_trigger\": \"...\", \"new_steps\": [...], \"reason\": \"...\"}\n"
            "For 'skip', only include action and reason."
        )

        try:
            raw = await asyncio.wait_for(
                llm.invoke_nothink(
                    [{"role": "user", "content": prompt}],
                    json_mode=True,
                    json_prefix="{",
                    max_tokens=400,
                    temperature=0.2,
                ),
                timeout=config.INTERNAL_LLM_TIMEOUT,
            )
            obj = llm.extract_json_object(raw)
            if not obj or obj.get("action") == "skip":
                return False

            action = obj.get("action", "skip")

            if action == "narrow" and obj.get("new_trigger"):
                new_trigger = obj["new_trigger"]
                try:
                    re.compile(new_trigger)
                except re.error:
                    return False
                if _is_redos_risk(new_trigger):
                    logger.warning("Skill #%d refinement rejected: ReDoS risk (%s)", skill_id, new_trigger)
                    return False
                if _is_too_broad(new_trigger):
                    logger.warning("Skill #%d refinement rejected: trigger too broad (%s)", skill_id, new_trigger)
                    return False
                self._db.execute(
                    "UPDATE skills SET trigger_pattern = ? WHERE id = ?",
                    (new_trigger, skill_id),
                )
                # Re-index with updated description
                updated = self.get_skill(skill_id)
                if updated:
                    self._embed_skill(skill_id, updated.name, new_trigger)
                logger.info("Skill #%d refined: narrowed trigger to '%s'", skill_id, new_trigger)
                return True

            elif action == "adjust" and obj.get("new_steps"):
                new_steps = obj["new_steps"]
                if not isinstance(new_steps, list):
                    return False
                valid_tools = _get_tool_names()
                for step in new_steps:
                    if not isinstance(step, dict) or step.get("tool") not in valid_tools:
                        return False
                self._db.execute(
                    "UPDATE skills SET steps = ? WHERE id = ?",
                    (json.dumps(new_steps), skill_id),
                )
                logger.info("Skill #%d refined: adjusted steps", skill_id)
                return True

        except Exception as e:
            logger.debug("Skill refinement failed: %s", e)

        return False

    # ------------------------------------------------------------------
    # Skill composition
    # ------------------------------------------------------------------

    def get_composed_steps(self, skill: Skill) -> list[dict]:
        """Expand a composed skill into its full ordered step list.

        Fetches each sub-skill by ID in order, concatenates their steps,
        then appends the composing skill's own steps last. Cycles and
        missing/disabled sub-skills are skipped silently.

        Returns skill.steps unchanged if composed_of is empty.
        """
        if not skill.composed_of:
            return skill.steps

        seen: set[int] = {skill.id}
        all_steps: list[dict] = []

        for sub_id in skill.composed_of[:10]:  # cap at 10 to prevent runaway chains
            if sub_id in seen:
                logger.debug("Skill composition cycle detected at #%d, skipping", sub_id)
                continue
            seen.add(sub_id)
            sub = self.get_skill(sub_id)
            if sub and sub.enabled:
                all_steps.extend(sub.steps)
            else:
                logger.debug("Composed sub-skill #%d unavailable, skipping", sub_id)

        # Append the composing skill's own steps last (may be empty for pure compositions)
        all_steps.extend(skill.steps)
        return all_steps or skill.steps

    # ------------------------------------------------------------------
    # Metrics & maintenance
    # ------------------------------------------------------------------

    def get_skill_stats(self) -> dict:
        """Return aggregate skill system metrics."""
        rows = self._db.fetchall("SELECT * FROM skills")
        if not rows:
            return {
                "total": 0, "enabled": 0, "disabled": 0,
                "total_uses": 0, "avg_success_rate": 0.0,
                "stale_count": 0, "top_skills": [],
                "by_source": {},
            }

        total = len(rows)
        enabled = sum(1 for r in rows if r["enabled"])
        total_uses = sum(r["times_used"] for r in rows)
        avg_rate = sum(r["success_rate"] for r in rows) / total

        # Source breakdown
        by_source: dict[str, int] = {}
        for r in rows:
            src = _safe_col(r, "source", "correction")
            by_source[src] = by_source.get(src, 0) + 1

        # Stale: not used in SKILL_STALE_DAYS days (NULL last_used_at = never used = stale)
        stale_row = self._db.fetchone(
            "SELECT COUNT(*) FROM skills WHERE "
            "last_used_at IS NULL OR last_used_at < datetime('now', ?)",
            (f"-{config.SKILL_STALE_DAYS} days",),
        )
        stale_count = stale_row[0] if stale_row else 0

        top = self._db.fetchall(
            "SELECT id, name, times_used, success_rate FROM skills WHERE enabled = 1 "
            "ORDER BY times_used DESC LIMIT 5"
        )

        return {
            "total": total,
            "enabled": enabled,
            "disabled": total - enabled,
            "total_uses": total_uses,
            "avg_success_rate": round(avg_rate, 3),
            "stale_count": stale_count,
            "by_source": by_source,
            "top_skills": [
                {
                    "id": r["id"],
                    "name": r["name"],
                    "uses": r["times_used"],
                    "success_rate": round(r["success_rate"], 3),
                }
                for r in top
            ],
        }

    def decay_stale_skills(self) -> int:
        """Reduce success_rate on skills unused for SKILL_STALE_DAYS days.

        Each decay pass reduces success_rate by 5% (multiplicative).
        Skills that cross below 0.3 after decay are auto-disabled.
        Returns number of skills decayed.
        """
        rows = self._db.fetchall(
            "SELECT id, name, success_rate FROM skills WHERE enabled = 1 AND "
            "(last_used_at IS NULL OR last_used_at < datetime('now', ?))",
            (f"-{config.SKILL_STALE_DAYS} days",),
        )
        if not rows:
            return 0

        decayed = 0
        for row in rows:
            new_rate = row["success_rate"] * 0.95
            if new_rate < 0.3:
                self._db.execute(
                    "UPDATE skills SET success_rate = ?, enabled = 0 WHERE id = ?",
                    (new_rate, row["id"]),
                )
                logger.info(
                    "Disabled stale skill #%d '%s': success_rate decayed to %.2f",
                    row["id"], row["name"], new_rate,
                )
            else:
                self._db.execute(
                    "UPDATE skills SET success_rate = ? WHERE id = ?",
                    (new_rate, row["id"]),
                )
            decayed += 1

        logger.info("Skill staleness decay: %d skills decayed", decayed)
        return decayed

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _row_to_skill(self, row) -> Skill:
        """Convert a DB row to a Skill dataclass."""
        steps = json.loads(row["steps"]) if isinstance(row["steps"], str) else row["steps"]
        row_keys = row.keys()
        composed_raw = row["composed_of"] if "composed_of" in row_keys else "[]"
        try:
            composed_of = json.loads(composed_raw) if composed_raw else []
        except (json.JSONDecodeError, TypeError):
            composed_of = []
        return Skill(
            id=row["id"],
            name=row["name"],
            trigger_pattern=row["trigger_pattern"],
            steps=steps,
            answer_template=row["answer_template"],
            learned_from=row["learned_from"],
            times_used=row["times_used"],
            success_rate=row["success_rate"],
            enabled=bool(row["enabled"]),
            created_at=row["created_at"],
            last_used_at=row["last_used_at"] if "last_used_at" in row_keys else None,
            consecutive_failures=row["consecutive_failures"] if "consecutive_failures" in row_keys else 0,
            source=row["source"] if "source" in row_keys else "correction",
            composed_of=composed_of,
        )


# ---------------------------------------------------------------------------
# ChromaDB index helpers (module-level, reused by SkillStore)
# ---------------------------------------------------------------------------

def _skill_embed_text(skill: Skill) -> str:
    """Build a natural-language description of a skill for embedding.

    Strips regex syntax noise so the embedder sees meaningful tokens.
    """
    pattern_hint = (
        skill.trigger_pattern
        .replace("(?i)", "")
        .replace("(?:", "(")
        .replace(r"\b", " ")
        .replace(r"\s+", " ")
        .replace(r"\w+", "word")
        [:100]
    )
    return f"{skill.name}. Handles queries like: {pattern_hint}"


def _index_skill(collection, skill: Skill) -> None:
    """Upsert a skill into ChromaDB (idempotent)."""
    try:
        collection.upsert(
            ids=[f"skill_{skill.id}"],
            documents=[_skill_embed_text(skill)],
            metadatas=[{"skill_id": skill.id, "name": skill.name}],
        )
    except Exception as e:
        logger.debug("Skill index upsert failed for #%d: %s", skill.id, e)


# ---------------------------------------------------------------------------
# Guards — ported from old nova's hard-won lessons
# ---------------------------------------------------------------------------

def _is_redos_risk(pattern: str) -> bool:
    """Heuristic check for regex patterns likely to cause catastrophic backtracking.

    Detects nested quantifiers like (a+)+, (a*)+, (a+)*, overlapping
    alternations in quantified groups, and similar ReDoS-prone constructs.
    """
    # Nested quantifiers: (X+)+, (X*)+, (X+)*, (X*)*
    if re.search(r'\([^)]*[+*][^)]*\)[+*{]', pattern):
        return True
    # Overlapping quantifiers without anchors: \w+\w+, .+.+
    if re.search(r'(?:\\w|\.)[+*].*(?:\\w|\.)[+*].*[+*$]', pattern):
        return True
    # Alternation in quantified groups: (a|b)+, (x|y)*
    if re.search(r'\([^)]*\|[^)]*\)[+*{]', pattern):
        return True
    return False


_BROADNESS_TEST_QUERIES = [
    "What's the weather like today?",
    "Tell me a joke",
    "How do I cook pasta?",
    "What is quantum computing?",
    "Recommend a good book",
    "How tall is Mount Everest?",
    "Translate hello to Spanish",
    "What time is it in Tokyo?",
    "How much does a Tesla cost?",
    "What's the price of gold?",
    "Compare Python and JavaScript",
    "How do I fix a flat tire?",
    "Who won the World Cup?",
    "What should I eat for dinner?",
    "How much is a flight to Paris?",
    # Temporal-framing unrelated queries — catch patterns anchored on
    # "today / current / latest / now / recent" without a domain constraint.
    # Audit P0-2: patterns like (?i)(current|today|latest) matched 1 existing
    # query (the weather one above) but needed a second hit to be flagged.
    "Today I want to learn how to play chess",
    "Give me the current bus schedule for downtown",
    "What's the latest gossip about celebrity drama?",
    "Tell me what's happening right now in my neighborhood",
    "I need the most recent study tips for exams",
    # Non-English queries to prevent non-English patterns from always passing
    "¿Cuál es el clima hoy?",          # Spanish
    "今天天气怎么样？",                    # Chinese
    "ما هو الطقس اليوم؟",              # Arabic
    "आज मौसम कैसा है?",                # Hindi
    "今日の天気は何ですか？",               # Japanese
]


def _is_too_broad(pattern: str) -> bool:
    """Test a trigger regex against 20 unrelated queries. Reject if ≥2 match."""
    try:
        regex = re.compile(pattern, re.IGNORECASE)
    except re.error:
        return True
    matches = sum(1 for q in _BROADNESS_TEST_QUERIES if regex.search(q))
    return matches >= 2


def _has_capture_group_mismatch(pattern: str, steps: list[dict], answer_template: str | None) -> bool:
    """Check if templates reference capture groups or named placeholders that don't exist.

    Catches three classes of mismatch in args_template (step tool arguments):
    - $N  numbered back-references where N > actual group count
    - {capture_N} references where N > actual group count
    - {named_placeholder} that is not {query}, not a named capture group
      (?P<name>…), and not an output_key produced by an earlier step

    answer_template is injected as raw LLM guidance text (never Python-substituted),
    so only $N / {capture_N} numeric mismatches are checked there — not {named}.
    """
    try:
        compiled = re.compile(pattern)
        num_groups = compiled.groups
        named_groups: set[str] = set(compiled.groupindex.keys())
    except re.error:
        return True

    # Valid named bindings available to every args_template:
    #   {query}          -- always available (the user's raw query)
    #   {capture_N}      -- handled by the numeric check below; skip in named check
    #   {output_key}     -- each step's output_key is available to subsequent steps
    #   (?P<name>...)    -- named capture groups from the trigger pattern
    output_keys: set[str] = set()
    for step in steps:
        ok = step.get("output_key", "")
        if ok and isinstance(ok, str):
            output_keys.add(ok)

    _EXEMPT = frozenset({"query"})
    valid_named = _EXEMPT | named_groups | output_keys

    def _check_numeric(tmpl: str) -> bool:
        """Return True if tmpl has a $N or {capture_N} that exceeds num_groups."""
        for match in re.finditer(r"\$(\d+)", tmpl):
            if int(match.group(1)) > num_groups:
                return True
        for match in re.finditer(r"\{capture_(\d+)\}", tmpl):
            if int(match.group(1)) > num_groups:
                return True
        return False

    # answer_template: only check numeric back-references — {named} are LLM hints.
    if answer_template and _check_numeric(answer_template):
        return True

    # args_template in each step: full check including {named} placeholders.
    for step in steps:
        args = step.get("args_template", {})
        templates: list[str] = []
        if isinstance(args, dict):
            templates.extend(str(v) for v in args.values())
        elif isinstance(args, str):
            templates.append(args)

        for tmpl in templates:
            if _check_numeric(tmpl):
                return True
            # {named_placeholder} — must resolve to a known binding
            for match in re.finditer(r"\{(\w+)\}", tmpl):
                name = match.group(1)
                if re.match(r"^capture_\d+$", name):
                    continue
                if name not in valid_named:
                    return True

    return False


# Fallback tool names for validation when ToolRegistry is unavailable
_FALLBACK_TOOL_NAMES = frozenset({
    "web_search", "calculator", "http_fetch", "knowledge_search",
    "code_exec", "memory_search", "file_ops", "shell_exec", "browser",
    "integration", "screenshot", "monitor", "email_send", "calendar",
    "reminder", "webhook", "delegate", "desktop", "background_task",
})


def _get_tool_names() -> set[str]:
    """Get valid tool names from ToolRegistry (dynamic), falling back to hardcoded set."""
    try:
        from app.tools.base import ToolRegistry
        from app.core.brain import get_services
        svc = get_services()
        if svc.tool_registry:
            names = set(svc.tool_registry.tool_names)
            if names:
                return names
    except Exception:
        pass
    return _FALLBACK_TOOL_NAMES


def _mentions_tool_procedure(text: str) -> bool:
    """Check if a correction message describes a tool-based procedure."""
    lower = text.lower()
    if any(tool in lower for tool in _get_tool_names()):
        return True
    procedural = re.search(
        r"(?i)\b(?:search|look\s+up|fetch|calculate|check|use|try)\b.*\b(?:first|then|instead|always|next)\b",
        text,
    )
    return bool(procedural)


def _safe_col(row, key: str, default):
    """Safely read a column from a sqlite3.Row, returning default if absent."""
    try:
        return row[key]
    except (IndexError, KeyError):
        return default


async def extract_skill_from_correction(
    correction_context: str,
    tool_history: list[dict],
    lesson_id: int | None = None,
) -> dict | None:
    """Try to extract a reusable skill from a correction.

    Creates a skill if:
    - The correction involved tool use, OR
    - The correction describes a tool-based procedure
    Returns skill dict or None.
    """
    if not tool_history and not _mentions_tool_procedure(correction_context):
        return None

    tool_info = ""
    if tool_history:
        tool_info = f"\n\nTool calls that happened: {json.dumps(tool_history)}"

    try:
        from app.core.prompt_optimizer import get_active_module
        _skill_extraction_default = (
            "You extract reusable skills from corrections. A skill is a "
            "trigger pattern (regex) and a sequence of tool calls.\n\n"
            "Given a correction, create a skill if the user describes a "
            "reusable procedure involving tool calls.\n\n"
            "Respond with JSON:\n"
            '{"name": "short_name", "trigger_pattern": "regex_to_match_queries", '
            '"steps": [{"tool": "tool_name", "args_template": {"key": "{query}"}}], '
            '"answer_template": "Use the result to answer: {result}"}\n\n'
            "IMPORTANT:\n"
            "- trigger_pattern must be a valid regex that matches similar future queries\n"
            f"- steps must reference actual tools: {', '.join(_get_tool_names())}\n"
            "- Use {query} as placeholder for the user's query in args_template\n\n"
            'If this is NOT a reusable tool procedure, respond: {"skip": true}'
        )
        skill_extraction_prompt = (
            get_active_module("skill_extraction_prompt") or _skill_extraction_default
        )
        result = await asyncio.wait_for(
            llm.invoke_nothink(
                [
                    {
                        "role": "system",
                        "content": skill_extraction_prompt,
                    },
                    {
                        "role": "user",
                        "content": f"Correction: {correction_context}{tool_info}",
                    },
                ],
                json_mode=True,
                json_prefix="{",
            ),
            timeout=config.INTERNAL_LLM_TIMEOUT,
        )

        obj = llm.extract_json_object(result)
        if not obj or obj.get("skip") or not obj.get("name"):
            return None

        pattern = obj.get("trigger_pattern", "")
        if pattern:
            try:
                re.compile(pattern)
            except re.error:
                logger.warning("Skill extraction produced invalid regex: %s", pattern)
                return None

        steps = obj.get("steps", [])
        valid_tools = _get_tool_names()
        if steps:
            for step in steps:
                if not isinstance(step, dict):
                    return None
                if step.get("tool") not in valid_tools:
                    return None
                if "args_template" not in step:
                    return None
                output_key = step.get("output_key", "")
                if output_key and not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", output_key):
                    return None

        answer_template = obj.get("answer_template")

        if _has_capture_group_mismatch(pattern, steps, answer_template):
            logger.warning("Skill extraction has capture group mismatch: %s", pattern)
            return None

        return {
            "name": obj["name"],
            "trigger_pattern": pattern,
            "steps": steps,
            "answer_template": answer_template,
            "learned_from": lesson_id,
            "source": "correction",
        }

    except Exception as e:
        logger.warning("Skill extraction failed: %s", e)
        return None
