"""Knowledge graph — structured facts as (subject, predicate, object) triples.

SQLite-only, no NetworkX. 1-hop graph queries via recursive CTE.
Predicate normalization to ~20 canonical forms.
Temporal tracking: facts have valid_from/valid_to for historical queries.
"""

from __future__ import annotations

import asyncio
import logging
import math
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

logger = logging.getLogger(__name__)

_PRUNE_BATCH_SIZE = 50

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class Fact:
    id: int
    subject: str
    predicate: str
    object: str
    confidence: float
    source: str
    created_at: str
    valid_from: str | None = None
    valid_to: str | None = None
    provenance: str = ""
    superseded_by: int | None = None


# ---------------------------------------------------------------------------
# Predicate normalization
# ---------------------------------------------------------------------------

CANONICAL_PREDICATES = frozenset({
    "is_a", "part_of", "located_in", "created_by", "used_for",
    "known_for", "related_to", "belongs_to", "has_property",
    "born_in", "founded_in", "capital_of", "currency_of",
    "spoken_in", "developed_by", "written_by", "caused_by",
    "contains", "produces", "leads",
    "works_at", "employed_by", "lives_in", "studied_at",
    "married_to", "member_of", "invented_by", "successor_of",
    "succeeded_by", "price_of", "version_of",
})

_PREDICATE_ALIASES: dict[str, str] = {
    "is a": "is_a", "is an": "is_a", "type of": "is_a",
    "is part of": "part_of", "part of": "part_of",
    "located in": "located_in", "is in": "located_in", "is located in": "located_in",
    "created by": "created_by", "made by": "created_by", "built by": "created_by",
    "used for": "used_for", "used in": "used_for",
    "known for": "known_for", "famous for": "known_for",
    "related to": "related_to",
    "belongs to": "belongs_to",
    "has property": "has_property", "has": "has_property",
    "born in": "born_in",
    "founded in": "founded_in", "established in": "founded_in",
    "capital of": "capital_of", "is capital of": "capital_of",
    "currency of": "currency_of",
    "spoken in": "spoken_in",
    "developed by": "developed_by",
    "written by": "written_by", "authored by": "written_by",
    "caused by": "caused_by",
    "contains": "contains", "includes": "contains",
    "produces": "produces",
    "leads": "leads",
    "works at": "works_at", "works for": "works_at", "employed at": "works_at",
    "employed by": "employed_by", "hired by": "employed_by",
    "lives in": "lives_in", "resides in": "lives_in",
    "studied at": "studied_at", "graduated from": "studied_at", "attended": "studied_at",
    "married to": "married_to", "spouse of": "married_to",
    "member of": "member_of",
    "invented by": "invented_by", "discovered by": "invented_by",
    "successor of": "successor_of", "succeeded by": "succeeded_by", "replaced by": "succeeded_by",
    "price of": "price_of", "cost of": "price_of", "costs": "price_of",
    "version of": "version_of", "variant of": "version_of",
}


def normalize_predicate(pred: str) -> str:
    """Normalize a predicate to a canonical form."""
    p = pred.strip().lower()

    # Check alias map (before underscore conversion)
    if p in _PREDICATE_ALIASES:
        return _PREDICATE_ALIASES[p]

    # Underscores
    p = p.replace(" ", "_")

    # Already canonical?
    if p in CANONICAL_PREDICATES:
        return p

    # Strip common prefixes and re-check
    for prefix in ("is_", "has_", "was_", "does_", "are_"):
        if p.startswith(prefix):
            short = p[len(prefix):]
            if short in CANONICAL_PREDICATES:
                return short
            # Check with common suffixes
            for alias_key, canon in _PREDICATE_ALIASES.items():
                if short == alias_key.replace(" ", "_"):
                    return canon

    # Allow well-formed custom predicates (lowercase, underscores, 3-31 chars, max 3 underscores)
    if re.match(r"^[a-z][a-z_]{2,30}$", p) and p.count("_") <= 3:
        return p

    return "related_to"


# ---------------------------------------------------------------------------
# Stop words and normalization — shared via text_utils
# ---------------------------------------------------------------------------

from app.core.text_utils import normalize_words as _base_normalize  # noqa: E402


def _normalize_words(text: str) -> set[str]:
    """Lowercase, strip punctuation, split into word set (min length 2)."""
    return _base_normalize(text, min_length=2)


# ---------------------------------------------------------------------------
# Triple quality gate — heuristic pre-filter
# ---------------------------------------------------------------------------

_GARBAGE_PATTERNS = [
    re.compile(r"^[\d\s\.\+\-\*\/\=\(\)]+$"),        # math expressions
    re.compile(r"[/\\][\w/\\]+\.\w+"),                 # file paths
    re.compile(r"^\d+(\.\d+)?$"),                      # bare numbers
]

_GARBAGE_VALUES = frozenset({
    "testuser", "test", "foo", "bar", "baz", "example",
    "null", "none", "undefined", "n/a", "na",
})


_SHORT_ENTITY_ALLOWLIST = frozenset({
    "c", "r", "go", "us", "uk", "eu", "ai", "ml",
    "os", "js", "ts", "py", "c#", "c++", "f#", "qt", "vi",
})


def is_garbage_triple(subject: str, predicate: str, object_: str) -> bool:
    """Return True if a triple is obvious garbage that should not be stored."""
    s, o = subject.strip().lower(), object_.strip().lower()

    # Too short (unless in the allowlist of legitimate short entities)
    if len(s) < 2 and s not in _SHORT_ENTITY_ALLOWLIST:
        return True
    if len(o) < 2 and o not in _SHORT_ENTITY_ALLOWLIST:
        return True

    # Self-referential
    if s == o:
        return True

    # Known garbage values
    if s in _GARBAGE_VALUES or o in _GARBAGE_VALUES:
        return True

    # Pattern-based rejection
    for pat in _GARBAGE_PATTERNS:
        if pat.match(s) or pat.match(o):
            return True

    return False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    """Return current UTC time as ISO string (SQLite CURRENT_TIMESTAMP format)."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _is_recent(ts: str | None, days: int = 7) -> bool:
    """Return True if a timestamp string is within the last N days."""
    if not ts:
        return False
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        # If naive, assume UTC
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        return dt >= cutoff
    except (ValueError, TypeError):
        return False


# ---------------------------------------------------------------------------
# KnowledgeGraph
# ---------------------------------------------------------------------------

from app.config import config as _config


class KnowledgeGraph:
    """Structured fact store with 1-hop graph queries and temporal tracking."""

    _SCHEMA = """
    CREATE TABLE IF NOT EXISTS kg_facts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        subject TEXT NOT NULL,
        predicate TEXT NOT NULL,
        object TEXT NOT NULL,
        confidence REAL DEFAULT 0.8,
        source TEXT DEFAULT 'extracted',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        valid_from TIMESTAMP,
        valid_to TIMESTAMP,
        provenance TEXT DEFAULT '',
        superseded_by INTEGER,
        UNIQUE(subject, predicate, object)
    );
    CREATE INDEX IF NOT EXISTS idx_kg_subject ON kg_facts(subject);
    CREATE INDEX IF NOT EXISTS idx_kg_object ON kg_facts(object);
    """

    def __init__(self, db):
        self._db = db
        # Create table if not exists (safe to call multiple times)
        for stmt in self._SCHEMA.strip().split(";"):
            stmt = stmt.strip()
            if stmt:
                self._db.execute(stmt)

        # Migration: add temporal columns if missing (for existing databases)
        for col, typedef in [
            ("valid_from", "TIMESTAMP"),
            ("valid_to", "TIMESTAMP"),
            ("provenance", "TEXT DEFAULT ''"),
            ("superseded_by", "INTEGER"),
            ("times_retrieved", "INTEGER DEFAULT 0"),
        ]:
            try:
                self._db.execute(f"ALTER TABLE kg_facts ADD COLUMN {col} {typedef}")
            except Exception:
                pass  # Column already exists

        # Create index on valid_to (must come after migration adds the column)
        try:
            self._db.execute(
                "CREATE INDEX IF NOT EXISTS idx_kg_valid_to ON kg_facts(valid_to)"
            )
        except Exception:
            pass

        # Backfill valid_from from created_at for existing rows
        self._db.execute(
            "UPDATE kg_facts SET valid_from = created_at WHERE valid_from IS NULL"
        )

        # Insert counter for batched pruning (only prune every 50 inserts)
        self._inserts_since_prune = 0
        # Lock for concurrent supersession safety
        self._write_lock = asyncio.Lock()

    # --- Core operations ---

    async def add_fact(
        self,
        subject: str,
        predicate: str,
        object_: str,
        confidence: float = 0.8,
        source: str = "extracted",
        valid_from: str | None = None,
        valid_to: str | None = None,
        provenance: str = "",
    ) -> bool:
        """Add or update a fact. Returns True if added/updated.

        When a fact contradicts an existing one (same subject+predicate,
        different object), the old fact is superseded rather than deleted,
        creating a temporal trail.
        """
        subject = subject.strip()
        predicate = normalize_predicate(predicate)
        object_ = object_.strip()

        if not subject or not object_ or len(subject) > 200 or len(object_) > 200:
            return False

        # Sanitize confidence: NaN, Inf, negative → clamp to valid range
        if not isinstance(confidence, (int, float)) or math.isnan(confidence) or math.isinf(confidence):
            confidence = 0.8  # default
        confidence = max(0.0, min(1.0, confidence))

        now = _now_iso()
        fact_valid_from = valid_from or now

        # All DB operations under the write lock to prevent TOCTOU races
        # (duplicate check, conflict resolution, insert, prune counter)
        async with self._write_lock:
            # Check for exact duplicate (same subject+predicate+object) — current
            # Use LOWER() for case-insensitive matching to preserve original casing
            existing = self._db.fetchone(
                "SELECT id, confidence FROM kg_facts "
                "WHERE LOWER(subject) = LOWER(?) AND predicate = ? AND LOWER(object) = LOWER(?) "
                "AND valid_to IS NULL",
                (subject, predicate, object_),
            )

            if existing:
                if confidence > existing["confidence"]:
                    self._db.execute(
                        "UPDATE kg_facts SET confidence = ?, source = ?, "
                        "provenance = CASE WHEN ? != '' THEN ? ELSE provenance END "
                        "WHERE id = ?",
                        (confidence, source, provenance, provenance, existing["id"]),
                    )
                    return True
                return False

            # Check for contradicting facts (same subject+predicate, different object, still current)
            # Also check reversed direction: same object+predicate, different subject
            # (e.g. "Arsenal leads Premier League" contradicts "Man Utd leads Premier League")
            conflicts = self._db.fetchall(
                "SELECT id, object, confidence FROM kg_facts "
                "WHERE LOWER(subject) = LOWER(?) AND predicate = ? AND LOWER(object) != LOWER(?) "
                "AND valid_to IS NULL",
                (subject, predicate, object_),
            )
            # Also find inverse contradictions: same object being "led" by a different subject
            # Only for predicates that imply uniqueness (leads, is_leader_of, etc.)
            _UNIQUE_PREDICATES = {"leads", "is_leader_of", "is_president_of", "is_ceo_of",
                                  "is_capital_of", "is_champion_of"}
            if predicate in _UNIQUE_PREDICATES:
                inverse_conflicts = self._db.fetchall(
                    "SELECT id, object, confidence FROM kg_facts "
                    "WHERE LOWER(subject) != LOWER(?) AND predicate = ? AND LOWER(object) = LOWER(?) "
                    "AND valid_to IS NULL",
                    (subject, predicate, object_),
                )
                conflicts = list(conflicts) + list(inverse_conflicts)

            # Supersede conflicting facts + insert new fact atomically
            with self._db.transaction() as tx:
                for conflict in conflicts:
                    tx.execute(
                        "UPDATE kg_facts SET valid_to = ? WHERE id = ?",
                        (now, conflict["id"]),
                    )

                # Check for a previously-superseded row with the same triple
                # (the UNIQUE constraint means we can't INSERT a duplicate).
                # If found, reactivate it instead of inserting.
                old_superseded = tx.fetchone(
                    "SELECT id FROM kg_facts "
                    "WHERE LOWER(subject) = LOWER(?) AND predicate = ? AND LOWER(object) = LOWER(?) "
                    "AND valid_to IS NOT NULL",
                    (subject, predicate, object_),
                )

                if old_superseded:
                    # Reactivate: clear valid_to/superseded_by, update metadata
                    tx.execute(
                        "UPDATE kg_facts SET valid_from = ?, valid_to = NULL, "
                        "superseded_by = NULL, confidence = ?, source = ?, "
                        "provenance = ? WHERE id = ?",
                        (fact_valid_from, confidence, source, provenance, old_superseded["id"]),
                    )
                    new_id = old_superseded["id"]
                else:
                    # Insert the new fact
                    tx.execute(
                        "INSERT INTO kg_facts "
                        "(subject, predicate, object, confidence, source, valid_from, valid_to, provenance) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        (subject, predicate, object_, confidence, source, fact_valid_from, valid_to, provenance),
                    )
                    # Get the new fact's ID
                    new_row = tx.fetchone(
                        "SELECT id FROM kg_facts "
                        "WHERE LOWER(subject) = LOWER(?) AND predicate = ? AND LOWER(object) = LOWER(?) "
                        "AND valid_to IS NULL ORDER BY id DESC LIMIT 1",
                        (subject, predicate, object_),
                    )
                    new_id = new_row["id"] if new_row else None

                # Set superseded_by on old conflicting facts
                if conflicts and new_id is not None:
                    for conflict in conflicts:
                        tx.execute(
                            "UPDATE kg_facts SET superseded_by = ? WHERE id = ?",
                            (new_id, conflict["id"]),
                        )
                logger.info(
                    "KG: superseded %d fact(s) for %s/%s -> %s",
                    len(conflicts), subject, predicate, object_,
                )

            self._inserts_since_prune += 1
            if self._inserts_since_prune >= _PRUNE_BATCH_SIZE:
                self._prune()
                self._inserts_since_prune = 0

        return True

    def _retire_fact(self, fact_id: int) -> bool:
        """Retire a fact by setting valid_to instead of deleting.

        This preserves temporal history. Works for single fact retirement.
        Returns True if a row was updated.
        """
        cursor = self._db.execute(
            "UPDATE kg_facts SET valid_to = CURRENT_TIMESTAMP WHERE id = ? AND valid_to IS NULL",
            (fact_id,),
        )
        return cursor.rowcount > 0

    def _retire_facts_batch(self, fact_ids: list[int]) -> int:
        """Retire multiple facts by setting valid_to. Returns count retired."""
        if not fact_ids:
            return 0
        placeholders = ",".join("?" for _ in fact_ids)
        cursor = self._db.execute(
            f"UPDATE kg_facts SET valid_to = CURRENT_TIMESTAMP "
            f"WHERE id IN ({placeholders}) AND valid_to IS NULL",
            tuple(fact_ids),
        )
        return cursor.rowcount

    async def delete_fact(self, subject: str, predicate: str, object_: str) -> bool:
        """Retire a specific fact triple (temporal retirement, not hard delete)."""
        async with self._write_lock:
            row = self._db.fetchone(
                "SELECT id FROM kg_facts WHERE LOWER(subject) = LOWER(?) AND predicate = ? AND LOWER(object) = LOWER(?) AND valid_to IS NULL",
                (subject.strip(), normalize_predicate(predicate), object_.strip()),
            )
            if row:
                return self._retire_fact(row["id"])
            return False

    async def check_and_resolve_contradictions(
        self,
        subject: str,
        predicate: str,
        new_object: str,
        new_confidence: float = 0.8,
    ) -> bool:
        """Check for contradicting facts and resolve via LLM. Returns True if safe to add.

        Uses read-under-lock -> LLM call (no lock) -> re-read-and-write-under-lock
        pattern to avoid holding the lock during slow LLM calls while still
        preventing stale-data races.
        """
        subject = subject.strip()
        predicate = normalize_predicate(predicate)
        new_object = new_object.strip()

        # Phase 1: Read under lock — snapshot the conflicts
        async with self._write_lock:
            conflicts = self._db.fetchall(
                "SELECT id, object, confidence FROM kg_facts "
                "WHERE LOWER(subject) = LOWER(?) AND predicate = ? AND LOWER(object) != LOWER(?) "
                "AND valid_to IS NULL",
                (subject, predicate, new_object),
            )
        if not conflicts:
            return True  # no contradiction

        # Phase 2: LLM calls outside the lock (slow I/O, no DB mutation)
        from app.core import llm

        decisions: list[tuple[dict, str]] = []  # (conflict_row, keep_verdict)
        for conflict in conflicts:
            old_object = conflict["object"]

            prompt = (
                f"Two facts conflict. Which is correct?\n"
                f"A: {subject} {predicate.replace('_', ' ')} {old_object}\n"
                f"B: {subject} {predicate.replace('_', ' ')} {new_object}\n\n"
                'Reply with JSON: {"keep": "A"} or {"keep": "B"} or {"keep": "both"} '
                'if they are not actually contradictory.'
            )
            try:
                raw = await llm.invoke_nothink(
                    [{"role": "user", "content": prompt}],
                    json_mode=True,
                    json_prefix="{",
                    max_tokens=50,
                    temperature=0.1,
                )
                obj = llm.extract_json_object(raw)
                if not obj:
                    continue
                keep = str(obj.get("keep", "both")).upper()
                decisions.append((conflict, keep))
            except Exception as e:
                logger.debug("KG contradiction check failed (allowing both): %s", e)

        # Phase 3: Re-read and write under lock — verify data hasn't gone stale
        async with self._write_lock:
            for conflict, keep in decisions:
                if keep == "B":
                    # Re-check that the conflict row is still current (not retired by another task)
                    still_current = self._db.fetchone(
                        "SELECT id FROM kg_facts WHERE id = ? AND valid_to IS NULL",
                        (conflict["id"],),
                    )
                    if not still_current:
                        logger.debug("KG contradiction: conflict id=%d already retired, skipping", conflict["id"])
                        continue
                    now = _now_iso()
                    self._db.execute(
                        "UPDATE kg_facts SET valid_to = ? WHERE id = ? AND valid_to IS NULL",
                        (now, conflict["id"]),
                    )
                    logger.info("KG contradiction resolved: superseded old '%s' for new '%s'", conflict["object"], new_object)
                elif keep == "A":
                    # Old fact wins — don't add new
                    logger.info("KG contradiction resolved: kept old '%s', rejected new '%s'", conflict["object"], new_object)
                    return False
                # "both" — not a real contradiction, allow both

        return True

    async def curate(self, sample_size: int = 20, *, heuristic: bool = True) -> dict:
        """Run curation: heuristic cleanup + LLM validation of low-confidence facts.

        Only curates current facts (valid_to IS NULL). Superseded facts are
        preserved as historical records.

        Args:
            sample_size: Number of low-confidence facts to validate via LLM (0 to skip).
            heuristic: Whether to run the heuristic filter pass.

        Returns dict with counts of deleted facts.
        """
        deleted_heuristic = 0
        deleted_llm = 0

        # Pass 1: Heuristic filters (only current facts)
        if heuristic:
            all_facts = self._db.fetchall(
                "SELECT id, subject, predicate, object FROM kg_facts "
                "WHERE valid_to IS NULL"
            )
            ids_to_delete = []
            for row in all_facts:
                if is_garbage_triple(row["subject"], row["predicate"], row["object"]):
                    ids_to_delete.append(row["id"])

            if ids_to_delete:
                async with self._write_lock:
                    deleted_heuristic = self._retire_facts_batch(ids_to_delete)
                logger.info("KG curation: retired %d garbage facts (heuristic)", deleted_heuristic)

        if sample_size <= 0:
            return {"heuristic": deleted_heuristic, "llm": 0}

        # Pass 2: LLM validation of lowest-confidence current facts
        low_facts = self._db.fetchall(
            "SELECT id, subject, predicate, object, confidence FROM kg_facts "
            "WHERE valid_to IS NULL "
            "ORDER BY confidence ASC LIMIT ?",
            (sample_size,),
        )
        if not low_facts:
            return {"heuristic": deleted_heuristic, "llm": 0}

        # Batch into a single LLM call
        lines = []
        for i, f in enumerate(low_facts):
            lines.append(f"{i+1}. {f['subject']} {f['predicate'].replace('_', ' ')} {f['object']}")
        batch_text = "\n".join(lines)

        from app.core import llm as llm_mod

        prompt = (
            f"Rate each fact as 'keep' or 'garbage'. Garbage = obviously wrong, "
            f"nonsensical, test data, or trivially useless.\n\n{batch_text}\n\n"
            f'Return JSON: {{"results": [{{"id": 1, "verdict": "keep"}}, ...]}}'
        )
        try:
            raw = await llm_mod.invoke_nothink(
                [{"role": "user", "content": prompt}],
                json_mode=True,
                json_prefix="{",
                max_tokens=500,
                temperature=0.1,
            )
            obj = llm_mod.extract_json_object(raw)
            if obj and "results" in obj:
                garbage_ids = []
                for r in obj["results"]:
                    idx = r.get("id", 0)
                    if 1 <= idx <= len(low_facts) and r.get("verdict") == "garbage":
                        garbage_ids.append(low_facts[idx - 1]["id"])
                if garbage_ids:
                    async with self._write_lock:
                        deleted_llm = self._retire_facts_batch(garbage_ids)
                    logger.info("KG curation: retired %d garbage facts (LLM)", deleted_llm)
        except Exception as e:
            logger.warning("KG LLM curation failed (heuristic pass still ran): %s", e)

        return {"heuristic": deleted_heuristic, "llm": deleted_llm}

    # --- Querying ---

    def query(
        self,
        entity: str,
        hops: int = 1,
        max_results: int = 200,
        include_superseded: bool = False,
    ) -> list[dict]:
        """Get facts within N hops of an entity.

        Uses iterative BFS (1 query per hop) instead of recursive CTE
        to avoid SQLite limitations with multiple self-references.

        Args:
            entity: The entity to start from.
            hops: Number of hops to traverse.
            max_results: Maximum number of results.
            include_superseded: If False (default), only return current facts.
        """
        entity = entity.strip().lower()
        if not entity:
            return []

        validity_filter = "" if include_superseded else "AND valid_to IS NULL"

        seen_ids: set[int] = set()
        visited: set[str] = set()
        results: list[dict] = []
        frontier: set[str] = {entity}

        for depth in range(hops + 1):
            if not frontier or len(results) >= max_results:
                break

            placeholders = ",".join("?" for _ in frontier)
            params = tuple(frontier) + tuple(frontier)
            rows = self._db.fetchall(
                f"SELECT id, subject, predicate, object, confidence, source "
                f"FROM kg_facts "
                f"WHERE (LOWER(subject) IN ({placeholders}) OR LOWER(object) IN ({placeholders})) "
                f"{validity_filter}",
                params,
            )

            next_entities: set[str] = set()
            for r in rows:
                if r["id"] in seen_ids:
                    continue
                seen_ids.add(r["id"])
                results.append({
                    "id": r["id"],
                    "subject": r["subject"],
                    "predicate": r["predicate"],
                    "object": r["object"],
                    "confidence": r["confidence"],
                    "source": r["source"],
                    "depth": depth,
                })
                next_entities.add(r["subject"].lower())
                next_entities.add(r["object"].lower())

            visited.update(frontier)
            frontier = next_entities - visited  # only truly new entities
            # Cap frontier size to prevent query explosion on highly-connected graphs
            if len(frontier) > _config.KG_GRAPH_MAX_FRONTIER:
                frontier = set(list(frontier)[:_config.KG_GRAPH_MAX_FRONTIER])

        results.sort(key=lambda x: (x["depth"], -(x["confidence"] or 0)))
        final = results[:max_results]

        # Batch-update times_retrieved for all returned facts
        if final:
            ret_ids = [r["id"] for r in final if r.get("id") is not None]
            if ret_ids:
                placeholders = ",".join("?" for _ in ret_ids)
                try:
                    self._db.execute(
                        f"UPDATE kg_facts SET times_retrieved = times_retrieved + 1 "
                        f"WHERE id IN ({placeholders})",
                        tuple(ret_ids),
                    )
                except Exception:
                    pass  # backward compat if column missing

        return final

    def search(
        self,
        text: str,
        limit: int = 10,
        include_history: bool = False,
    ) -> list[dict]:
        """Search facts by text in subject or object.

        Args:
            text: Search term.
            limit: Maximum results.
            include_history: If True, include superseded facts.
        """
        text = text.strip().lower()
        if not text:
            return []

        # Escape LIKE wildcards
        escaped = text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")

        validity_filter = "" if include_history else "AND valid_to IS NULL"

        rows = self._db.fetchall(
            f"SELECT id, subject, predicate, object, confidence, source "
            f"FROM kg_facts "
            f"WHERE (subject LIKE ? ESCAPE '\\' OR object LIKE ? ESCAPE '\\') "
            f"{validity_filter} "
            f"ORDER BY confidence DESC LIMIT ?",
            (f"%{escaped}%", f"%{escaped}%", limit),
        )
        results = [dict(r) for r in rows]

        # Batch-update times_retrieved for all returned facts
        if results:
            ret_ids = [r["id"] for r in results if r.get("id") is not None]
            if ret_ids:
                placeholders = ",".join("?" for _ in ret_ids)
                try:
                    self._db.execute(
                        f"UPDATE kg_facts SET times_retrieved = times_retrieved + 1 "
                        f"WHERE id IN ({placeholders})",
                        tuple(ret_ids),
                    )
                except Exception:
                    pass  # backward compat if column missing

        return results

    def get_relevant_facts(self, query: str, limit: int = 8) -> list[Fact]:
        """Get facts relevant to a query by keyword overlap.

        Same scoring pattern as LearningEngine.get_relevant_lessons().
        Only returns current facts (valid_to IS NULL).
        """
        all_facts = self._db.fetchall(
            "SELECT * FROM kg_facts WHERE valid_to IS NULL "
            "ORDER BY confidence DESC LIMIT 500"
        )
        if not all_facts:
            return []

        query_words = _normalize_words(query)
        if not query_words:
            return []

        scored: list[tuple[int, dict]] = []

        for row in all_facts:
            fact_words = (
                _normalize_words(row["subject"])
                | _normalize_words(row["predicate"].replace("_", " "))
                | _normalize_words(row["object"])
            )
            overlap = len(query_words & fact_words)
            if overlap >= 2:
                scored.append((overlap, row))

        scored.sort(key=lambda x: (-x[0], -x[1]["confidence"]))

        top = scored[:limit]

        # Batch increment retrieval counts and update last_retrieved_at
        if top:
            retrieved_ids = [row["id"] for _, row in top]
            placeholders = ",".join("?" for _ in retrieved_ids)
            try:
                self._db.execute(
                    f"UPDATE kg_facts SET times_retrieved = times_retrieved + 1, "
                    f"last_retrieved_at = datetime('now') WHERE id IN ({placeholders})",
                    tuple(retrieved_ids),
                )
            except Exception:
                pass  # backward compat if column missing

        return [
            Fact(
                id=row["id"],
                subject=row["subject"],
                predicate=row["predicate"],
                object=row["object"],
                confidence=row["confidence"],
                source=row["source"],
                created_at=row["created_at"],
                valid_from=row["valid_from"] if "valid_from" in row.keys() else None,
                valid_to=row["valid_to"] if "valid_to" in row.keys() else None,
                provenance=row["provenance"] if "provenance" in row.keys() else "",
                superseded_by=row["superseded_by"] if "superseded_by" in row.keys() else None,
            )
            for _, row in top
        ]

    # --- Temporal query methods ---

    def query_at(self, entity: str, at_time: str | None = None) -> list[dict]:
        """Query facts that were valid at a specific point in time.

        Args:
            entity: The entity to query (matched against subject or object).
            at_time: ISO timestamp string. If None, returns current facts
                     (where valid_to IS NULL).

        Returns:
            List of fact dicts valid at the given time.
        """
        entity = entity.strip().lower()
        if not entity:
            return []

        if at_time is None:
            # Return current facts
            rows = self._db.fetchall(
                "SELECT id, subject, predicate, object, confidence, source, "
                "created_at, valid_from, valid_to, provenance "
                "FROM kg_facts "
                "WHERE (subject = ? OR object = ?) AND valid_to IS NULL "
                "ORDER BY confidence DESC",
                (entity, entity),
            )
        else:
            rows = self._db.fetchall(
                "SELECT id, subject, predicate, object, confidence, source, "
                "created_at, valid_from, valid_to, provenance "
                "FROM kg_facts "
                "WHERE (subject = ? OR object = ?) "
                "AND COALESCE(valid_from, created_at) <= ? "
                "AND (valid_to IS NULL OR valid_to > ?) "
                "ORDER BY confidence DESC",
                (entity, entity, at_time, at_time),
            )

        return [dict(r) for r in rows]

    def get_fact_history(self, subject: str, predicate: str) -> list[dict]:
        """Return all versions of a fact over time (current + superseded).

        Args:
            subject: The subject entity.
            predicate: The predicate (will be normalized).

        Returns:
            List of fact dicts ordered by valid_from DESC (most recent first).
        """
        subject = subject.strip().lower()
        predicate = normalize_predicate(predicate)

        rows = self._db.fetchall(
            "SELECT id, subject, predicate, object, confidence, source, "
            "created_at, valid_from, valid_to, provenance, superseded_by "
            "FROM kg_facts "
            "WHERE subject = ? AND predicate = ? "
            "ORDER BY valid_from DESC",
            (subject, predicate),
        )
        return [dict(r) for r in rows]

    def get_changes_since(self, since: str, limit: int = 50) -> list[dict]:
        """Return facts created or superseded since a given timestamp.

        Useful for "what changed in the last week?" queries.

        Args:
            since: ISO timestamp string.
            limit: Maximum results.

        Returns:
            List of fact dicts that were created or had their valid_to set
            since the given timestamp.
        """
        rows = self._db.fetchall(
            "SELECT id, subject, predicate, object, confidence, source, "
            "created_at, valid_from, valid_to, provenance, superseded_by "
            "FROM kg_facts "
            "WHERE valid_from >= ? OR (valid_to IS NOT NULL AND valid_to >= ?) "
            "ORDER BY COALESCE(valid_to, valid_from) DESC "
            "LIMIT ?",
            (since, since, limit),
        )
        return [dict(r) for r in rows]

    # --- Formatting ---

    @staticmethod
    def format_for_prompt(facts: list[Fact]) -> str:
        """Format facts as a prompt-ready string with confidence and temporal labels.

        Facts with valid_from within the last 7 days get a [NEW] label.
        Superseded facts are excluded.
        """
        if not facts:
            return ""
        lines = []
        for f in facts:
            # Skip superseded facts
            if f.superseded_by is not None or f.valid_to is not None:
                continue

            pred = f.predicate.replace("_", " ")
            conf = f.confidence if f.confidence is not None else 0
            label = "[HIGH]" if conf >= 0.8 else ("[MED]" if conf >= 0.5 else "[LOW]")

            # Add [NEW] for recently-added facts
            new_tag = ""
            if _is_recent(f.valid_from, days=7):
                new_tag = "[NEW] "

            lines.append(
                f"- {new_tag}{label} {f.subject} {pred} {f.object} "
                f"[confidence: {conf:.2f}]"
            )
        return "\n".join(lines)

    # --- Management ---

    def get_all_facts(self, limit: int = 100, offset: int = 0) -> list[Fact]:
        """Paginated fact listing."""
        rows = self._db.fetchall(
            "SELECT * FROM kg_facts ORDER BY created_at DESC LIMIT ? OFFSET ?",
            (limit, offset),
        )
        return [
            Fact(
                id=r["id"], subject=r["subject"], predicate=r["predicate"],
                object=r["object"], confidence=r["confidence"],
                source=r["source"], created_at=r["created_at"],
                valid_from=r["valid_from"] if "valid_from" in r.keys() else None,
                valid_to=r["valid_to"] if "valid_to" in r.keys() else None,
                provenance=r["provenance"] if "provenance" in r.keys() else "",
                superseded_by=r["superseded_by"] if "superseded_by" in r.keys() else None,
            )
            for r in rows
        ]

    def get_top_entities(self, limit: int = 10) -> list[dict]:
        """Return the top entities by fact count (current facts only).

        Returns list of dicts with 'subject' and 'cnt' keys, ordered by count descending.
        """
        rows = self._db.fetchall(
            "SELECT subject, COUNT(*) as cnt FROM kg_facts "
            "WHERE valid_to IS NULL GROUP BY subject ORDER BY cnt DESC LIMIT ?",
            (limit,),
        )
        return [dict(r) for r in rows]

    def get_stats(self) -> dict:
        """Return KG statistics."""
        total = self._db.fetchone("SELECT COUNT(*) AS c FROM kg_facts")["c"]
        current = self._db.fetchone(
            "SELECT COUNT(*) AS c FROM kg_facts WHERE valid_to IS NULL"
        )["c"]
        superseded = total - current
        entities_row = self._db.fetchone(
            "SELECT COUNT(*) AS c FROM ("
            "SELECT subject AS e FROM kg_facts WHERE valid_to IS NULL "
            "UNION SELECT object FROM kg_facts WHERE valid_to IS NULL)"
        )
        predicates = self._db.fetchone(
            "SELECT COUNT(DISTINCT predicate) AS c FROM kg_facts WHERE valid_to IS NULL"
        )["c"]
        return {
            "total_facts": total,
            "current_facts": current,
            "superseded_facts": superseded,
            "unique_entities": entities_row["c"] if entities_row else 0,
            "unique_predicates": predicates,
        }

    def _prune(self) -> None:
        """If current kg_facts exceed _config.MAX_KG_FACTS, delete oldest low-confidence ones.

        Only prunes current facts. Superseded facts are historical and not counted.
        """
        count_row = self._db.fetchone(
            "SELECT COUNT(*) AS c FROM kg_facts WHERE valid_to IS NULL"
        )
        count = count_row["c"] if count_row else 0
        if count <= _config.MAX_KG_FACTS:
            return
        excess = count - _config.MAX_KG_FACTS
        # Retire (set valid_to) instead of hard-deleting to preserve temporal history
        prune_rows = self._db.fetchall(
            "SELECT id FROM kg_facts "
            "WHERE valid_to IS NULL "
            "ORDER BY times_retrieved ASC, confidence ASC, created_at ASC "
            "LIMIT ?",
            (excess,),
        )
        prune_ids = [r["id"] for r in prune_rows]
        retired = self._retire_facts_batch(prune_ids)
        logger.info("Pruned (retired) %d KG facts (over %d limit)", retired, _config.MAX_KG_FACTS)

    async def decay_stale(self, days: int = 60, decay_amount: float = 0.05) -> int:
        """Lower confidence on old current facts. Returns count affected."""
        cutoff = f"-{days} days"
        async with self._write_lock:
            cursor = self._db.execute(
                "UPDATE kg_facts SET confidence = MAX(0.1, confidence - ?) "
                "WHERE created_at < datetime('now', ?) AND valid_to IS NULL "
                "AND (last_retrieved_at IS NULL OR last_retrieved_at < datetime('now', ?))",
                (decay_amount, cutoff, cutoff),
            )
            return cursor.rowcount
