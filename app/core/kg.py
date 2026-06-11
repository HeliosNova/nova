"""Knowledge graph — structured facts as (subject, predicate, object) triples.

SQLite-only, no NetworkX. 1-hop graph queries via recursive CTE.
Predicate normalization to 31 canonical forms.
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

    # Hard allow-list — only canonical predicates and explicit aliases pass.
    # Permissive custom-predicate matching was removed 2026-05-13: LLM
    # extractions like "founded_in_year" or "custom_metric_v2" would orphan
    # facts (stored under a unique key that no canonical query ever hits).
    # The 31 canonical predicates + ~50 alias phrases cover the common shapes;
    # anything else degrades to `related_to`, preserving the relationship
    # without splintering the predicate space.
    return "related_to"


# ---------------------------------------------------------------------------
# Stop words and normalization — shared via text_utils
# ---------------------------------------------------------------------------

from app.core.text_utils import normalize_words as _base_normalize  # noqa: E402


def _normalize_words(text: str) -> set[str]:
    """Lowercase, strip punctuation, split into word set (min length 2)."""
    return _base_normalize(text, min_length=2)


def normalize_entity(name: str) -> str:
    """Light cleanup only — strip and collapse whitespace. Casing is PRESERVED.

    The old implementation title-cased every word, which mangled the extractor's
    correct casing (BlackRock->Blackrock, OpenAI->Openai, iPhone->Iphone) and
    diverged on acronyms ("AMD" kept but "amd"->"Amd"), fragmenting entities into
    casing-only variants. Cross-variant consistency is now handled by
    KnowledgeGraph._canonical_entity (a kg_entity_aliases registry), which maps
    every casing variant to one canonical form chosen by casing richness.
    """
    return " ".join(name.split()) if name else name


def _casing_score(s: str) -> int:
    """Rank casing 'intentionality' for choosing a canonical form. Counts
    uppercase letters: BlackRock=2 > Blackrock=1; AMD=3 > Amd=1; OpenAI=3 >
    Openai=1. Naive .capitalize() artifacts (one leading capital) score lowest."""
    return sum(1 for c in s if c.isupper())


# ---------------------------------------------------------------------------
# Triple quality gate — heuristic pre-filter
# ---------------------------------------------------------------------------

_GARBAGE_PATTERNS = [
    re.compile(r"^[\d\s\.\+\-\*\/\=\(\)]+$"),        # math expressions
    re.compile(r"[/\\][\w/\\]+\.\w+"),                 # file paths
    re.compile(r"^\d+(\.\d+)?$"),                      # bare numbers
    # Monitor category labels — NEVER a real entity
    re.compile(r"^domain study[:\s]", re.IGNORECASE),
    re.compile(r"^monitor(ing)?\s+system\b", re.IGNORECASE),
    # Underscored/lowercase pseudo-entities generated from monitor names:
    # "energy_and_climate_intelligence", "trading_and_positioning_intelligence", etc.
    re.compile(r"^[a-z][a-z0-9_]*_intelligence$"),
    # Title-cased "X Architecture" as an Intel/CPU thing when conflated with Nova.
    # Nova's "nova architecture" question caused "Nova Architecture is_a Intel Cpu Platform".
    re.compile(r"^Nova\s+Architecture$", re.IGNORECASE),
]

_GARBAGE_VALUES = frozenset({
    "testuser", "test", "foo", "bar", "baz", "example",
    "null", "none", "undefined", "n/a", "na",
})


_SHORT_ENTITY_ALLOWLIST = frozenset({
    "c", "r", "go", "us", "uk", "eu", "ai", "ml",
    "os", "js", "ts", "py", "c#", "c++", "f#", "qt", "vi",
})

# Predicate-direction sanity checks. Catches the most common LLM reversals.
# Subject and object are checked against small known-entity lists; if the
# triple has the WRONG entity on the wrong side, reject (the LLM can re-emit
# it correctly next time). False negatives are fine — these are heuristics
# meant to filter the obvious garbage we observed in production.

# Known countries/regions that should never be the SUBJECT of capital_of
_KNOWN_COUNTRIES = frozenset({
    "us", "usa", "u.s.", "u.s.a.", "united states", "united states of america",
    "uk", "u.k.", "united kingdom", "great britain", "britain", "england",
    "russia", "russian federation", "soviet union", "ussr",
    "china", "people's republic of china", "prc",
    "india", "japan", "germany", "france", "italy", "spain", "canada", "mexico",
    "brazil", "argentina", "australia", "south korea", "north korea", "vietnam",
    "thailand", "indonesia", "philippines", "malaysia", "singapore", "ukraine",
    "poland", "turkey", "egypt", "israel", "iran", "iraq", "saudi arabia", "uae",
    "south africa", "nigeria", "kenya", "ethiopia", "morocco", "algeria",
    "pakistan", "bangladesh", "afghanistan", "switzerland", "sweden", "norway",
    "finland", "denmark", "netherlands", "belgium", "austria", "greece",
    "portugal", "ireland", "new zealand", "chile", "peru", "colombia", "venezuela",
    "european union", "eu", "scotland", "wales", "northern ireland",
    "taiwan", "hong kong", "korea", "czech republic", "hungary", "romania",
    "bulgaria", "serbia", "croatia", "slovenia", "slovakia", "kazakhstan",
    "uzbekistan", "syria", "lebanon", "jordan", "palestine", "qatar", "kuwait",
    "bahrain", "oman", "yemen", "libya", "tunisia", "ghana", "tanzania",
    "uganda", "zimbabwe", "angola", "mozambique", "cuba", "panama", "costa rica",
    "ecuador", "bolivia", "uruguay", "paraguay",
})

# Known orgs (companies, government agencies, sports teams) — never the SUBJECT
# of works_at / leads (those should have a person as subject).
_KNOWN_ORGS = frozenset({
    "tesla", "spacex", "apple", "google", "alphabet", "microsoft", "amazon",
    "meta", "facebook", "instagram", "twitter", "x", "openai", "anthropic",
    "nvidia", "amd", "intel", "tsmc", "samsung", "sony", "ibm", "oracle",
    "salesforce", "adobe", "uber", "lyft", "airbnb", "netflix", "disney",
    "spotify", "shopify", "stripe", "paypal", "visa", "mastercard",
    "berkshire hathaway", "jpmorgan", "goldman sachs", "morgan stanley",
    "blackrock", "bank of america", "wells fargo", "citigroup",
    "boeing", "lockheed martin", "northrop grumman", "raytheon",
    "ford", "general motors", "toyota", "honda", "bmw", "mercedes-benz",
    "exxon", "chevron", "shell", "bp", "saudi aramco",
    "deepseek", "alibaba", "tencent", "baidu", "huawei", "xiaomi", "byd",
    "sec", "fbi", "cia", "doj", "fda", "epa", "irs", "fed", "federal reserve",
    "ecb", "european central bank", "imf", "world bank",
    "un", "united nations", "nato", "who", "world health organization",
    "office of the us trade representative",
    "premier league", "nfl", "nba", "mlb", "nhl", "fifa",
    "arsenal", "chelsea", "manchester united", "manchester city", "liverpool",
    "real madrid", "barcelona", "los angeles dodgers", "new york yankees",
})


def _is_country(name: str) -> bool:
    return name.strip().lower() in _KNOWN_COUNTRIES


def _is_org(name: str) -> bool:
    return name.strip().lower() in _KNOWN_ORGS


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

    # Predicate-direction sanity (rejects obvious reversals)
    p = predicate.strip().lower()
    if p == "capital_of" and _is_country(s):
        # "Russia capital_of Moscow" — backwards
        return True
    if p in ("works_at", "leads") and _is_org(s) and not _is_org(o):
        # "Tesla works_at Elon Musk" or "Federal Reserve leads Jerome Powell"
        return True
    if p in ("created_by", "invented_by", "founded_by") and _is_org(o) and _is_org(s):
        # Two orgs in created_by is almost always wrong (acquisitions/parents
        # use different predicates — part_of, owned_by, contains)
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
    CREATE TABLE IF NOT EXISTS kg_entity_aliases (
        alias_lower TEXT PRIMARY KEY,
        canonical TEXT NOT NULL
    );
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
            # Bitemporal: transaction time of supersession (added 2026-05-16, task #29).
            # Distinct from valid_to: valid_to = when the fact stopped being true in
            # the world; superseded_at = when WE recorded the supersession. For
            # current ingest they coincide, but the column lets us answer "what did
            # Nova believe about X on date Y" by filtering out facts that were
            # logically deleted by Y. created_at is the partner column (transaction
            # time of insertion).
            ("superseded_at", "TIMESTAMP"),
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
        try:
            # Index for bitemporal as-of queries on transaction time
            self._db.execute(
                "CREATE INDEX IF NOT EXISTS idx_kg_superseded_at ON kg_facts(superseded_at)"
            )
        except Exception:
            pass

        # Backfill valid_from from created_at for existing rows
        self._db.execute(
            "UPDATE kg_facts SET valid_from = created_at WHERE valid_from IS NULL"
        )
        # Backfill superseded_at from valid_to for rows that were logically
        # deleted before this migration ran. valid_to was conflating world-validity
        # with transaction-time supersession; for historical rows the two coincide.
        self._db.execute(
            "UPDATE kg_facts SET superseded_at = valid_to "
            "WHERE superseded_at IS NULL AND valid_to IS NOT NULL"
        )

        # Insert counter for batched pruning (only prune every 50 inserts)
        self._inserts_since_prune = 0
        # Lock for concurrent supersession safety
        self._write_lock = asyncio.Lock()
        # ChromaDB collection for semantic search (lazy init)
        self._collection = None

    # --- ChromaDB vector collection for semantic KG search ---

    def _get_collection(self):
        """Lazy-init ChromaDB collection for semantic KG fact search."""
        if self._collection is None:
            try:
                import chromadb
                from ..config import config
                from .embedding import open_collection
                client = chromadb.PersistentClient(path=config.CHROMADB_PATH)
                self._collection = open_collection(
                    client, "kg_facts", reindex=self._backfill_collection,
                )
            except Exception as e:
                logger.warning("Failed to init kg_facts ChromaDB collection: %s", e)
                return None
        return self._collection

    def _backfill_collection(self, collection) -> int:
        """Populate `collection` from current KG facts, unconditionally.
        Shared by reindex_kg_facts (guarded) and the embedder-rebuild path."""
        all_rows = self._db.fetchall(
            "SELECT id, subject, predicate, object FROM kg_facts WHERE valid_to IS NULL"
        )
        if not all_rows:
            return 0
        ids, documents, metadatas = [], [], []
        for row in all_rows:
            searchable = f"{row['subject']} {row['predicate'].replace('_', ' ')} {row['object']}"
            ids.append(str(row["id"]))
            documents.append(searchable)
            metadatas.append({"subject": row["subject"], "predicate": row["predicate"]})
        if ids:
            collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
            logger.info("Reindexed %d KG facts into ChromaDB", len(ids))
        return len(ids)

    def reindex_kg_facts(self) -> int:
        """One-time backfill of existing KG facts into ChromaDB. Returns count indexed."""
        collection = self._get_collection()
        if collection is None:
            return 0
        if collection.count() > 0:
            logger.info("KG facts collection already has %d entries, skipping reindex", collection.count())
            return 0
        return self._backfill_collection(collection)

    def _add_to_vector(self, fact_id: int, subject: str, predicate: str, object_: str) -> None:
        """Add a single fact to the vector collection."""
        collection = self._get_collection()
        if collection is None:
            return
        try:
            searchable = f"{subject} {predicate.replace('_', ' ')} {object_}"
            collection.upsert(
                ids=[str(fact_id)],
                documents=[searchable],
                metadatas=[{"subject": subject, "predicate": predicate}],
            )
        except Exception as e:
            logger.debug("Failed to add KG fact %d to vector store: %s", fact_id, e)

    def _remove_from_vector(self, fact_id: int) -> None:
        """Remove a fact from the vector collection (on supersession/deletion)."""
        collection = self._get_collection()
        if collection is None:
            return
        try:
            collection.delete(ids=[str(fact_id)])
        except Exception:
            pass

    @staticmethod
    def _rrf_fuse(
        keyword_ids: list[int],
        vector_ids: list[int],
        ppr_ids: list[int] | None = None,
        k: int = 60,
    ) -> list[int]:
        """Reciprocal Rank Fusion of up to three ranked ID lists.

        Each list contributes 1/(k + rank + 1) to a fact's score; final order
        is by descending sum. The PPR list is the HippoRAG 2 graph-walk signal
        — facts reachable from query entities via the kg_facts graph rank up
        even when their literal text doesn't match the query.
        """
        scores: dict[int, float] = {}
        for rank, rid in enumerate(keyword_ids):
            scores[rid] = scores.get(rid, 0.0) + 1.0 / (k + rank + 1)
        for rank, rid in enumerate(vector_ids):
            scores[rid] = scores.get(rid, 0.0) + 1.0 / (k + rank + 1)
        if ppr_ids:
            for rank, rid in enumerate(ppr_ids):
                scores[rid] = scores.get(rid, 0.0) + 1.0 / (k + rank + 1)
        return sorted(scores, key=lambda x: scores[x], reverse=True)

    # --- Entity canonicalization ---

    def _canonical_entity(self, name: str, register: bool = False) -> str:
        """Map an entity to its one canonical casing via the kg_entity_aliases
        registry, so casing variants ("AMD"/"amd", "BlackRock"/"Blackrock") never
        fragment into separate graph nodes or duplicate facts.

        Lookup is case-insensitive (keyed on lower(name)). On a write path pass
        register=True: an unseen entity registers its (whitespace-cleaned) form as
        the canonical; a richer-cased form upgrades the canonical AND rewrites the
        entity's existing facts so storage stays consistent. Query paths pass
        register=False (lookup only — never pollute the registry with query casing).
        """
        clean = normalize_entity(name)
        if not clean:
            return clean
        low = clean.lower()
        row = self._db.fetchone(
            "SELECT canonical FROM kg_entity_aliases WHERE alias_lower = ?", (low,)
        )
        if row is not None:
            current = row["canonical"]
            if register and _casing_score(clean) > _casing_score(current):
                # A better-cased form arrived — upgrade canonical + rewrite facts.
                try:
                    self._db.execute(
                        "UPDATE kg_entity_aliases SET canonical = ? WHERE alias_lower = ?",
                        (clean, low),
                    )
                    self._db.execute(
                        "UPDATE kg_facts SET subject = ? WHERE LOWER(subject) = ?",
                        (clean, low),
                    )
                    self._db.execute(
                        "UPDATE kg_facts SET object = ? WHERE LOWER(object) = ?",
                        (clean, low),
                    )
                except Exception as e:
                    logger.debug("canonical upgrade failed for %r: %s", clean, e)
                return clean
            return current
        if register:
            try:
                self._db.execute(
                    "INSERT OR IGNORE INTO kg_entity_aliases (alias_lower, canonical) "
                    "VALUES (?, ?)", (low, clean),
                )
            except Exception as e:
                logger.debug("alias register failed for %r: %s", clean, e)
        return clean

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
        subject = normalize_entity(subject)
        predicate = normalize_predicate(predicate)
        object_ = normalize_entity(object_)

        if not subject or not object_ or len(subject) > 200 or len(object_) > 200:
            return False

        # Sanitize confidence: NaN, Inf, negative → clamp to valid range
        if not isinstance(confidence, (int, float)) or math.isnan(confidence) or math.isinf(confidence):
            confidence = 0.8  # default
        confidence = max(0.0, min(1.0, confidence))

        now = _now_iso()
        fact_valid_from = valid_from or now

        # All DB operations under the write lock to prevent TOCTOU races.
        # Sync DB work runs in a thread to avoid blocking the event loop.
        async with self._write_lock:
            result = await asyncio.to_thread(
                self._sync_add_fact, subject, predicate, object_,
                confidence, source, fact_valid_from, valid_to, provenance, now,
            )
        return result

    def _sync_add_fact(
        self, subject, predicate, object_, confidence, source,
        fact_valid_from, valid_to, provenance, now,
    ) -> bool:
        """Sync helper for add_fact — all DB operations happen here (off event loop)."""
        # Canonicalize entities (under the write lock) so casing variants collapse
        # to one form before storage — no fragmented graph nodes or dup facts.
        subject = self._canonical_entity(subject, register=True)
        object_ = self._canonical_entity(object_, register=True)
        # Check for exact duplicate
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

        # Check for contradicting facts
        conflicts = self._db.fetchall(
            "SELECT id, object, confidence FROM kg_facts "
            "WHERE LOWER(subject) = LOWER(?) AND predicate = ? AND LOWER(object) != LOWER(?) "
            "AND valid_to IS NULL",
            (subject, predicate, object_),
        )
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

            old_superseded = tx.fetchone(
                "SELECT id FROM kg_facts "
                "WHERE LOWER(subject) = LOWER(?) AND predicate = ? AND LOWER(object) = LOWER(?) "
                "AND valid_to IS NOT NULL",
                (subject, predicate, object_),
            )

            if old_superseded:
                tx.execute(
                    "UPDATE kg_facts SET valid_from = ?, valid_to = NULL, "
                    "superseded_by = NULL, confidence = ?, source = ?, "
                    "provenance = ? WHERE id = ?",
                    (fact_valid_from, confidence, source, provenance, old_superseded["id"]),
                )
                new_id = old_superseded["id"]
            else:
                # Explicitly set created_at = now (the bitemporal transaction
                # time) rather than letting SQLite's CURRENT_TIMESTAMP default
                # take it — keeps add_fact's own time control in one place and
                # makes the column reliable as a "when we recorded" filter.
                tx.execute(
                    "INSERT INTO kg_facts "
                    "(subject, predicate, object, confidence, source, "
                    " created_at, valid_from, valid_to, provenance) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (subject, predicate, object_, confidence, source,
                     now, fact_valid_from, valid_to, provenance),
                )
                new_row = tx.fetchone(
                    "SELECT id FROM kg_facts "
                    "WHERE LOWER(subject) = LOWER(?) AND predicate = ? AND LOWER(object) = LOWER(?) "
                    "AND valid_to IS NULL ORDER BY id DESC LIMIT 1",
                    (subject, predicate, object_),
                )
                new_id = new_row["id"] if new_row else None

            if conflicts and new_id is not None:
                for conflict in conflicts:
                    # Bitemporal: superseded_at records the transaction-time
                    # of the supersession (when we LEARNED the new fact),
                    # parallel to valid_to which records world-time.
                    tx.execute(
                        "UPDATE kg_facts SET superseded_by = ?, superseded_at = ? WHERE id = ?",
                        (new_id, now, conflict["id"]),
                    )
            logger.info(
                "KG: superseded %d fact(s) for %s/%s -> %s",
                len(conflicts), subject, predicate, object_,
            )

        # Add new fact to vector store for semantic search
        if new_id is not None:
            self._add_to_vector(new_id, subject, predicate, object_)
        # Remove superseded facts from vector store
        for conflict in conflicts:
            self._remove_from_vector(conflict["id"])

        self._inserts_since_prune += 1
        if self._inserts_since_prune >= _PRUNE_BATCH_SIZE:
            self._prune()
            self._inserts_since_prune = 0

        # Emit event for event-driven triggers
        try:
            from app.monitors.event_trigger import emit_event
            emit_event("internal:kg_fact_added", {"subject": subject, "predicate": predicate, "object": object_})
        except Exception:
            pass

        # PPR adjacency cache invalidation — only invalidate when a contradicting
        # supersede happened (graph topology changed) or when this is a high-confidence
        # user/correction-sourced fact. Routine extraction flows (confidence < 0.85,
        # no supersede) coast on the 5 min TTL to avoid cache thrash.
        try:
            if conflicts or (confidence >= 0.85 and source in ("user", "correction", "user_stated")):
                from app.core import ppr as _ppr
                _ppr.invalidate_cache()
        except Exception:
            pass

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
            return await asyncio.to_thread(
                self._sync_delete_fact,
                subject.strip(), normalize_predicate(predicate), object_.strip(),
            )

    def _sync_delete_fact(self, subject: str, predicate: str, object_: str) -> bool:
        """Sync helper for delete_fact."""
        row = self._db.fetchone(
            "SELECT id FROM kg_facts WHERE LOWER(subject) = LOWER(?) AND predicate = ? AND LOWER(object) = LOWER(?) AND valid_to IS NULL",
            (subject, predicate, object_),
        )
        if row:
            ok = self._retire_fact(row["id"])
            if ok:
                try:
                    from app.core import ppr as _ppr
                    _ppr.invalidate_cache()
                except Exception:
                    pass
            return ok
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
            conflicts = await asyncio.to_thread(
                self._db.fetchall,
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
        def _sync_resolve() -> bool | None:
            """Returns False to reject new fact, None to continue (allow)."""
            for conflict, keep in decisions:
                if keep == "B":
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
                    logger.info("KG contradiction resolved: kept old '%s', rejected new '%s'", conflict["object"], new_object)
                    return False
            return None

        async with self._write_lock:
            result = await asyncio.to_thread(_sync_resolve)
            if result is False:
                return False

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
            all_facts = await asyncio.to_thread(
                self._db.fetchall,
                "SELECT id, subject, predicate, object FROM kg_facts "
                "WHERE valid_to IS NULL",
            )
            ids_to_delete = []
            for row in all_facts:
                if is_garbage_triple(row["subject"], row["predicate"], row["object"]):
                    ids_to_delete.append(row["id"])

            if ids_to_delete:
                async with self._write_lock:
                    deleted_heuristic = await asyncio.to_thread(
                        self._retire_facts_batch, ids_to_delete
                    )
                logger.info("KG curation: retired %d garbage facts (heuristic)", deleted_heuristic)

        if sample_size <= 0:
            return {"heuristic": deleted_heuristic, "llm": 0}

        # Pass 2: LLM validation of lowest-confidence current facts
        low_facts = await asyncio.to_thread(
            self._db.fetchall,
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
        entity = self._canonical_entity(entity)
        if not entity:
            return []

        validity_filter = "" if include_superseded else "AND valid_to IS NULL"

        seen_ids: set[int] = set()
        visited: set[str] = set()
        results: list[dict] = []
        frontier: set[str] = {entity.lower()}

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
        """Get facts relevant to a query by hybrid keyword + semantic search.

        Uses RRF fusion of keyword overlap and ChromaDB vector similarity.
        Only returns current facts (valid_to IS NULL).
        """
        # Candidate set for relevance scoring. This MUST cover all valid facts:
        # a hard "LIMIT 500 ORDER BY confidence" silently made the majority of
        # facts unretrievable once the KG grew past 500 (found 2026-05-30 — 2734
        # valid facts, 1863 at conf>=0.95, so a freshly-relevant fact fell
        # outside the confidence window and get_relevant_facts returned []).
        # Use the KG cap so every valid fact is a candidate; keyword/vector/PPR
        # then rank by actual query relevance, not by a confidence pre-truncation.
        _cand_limit = int(getattr(_config, "MAX_KG_FACTS", 5000))
        all_facts = self._db.fetchall(
            "SELECT * FROM kg_facts WHERE valid_to IS NULL "
            "ORDER BY confidence DESC LIMIT ?",
            (_cand_limit,),
        )
        if not all_facts:
            return []

        rows_by_id = {row["id"]: row for row in all_facts}

        # --- Keyword search (existing approach) ---
        query_words = _normalize_words(query)
        keyword_ids: list[int] = []
        if query_words:
            scored: list[tuple[int, int]] = []
            for row in all_facts:
                fact_words = (
                    _normalize_words(row["subject"])
                    | _normalize_words(row["predicate"].replace("_", " "))
                    | _normalize_words(row["object"])
                )
                overlap = len(query_words & fact_words)
                if overlap >= 2:
                    scored.append((overlap, row["id"]))
            scored.sort(key=lambda x: -x[0])
            keyword_ids = [rid for _, rid in scored[:limit * 3]]

        # --- Vector search (semantic similarity via ChromaDB) ---
        vector_ids: list[int] = []
        collection = self._get_collection()
        if collection is not None and collection.count() > 0:
            try:
                results = collection.query(
                    query_texts=[query],
                    n_results=min(limit * 3, collection.count()),
                    include=["distances"],
                )
                if results and results["ids"] and results["ids"][0]:
                    # Filter by cosine distance threshold (0 = identical, 2 = opposite).
                    # Default 0.8 (sim > 0.6) suits MiniLM; configurable because
                    # modern embedders (bge-m3) place relevant pairs at different
                    # distances. RRF fusion is the real ranker; this is a coarse gate.
                    _MAX_DISTANCE = float(getattr(_config, "KG_VECTOR_MAX_DISTANCE", 0.8))
                    distances = results.get("distances", [[]])[0]
                    for rid_str, dist in zip(results["ids"][0], distances):
                        rid = int(rid_str)
                        if rid in rows_by_id and dist < _MAX_DISTANCE:
                            vector_ids.append(rid)
            except Exception as e:
                logger.debug("KG vector search failed: %s", e)

        # --- PPR (HippoRAG 2 style graph walk) ---
        # Adds a third signal: facts whose endpoints are reachable from query
        # entities via the kg_facts graph score high even when their literal
        # text doesn't match the query keywords. Captures multi-hop reasoning
        # like "tell me about Apple" surfacing facts about "Tim Cook" and
        # "iOS" without those entities being in the query.
        ppr_ids: list[int] = []
        if getattr(_config, "ENABLE_PPR_RETRIEVAL", False):
            try:
                from app.core import ppr as ppr_mod
                seeds = ppr_mod.extract_entities(query, max_seeds=6)
                if seeds:
                    ranked = ppr_mod.rank_facts_by_ppr(all_facts, seeds, top_k=limit * 3)
                    ppr_ids = [rid for rid, _ in ranked]
                    if ppr_ids:
                        logger.debug(
                            "[KG/PPR] %d facts ranked by graph walk (seeds=%s)",
                            len(ppr_ids), seeds[:3],
                        )
            except Exception as e:
                logger.warning("[KG/PPR] graph walk failed: %s", e)

        # --- RRF fusion ---
        # Require keyword matches OR PPR signal for fusion — vector-only matches
        # are too noisy in small KGs where ChromaDB returns whatever it has
        # regardless of relevance. PPR-only allowed because graph reachability
        # is a strong relevance signal even without keyword overlap (HippoRAG 2).
        if keyword_ids or ppr_ids:
            fused_ids = self._rrf_fuse(keyword_ids, vector_ids, ppr_ids)
            # Filter to valid IDs and take top limit
            top_ids = [rid for rid in fused_ids if rid in rows_by_id][:limit]
        elif query_words:
            # No keyword matches with overlap >= 2, no PPR seeds in graph,
            # and no vector boost. Do NOT fall back to overlap >= 1 — single-
            # word matches are too noisy.
            top_ids = []
        else:
            return []

        # --- Graph-neighbor enrichment (1-hop) on the FUSED result ---
        # If the fused (keyword + vector + PPR) result is short of `limit`, add
        # high-confidence neighbors of the matched entities so related facts
        # cluster together. Fixed 2026-05-30: this previously rebuilt a
        # keyword-overlap-only list and OVERWROTE `top_ids`, silently discarding
        # the vector/PPR ranking from the RRF fusion above. The fused result is
        # now authoritative; keyword/vector/PPR all contribute to ranking.
        if top_ids and len(top_ids) < limit:
            seen_ids = set(top_ids)
            entities: set[str] = set()
            for rid in top_ids:
                row = rows_by_id.get(rid)
                if row:
                    entities.add(row["subject"].lower())
                    entities.add(row["object"].lower())
            neighbor_budget = limit - len(top_ids)
            if entities and neighbor_budget > 0:
                placeholders = ",".join("?" for _ in entities)
                neighbors = self._db.fetchall(
                    f"SELECT id FROM kg_facts WHERE valid_to IS NULL "
                    f"AND (LOWER(subject) IN ({placeholders}) OR LOWER(object) IN ({placeholders})) "
                    f"ORDER BY confidence DESC LIMIT ?",
                    tuple(entities) + tuple(entities) + (neighbor_budget * 3,),
                )
                for nrow in neighbors:
                    nid = nrow["id"]
                    if nid not in seen_ids and nid in rows_by_id:
                        seen_ids.add(nid)
                        top_ids.append(nid)
                        if len(top_ids) >= limit:
                            break

        # Batch increment retrieval counts and update last_retrieved_at
        if top_ids:
            placeholders = ",".join("?" for _ in top_ids)
            try:
                self._db.execute(
                    f"UPDATE kg_facts SET times_retrieved = times_retrieved + 1, "
                    f"last_retrieved_at = datetime('now') WHERE id IN ({placeholders})",
                    tuple(top_ids),
                )
            except Exception:
                pass

        return [
            Fact(
                id=rows_by_id[rid]["id"],
                subject=rows_by_id[rid]["subject"],
                predicate=rows_by_id[rid]["predicate"],
                object=rows_by_id[rid]["object"],
                confidence=rows_by_id[rid]["confidence"],
                source=rows_by_id[rid]["source"],
                created_at=rows_by_id[rid]["created_at"],
                valid_from=rows_by_id[rid]["valid_from"] if "valid_from" in rows_by_id[rid].keys() else None,
                valid_to=rows_by_id[rid]["valid_to"] if "valid_to" in rows_by_id[rid].keys() else None,
                provenance=rows_by_id[rid]["provenance"] if "provenance" in rows_by_id[rid].keys() else "",
                superseded_by=rows_by_id[rid]["superseded_by"] if "superseded_by" in rows_by_id[rid].keys() else None,
            )
            for rid in top_ids
            if rid in rows_by_id
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
        entity = self._canonical_entity(entity)
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

    def query_as_of(
        self,
        entity: str,
        *,
        valid_at: str | None = None,
        recorded_at: str | None = None,
    ) -> list[dict]:
        """Bitemporal query: what was the KG's belief about `entity` at a
        specific (valid_time, transaction_time) point?

        Two timelines, in Memento-style bitemporal logic:
          - valid_time: when the fact was/is true in the world
              (kept in `valid_from`/`valid_to`)
          - transaction_time: when the system recorded / superseded the fact
              (kept in `created_at`/`superseded_at`; superseded_at added
               2026-05-16, task #29)

        Args:
            entity: subject or object to look for (normalized).
            valid_at: ISO timestamp. If given, return only facts whose
                world-validity window contains this instant. If `None`,
                no valid-time filter (treat all rows as world-valid).
            recorded_at: ISO timestamp. If given, return only facts that
                the KG had recorded by this instant AND had not yet
                superseded by then. If `None`, no transaction-time filter
                (treat all rows as "currently recorded").

        Returns: list of fact dicts ordered by confidence DESC. Empty if
        the entity is unknown.

        Example — "What did we believe about Alice's job on 2026-04-01?":
            kg.query_as_of("Alice", recorded_at="2026-04-01T00:00:00")
        Returns rows we had in the DB on that date and hadn't superseded yet,
        regardless of whether those records were later overturned.
        """
        entity = self._canonical_entity(entity)
        if not entity:
            return []

        clauses = ["(subject = ? OR object = ?)"]
        params: list[Any] = [entity, entity]

        # Bitemporal filter cases — split on which arguments were given so each
        # has its own clean semantic (no double-filter conflation):
        if valid_at is not None and recorded_at is not None:
            # Audit query: "Which rows EXISTED in the DB by recorded_at AND
            # were world-valid at valid_at?" Row presence at recorded_at is
            # just `created_at <= RT`; supersession status at RT is irrelevant —
            # we want to reconstruct what records we *had* about that VT period.
            clauses.append("created_at <= ?")
            clauses.append("COALESCE(valid_from, created_at) <= ?")
            clauses.append("(valid_to IS NULL OR valid_to > ?)")
            params.extend([recorded_at, valid_at, valid_at])
        elif recorded_at is not None:
            # Belief query: "What did we believe at recorded_at?" — rows that
            # existed by RT and had not yet been superseded by RT.
            clauses.append("created_at <= ?")
            clauses.append("(superseded_at IS NULL OR superseded_at > ?)")
            params.extend([recorded_at, recorded_at])
        elif valid_at is not None:
            # Historical world-time query: any row world-valid at VT, including
            # ones currently superseded (we still HAVE the historical record).
            clauses.append("COALESCE(valid_from, created_at) <= ?")
            clauses.append("(valid_to IS NULL OR valid_to > ?)")
            params.extend([valid_at, valid_at])
        else:
            # No filters — currently believed facts (mirrors query_at(None)).
            clauses.append("superseded_at IS NULL")

        sql = (
            "SELECT id, subject, predicate, object, confidence, source, "
            "created_at, valid_from, valid_to, provenance, "
            "superseded_by, superseded_at "
            "FROM kg_facts WHERE " + " AND ".join(clauses) + " "
            "ORDER BY confidence DESC"
        )
        rows = self._db.fetchall(sql, tuple(params))
        return [dict(r) for r in rows]

    def get_fact_history(self, subject: str, predicate: str) -> list[dict]:
        """Return all versions of a fact over time (current + superseded).

        Args:
            subject: The subject entity.
            predicate: The predicate (will be normalized).

        Returns:
            List of fact dicts ordered by valid_from DESC (most recent first).
        """
        subject = self._canonical_entity(subject)
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
        Superseded facts are excluded. Source provenance is appended in parens
        when non-default so the model can weight evidence by where it came from
        (monitor vs chat extraction vs user-provided correction).
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

            # Surface source for grounding — skip plain "extracted" since that's the
            # default for chat answers and adds no signal. Real provenance (monitor
            # name, "user", "correction") is informative for the model.
            src = f.provenance or f.source or ""
            src_tag = ""
            if src and src not in ("extracted", "inferred"):
                src_tag = f", src: {src[:40]}"

            lines.append(
                f"- {new_tag}{label} {f.subject} {pred} {f.object} "
                f"[confidence: {conf:.2f}{src_tag}]"
            )
        return "\n".join(lines)

    @staticmethod
    def format_summary_for_prompt(facts: list[Fact]) -> str:
        """Format facts as compact one-line summaries with IDs for lazy retrieval.

        Each line includes the fact ID so the LLM can call
        context_detail(category='kg_fact', item_id=N) for full details.
        """
        if not facts:
            return ""
        lines = []
        for f in facts:
            if f.superseded_by is not None or f.valid_to is not None:
                continue
            pred = f.predicate.replace("_", " ")
            conf = f.confidence if f.confidence is not None else 0
            lines.append(f"- [K{f.id}] {f.subject} —{pred}→ {f.object} ({conf:.1f})")
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
            def _do_decay():
                cursor = self._db.execute(
                    "UPDATE kg_facts SET confidence = MAX(0.1, confidence - ?) "
                    "WHERE created_at < datetime('now', ?) AND valid_to IS NULL "
                    "AND (last_retrieved_at IS NULL OR last_retrieved_at < datetime('now', ?))",
                    (decay_amount, cutoff, cutoff),
                )
                return cursor.rowcount
            return await asyncio.to_thread(_do_decay)

    async def hard_prune_dead_facts(self, days: int = 60, max_count: int = 1000) -> int:
        """Permanently retire facts that have NEVER been retrieved and are
        older than `days`. Runtime audit (2026-05-06) found 92% of KG facts
        are never retrieved — they dilute the useful signal and grow the DB
        without payoff. Conservative: only marks valid_to (soft retire), not
        physical delete; max_count caps per-cycle work.
        """
        async with self._write_lock:
            def _do_prune():
                cursor = self._db.execute(
                    "UPDATE kg_facts SET valid_to = datetime('now') "
                    "WHERE id IN ("
                    "  SELECT id FROM kg_facts "
                    "  WHERE valid_to IS NULL "
                    "  AND last_retrieved_at IS NULL "
                    "  AND created_at < datetime('now', ?) "
                    "  ORDER BY created_at ASC LIMIT ?"
                    ")",
                    (f"-{days} days", max_count),
                )
                return cursor.rowcount
            return await asyncio.to_thread(_do_prune)

    def get_provenance_usage_stats(self, provenance: str) -> dict:
        """Return usage stats for facts with a given provenance.

        Used to validate whether speculative provenance like 'cross_synthesis'
        actually produces useful facts (high times_retrieved) or just noise
        (high count, zero retrieval).

        Matches by prefix — e.g. provenance='cross_synthesis' will match
        'cross_synthesis:3_monitors:24h' (cross_monitor writes suffixed
        provenance for breadth metadata).
        """
        try:
            like_pattern = f"{provenance}%"
            row = self._db.fetchone(
                "SELECT "
                "  COUNT(*) AS total, "
                "  SUM(CASE WHEN times_retrieved > 0 THEN 1 ELSE 0 END) AS used, "
                "  AVG(times_retrieved) AS avg_retrievals, "
                "  MAX(times_retrieved) AS max_retrievals "
                "FROM kg_facts WHERE provenance LIKE ? AND valid_to IS NULL",
                (like_pattern,),
            )
            if not row:
                return {"total": 0, "used": 0, "avg_retrievals": 0.0, "max_retrievals": 0}
            return {
                "total": int(row["total"] or 0),
                "used": int(row["used"] or 0),
                "avg_retrievals": float(row["avg_retrievals"] or 0.0),
                "max_retrievals": int(row["max_retrievals"] or 0),
            }
        except Exception as e:
            logger.warning("Provenance usage stats failed: %s", e)
            return {"total": 0, "used": 0, "avg_retrievals": 0.0, "max_retrievals": 0}

    async def decay_unused_speculative(self, provenance: str = "cross_synthesis",
                                        days: int = 14, decay_amount: float = 0.15) -> int:
        """Aggressively decay speculative facts (e.g. cross_synthesis) that
        weren't retrieved within `days`. Closes the loop on synthesis quality —
        useful synthesis gets retrieved; useless synthesis decays out fast.

        Matches by prefix so 'cross_synthesis' covers 'cross_synthesis:3_monitors:24h' etc.
        """
        cutoff = f"-{days} days"
        like_pattern = f"{provenance}%"
        async with self._write_lock:
            def _do():
                cursor = self._db.execute(
                    "UPDATE kg_facts SET confidence = MAX(0.05, confidence - ?) "
                    "WHERE provenance LIKE ? AND valid_to IS NULL "
                    "AND created_at < datetime('now', ?) "
                    "AND COALESCE(times_retrieved, 0) = 0",
                    (decay_amount, like_pattern, cutoff),
                )
                return cursor.rowcount
            return await asyncio.to_thread(_do)
