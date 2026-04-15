"""Tests for Feature 3: Semantic skill matching with ChromaDB embeddings.

Strategy: ChromaDB PersistentClient is mocked in all tests to avoid
model downloads and network calls. The mock simulates the collection API
so we can assert correct embed/query/delete calls.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch, call
import pytest

from app.core.skills import SkillStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_store(db, monkeypatch, *, semantic: bool = True, threshold: float = 0.65):
    """Return a SkillStore with semantic matching configured."""
    flag = "true" if semantic else "false"
    monkeypatch.setenv("ENABLE_SEMANTIC_SKILL_MATCHING", flag)
    monkeypatch.setenv("SKILL_SEMANTIC_THRESHOLD", str(threshold))
    from app.config import reset_config
    reset_config()
    return SkillStore(db=db)


def _mock_chroma_collection(similarity: float = 0.9, skill_id: int = 1):
    """Build a mock ChromaDB collection that returns one result at the given similarity."""
    collection = MagicMock()
    collection.count.return_value = 1
    # cosine distance = 2 * (1 - similarity)
    distance = 2.0 * (1.0 - similarity)
    collection.query.return_value = {
        "ids": [[f"skill_{skill_id}"]],
        "distances": [[distance]],
        "metadatas": [[{"skill_id": str(skill_id), "name": "test_skill"}]],
    }
    collection.get.return_value = {"ids": []}  # nothing pre-existing → sync adds it
    return collection


def _mock_empty_collection():
    """A collection with no skills."""
    collection = MagicMock()
    collection.count.return_value = 0
    collection.get.return_value = {"ids": []}
    return collection


# ---------------------------------------------------------------------------
# Regex match still works (fast path, no ChromaDB)
# ---------------------------------------------------------------------------

class TestRegexMatchStillWorks:
    def test_exact_regex_match_returns_without_semantic(self, db, monkeypatch):
        """Regex hit should short-circuit before semantic lookup."""
        store = _make_store(db, monkeypatch, semantic=True)
        store.create_skill(
            name="weather",
            trigger_pattern=r"what(?:'s| is) the weather",
            steps=[{"tool": "web_search", "args_template": {"query": "{query}"}}],
        )
        # _regex_match must never touch the ChromaDB collection
        with patch.object(store, "_semantic_match") as mock_sem:
            result = store._regex_match("what's the weather in London?")
        assert result is not None
        assert result.name == "weather"
        mock_sem.assert_not_called()

    def test_regex_match_wins_over_semantic(self, db, monkeypatch):
        """get_matching_skill returns regex result and never calls semantic."""
        store = _make_store(db, monkeypatch, semantic=True)
        store.create_skill(
            name="crypto_price",
            trigger_pattern=r"(?:price|value) of (\w+)",
            steps=[{"tool": "web_search", "args_template": {"query": "price {query}"}}],
        )
        # Even with semantic enabled, regex hit → semantic never invoked
        with patch.object(store, "_semantic_match", return_value=None) as mock_sem:
            result = store.get_matching_skill("price of bitcoin")
        assert result is not None
        assert result.name == "crypto_price"
        mock_sem.assert_not_called()


# ---------------------------------------------------------------------------
# Semantic fallback when regex misses
# ---------------------------------------------------------------------------

class TestSemanticFallback:
    def test_paraphrase_hit_above_threshold(self, db, monkeypatch):
        """A semantically similar query returns the skill even without regex match."""
        store = _make_store(db, monkeypatch, semantic=True, threshold=0.80)
        # Create skill with specific regex that won't match the paraphrase
        sid = store.create_skill(
            name="stock_lookup",
            trigger_pattern=r"stock price of (\w+)",
            steps=[{"tool": "web_search", "args_template": {"query": "stock {query}"}}],
        )
        assert sid is not None

        mock_col = _mock_chroma_collection(similarity=0.92, skill_id=sid)
        store._chroma_collection = mock_col

        # "share value for Apple" won't regex-match "stock price of (\w+)"
        result = store.get_matching_skill("share value for Apple")
        assert result is not None
        assert result.name == "stock_lookup"
        mock_col.query.assert_called_once()

    def test_low_similarity_returns_none(self, db, monkeypatch):
        """Below threshold → semantic returns None."""
        store = _make_store(db, monkeypatch, semantic=True, threshold=0.65)
        sid = store.create_skill(
            name="weather_skill",
            trigger_pattern=r"weather in (\w+)",
            steps=[{"tool": "web_search", "args_template": {"query": "weather {query}"}}],
        )
        assert sid is not None

        mock_col = _mock_chroma_collection(similarity=0.60, skill_id=sid)
        store._chroma_collection = mock_col

        result = store.get_matching_skill("recommend a book to read")
        assert result is None

    def test_empty_collection_returns_none(self, db, monkeypatch):
        """Empty ChromaDB collection → semantic returns None immediately."""
        store = _make_store(db, monkeypatch, semantic=True)
        mock_col = _mock_empty_collection()
        store._chroma_collection = mock_col

        result = store._semantic_match("anything")
        assert result is None
        mock_col.query.assert_not_called()

    def test_semantic_disabled_skips_lookup(self, db, monkeypatch):
        """With ENABLE_SEMANTIC_SKILL_MATCHING=false, semantic path never runs."""
        store = _make_store(db, monkeypatch, semantic=False)
        store.create_skill(
            name="some_skill",
            trigger_pattern=r"very specific trigger xyz123",
            steps=[{"tool": "web_search", "args_template": {"query": "{query}"}}],
        )
        with patch.object(store, "_semantic_match") as mock_sem:
            result = store.get_matching_skill("unrelated query that won't regex match")
        assert result is None
        mock_sem.assert_not_called()


# ---------------------------------------------------------------------------
# Embed on create / unembed on delete
# ---------------------------------------------------------------------------

class TestEmbedSync:
    def test_embed_called_on_create(self, db, monkeypatch):
        """_embed_skill is called once when a new skill is created."""
        store = _make_store(db, monkeypatch, semantic=True)
        mock_col = MagicMock()
        mock_col.get.return_value = {"ids": []}
        store._chroma_collection = mock_col

        store.create_skill(
            name="embed_test",
            trigger_pattern=r"test pattern (\w+)",
            steps=[{"tool": "web_search", "args_template": {"query": "{query}"}}],
        )

        mock_col.add.assert_called_once()
        call_kwargs = mock_col.add.call_args
        assert "skill_1" in call_kwargs.kwargs.get("ids", call_kwargs.args[0] if call_kwargs.args else [])

    def test_unembed_called_on_delete(self, db, monkeypatch):
        """_unembed_skill is called when a skill is deleted."""
        store = _make_store(db, monkeypatch, semantic=True)
        mock_col = MagicMock()
        mock_col.get.return_value = {"ids": []}
        store._chroma_collection = mock_col

        sid = store.create_skill(
            name="delete_test",
            trigger_pattern=r"delete test (\w+)",
            steps=[{"tool": "web_search", "args_template": {"query": "{query}"}}],
        )
        mock_col.reset_mock()

        deleted = store.delete_skill(sid)
        assert deleted is True
        mock_col.delete.assert_called_once_with(ids=[f"skill_{sid}"])

    def test_no_embed_when_semantic_disabled(self, db, monkeypatch):
        """When semantic matching is off, the ChromaDB collection is never initialised."""
        store = _make_store(db, monkeypatch, semantic=False)
        store.create_skill(
            name="no_embed",
            trigger_pattern=r"no embed test (\w+)",
            steps=[{"tool": "web_search", "args_template": {"query": "{query}"}}],
        )
        # Collection must never have been created
        assert store._chroma_collection is None

    def test_name_dedup_reembeds(self, db, monkeypatch):
        """Updating a skill by name dedup should re-embed with new trigger."""
        store = _make_store(db, monkeypatch, semantic=True)
        mock_col = MagicMock()
        mock_col.get.return_value = {"ids": []}
        store._chroma_collection = mock_col

        store.create_skill(
            name="dedup_skill",
            trigger_pattern=r"original trigger (\w+)",
            steps=[{"tool": "web_search", "args_template": {"query": "{query}"}}],
        )
        add_count_after_first = mock_col.add.call_count

        store.create_skill(
            name="dedup_skill",  # same name → name-dedup path
            trigger_pattern=r"updated trigger (\w+)",
            steps=[{"tool": "web_search", "args_template": {"query": "{query}"}}],
        )
        # add should have been called again for the update
        assert mock_col.add.call_count > add_count_after_first


# ---------------------------------------------------------------------------
# Regex fallback path (regex works, semantic disabled, still returns skill)
# ---------------------------------------------------------------------------

class TestRegexFallbackWithSemanticOff:
    def test_regex_still_works_when_semantic_disabled(self, db, monkeypatch):
        """Disabling semantic doesn't break regex matching."""
        store = _make_store(db, monkeypatch, semantic=False)
        store.create_skill(
            name="regex_only",
            trigger_pattern=r"find (\w+) on github",
            steps=[{"tool": "web_search", "args_template": {"query": "github {query}"}}],
        )
        result = store.get_matching_skill("find numpy on github")
        assert result is not None
        assert result.name == "regex_only"


# ---------------------------------------------------------------------------
# sync_embeddings
# ---------------------------------------------------------------------------

class TestSyncEmbeddings:
    def test_sync_skips_when_disabled(self, db, monkeypatch):
        """sync_embeddings no-ops when semantic matching is off — collection never created."""
        store = _make_store(db, monkeypatch, semantic=False)
        count = store.sync_embeddings()
        assert count == 0
        assert store._chroma_collection is None

    def test_sync_embeds_missing_skills(self, db, monkeypatch):
        """sync_embeddings adds skills that aren't yet in ChromaDB."""
        store = _make_store(db, monkeypatch, semantic=True)
        mock_col = MagicMock()
        # Simulate skill not present in ChromaDB
        mock_col.get.return_value = {"ids": []}
        store._chroma_collection = mock_col

        store.create_skill(
            name="sync_skill",
            trigger_pattern=r"sync test (\w+)",
            steps=[{"tool": "web_search", "args_template": {"query": "{query}"}}],
        )
        mock_col.reset_mock()
        mock_col.get.return_value = {"ids": []}  # still missing after reset

        count = store.sync_embeddings()
        assert count == 1
        mock_col.add.assert_called_once()

    def test_sync_skips_already_embedded(self, db, monkeypatch):
        """sync_embeddings doesn't re-embed skills already in ChromaDB."""
        store = _make_store(db, monkeypatch, semantic=True)
        mock_col = MagicMock()
        store._chroma_collection = mock_col

        store.create_skill(
            name="already_there",
            trigger_pattern=r"already there (\w+)",
            steps=[{"tool": "web_search", "args_template": {"query": "{query}"}}],
        )
        # Now simulate the skill is present in ChromaDB
        mock_col.get.return_value = {"ids": ["skill_1"]}
        mock_col.reset_mock()

        count = store.sync_embeddings()
        assert count == 0
        mock_col.add.assert_not_called()


# ---------------------------------------------------------------------------
# Semantic dedup guard — insert-time near-duplicate rejection
# ---------------------------------------------------------------------------

class TestSemanticDedupGuard:
    """_find_semantic_duplicate() is called during create_skill() to block
    near-duplicate skills from accumulating in the corpus."""

    def _dup_collection(self, similarity: float, existing_id: int = 1,
                        existing_name: str = "existing_skill") -> MagicMock:
        """Mock collection that returns one result at the given similarity."""
        col = MagicMock()
        col.count.return_value = 1
        distance = 2.0 * (1.0 - similarity)
        col.query.return_value = {
            "ids": [[f"skill_{existing_id}"]],
            "distances": [[distance]],
            "metadatas": [[{"skill_id": str(existing_id), "name": existing_name}]],
        }
        col.get.return_value = {"ids": []}
        return col

    def test_duplicate_above_threshold_rejected(self, db, monkeypatch):
        """create_skill() returns None when similarity ≥ threshold."""
        store = _make_store(db, monkeypatch, semantic=True, threshold=0.65)
        # Plant an existing skill in the DB so the name-check doesn't short-circuit
        db.execute(
            "INSERT INTO skills (name, trigger_pattern, steps, success_rate) "
            "VALUES (?, ?, ?, ?)",
            ("existing_skill", r"existing trigger (\w+)", "[]", 0.7),
        )
        mock_col = self._dup_collection(similarity=0.92, existing_id=1)
        store._chroma_collection = mock_col

        result = store.create_skill(
            name="new_duplicate_skill",
            trigger_pattern=r"duplicate trigger (\w+)",
            steps=[{"tool": "web_search", "args_template": {"query": "{query}"}}],
        )
        assert result is None, "Skill above dedup threshold should be rejected"
        # Nothing new should have been inserted
        rows = db.fetchall("SELECT * FROM skills")
        assert len(rows) == 1  # only the pre-existing one

    def test_unique_below_threshold_accepted(self, db, monkeypatch):
        """create_skill() succeeds when similarity is below threshold."""
        store = _make_store(db, monkeypatch, semantic=True, threshold=0.65)
        db.execute(
            "INSERT INTO skills (name, trigger_pattern, steps, success_rate) "
            "VALUES (?, ?, ?, ?)",
            ("existing_skill", r"existing trigger (\w+)", "[]", 0.7),
        )
        # Similarity 0.50 < 0.65 threshold → should pass
        mock_col = self._dup_collection(similarity=0.50, existing_id=1)
        store._chroma_collection = mock_col

        result = store.create_skill(
            name="genuinely_new_skill",
            trigger_pattern=r"totally different topic (\w+)",
            steps=[{"tool": "web_search", "args_template": {"query": "{query}"}}],
        )
        assert result is not None, "Skill below dedup threshold should be accepted"

    def test_empty_collection_skips_dedup(self, db, monkeypatch):
        """When the collection is empty, dedup check is skipped (nothing to compare)."""
        store = _make_store(db, monkeypatch, semantic=True, threshold=0.65)
        mock_col = MagicMock()
        mock_col.count.return_value = 0
        mock_col.get.return_value = {"ids": []}
        store._chroma_collection = mock_col

        result = store.create_skill(
            name="first_skill",
            trigger_pattern=r"first ever skill (\w+)",
            steps=[{"tool": "web_search", "args_template": {"query": "{query}"}}],
        )
        assert result is not None

    def test_semantic_disabled_skips_dedup(self, db, monkeypatch):
        """With ENABLE_SEMANTIC_SKILL_MATCHING=false, dedup check never runs."""
        store = _make_store(db, monkeypatch, semantic=False)
        # Even if collection somehow existed, query should never be called
        mock_col = MagicMock()
        store._chroma_collection = mock_col

        result = store.create_skill(
            name="semantic_off_skill",
            trigger_pattern=r"semantic off test (\w+)",
            steps=[{"tool": "web_search", "args_template": {"query": "{query}"}}],
        )
        assert result is not None
        mock_col.query.assert_not_called()

    def test_same_name_update_not_blocked(self, db, monkeypatch):
        """Name-dedup (update) path fires before semantic check — same-name update allowed."""
        store = _make_store(db, monkeypatch, semantic=True, threshold=0.65)
        mock_col = MagicMock()
        mock_col.get.return_value = {"ids": []}
        store._chroma_collection = mock_col

        # Create a skill
        sid = store.create_skill(
            name="updateable_skill",
            trigger_pattern=r"original trigger (\w+)",
            steps=[{"tool": "web_search", "args_template": {"query": "{query}"}}],
        )
        assert sid is not None

        # Now update with the same name but different trigger — name-dedup runs first
        # and should succeed even if semantic similarity would be high.
        mock_col.reset_mock()
        mock_col.get.return_value = {"ids": []}
        # Simulate high similarity (same conceptual skill, updated trigger)
        mock_col.query.return_value = {
            "ids": [[f"skill_{sid}"]],
            "distances": [[0.02]],  # very high similarity
            "metadatas": [[{"skill_id": str(sid), "name": "updateable_skill"}]],
        }

        result = store.create_skill(
            name="updateable_skill",  # same name → name-dedup path, no semantic check
            trigger_pattern=r"updated trigger (\w+)",
            steps=[{"tool": "web_search", "args_template": {"query": "{query}"}}],
        )
        assert result is not None, "Same-name update should bypass semantic dedup"

    def test_dedup_check_failure_is_non_critical(self, db, monkeypatch):
        """If ChromaDB query throws, dedup check is skipped and skill is accepted."""
        store = _make_store(db, monkeypatch, semantic=True, threshold=0.65)
        mock_col = MagicMock()
        mock_col.count.return_value = 5
        mock_col.query.side_effect = RuntimeError("ChromaDB unavailable")
        mock_col.get.return_value = {"ids": []}
        store._chroma_collection = mock_col

        result = store.create_skill(
            name="fault_tolerant_skill",
            trigger_pattern=r"fault tolerant test (\w+)",
            steps=[{"tool": "web_search", "args_template": {"query": "{query}"}}],
        )
        assert result is not None, "ChromaDB error should not block skill creation"
