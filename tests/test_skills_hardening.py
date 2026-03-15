"""Tests for skill hardening, learning guards, and retrieval entity relevance.

Covers: broadness guard, capture group safety, record_use + auto-disable,
lesson dedup, answer quality gate, training data rotation, entity relevance filter.
"""

from __future__ import annotations

import json
import os
from unittest.mock import AsyncMock, patch

import pytest

from app.database import SafeDB


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def skill_db(db):
    """DB + SkillStore."""
    from app.core.skills import SkillStore
    return SkillStore(db=db)


@pytest.fixture
def learning_db(db):
    """DB + LearningEngine."""
    from app.core.learning import LearningEngine
    return LearningEngine(db=db)


# ===========================================================================
# Broadness guard
# ===========================================================================

class TestBroadnessGuard:
    def test_rejects_catchall_pattern(self):
        from app.core.skills import _is_too_broad
        assert _is_too_broad("(?i).*") is True

    def test_rejects_overly_broad_price_pattern(self):
        from app.core.skills import _is_too_broad
        # "price" or "what" appears in many unrelated queries
        assert _is_too_broad("(?i).*(?:what|price).*") is True

    def test_accepts_specific_crypto_pattern(self):
        from app.core.skills import _is_too_broad
        assert _is_too_broad(r"(?i)\bcrypto\s+price\b") is False

    def test_accepts_specific_translate_pattern(self):
        from app.core.skills import _is_too_broad
        assert _is_too_broad(r"(?i)\btranslate\s+.+\s+to\s+\w+") is False

    def test_rejects_invalid_regex(self):
        from app.core.skills import _is_too_broad
        assert _is_too_broad("[invalid") is True

    def test_create_skill_returns_none_for_broad(self, skill_db):
        result = skill_db.create_skill(
            name="bad_skill",
            trigger_pattern="(?i).*",
            steps=[{"tool": "web_search", "args_template": {"query": "{query}"}}],
        )
        assert result is None

    def test_create_skill_returns_id_for_valid(self, skill_db):
        result = skill_db.create_skill(
            name="good_skill",
            trigger_pattern=r"(?i)\bcrypto\s+price\b",
            steps=[{"tool": "web_search", "args_template": {"query": "{query}"}}],
        )
        assert result is not None
        assert isinstance(result, int)


# ===========================================================================
# Capture group safety
# ===========================================================================

class TestCaptureGroupSafety:
    def test_mismatch_detected_no_groups(self):
        from app.core.skills import _has_capture_group_mismatch
        # Template references $1 but regex has no capture groups
        assert _has_capture_group_mismatch(
            r"(?i)\bprice\b",
            [{"tool": "web_search", "args_template": {"query": "$1"}}],
            None,
        ) is True

    def test_mismatch_detected_in_answer_template(self):
        from app.core.skills import _has_capture_group_mismatch
        assert _has_capture_group_mismatch(
            r"(?i)\bprice\b",
            [],
            "The price of $1 is {result}",
        ) is True

    def test_no_mismatch_with_groups(self):
        from app.core.skills import _has_capture_group_mismatch
        # Template references $1 and regex has 1 capture group
        assert _has_capture_group_mismatch(
            r"(?i)price\s+of\s+(\w+)",
            [{"tool": "web_search", "args_template": {"query": "$1 price"}}],
            None,
        ) is False

    def test_no_mismatch_without_refs(self):
        from app.core.skills import _has_capture_group_mismatch
        # No $N or {capture_N} references — always safe
        assert _has_capture_group_mismatch(
            r"(?i)\bprice\b",
            [{"tool": "web_search", "args_template": {"query": "{query}"}}],
            "Here's the result: {result}",
        ) is False

    @pytest.mark.asyncio
    async def test_extraction_rejects_capture_mismatch(self):
        """extract_skill_from_correction returns None when capture groups don't match."""
        from app.core.skills import extract_skill_from_correction

        mock_result = json.dumps({
            "name": "bad_captures",
            "trigger_pattern": r"(?i)\bprice\b",
            "steps": [{"tool": "web_search", "args_template": {"query": "$1 price"}}],
            "answer_template": "Price is $1",
        })

        with patch("app.core.skills.llm") as mock_llm:
            mock_llm.invoke_nothink = AsyncMock(return_value=mock_result)
            mock_llm.extract_json_object = lambda x: json.loads(x)

            result = await extract_skill_from_correction(
                "You should always search for crypto prices on CoinGecko",
                [{"tool": "web_search", "args": {"query": "bitcoin price"}}],
            )
            assert result is None


# ===========================================================================
# record_use + auto-disable
# ===========================================================================

class TestRecordUseAutoDisable:
    def test_record_use_updates_stats(self, skill_db):
        sid = skill_db.create_skill(
            name="test_skill",
            trigger_pattern=r"(?i)\btest\s+skill\b",
            steps=[{"tool": "web_search", "args_template": {"query": "{query}"}}],
        )
        skill_db.record_use(sid, success=True)
        skill = skill_db.get_skill(sid)
        assert skill.times_used == 1
        # EMA (alpha=0.15) from initial 0.7: 0.7 + 0.15*(1-0.7) = 0.745
        assert skill.success_rate == pytest.approx(0.745)

    def test_record_use_running_average(self, skill_db):
        sid = skill_db.create_skill(
            name="test_skill",
            trigger_pattern=r"(?i)\btest\s+skill\b",
            steps=[{"tool": "web_search", "args_template": {"query": "{query}"}}],
        )
        skill_db.record_use(sid, success=True)
        skill_db.record_use(sid, success=False)
        skill = skill_db.get_skill(sid)
        assert skill.times_used == 2
        # EMA (alpha=0.15) from initial 0.7: 0.7 -> 0.745 (success) -> 0.63325 (failure)
        assert skill.success_rate == pytest.approx(0.63325)

    def test_auto_disable_after_consecutive_failures(self, skill_db):
        sid = skill_db.create_skill(
            name="bad_skill",
            trigger_pattern=r"(?i)\bbad\s+skill\b",
            steps=[{"tool": "web_search", "args_template": {"query": "{query}"}}],
        )
        # EMA needs ~8 failures from 0.7 to drop below 0.3 (0.7 * 0.85^8 ≈ 0.19)
        for _ in range(8):
            skill_db.record_use(sid, success=False)

        skill = skill_db.get_skill(sid)
        assert skill.enabled is False
        assert skill.success_rate < 0.3

    def test_no_disable_if_rate_above_threshold(self, skill_db):
        sid = skill_db.create_skill(
            name="ok_skill",
            trigger_pattern=r"(?i)\bok\s+skill\b",
            steps=[{"tool": "web_search", "args_template": {"query": "{query}"}}],
        )
        # 2 success + 1 failure from 0.7 → rate ~0.67 → should NOT disable
        skill_db.record_use(sid, success=True)
        skill_db.record_use(sid, success=True)
        skill_db.record_use(sid, success=False)

        skill = skill_db.get_skill(sid)
        assert skill.enabled is True

    def test_disabled_skill_not_matched(self, skill_db):
        sid = skill_db.create_skill(
            name="disabled_skill",
            trigger_pattern=r"(?i)\bdisabled\s+query\b",
            steps=[{"tool": "web_search", "args_template": {"query": "{query}"}}],
        )
        # EMA needs ~8 failures from 0.7 to drop below 0.3 and auto-disable
        for _ in range(8):
            skill_db.record_use(sid, success=False)

        match = skill_db.get_matching_skill("disabled query test")
        assert match is None

    def test_no_disable_at_3_failures(self, skill_db):
        """Threshold raised from 3 to 5: 3 failures should NOT disable."""
        sid = skill_db.create_skill(
            name="surviving_skill",
            trigger_pattern=r"(?i)\bsurviving\s+skill\b",
            steps=[{"tool": "web_search", "args_template": {"query": "{query}"}}],
        )
        for _ in range(3):
            skill_db.record_use(sid, success=False)

        skill = skill_db.get_skill(sid)
        assert skill.enabled is True  # Still enabled at 3 uses


# ===========================================================================
# Lesson deduplication
# ===========================================================================

class TestLessonDedup:
    def test_duplicate_boosts_confidence(self, learning_db):
        from app.core.learning import Correction

        c1 = Correction(
            user_message="Actually the capital is Canberra",
            previous_answer="The capital of Australia is Sydney",
            topic="Capital of Australia",
            correct_answer="The capital of Australia is Canberra",
        )
        id1 = learning_db.save_lesson(c1)
        assert id1 > 0

        # Same correction again
        id2 = learning_db.save_lesson(c1)
        assert id2 == id1  # Same ID — dedup happened

        # Confidence should have increased
        lessons = learning_db.get_all_lessons()
        assert len(lessons) == 1
        assert lessons[0].confidence == pytest.approx(0.9)  # 0.8 + 0.1

    def test_different_topic_creates_new(self, learning_db):
        from app.core.learning import Correction

        c1 = Correction(
            user_message="Actually Python was created by Guido",
            previous_answer="Python was created by James Gosling",
            topic="Python creator",
            correct_answer="Python was created by Guido van Rossum",
        )
        c2 = Correction(
            user_message="Actually Java was created by James Gosling",
            previous_answer="Java was created by Guido",
            topic="Java creator",
            correct_answer="Java was created by James Gosling",
        )
        id1 = learning_db.save_lesson(c1)
        id2 = learning_db.save_lesson(c2)
        assert id1 != id2
        assert len(learning_db.get_all_lessons()) == 2


# ===========================================================================
# Answer quality gate
# ===========================================================================

class TestQualityGate:
    def test_rejects_short_content(self):
        from app.core.learning import _is_quality_content
        assert _is_quality_content("ok") is False
        assert _is_quality_content("") is False
        assert _is_quality_content(None) is False

    def test_rejects_error_phrases(self):
        from app.core.learning import _is_quality_content
        assert _is_quality_content("I don't know the answer to that question") is False
        assert _is_quality_content("I'm not sure about this topic") is False
        assert _is_quality_content("I cannot provide that information") is False
        assert _is_quality_content("I wasn't able to find anything") is False

    def test_accepts_normal_content(self):
        from app.core.learning import _is_quality_content
        assert _is_quality_content("The capital of Australia is Canberra") is True
        assert _is_quality_content("Python was created by Guido van Rossum in 1991") is True

    def test_save_lesson_rejects_garbage(self, learning_db):
        from app.core.learning import Correction

        c = Correction(
            user_message="fix this",
            previous_answer="wrong",
            topic="test",
            correct_answer="idk",  # Too short, fails quality gate
        )
        result = learning_db.save_lesson(c)
        assert result == -1
        assert len(learning_db.get_all_lessons()) == 0

    @pytest.mark.asyncio
    async def test_save_training_pair_skips_garbage(self, learning_db, tmp_path):
        import importlib
        import app.config
        os.environ["TRAINING_DATA_PATH"] = str(tmp_path / "training.jsonl")
        importlib.reload(app.config)

        await learning_db.save_training_pair(
            query="What is X?",
            bad_answer="Wrong answer here",
            good_answer="I don't know",  # Fails quality gate
        )
        path = tmp_path / "training.jsonl"
        assert not path.exists() or path.read_text().strip() == ""


# ===========================================================================
# Training data rotation
# ===========================================================================

class TestTrainingRotation:
    def test_rotation_triggers_when_over_limit(self, tmp_path):
        from app.core.learning import _rotate_training_data
        from unittest.mock import patch

        path = tmp_path / "training.jsonl"
        # Write 15 lines
        with open(path, "w") as f:
            for i in range(15):
                f.write(json.dumps({"query": f"q{i}", "chosen": f"a{i}", "rejected": "bad"}) + "\n")

        # Mock config to set low limit
        with patch("app.core.learning.config") as mock_config:
            mock_config.MAX_TRAINING_PAIRS = 10
            _rotate_training_data(path)

        with open(path) as f:
            lines = f.readlines()

        # Should keep exactly MAX (10), from the end (lines 5-14)
        assert len(lines) == 10
        assert json.loads(lines[0])["query"] == "q5"
        assert json.loads(lines[-1])["query"] == "q14"

    def test_no_rotation_under_limit(self, tmp_path):
        from app.core.learning import _rotate_training_data
        from unittest.mock import patch

        path = tmp_path / "training.jsonl"
        with open(path, "w") as f:
            for i in range(5):
                f.write(json.dumps({"query": f"q{i}"}) + "\n")

        with patch("app.core.learning.config") as mock_config:
            mock_config.MAX_TRAINING_PAIRS = 100
            _rotate_training_data(path)

        with open(path) as f:
            lines = f.readlines()
        assert len(lines) == 5  # Untouched


# ===========================================================================
# Entity relevance filter
# ===========================================================================

class TestEntityRelevanceFilter:
    def test_passes_relevant_chunks(self):
        from app.core.retriever import Chunk, _entity_relevance_filter

        chunks = [
            Chunk(chunk_id="1", document_id="d1",
                  content="The capital of France is Paris, a beautiful city.", score=0.9),
            Chunk(chunk_id="2", document_id="d2",
                  content="France has many famous landmarks and culture.", score=0.8),
        ]
        result = _entity_relevance_filter("What is the capital of France?", chunks)
        assert len(result) == 2  # Both mention France

    def test_drops_irrelevant_chunks(self):
        from app.core.retriever import Chunk, _entity_relevance_filter

        chunks = [
            Chunk(chunk_id="1", document_id="d1",
                  content="The capital of France is Paris.", score=0.9),
            Chunk(chunk_id="2", document_id="d2",
                  content="The capital of Australia is Canberra.", score=0.85),
        ]
        result = _entity_relevance_filter("What is the capital of France?", chunks)
        # "Australia" chunk should be dropped — it shares "capital" but not "France"
        assert len(result) >= 1
        assert any("France" in c.content for c in result)

    def test_always_returns_at_least_one(self):
        from app.core.retriever import Chunk, _entity_relevance_filter

        chunks = [
            Chunk(chunk_id="1", document_id="d1",
                  content="Completely unrelated content about cooking recipes.", score=0.5),
        ]
        result = _entity_relevance_filter("quantum physics experiments", chunks)
        assert len(result) >= 1  # Never returns empty

    def test_skips_filter_for_short_queries(self):
        from app.core.retriever import Chunk, _entity_relevance_filter

        chunks = [
            Chunk(chunk_id="1", document_id="d1", content="Some content.", score=0.9),
            Chunk(chunk_id="2", document_id="d2", content="Other content.", score=0.8),
        ]
        # Single content word after stop word removal
        result = _entity_relevance_filter("hello", chunks)
        assert len(result) == 2  # All returned, no filtering


# ===========================================================================
# End-to-end: correction → skill extraction → broadness guard
# ===========================================================================

class TestE2ESkillExtraction:
    @pytest.mark.asyncio
    async def test_broad_skill_rejected_e2e(self, db):
        """LLM returns a broad trigger → create_skill returns None."""
        from app.core.skills import SkillStore, extract_skill_from_correction

        store = SkillStore(db=db)

        # Mock LLM to return a broad pattern
        mock_result = json.dumps({
            "name": "answer_anything",
            "trigger_pattern": "(?i).*",  # Overly broad
            "steps": [{"tool": "web_search", "args_template": {"query": "{query}"}}],
            "answer_template": "{result}",
        })

        with patch("app.core.skills.llm") as mock_llm:
            mock_llm.invoke_nothink = AsyncMock(return_value=mock_result)
            mock_llm.extract_json_object = lambda x: json.loads(x)

            skill_data = await extract_skill_from_correction(
                "You should always search the web first",
                [{"tool": "web_search", "args": {"query": "test"}}],
                lesson_id=None,
            )

            # Extraction succeeds (LLM produced valid output)
            assert skill_data is not None
            # But creation should fail due to broadness guard
            skill_id = store.create_skill(**skill_data)
            assert skill_id is None

    @pytest.mark.asyncio
    async def test_valid_skill_created_and_matched(self, db):
        """LLM returns a valid trigger → skill created → query matches → record_use works."""
        from app.core.skills import SkillStore, extract_skill_from_correction

        store = SkillStore(db=db)

        mock_result = json.dumps({
            "name": "check_crypto_price",
            "trigger_pattern": r"(?i)\bcrypto\s+price\b",
            "steps": [{"tool": "web_search", "args_template": {"query": "{query} site:coingecko.com"}}],
            "answer_template": "The price is: {result}",
        })

        with patch("app.core.skills.llm") as mock_llm:
            mock_llm.invoke_nothink = AsyncMock(return_value=mock_result)
            mock_llm.extract_json_object = lambda x: json.loads(x)

            skill_data = await extract_skill_from_correction(
                "For crypto price queries, always search CoinGecko",
                [{"tool": "web_search", "args": {"query": "bitcoin price"}}],
            )

            assert skill_data is not None
            skill_id = store.create_skill(**skill_data)
            assert skill_id is not None

            # Query should match
            match = store.get_matching_skill("What is the crypto price of bitcoin?")
            assert match is not None
            assert match.name == "check_crypto_price"

            # Record use
            store.record_use(skill_id, success=True)
            skill = store.get_skill(skill_id)
            assert skill.times_used == 1
            # EMA (alpha=0.15) from initial 0.7: 0.7 + 0.15*(1-0.7) = 0.745
            assert skill.success_rate == pytest.approx(0.745)


# ===========================================================================
# Skill conflict resolution (Item 1)
# ===========================================================================

class TestSkillConflictResolution:
    def test_tiebreaker_prefers_more_specific(self, skill_db):
        """When multiple skills match, longer (more specific) pattern wins."""
        s1 = skill_db.create_skill("skill_a", r"(?i)\bconflict\s+alpha\b", [])
        s2 = skill_db.create_skill("skill_b", r"(?i)\bconflict\b", [])

        match = skill_db.get_matching_skill("conflict alpha test")
        assert match is not None
        assert match.id == s1  # Longer pattern is more specific, wins

    def test_tiebreaker_longest_pattern_wins(self, skill_db):
        """When multiple skills match, the longest pattern (most specific) wins."""
        s1 = skill_db.create_skill("first", r"(?i)\bsamestat\b", [])
        s2 = skill_db.create_skill("second", r"(?i)\bsamestat\s+extra\b", [])
        # Both match "samestat extra", s2 has longer pattern

        match = skill_db.get_matching_skill("samestat extra")
        assert match is not None
        assert match.id == s2  # Longer pattern is more specific, wins


# ===========================================================================
# Skill deduplication (Item 5)
# ===========================================================================

class TestSkillDedup:
    def test_duplicate_trigger_returns_same_id(self, skill_db):
        """Creating a skill with the same trigger_pattern should dedup."""
        s1 = skill_db.create_skill("crypto_v1", r"(?i)\bcrypto\s+price\b", [])
        s2 = skill_db.create_skill("crypto_v2", r"(?i)\bcrypto\s+price\b", [])
        assert s1 == s2  # Same ID returned

    def test_duplicate_boosts_success_rate(self, skill_db):
        """Deduped skill should have success_rate boosted by 0.1."""
        s1 = skill_db.create_skill("test_skill", r"(?i)\bdedup\s+test\b", [])
        skill_db.create_skill("test_v2", r"(?i)\bdedup\s+test\b", [])
        skill = skill_db.get_skill(s1)
        assert skill.success_rate == pytest.approx(0.8)  # min(1.0, 0.7 + 0.1) = 0.8

    def test_different_trigger_creates_new(self, skill_db):
        """Different trigger patterns should create separate skills."""
        s1 = skill_db.create_skill("skill_a", r"(?i)\bfoo\s+bar\b", [])
        s2 = skill_db.create_skill("skill_b", r"(?i)\bbaz\s+qux\b", [])
        assert s1 != s2


# ===========================================================================
# Expanded broadness probes (Fix 3)
# ===========================================================================

class TestExpandedBroadnessProbes:
    def test_price_cost_pattern_rejected(self):
        """Fix 3: Overly broad price/cost pattern should now be rejected."""
        from app.core.skills import _is_too_broad
        # This was skill #3's trigger — matches "how much" in many queries
        assert _is_too_broad(r"(price|cost|how much is|how much does it cost).*") is True

    def test_broadness_probes_count(self):
        """Fix 3: Should have at least 15 broadness probes."""
        from app.core.skills import _BROADNESS_TEST_QUERIES
        assert len(_BROADNESS_TEST_QUERIES) >= 15

    def test_probes_include_price_queries(self):
        """Fix 3: Probes should include price/cost related queries."""
        from app.core.skills import _BROADNESS_TEST_QUERIES
        combined = " ".join(_BROADNESS_TEST_QUERIES).lower()
        assert "price" in combined or "cost" in combined or "how much" in combined

    def test_dedup_re_enables_disabled_skill(self, skill_db):
        """Dedup on a disabled skill should re-enable it."""
        s1 = skill_db.create_skill("test", r"(?i)\breenabled\s+skill\b", [])
        skill_db.toggle_skill(s1, False)

        s2 = skill_db.create_skill("test_v2", r"(?i)\breenabled\s+skill\b", [])
        assert s1 == s2

        skill = skill_db.get_skill(s1)
        assert skill.enabled is True


# ===========================================================================
# Retriever utility functions (Item 10 partial)
# ===========================================================================

class TestContentWords:
    def test_removes_stop_words(self):
        from app.core.retriever import _content_words
        words = _content_words("What is the capital of France?")
        # With stemming: "capital" -> "capit", "France" -> "france"
        assert "capit" in words or "capital" in words
        assert "france" in words
        assert "what" not in words
        assert "the" not in words

    def test_strips_punctuation(self):
        from app.core.retriever import _content_words
        words = _content_words("Hello, world! It's great.")
        # "hello" -> "hello" (no suffix to strip), "world" -> "world"
        assert "hello" in words
        assert "world" in words

    def test_empty_string(self):
        from app.core.retriever import _content_words
        words = _content_words("")
        assert len(words) == 0


class TestSplitBySeparator:
    def test_paragraph_split(self):
        from app.core.retriever import _split_by_separator
        text = "Paragraph one here.\n\nParagraph two here.\n\nParagraph three here."
        chunks = _split_by_separator(text, "\n\n", 100, 0)
        assert len(chunks) >= 1

    def test_single_paragraph_returns_empty(self):
        from app.core.retriever import _split_by_separator
        chunks = _split_by_separator("No separators in this text", "\n\n", 100, 0)
        assert chunks == []

    def test_respects_max_chars(self):
        from app.core.retriever import _split_by_separator
        text = "A" * 50 + "\n\n" + "B" * 50 + "\n\n" + "C" * 50
        chunks = _split_by_separator(text, "\n\n", 60, 0)
        assert all(len(c) <= 60 for c in chunks)


# ===========================================================================
# Skill Refinement (Phase 4)
# ===========================================================================

class TestSkillRefinement:
    @pytest.mark.asyncio
    async def test_refine_narrows_trigger(self, skill_db):
        sid = skill_db.create_skill(
            name="price_skill",
            trigger_pattern=r"(?i)\bprice\b",
            steps=[{"tool": "web_search", "args_template": {"query": "{query}"}}],
        )
        mock_response = json.dumps({
            "action": "narrow",
            "new_trigger": r"(?i)\bcrypto\s+price\b",
            "reason": "Too broad, narrowing to crypto-specific",
        })
        with patch("app.core.skills.llm") as mock_llm:
            mock_llm.invoke_nothink = AsyncMock(return_value=mock_response)
            mock_llm.extract_json_object = lambda raw: json.loads(raw)
            result = await skill_db.refine_skill(sid, "Matched unrelated price query")

        assert result is True
        skill = skill_db.get_skill(sid)
        assert skill.trigger_pattern == r"(?i)\bcrypto\s+price\b"

    @pytest.mark.asyncio
    async def test_refine_adjusts_steps(self, skill_db):
        sid = skill_db.create_skill(
            name="search_skill",
            trigger_pattern=r"(?i)\bsearch\s+test\b",
            steps=[{"tool": "web_search", "args_template": {"query": "{query}"}}],
        )
        new_steps = [
            {"tool": "web_search", "args_template": {"query": "{query} site:example.com"}},
            {"tool": "calculator", "args_template": {"expression": "1+1"}},
        ]
        mock_response = json.dumps({
            "action": "adjust",
            "new_steps": new_steps,
            "reason": "Adding calculator step",
        })
        with patch("app.core.skills.llm") as mock_llm:
            mock_llm.invoke_nothink = AsyncMock(return_value=mock_response)
            mock_llm.extract_json_object = lambda raw: json.loads(raw)
            result = await skill_db.refine_skill(sid, "Missing calculation step")

        assert result is True
        skill = skill_db.get_skill(sid)
        assert len(skill.steps) == 2

    @pytest.mark.asyncio
    async def test_refine_skip_returns_false(self, skill_db):
        sid = skill_db.create_skill(
            name="skip_skill",
            trigger_pattern=r"(?i)\bskip\s+test\b",
            steps=[{"tool": "web_search", "args_template": {"query": "{query}"}}],
        )
        mock_response = json.dumps({"action": "skip", "reason": "Not refinable"})
        with patch("app.core.skills.llm") as mock_llm:
            mock_llm.invoke_nothink = AsyncMock(return_value=mock_response)
            mock_llm.extract_json_object = lambda raw: json.loads(raw)
            result = await skill_db.refine_skill(sid, "Some failure context")

        assert result is False

    @pytest.mark.asyncio
    async def test_refine_rejects_broad_new_trigger(self, skill_db):
        sid = skill_db.create_skill(
            name="broad_refine",
            trigger_pattern=r"(?i)\bspecific\s+test\b",
            steps=[{"tool": "web_search", "args_template": {"query": "{query}"}}],
        )
        mock_response = json.dumps({
            "action": "narrow",
            "new_trigger": "(?i).*",  # Too broad
            "reason": "Made broader by mistake",
        })
        with patch("app.core.skills.llm") as mock_llm:
            mock_llm.invoke_nothink = AsyncMock(return_value=mock_response)
            mock_llm.extract_json_object = lambda raw: json.loads(raw)
            result = await skill_db.refine_skill(sid, "Context")

        assert result is False
        # Original trigger preserved
        skill = skill_db.get_skill(sid)
        assert skill.trigger_pattern == r"(?i)\bspecific\s+test\b"

    @pytest.mark.asyncio
    async def test_refine_rejects_invalid_tool(self, skill_db):
        sid = skill_db.create_skill(
            name="bad_tool_refine",
            trigger_pattern=r"(?i)\btool\s+test\b",
            steps=[{"tool": "web_search", "args_template": {"query": "{query}"}}],
        )
        mock_response = json.dumps({
            "action": "adjust",
            "new_steps": [{"tool": "nonexistent_tool", "args_template": {}}],
            "reason": "Bad tool",
        })
        with patch("app.core.skills.llm") as mock_llm:
            mock_llm.invoke_nothink = AsyncMock(return_value=mock_response)
            mock_llm.extract_json_object = lambda raw: json.loads(raw)
            result = await skill_db.refine_skill(sid, "Context")

        assert result is False

    @pytest.mark.asyncio
    async def test_refine_nonexistent_skill(self, skill_db):
        result = await skill_db.refine_skill(9999, "failure context")
        assert result is False
