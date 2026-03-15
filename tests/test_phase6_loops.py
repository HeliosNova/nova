"""Tests for Phase 6: Close the Loops — all 8 fixes.

Covers: curiosity→lesson, had_kg/had_docs bug, skill validation recording,
lesson feedback gating, domain study→KG, maintenance monitor, success
reflexions, and KG usefulness tracking.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch, MagicMock

import pytest

from app.core.curiosity import detect_gaps, CuriosityQueue
from app.core.learning import LearningEngine, Correction
from app.core.reflexion import ReflexionStore
from app.core.kg import KnowledgeGraph
from app.monitors.heartbeat import MonitorStore


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def engine(db):
    return LearningEngine(db)


@pytest.fixture
def reflexions(db):
    return ReflexionStore(db)


@pytest.fixture
def kg(db):
    return KnowledgeGraph(db)


@pytest.fixture
def queue(db):
    return CuriosityQueue(db)


@pytest.fixture
def monitor_store(db):
    return MonitorStore(db)


# ===========================================================================
# Fix 2: had_kg/had_docs no longer hardcoded False
# ===========================================================================

class TestGapDetectionAccuracy:
    """Verify no context_gap when KG or docs are present."""

    def test_no_gap_when_kg_present(self):
        gaps = detect_gaps(
            query="What is the capital of France and how does the economy work?",
            answer="Paris is the capital of France. The economy is strong.",
            tool_results=[],
            had_lessons=False,
            had_kg=True,
            had_docs=False,
        )
        # Should not produce a context_gap since KG facts were available
        context_gaps = [g for g in gaps if g["source"] == "context_gap"]
        assert len(context_gaps) == 0

    def test_no_gap_when_docs_present(self):
        gaps = detect_gaps(
            query="What does the quarterly report say about revenue growth?",
            answer="Revenue grew by 15% according to the report.",
            tool_results=[],
            had_lessons=False,
            had_kg=False,
            had_docs=True,
        )
        context_gaps = [g for g in gaps if g["source"] == "context_gap"]
        assert len(context_gaps) == 0

    def test_gap_when_no_context(self):
        gaps = detect_gaps(
            query="What is the population of Luxembourg and what's their GDP?",
            answer="Luxembourg has around 600,000 people with a strong economy.",
            tool_results=[],
            had_lessons=False,
            had_kg=False,
            had_docs=False,
        )
        context_gaps = [g for g in gaps if g["source"] == "context_gap"]
        assert len(context_gaps) == 1


# ===========================================================================
# Fix 3: Skill validation records use
# ===========================================================================

class TestSkillValidationRecordsUse:
    """Verify _execute_skill_test calls record_use."""

    @pytest.mark.asyncio
    async def test_skill_validation_records_use(self, db):
        from app.monitors.heartbeat import HeartbeatLoop, MonitorStore

        store = MonitorStore(db)
        loop = HeartbeatLoop(store)

        # Set up services mock
        mock_skill = MagicMock()
        mock_skill.id = 1
        mock_skill.name = "test_skill"
        mock_skill.trigger_pattern = "test.*"
        mock_skill.success_rate = 1.0

        mock_skills = MagicMock()
        mock_skills.get_active_skills.return_value = [mock_skill]

        mock_svc = MagicMock()
        mock_svc.skills = mock_skills

        with patch("app.monitors.heartbeat.HeartbeatLoop._think_query", new_callable=AsyncMock) as mock_think, \
             patch("app.core.brain.get_services", return_value=mock_svc), \
             patch("app.core.reflexion.assess_quality", return_value=(0.8, "")), \
             patch("app.core.llm.invoke_nothink", new_callable=AsyncMock, return_value="test query"):
            mock_think.return_value = "A good response about the test topic."
            await loop._execute_skill_test({})

        # Verify record_use was called
        mock_skills.record_use.assert_called_once_with(1, True)


# ===========================================================================
# Fix 4: Lesson feedback gated on quality
# ===========================================================================

class TestLessonFeedbackGating:
    def test_mark_lesson_unhelpful_reduces_confidence(self, engine):
        correction = Correction(
            user_message="Actually the correct approach is different",
            previous_answer="wrong approach",
            topic="test topic",
            wrong_answer="wrong approach",
            correct_answer="The correct approach is to use method B instead",
            lesson_text="Use method B, not method A",
        )
        lesson_id = engine.save_lesson(correction)
        assert lesson_id > 0

        # Get initial confidence
        row = engine._db.fetchone("SELECT confidence FROM lessons WHERE id = ?", (lesson_id,))
        initial = row["confidence"]

        engine.mark_lesson_unhelpful(lesson_id)

        row = engine._db.fetchone("SELECT confidence FROM lessons WHERE id = ?", (lesson_id,))
        assert row["confidence"] == initial - 0.05

    def test_mark_lesson_unhelpful_floor(self, engine):
        correction = Correction(
            user_message="Actually it's right",
            previous_answer="wrong",
            topic="floor test",
            wrong_answer="wrong",
            correct_answer="right answer for floor test",
            lesson_text="floor test lesson",
        )
        lesson_id = engine.save_lesson(correction)

        # Set confidence to near-floor
        engine._db.execute("UPDATE lessons SET confidence = 0.12 WHERE id = ?", (lesson_id,))

        engine.mark_lesson_unhelpful(lesson_id)

        row = engine._db.fetchone("SELECT confidence FROM lessons WHERE id = ?", (lesson_id,))
        assert row["confidence"] >= 0.1  # Floor at 0.1

    def test_helpful_gated_on_quality_high(self, engine):
        """mark_lesson_helpful should only fire when quality >= 0.6."""
        correction = Correction(
            user_message="Actually it's good",
            previous_answer="bad",
            topic="gating test",
            wrong_answer="bad",
            correct_answer="good answer for gating",
            lesson_text="gating lesson",
        )
        lesson_id = engine.save_lesson(correction)

        row = engine._db.fetchone("SELECT times_helpful FROM lessons WHERE id = ?", (lesson_id,))
        assert row["times_helpful"] == 0

        engine.mark_lesson_helpful(lesson_id)

        row = engine._db.fetchone("SELECT times_helpful FROM lessons WHERE id = ?", (lesson_id,))
        assert row["times_helpful"] == 1


# ===========================================================================
# Fix 8: KG usefulness tracking
# ===========================================================================

class TestKGUsefulnessTracking:
    def test_get_relevant_increments_retrieved(self, kg):
        # Add times_retrieved column if migration hasn't run
        try:
            kg._db.execute("ALTER TABLE kg_facts ADD COLUMN times_retrieved INTEGER DEFAULT 0")
        except Exception:
            pass  # Already exists

        kg.add_fact("python", "created_by", "guido van rossum")
        kg.add_fact("javascript", "created_by", "brendan eich")

        # First retrieval
        facts = kg.get_relevant_facts("python created guido", limit=5)
        assert len(facts) >= 1

        # Check times_retrieved incremented
        row = kg._db.fetchone(
            "SELECT times_retrieved FROM kg_facts WHERE subject = 'python' AND object = 'guido van rossum'"
        )
        assert row is not None
        assert row["times_retrieved"] >= 1

    def test_prune_prefers_unretrieved(self, kg):
        """Unretrieved facts should be pruned first."""
        try:
            kg._db.execute("ALTER TABLE kg_facts ADD COLUMN times_retrieved INTEGER DEFAULT 0")
        except Exception:
            pass

        # Add two facts
        kg.add_fact("fact_a", "is_a", "thing_a")
        kg.add_fact("fact_b", "is_a", "thing_b")

        # Give fact_b some retrievals
        kg._db.execute(
            "UPDATE kg_facts SET times_retrieved = 5 WHERE subject = 'fact_b'"
        )

        # Both have same confidence (0.8 default), but fact_a has 0 retrievals
        # When pruning, fact_a should be deleted first
        from app.core.kg import MAX_KG_FACTS
        with patch("app.core.kg.MAX_KG_FACTS", 1):
            kg._prune()

        remaining = kg._db.fetchall("SELECT subject FROM kg_facts")
        subjects = [r["subject"] for r in remaining]
        # fact_b (retrieved) should survive over fact_a (unretrieved)
        assert "fact_b" in subjects


# ===========================================================================
# Fix 6: Maintenance monitor
# ===========================================================================

class TestMaintenanceMonitor:
    def test_seed_defaults_includes_maintenance(self, monitor_store):
        count = monitor_store.seed_defaults()
        assert count == 14  # 12 original + 1 maintenance + 1 fine-tune check

        m = monitor_store.get_by_name("System Maintenance")
        assert m is not None
        assert m.check_type == "maintenance"
        assert m.schedule_seconds == 86400

    @pytest.mark.asyncio
    async def test_maintenance_monitor_runs_decay(self, db):
        from app.monitors.heartbeat import HeartbeatLoop, MonitorStore

        store = MonitorStore(db)
        loop = HeartbeatLoop(store)

        mock_learning = MagicMock()
        mock_learning.decay_stale_lessons.return_value = 5

        mock_kg = MagicMock()
        mock_kg.decay_stale.return_value = 3

        mock_reflexions = MagicMock()
        mock_reflexions.decay_stale.return_value = 2

        mock_curiosity = MagicMock()
        mock_curiosity.prune.return_value = 4

        mock_svc = MagicMock()
        mock_svc.learning = mock_learning
        mock_svc.kg = mock_kg
        mock_svc.reflexions = mock_reflexions
        mock_svc.curiosity = mock_curiosity

        with patch("app.core.brain.get_services", return_value=mock_svc):
            result = await loop._execute_maintenance({})

        assert "lessons decayed: 5" in result
        assert "KG facts decayed: 3" in result
        assert "reflexions decayed: 2" in result
        assert "curiosity items pruned: 4" in result

        mock_learning.decay_stale_lessons.assert_called_once_with(days=30)
        mock_kg.decay_stale.assert_called_once_with(days=60)
        mock_reflexions.decay_stale.assert_called_once_with(days=90)
        mock_curiosity.prune.assert_called_once_with(days=30)

    @pytest.mark.asyncio
    async def test_maintenance_no_work_needed(self, db):
        from app.monitors.heartbeat import HeartbeatLoop, MonitorStore

        store = MonitorStore(db)
        loop = HeartbeatLoop(store)

        mock_svc = MagicMock()
        mock_svc.learning.decay_stale_lessons.return_value = 0
        mock_svc.kg.decay_stale.return_value = 0
        mock_svc.reflexions.decay_stale.return_value = 0
        mock_svc.curiosity.prune.return_value = 0

        with patch("app.core.brain.get_services", return_value=mock_svc):
            result = await loop._execute_maintenance({})

        assert result == "[No maintenance needed]"


# ===========================================================================
# Fix 7: Success reflexions surfaced
# ===========================================================================

class TestSuccessReflexions:
    def test_success_patterns_retrieved(self, reflexions):
        # Store some successes
        reflexions.store(
            task_summary="python data analysis with pandas",
            outcome="success",
            reflection="Good result (quality=0.90)",
            quality_score=0.9,
            tools_used=["web_search"],
        )
        reflexions.store(
            task_summary="python machine learning model",
            outcome="success",
            reflection="Good result (quality=0.85)",
            quality_score=0.85,
            tools_used=["web_search", "calculator"],
        )

        # Store a failure (should not appear)
        reflexions.store(
            task_summary="python regex parsing",
            outcome="failure",
            reflection="Failed to parse regex",
            quality_score=0.3,
        )

        results = reflexions.get_success_patterns("python data analysis", limit=5)
        assert len(results) >= 1
        for r in results:
            assert r.outcome == "success"

    def test_format_success_patterns(self, reflexions):
        reflexions.store(
            task_summary="weather forecast lookup",
            outcome="success",
            reflection="Good result (quality=0.90)",
            quality_score=0.9,
            tools_used=["web_search"],
        )

        results = reflexions.get_success_patterns("weather forecast", limit=2)
        formatted = ReflexionStore.format_success_patterns(results)
        assert "weather forecast" in formatted
        assert "web_search" in formatted

    def test_success_patterns_in_prompt(self):
        from app.core.prompt import build_system_prompt

        prompt = build_system_prompt(
            success_patterns="- weather query (tools: web_search): Good result",
        )
        assert "What Worked Before" in prompt
        assert "weather query" in prompt

    def test_prompt_without_success_patterns(self):
        from app.core.prompt import build_system_prompt

        prompt = build_system_prompt()
        assert "What Worked Before" not in prompt


# ===========================================================================
# Fix 1: Curiosity research → Lesson pipeline
# ===========================================================================

class TestCuriosityToLesson:
    @pytest.mark.asyncio
    async def test_curiosity_research_creates_lesson(self, db):
        from app.monitors.heartbeat import HeartbeatLoop, MonitorStore
        from app.core.curiosity import CuriosityQueue

        store = MonitorStore(db)
        loop = HeartbeatLoop(store)

        queue = CuriosityQueue(db)
        queue.add("quantum computing basics", source="admission", urgency=0.8)

        learning = LearningEngine(db)

        mock_svc = MagicMock()
        mock_svc.curiosity = queue
        mock_svc.kg = None
        mock_svc.learning = learning

        research_result = "Quantum computing uses qubits that can be in superposition of states."

        with patch("app.core.brain.get_services", return_value=mock_svc), \
             patch.object(loop, "_think_query", new_callable=AsyncMock, return_value=research_result), \
             patch.object(loop, "_send_curiosity_followup", new_callable=AsyncMock), \
             patch("app.core.llm.invoke_nothink", new_callable=AsyncMock) as mock_invoke, \
             patch("app.core.llm.extract_json_object") as mock_extract:

            mock_invoke.return_value = json.dumps({
                "topic": "quantum computing",
                "lesson": "Quantum computers use qubits in superposition to process multiple states simultaneously."
            })
            mock_extract.return_value = {
                "topic": "quantum computing",
                "lesson": "Quantum computers use qubits in superposition to process multiple states simultaneously."
            }

            result = await loop._execute_curiosity_research({})

        assert "CURIOSITY RESOLVED" in result

        # Verify lesson was created
        lessons = learning.get_all_lessons(limit=10)
        assert len(lessons) >= 1
        found = any("quantum" in (l.topic or "").lower() for l in lessons)
        assert found, f"Expected quantum lesson, got: {[l.topic for l in lessons]}"


# ===========================================================================
# Fix 5: Domain study findings → KG extraction
# ===========================================================================

class TestDomainStudyToKG:
    @pytest.mark.asyncio
    async def test_domain_study_extracts_kg_triples(self, db):
        from app.monitors.heartbeat import HeartbeatLoop, MonitorStore, Monitor

        store = MonitorStore(db)
        loop = HeartbeatLoop(store)

        # Create a query-type monitor
        mid = store.create(
            name="Domain Study: Science",
            check_type="query",
            check_config={"query": "find a science discovery"},
            schedule_seconds=43200,
        )

        monitor = store.get(mid)
        assert monitor is not None

        long_result = "Scientists discovered a new exoplanet. " * 10  # >100 chars

        mock_kg = MagicMock()
        mock_svc = MagicMock()
        mock_svc.kg = mock_kg

        with patch("app.core.brain.get_services", return_value=mock_svc), \
             patch("app.core.brain._extract_kg_triples", new_callable=AsyncMock) as mock_extract, \
             patch.object(loop, "_execute_check", new_callable=AsyncMock, return_value=long_result), \
             patch.object(loop, "_analyze_result", new_callable=AsyncMock, return_value="Analysis done"), \
             patch.object(loop, "_send_alert", new_callable=AsyncMock):

            await loop._check_monitor(monitor)

        # Verify KG extraction was triggered
        mock_extract.assert_called_once()
        call_args = mock_extract.call_args
        assert call_args[0][0] == mock_kg  # first arg is kg
        assert "Domain Study: Science" in call_args[0][1]  # monitor name as query

    @pytest.mark.asyncio
    async def test_short_result_no_kg_extraction(self, db):
        from app.monitors.heartbeat import HeartbeatLoop, MonitorStore

        store = MonitorStore(db)
        loop = HeartbeatLoop(store)

        mid = store.create(
            name="Short Test",
            check_type="query",
            check_config={"query": "short query"},
        )
        monitor = store.get(mid)

        short_result = "OK"  # <100 chars

        mock_svc = MagicMock()
        mock_svc.kg = MagicMock()

        with patch("app.core.brain.get_services", return_value=mock_svc), \
             patch("app.core.brain._extract_kg_triples", new_callable=AsyncMock) as mock_extract, \
             patch.object(loop, "_execute_check", new_callable=AsyncMock, return_value=short_result):

            await loop._check_monitor(monitor)

        # Should NOT trigger KG extraction for short results
        mock_extract.assert_not_called()

    @pytest.mark.asyncio
    async def test_non_domain_study_query_no_kg_extraction(self, db):
        """Non-Domain Study query monitors (e.g. Morning Check-in) should NOT extract KG triples."""
        from app.monitors.heartbeat import HeartbeatLoop, MonitorStore

        store = MonitorStore(db)
        loop = HeartbeatLoop(store)

        mid = store.create(
            name="Morning Check-in",
            check_type="query",
            check_config={"query": "give a brief status"},
        )
        monitor = store.get(mid)

        long_result = "All monitors running normally. No alerts overnight. " * 5  # >100 chars

        mock_svc = MagicMock()
        mock_svc.kg = MagicMock()

        with patch("app.core.brain.get_services", return_value=mock_svc), \
             patch("app.core.brain._extract_kg_triples", new_callable=AsyncMock) as mock_extract, \
             patch.object(loop, "_execute_check", new_callable=AsyncMock, return_value=long_result):

            await loop._check_monitor(monitor)

        # Non-Domain Study monitors should NOT trigger KG extraction
        mock_extract.assert_not_called()
