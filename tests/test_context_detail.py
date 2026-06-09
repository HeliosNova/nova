"""Tests for context_detail tool and summary formatters."""

import pytest

from app.core.prompt import format_lessons_summary_for_prompt


# ---------------------------------------------------------------------------
# Summary formatters
# ---------------------------------------------------------------------------


class FakeLesson:
    def __init__(self, id, topic, lesson_text="", correct_answer="", confidence=0.8):
        self.id = id
        self.topic = topic
        self.lesson_text = lesson_text
        self.correct_answer = correct_answer
        self.wrong_answer = ""
        self.confidence = confidence


def test_lessons_summary_empty():
    assert format_lessons_summary_for_prompt([]) == ""


def test_lessons_summary_single():
    lessons = [FakeLesson(42, "Python print", "print() is a function in Python 3")]
    result = format_lessons_summary_for_prompt(lessons)
    assert "[L42]" in result
    assert "[HIGH]" in result
    assert "Python print" in result


def test_lessons_summary_truncates():
    long_text = "A" * 200
    lessons = [FakeLesson(1, "test", long_text)]
    result = format_lessons_summary_for_prompt(lessons)
    assert "..." in result
    # Should not exceed 80 chars for the summary part
    lines = result.strip().split("\n")
    summary_line = [l for l in lines if l.startswith("- [L")][0]
    # The ID + label + topic + summary total can exceed 80, but the summary part itself is truncated
    assert len(long_text) > len(summary_line)


def test_lessons_summary_confidence_labels():
    lessons = [
        FakeLesson(1, "high", "text1", confidence=0.9),
        FakeLesson(2, "med", "text2", confidence=0.6),
        FakeLesson(3, "low", "text3", confidence=0.3),
    ]
    result = format_lessons_summary_for_prompt(lessons)
    assert "[HIGH]" in result
    assert "[MED]" in result
    assert "[LOW]" in result


def test_lessons_summary_includes_header():
    lessons = [FakeLesson(1, "test", "text")]
    result = format_lessons_summary_for_prompt(lessons)
    assert "## Lessons (Summaries)" in result
    assert "context_detail" in result


# ---------------------------------------------------------------------------
# KG summary
# ---------------------------------------------------------------------------


def test_kg_summary():
    from app.core.kg import Fact, KnowledgeGraph

    facts = [
        Fact(id=7, subject="Python", predicate="is_a", object="programming language",
             confidence=0.95, source="extracted", created_at="2026-01-01",
             valid_from=None, valid_to=None, provenance="", superseded_by=None),
    ]
    result = KnowledgeGraph.format_summary_for_prompt(facts)
    assert "[K7]" in result
    assert "Python" in result
    assert "is a" in result  # predicate with underscores replaced


def test_kg_summary_skips_superseded():
    from app.core.kg import Fact, KnowledgeGraph

    facts = [
        Fact(id=1, subject="old", predicate="is_a", object="thing",
             confidence=0.8, source="test", created_at="2026-01-01",
             valid_from=None, valid_to="2026-01-02", provenance="", superseded_by=2),
    ]
    result = KnowledgeGraph.format_summary_for_prompt(facts)
    assert result == ""


# ---------------------------------------------------------------------------
# Reflexion summary
# ---------------------------------------------------------------------------


def test_reflexion_summary():
    from app.core.reflexion import Reflexion, ReflexionStore

    reflexions = [
        Reflexion(id=15, task_summary="weather query", outcome="failure",
                  reflection="web_search timed out, should use http_fetch",
                  quality_score=0.3, tools_used="web_search", revision_count=3,
                  created_at="2026-01-01"),
    ]
    result = ReflexionStore.format_for_prompt(reflexions)
    # format_for_prompt outputs a "Previous failure" / "Previous success" prefix
    # plus the reflection text and tools_used. It does NOT include the raw R-id
    # or task summary (those went out when the prompt format was tightened).
    assert "Previous failure" in result
    assert "web_search" in result  # tools_used appears in output
