"""Tests for dream's procedural memory consolidation pass.

Covers _consolidate_procedural_memory: clusters similar lessons, generalizes
via LLM into a single canonical lesson, demotes subsumed members, and persists
the cluster signature so the next dream cycle skips duplicates.
"""

from __future__ import annotations

import asyncio
import json
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.core.dream import DreamConsolidator, ConsolidationResult
from app.database import AsyncSafeDB, SafeDB, get_db, _instances


@pytest.fixture
def dream_db(tmp_path, monkeypatch):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "dream.db"))
    monkeypatch.setenv("ENABLE_PROCEDURAL_CONSOLIDATION", "true")
    from app.config import reset_config
    reset_config()
    _instances.clear()
    sync = SafeDB(str(tmp_path / "dream.db"))
    sync.init_schema()
    adb = AsyncSafeDB(sync)
    yield adb, sync
    sync.close()
    _instances.clear()


def _seed_lessons(db, rows):
    for topic, answer, conf in rows:
        db.execute(
            "INSERT INTO lessons (topic, correct_answer, lesson_text, confidence, times_helpful) "
            "VALUES (?, ?, '', ?, 1)",
            (topic, answer, conf),
        )


def test_consolidates_three_similar_lessons(dream_db, monkeypatch):
    adb, sync = dream_db
    # Three lessons all about "always validate JSON before parsing"
    # Topic tokens overlap >= 0.6 across each pair; answers share majority of tokens.
    common_answer = (
        "always validate JSON schema before passing input to parser "
        "reject malformed input early"
    )
    _seed_lessons(sync, [
        ("validate json before parse",       common_answer + " always",   0.8),
        ("validate json before parser",      common_answer + " strictly", 0.85),
        ("validate json before parsing",     common_answer + " always",   0.78),
    ])
    # Mock the LLM to return a generalized lesson
    mock_resp = json.dumps({
        "topic": "validate JSON before parser",
        "correct_answer": "Always validate JSON schema before passing to a parser.",
        "rationale": "all three lessons agree on early schema validation",
    })
    monkeypatch.setattr(
        "app.core.llm.invoke_nothink",
        AsyncMock(return_value=mock_resp),
    )
    monkeypatch.setattr(
        "app.core.llm.extract_json_object",
        lambda raw: json.loads(raw),
    )

    consolidator = DreamConsolidator(adb)
    result = ConsolidationResult()
    fake_svc = MagicMock()
    asyncio.run(consolidator._consolidate_procedural_memory(result, fake_svc))

    assert result.procedural_clusters_consolidated == 1
    assert result.procedural_lessons_subsumed == 3

    # Canonical lesson exists
    rows = sync.fetchall(
        "SELECT topic, correct_answer, confidence "
        "FROM lessons WHERE lesson_text LIKE 'Procedural-consolidation:%'"
    )
    assert len(rows) == 1
    assert "validate" in rows[0]["topic"].lower()
    assert "json" in rows[0]["topic"].lower()
    assert rows[0]["confidence"] >= 0.85  # max member confidence
    # Source members demoted
    rows = sync.fetchall(
        "SELECT confidence FROM lessons "
        "WHERE topic LIKE '%JSON%' AND lesson_text NOT LIKE 'Procedural-consolidation:%'"
    )
    for r in rows:
        assert r["confidence"] <= 0.46  # 0.85 - 0.4 - epsilon
    # Cluster signature persisted
    cluster_rows = sync.fetchall("SELECT * FROM procedural_clusters")
    assert len(cluster_rows) == 1


def test_skips_when_cluster_too_small(dream_db, monkeypatch):
    adb, sync = dream_db
    # Only 2 similar lessons — below cluster threshold of 3
    _seed_lessons(sync, [
        ("alpha topic", "do alpha thing X then Y then Z", 0.7),
        ("alpha topics", "do alpha thing X then Y then Z again", 0.7),
    ])
    monkeypatch.setattr(
        "app.core.llm.invoke_nothink",
        AsyncMock(return_value="should not be called"),
    )

    consolidator = DreamConsolidator(adb)
    result = ConsolidationResult()
    asyncio.run(consolidator._consolidate_procedural_memory(result, MagicMock()))
    assert result.procedural_clusters_consolidated == 0


def test_skips_already_consolidated_cluster(dream_db, monkeypatch):
    adb, sync = dream_db
    common_answer = (
        "always validate inputs before pipeline processing tasks "
        "reject malformed early in the pipeline"
    )
    _seed_lessons(sync, [
        ("validate inputs early pipeline",     common_answer + " strictly", 0.8),
        ("validate inputs early pipelines",    common_answer + " always",   0.8),
        ("validate input early pipeline",      common_answer + " always",   0.8),
    ])
    # Pre-seed cluster signature so the run sees them as already done
    rows = sync.fetchall("SELECT id FROM lessons ORDER BY id")
    cluster_ids = sorted(r["id"] for r in rows)
    cluster_key = "ids:" + ",".join(str(i) for i in cluster_ids)
    sync.execute(
        "INSERT INTO procedural_clusters (cluster_key, member_lesson_ids, member_count) "
        "VALUES (?, ?, ?)",
        (cluster_key, json.dumps(cluster_ids), len(cluster_ids)),
    )
    mock_llm = AsyncMock(return_value="should not run")
    monkeypatch.setattr("app.core.llm.invoke_nothink", mock_llm)

    consolidator = DreamConsolidator(adb)
    result = ConsolidationResult()
    asyncio.run(consolidator._consolidate_procedural_memory(result, MagicMock()))
    assert result.procedural_clusters_consolidated == 0
    assert mock_llm.call_count == 0


def test_handles_llm_failure_gracefully(dream_db, monkeypatch):
    adb, sync = dream_db
    common_answer = (
        "retry transient network errors with exponential backoff between attempts "
        "until success or limit"
    )
    _seed_lessons(sync, [
        ("retry transient network errors",  common_answer + " always",  0.8),
        ("retry transient network error",   common_answer + " always",  0.8),
        ("retry transient network errored", common_answer + " strict",  0.8),
    ])
    monkeypatch.setattr(
        "app.core.llm.invoke_nothink",
        AsyncMock(side_effect=RuntimeError("Ollama 500")),
    )

    consolidator = DreamConsolidator(adb)
    result = ConsolidationResult()
    asyncio.run(consolidator._consolidate_procedural_memory(result, MagicMock()))
    # No crash; cluster not consolidated; error recorded
    assert result.procedural_clusters_consolidated == 0
    assert any("procedural" in e for e in result.errors)


def test_handles_malformed_llm_response(dream_db, monkeypatch):
    adb, sync = dream_db
    common_answer = (
        "use CDN for static assets and Redis for hot session keys "
        "consider stale-while-revalidate"
    )
    _seed_lessons(sync, [
        ("caching strategies web servers",   common_answer + " always", 0.8),
        ("caching strategies web server",    common_answer + " always", 0.8),
        ("caching strategies web serverless", common_answer + " strict", 0.8),
    ])
    # Returns garbage JSON
    monkeypatch.setattr(
        "app.core.llm.invoke_nothink",
        AsyncMock(return_value="not json {{{"),
    )
    monkeypatch.setattr(
        "app.core.llm.extract_json_object",
        lambda raw: None,
    )
    consolidator = DreamConsolidator(adb)
    result = ConsolidationResult()
    asyncio.run(consolidator._consolidate_procedural_memory(result, MagicMock()))
    assert result.procedural_clusters_consolidated == 0


def test_low_confidence_lessons_excluded(dream_db, monkeypatch):
    adb, sync = dream_db
    # Confidence below 0.5 — excluded from candidate pool
    common_answer = (
        "do alpha thing always when bravo signal seen in pipeline processing"
    )
    _seed_lessons(sync, [
        ("lower confidence rule alpha bravo",  common_answer, 0.3),
        ("lower confidence rule alpha bravoo", common_answer, 0.3),
        ("lower confidence rule alpha bravos", common_answer, 0.3),
    ])
    monkeypatch.setattr(
        "app.core.llm.invoke_nothink",
        AsyncMock(return_value="not called"),
    )
    consolidator = DreamConsolidator(adb)
    result = ConsolidationResult()
    asyncio.run(consolidator._consolidate_procedural_memory(result, MagicMock()))
    assert result.procedural_clusters_consolidated == 0
