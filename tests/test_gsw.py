"""Tests for app.core.gsw — Generative Semantic Workspace episodic memory."""

from __future__ import annotations

import asyncio
import json

import pytest

from app.core import gsw
from app.database import get_db, _instances


@pytest.fixture
def gsw_db(tmp_path, monkeypatch):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "gsw.db"))
    monkeypatch.setenv("ENABLE_GSW_EPISODIC", "true")
    from app.config import reset_config
    reset_config()
    _instances.clear()
    db = get_db()
    db.init_schema()
    # Need a conversation row for the FK
    db.execute(
        "INSERT INTO conversations (id, title) VALUES (?, ?)",
        ("conv-1", "Test conversation"),
    )
    yield db
    db.close()
    _instances.clear()


# ---- _parse_summary_json (private helper) -------------------------------

def test_parse_summary_json_valid_object():
    raw = json.dumps({
        "summary": "we worked on bitcoin price tracking",
        "narrative": "Set up a price alert above $80k.",
        "key_entities": ["bitcoin", "price alert", "80k"],
    })
    out = gsw._parse_summary_json(raw)
    assert out is not None
    assert "bitcoin" in out["summary"]
    assert isinstance(out["key_entities"], list)


def test_parse_summary_json_missing_summary():
    raw = json.dumps({"narrative": "x", "key_entities": []})
    assert gsw._parse_summary_json(raw) is None


def test_parse_summary_json_garbage():
    assert gsw._parse_summary_json("") is None
    assert gsw._parse_summary_json("not json at all") is None


def test_parse_summary_json_nested_braces_extracted():
    raw = "Sure! Here is the result: " + json.dumps({
        "summary": "fixed bug",
        "narrative": "the bug was in line 42",
        "key_entities": ["bug", "fix"],
    })
    out = gsw._parse_summary_json(raw)
    assert out is not None
    assert "bug" in out["summary"]


def test_parse_summary_json_normalizes_entities():
    raw = json.dumps({
        "summary": "x",
        "key_entities": ["BiTCoIn", "  Apple  ", "", None, "x" * 200],
    })
    out = gsw._parse_summary_json(raw)
    assert out["key_entities"] == ["bitcoin", "apple"]


# ---- save_summary / get_current_summary ---------------------------------

def test_save_and_get_current_summary(gsw_db):
    summary_dict = {
        "summary": "discussed retrieval improvements",
        "narrative": "tried RRF + PPR fusion",
        "key_entities": ["retrieval", "rrf", "ppr"],
    }
    new_id = gsw.save_summary(gsw_db, "conv-1", summary_dict, message_count=10)
    assert new_id is not None
    out = gsw.get_current_summary(gsw_db, "conv-1")
    assert out is not None
    assert out["summary"] == "discussed retrieval improvements"
    assert out["key_entities"] == ["retrieval", "rrf", "ppr"]
    assert out["message_count"] == 10


def test_save_summary_retires_prior(gsw_db):
    gsw.save_summary(
        gsw_db, "conv-1",
        {"summary": "first", "key_entities": ["a"]},
        message_count=4,
    )
    gsw.save_summary(
        gsw_db, "conv-1",
        {"summary": "second", "key_entities": ["a", "b"]},
        message_count=10,
    )
    # Only the second is current
    rows = gsw_db.fetchall(
        "SELECT summary, valid_to FROM conversation_summaries "
        "WHERE conversation_id='conv-1' ORDER BY id"
    )
    assert len(rows) == 2
    assert rows[0]["valid_to"] is not None  # retired
    assert rows[1]["valid_to"] is None      # current
    cur = gsw.get_current_summary(gsw_db, "conv-1")
    assert cur["summary"] == "second"


def test_save_summary_drops_empty(gsw_db):
    assert gsw.save_summary(gsw_db, "conv-1", {}) is None
    assert gsw.save_summary(gsw_db, "conv-1", {"summary": ""}) is None


def test_get_current_summary_missing(gsw_db):
    assert gsw.get_current_summary(gsw_db, "no-such-conv") is None


# ---- get_relevant_summaries --------------------------------------------

def test_get_relevant_summaries_substring_match(gsw_db):
    gsw_db.execute(
        "INSERT INTO conversations (id, title) VALUES (?, ?)",
        ("conv-2", "second conv"),
    )
    gsw.save_summary(
        gsw_db, "conv-1",
        {"summary": "RRF tuning", "key_entities": ["rrf", "retrieval", "ppr"]},
        message_count=10,
    )
    gsw.save_summary(
        gsw_db, "conv-2",
        {"summary": "snake game", "key_entities": ["snake", "javascript"]},
        message_count=8,
    )
    out = gsw.get_relevant_summaries(gsw_db, "tell me about retrieval", limit=5)
    assert len(out) >= 1
    assert any("rrf" in s["summary"].lower() for s in out)
    # snake game must NOT match "retrieval"
    assert all("snake" not in s["summary"].lower() for s in out)


def test_get_relevant_summaries_empty_query(gsw_db):
    assert gsw.get_relevant_summaries(gsw_db, "") == []
    assert gsw.get_relevant_summaries(gsw_db, "   ") == []


def test_get_relevant_summaries_no_overlap(gsw_db):
    gsw.save_summary(
        gsw_db, "conv-1",
        {"summary": "rocket science", "key_entities": ["rockets", "physics"]},
        message_count=5,
    )
    out = gsw.get_relevant_summaries(gsw_db, "what is for dinner")
    assert out == []


# ---- format_for_prompt --------------------------------------------------

def test_format_for_prompt_compact():
    summaries = [
        {
            "narrative": "Set up bitcoin price alert at 80k.",
            "key_entities": ["bitcoin", "price"],
            "created_at": "2026-04-10T10:00:00Z",
        },
        {
            "narrative": "Tried RRF + PPR retrieval.",
            "key_entities": ["rrf", "ppr"],
            "created_at": "",
        },
    ]
    out = gsw.format_for_prompt(summaries)
    assert "bitcoin" in out
    assert "RRF" in out
    assert "2026-04-10" in out
    assert "[prior]" in out


def test_format_for_prompt_empty():
    assert gsw.format_for_prompt([]) == ""


# ---- maybe_update_summary (integration with mocked LLM) -----------------

def test_maybe_update_summary_disabled(monkeypatch, gsw_db):
    monkeypatch.setenv("ENABLE_GSW_EPISODIC", "false")
    from app.config import reset_config
    reset_config()
    out = asyncio.run(gsw.maybe_update_summary(gsw_db, "conv-1"))
    assert out is False


def test_maybe_update_summary_too_few_messages(gsw_db):
    # Only 2 messages — below _GSW_MIN_MESSAGES = 4
    for i in range(2):
        gsw_db.execute(
            "INSERT INTO messages (id, conversation_id, role, content) "
            "VALUES (?, ?, ?, ?)",
            (f"m-{i}", "conv-1", "user", f"msg {i}"),
        )
    out = asyncio.run(gsw.maybe_update_summary(gsw_db, "conv-1"))
    assert out is False
