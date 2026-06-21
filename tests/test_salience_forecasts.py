"""Salience filtering + self-scoring forecasts (Monitor Intelligence v2, Phase C)."""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from app.core import salience, forecasts

# `db` fixture (init_schema → migration 23 tables) comes from conftest.py


# --- Salience -------------------------------------------------------------

def _seed_topics(db):
    # topic_frequency is created lazily by TopicTracker at startup, not by
    # init_schema — instantiate it so the table exists in the test DB.
    from app.core.curiosity import TopicTracker
    TopicTracker(db)


def test_score_rewards_owner_interest(db):
    # Owner repeatedly queries about NVIDIA → NVIDIA items score higher than noise.
    _seed_topics(db)
    db.execute("INSERT INTO topic_frequency (topic, query_count) VALUES ('nvidia gpus', 9)")
    hot = salience.score_text(db, "NVIDIA unveils new datacenter GPUs for AI training.")
    cold = salience.score_text(db, "Local council debates parking regulations downtown.")
    assert hot > cold


def test_corroboration_raises_score(db):
    high = salience.score_text(db, "Major story. Confirmed by 8 outlets across the world.")
    low = salience.score_text(db, "Minor item with a single mention and nothing else here.")
    assert high > low


def test_learn_from_rating_moves_weight(db):
    salience.learn_from_rating(db, "NVIDIA semiconductor earnings beat estimates", 1)
    row = db.fetchone("SELECT weight FROM salience_weights WHERE topic='nvidia'")
    assert row is not None and row["weight"] > 0
    salience.learn_from_rating(db, "NVIDIA semiconductor earnings beat estimates", -1)
    row2 = db.fetchone("SELECT weight FROM salience_weights WHERE topic='nvidia'")
    assert row2["weight"] < row["weight"]  # downvote pulled it back down


def test_rank_keeps_small_digest_intact(db):
    items = [("A", "x"), ("B", "y")]
    assert salience.rank_digest_items(db, items) == items  # never thin a tiny digest


def test_rank_orders_by_salience(db):
    _seed_topics(db)
    db.execute("INSERT INTO topic_frequency (topic, query_count) VALUES ('bitcoin', 9)")
    items = [
        ("Parking", "Council debates downtown parking permits and zoning today."),
        ("Crypto", "Bitcoin surges. Confirmed by 9 outlets as institutions pile in."),
        ("Weather", "Mild temperatures expected across the region this weekend."),
    ]
    ranked = salience.rank_digest_items(db, items)
    assert ranked[0][0] == "Crypto"  # highest salience leads


# --- Forecasts ------------------------------------------------------------

def test_create_and_list_due(db):
    # A forecast due in the past should be listed; a future one should not.
    fid = forecasts.create_forecast(db, "The Fed will cut rates within two weeks.", days=1, confidence=0.6)
    assert fid
    # backdate it so it's due
    db.execute("UPDATE forecasts SET resolves_at = datetime('now','-1 hour') WHERE id=?", (fid,))
    due = forecasts.list_due(db)
    assert any(f["id"] == fid for f in due)


def test_parse_forecast_line(db):
    text = ("Updated state of the story.\n"
            "CHANGED: tensions rose.\n"
            "FORECAST: Oil breaks $100 within the period | 10 days | 0.65")
    fid = forecasts.parse_and_store_forecast(db, text, source_monitor="Storyline Tracker")
    assert fid
    row = db.fetchone("SELECT claim, confidence FROM forecasts WHERE id=?", (fid,))
    assert "Oil breaks" in row["claim"] and abs(row["confidence"] - 0.65) < 1e-6


def test_no_forecast_line_returns_none(db):
    assert forecasts.parse_and_store_forecast(db, "No forecast here. CHANGED: stuff.") is None


@pytest.mark.asyncio
async def test_resolve_grades_hit_and_updates_accuracy(db):
    fid = forecasts.create_forecast(db, "X will happen within a week.", days=1, confidence=0.7)
    db.execute("UPDATE forecasts SET resolves_at = datetime('now','-1 hour') WHERE id=?", (fid,))
    verdict_json = '{"verdict": "hit", "reason": "It happened as predicted."}'
    with patch("app.core.forecasts.llm.invoke_nothink", AsyncMock(return_value=verdict_json)), \
         patch("app.core.forecasts.llm.extract_json_object", return_value={"verdict": "hit", "reason": "ok"}):
        out = await forecasts.resolve_due(db)
    row = db.fetchone("SELECT status FROM forecasts WHERE id=?", (fid,))
    assert row["status"] == "hit"
    acc = forecasts.accuracy(db)
    assert acc["resolved"] == 1 and acc["hits"] == 1 and acc["rate"] == 1.0
    assert "FORECAST" in out
