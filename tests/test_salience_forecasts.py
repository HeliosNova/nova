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


# --- Knowing signal (dossier-primed salience, 2026-08-12) -------------------

def _seed_dossier(db, title, open_questions):
    db.execute(
        "INSERT INTO dossiers (kind, dkey, title, body) VALUES ('domain', ?, ?, ?)",
        (title.lower().replace(" ", "-"), title,
         f"## Current understanding\nstuff\n\n## Open questions\n{open_questions}\n"),
    )


def test_knowing_signal_lifts_open_question_matches(db):
    # An item that speaks to a dossier's Open question outranks generic news —
    # even with zero owner-query signal (Nova's OWN curiosity is a signal).
    _seed_topics(db)
    db.execute("INSERT INTO topic_frequency (topic, query_count) VALUES ('markets', 3)")
    _seed_dossier(db, "Perovskite Solar", "- Will perovskite tandem modules reach commercial durability certification?")
    hot = salience.score_text(db, "Perovskite tandem modules pass durability certification milestone.")
    cold = salience.score_text(db, "Regional festival attendance grows modestly this summer season.")
    assert hot > cold


def test_contradiction_flag_boosts_score(db):
    _seed_topics(db)
    db.execute("INSERT INTO topic_frequency (topic, query_count) VALUES ('markets', 3)")
    base = salience.score_text(db, "Chip production output rose in the second quarter.")
    flagged = salience.score_text(
        db, "Chip production output rose in the second quarter. CONTRADICTS PRIOR UNDERSTANDING.")
    assert flagged > base


def test_cold_start_uses_knowing_only(db):
    # No owner queries, no learned weights — but dossiers exist: standing
    # knowledge still ranks, corroboration still counts, nothing crashes.
    _seed_dossier(db, "Quantum Computing", "- When will logical qubit counts cross 100?")
    know = salience.score_text(db, "Logical qubit counts cross 100 in new quantum computing record.")
    generic = salience.score_text(db, "Generic item mentioning nothing from standing knowledge here.")
    assert know > generic


def test_irrelevant_item_can_be_dropped(db):
    # Regression (2026-06-21 review): the drop-floor was inert (base==floor).
    # With owner signal present, a truly-irrelevant item must score BELOW floor.
    _seed_topics(db)
    db.execute("INSERT INTO topic_frequency (topic, query_count) VALUES ('nvidia gpus', 9)")
    s = salience.score_text(db, "Council debates downtown parking permits and zoning rules.")
    assert s < 0.4, f"irrelevant item should fall below drop floor, got {s}"
    # and a digest with clear noise actually drops something
    items = [("AI", "NVIDIA GPU roadmap update. Confirmed by 9 outlets."),
             ("AI2", "New NVIDIA datacenter GPUs announced for AI training."),
             ("Noise", "Council debates downtown parking permits and zoning rules."),
             ("Noise2", "Mild weather expected across the region this weekend.")]
    kept = salience.rank_digest_items(db, items)
    assert len(kept) < len(items)  # noise dropped


def test_cold_start_drops_nothing(db):
    # No owner queries AND no learned weights → don't gut the digest.
    items = [("A", "alpha item one with content here"),
             ("B", "beta item two with content here"),
             ("C", "gamma item three with content here")]
    assert len(salience.rank_digest_items(db, items)) == len(items)


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


def test_parse_forecast_with_confidence_word(db):
    # The dossier prompt's template reads '<0.x confidence>' and the 27B writes
    # the word — probe-confirmed 2026-08-13. The v1 regex demanded end-of-line
    # after the number, so the prompt's OWN format never parsed (one of two
    # stacked bugs behind zero minted forecasts across 60+ consolidations).
    text = ("CHANGED: initial dossier\n"
            "FORECAST: The Fed announces a 25 basis point cut at the September 17 "
            "FOMC meeting | 36 days | 0.78 confidence")
    fid = forecasts.parse_and_store_forecast(db, text, source_monitor="Knowledge Consolidation")
    assert fid
    row = db.fetchone("SELECT claim, confidence FROM forecasts WHERE id=?", (fid,))
    assert "September 17" in row["claim"] and abs(row["confidence"] - 0.78) < 1e-6
    assert "confidence" not in row["claim"]


def test_forecast_none_optout_not_stored(db):
    # The v2 mandatory line contract: 'FORECAST: none' is the explicit opt-out
    # and must never create a forecast row.
    assert forecasts.parse_and_store_forecast(db, "CHANGED: quiet cycle.\nFORECAST: none") is None


def test_calibration_gap_and_scoping(db):
    # Judgment rung (2026-08-14): stated confidence must be WORTH its number.
    for conf, status, key in ((0.9, "hit", "dossier:finance"), (0.9, "miss", "dossier:finance"),
                              (0.8, "miss", "dossier:finance"), (0.6, "hit", "dossier:ai-and-ml")):
        fid = forecasts.create_forecast(db, f"claim {conf}/{status}", days=1,
                                        confidence=conf, storyline_key=key)
        db.execute("UPDATE forecasts SET status=? WHERE id=?", (status, fid))
    cal = forecasts.calibration(db, key_prefix="dossier:finance")
    assert cal["n"] == 3
    assert abs(cal["mean_conf"] - 0.867) < 0.01
    assert abs(cal["hit_rate"] - 0.333) < 0.01
    assert cal["gap"] > 0.5                      # ran badly overconfident
    # min_n gate: no lessons from tiny samples
    assert forecasts.calibration(db, key_prefix="dossier:ai-and-ml", min_n=5) is None


@pytest.mark.asyncio
async def test_unparseable_forecast_auto_retires(db):
    # Regression (2026-06-21 review): a forecast the LLM never grades must not
    # re-process forever — it auto-retires to 'unresolvable' after N attempts.
    fid = forecasts.create_forecast(db, "Some vague claim that cannot be judged.", days=1, confidence=0.5)
    db.execute("UPDATE forecasts SET resolves_at = datetime('now','-1 hour') WHERE id=?", (fid,))
    # LLM always returns an invalid verdict → non-terminal each time. Evidence is
    # stubbed present so grading is reached (resolution is web-grounded now).
    with patch("app.core.forecasts._gather_evidence", AsyncMock(return_value="- headline [news.com]")), \
         patch("app.core.forecasts.llm.invoke_nothink", AsyncMock(return_value='{"verdict":"maybe"}')), \
         patch("app.core.forecasts.llm.extract_json_object", return_value={"verdict": "maybe"}):
        for _ in range(3):
            await forecasts.resolve_due(db)
    row = db.fetchone("SELECT status, attempts FROM forecasts WHERE id=?", (fid,))
    assert row["status"] == "unresolvable" and row["attempts"] >= 3
    assert forecasts.list_due(db) == []  # no longer re-selected → no starvation


@pytest.mark.asyncio
async def test_resolve_grades_hit_and_updates_accuracy(db):
    fid = forecasts.create_forecast(db, "X will happen within a week.", days=1, confidence=0.7)
    db.execute("UPDATE forecasts SET resolves_at = datetime('now','-1 hour') WHERE id=?", (fid,))
    verdict_json = '{"verdict": "hit", "reason": "It happened as predicted."}'
    with patch("app.core.forecasts._gather_evidence",
               AsyncMock(return_value="- X happened on schedule [news.com]")), \
         patch("app.core.forecasts.llm.invoke_nothink", AsyncMock(return_value=verdict_json)), \
         patch("app.core.forecasts.llm.extract_json_object", return_value={"verdict": "hit", "reason": "ok"}):
        out = await forecasts.resolve_due(db)
    row = db.fetchone("SELECT status FROM forecasts WHERE id=?", (fid,))
    assert row["status"] == "hit"
    acc = forecasts.accuracy(db)
    assert acc["resolved"] == 1 and acc["hits"] == 1 and acc["rate"] == 1.0
    assert "FORECAST" in out


@pytest.mark.asyncio
async def test_resolve_defers_without_evidence(db):
    # Grounding guard (2026-06-24 audit): with NO live web evidence, the model must
    # NOT be asked to guess — defer (count an attempt), leave the forecast open
    # rather than fabricate a verdict from a frozen-cutoff recollection.
    fid = forecasts.create_forecast(db, "Something will occur within a week.", days=1, confidence=0.6)
    db.execute("UPDATE forecasts SET resolves_at = datetime('now','-1 hour') WHERE id=?", (fid,))
    called = {"llm": False}

    async def _should_not_run(*a, **k):
        called["llm"] = True
        return '{"verdict":"hit"}'

    with patch("app.core.forecasts._gather_evidence", AsyncMock(return_value="")), \
         patch("app.core.forecasts.llm.invoke_nothink", _should_not_run):
        await forecasts.resolve_due(db)
    row = db.fetchone("SELECT status, attempts FROM forecasts WHERE id=?", (fid,))
    assert row["status"] == "open" and row["attempts"] == 1
    assert called["llm"] is False  # never graded without evidence
