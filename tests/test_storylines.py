"""Storyline tracking + change detection (Monitor Intelligence v2, Phase A).

The narrative engine clusters monitor items into ongoing threads, diffs new
developments against each thread's stored state, and surfaces ONLY moved threads.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

import app.core.storylines as sl

# `db` fixture (tmp_path + init_schema, runs migration 23) comes from conftest.py


def test_story_key_is_stable_slug():
    assert sl._story_key("Iran-Hormuz Tensions!") == "iran-hormuz-tensions"
    assert sl._story_key("NVIDIA  export   controls") == "nvidia-export-controls"
    # same title → same key (matching across cycles)
    assert sl._story_key("Fed rate path") == sl._story_key("Fed rate path")


@pytest.mark.asyncio
async def test_new_story_is_created_and_summarized(db):
    story = {
        "title": "Iran-Hormuz tensions",
        "key": "iran-hormuz-tensions",
        "monitors": ["Domain Study: Geopolitics"],
        "developments": ["Iran says it closed the Strait of Hormuz after strikes."],
    }
    fake_update = "Tensions escalate as Iran claims closure; US disputes it.\nCHANGED: Iran announced a closure of the strait."
    with patch.object(sl.llm, "invoke_nothink", AsyncMock(return_value=fake_update)):
        out = await sl._update_story(db, story)
    assert out is not None
    assert out["new"] is True
    assert "closure" in out["changed"].lower()
    # persisted
    row = db.fetchone("SELECT * FROM storylines WHERE story_key = ?", ("iran-hormuz-tensions",))
    assert row is not None and row["update_count"] == 1
    ev = db.fetchall("SELECT * FROM storyline_events WHERE storyline_id = ?", (row["id"],))
    assert len(ev) == 1


@pytest.mark.asyncio
async def test_known_thread_with_no_new_items_is_skipped(db):
    # Seed an existing storyline + an event matching the incoming development.
    db.execute("INSERT INTO storylines (story_key, title, summary, update_count) VALUES (?,?,?,1)",
               ("iran-hormuz-tensions", "Iran-Hormuz tensions", "prior state"))
    sid = db.fetchone("SELECT id FROM storylines WHERE story_key='iran-hormuz-tensions'")["id"]
    dev = "Iran says it closed the Strait of Hormuz after strikes."
    db.execute("INSERT INTO storyline_events (storyline_id, summary) VALUES (?, ?)", (sid, dev))

    story = {"title": "Iran-Hormuz tensions", "key": "iran-hormuz-tensions",
             "monitors": ["Domain Study: Geopolitics"], "developments": [dev]}
    # No fresh items → returns None WITHOUT calling the LLM.
    llm_mock = AsyncMock(return_value="should not be called")
    with patch.object(sl.llm, "invoke_nothink", llm_mock):
        out = await sl._update_story(db, story)
    assert out is None
    assert llm_mock.await_count == 0


@pytest.mark.asyncio
async def test_no_change_verdict_records_but_does_not_surface(db):
    db.execute("INSERT INTO storylines (story_key, title, summary, update_count) VALUES (?,?,?,1)",
               ("fed-rate-path", "Fed rate path", "Fed holding steady"))
    story = {"title": "Fed rate path", "key": "fed-rate-path",
             "monitors": ["Domain Study: Finance"],
             "developments": ["Another analyst reiterates the hold."]}
    with patch.object(sl.llm, "invoke_nothink", AsyncMock(return_value="NO CHANGE")):
        out = await sl._update_story(db, story)
    assert out is None  # not surfaced in the digest


def test_fuzzy_match_merges_differently_titled_continuation(db):
    # Same underlying story, different title across cycles → must merge, not fork.
    db.execute(
        "INSERT INTO storylines (story_key, title, summary, status, update_count) "
        "VALUES ('mideast-hormuz', 'Middle East Peace Talks & Strait of Hormuz Closure Threats', "
        "'US VP Vance in Switzerland; Iran threatens Hormuz closure amid Lebanon strikes.', 'active', 2)"
    )
    story = {
        "title": "US-Iran Diplomatic Talks & Hormuz Tensions",
        "key": "us-iran-diplomatic-talks-hormuz-tensions",
        "monitors": ["World Awareness"],
        "developments": ["US VP JD Vance arrives in Switzerland for Iran negotiations.",
                         "Iran asserts it closed the Strait of Hormuz."],
    }
    matched = sl._find_matching_storyline(db, story)
    assert matched is not None and matched["story_key"] == "mideast-hormuz"


def test_fuzzy_match_does_not_merge_unrelated_story(db):
    db.execute(
        "INSERT INTO storylines (story_key, title, summary, status, update_count) "
        "VALUES ('mideast-hormuz', 'Middle East Strait of Hormuz Closure', "
        "'Iran threatens Hormuz closure amid Lebanon strikes.', 'active', 2)"
    )
    story = {
        "title": "NVIDIA export controls on China chips",
        "key": "nvidia-export-controls-on-china-chips",
        "monitors": ["Domain Study: AI and ML"],
        "developments": ["NVIDIA cuts guidance citing China semiconductor export limits."],
    }
    assert sl._find_matching_storyline(db, story) is None  # unrelated → new thread


def test_common_entity_does_not_drive_overmerge(db):
    # Several distinct stories about a common actor ('trump') exist. A NEW distinct
    # trump story must NOT merge into them just for sharing 'trump' — only rare
    # shared entities should merge (document-frequency down-weighting).
    for k, t, s in [
        ("t-market", "Trump Administration Market Impact", "markets react to trump policy and inflation"),
        ("t-algae", "Trump Lincoln Memorial Algae Bloom", "trump memorial algae bloom potomac"),
        ("t-af1", "Qatar Air Force One for Trump", "trump qatar air force one gift"),
    ]:
        db.execute("INSERT INTO storylines (story_key,title,summary,status) VALUES (?,?,?,'active')", (k, t, s))
    story = {"title": "Trump statue vandalized in DC", "key": "trump-statue-vandalized-in-dc",
             "monitors": ["World Awareness"], "developments": ["A Trump statue was vandalized downtown."]}
    assert sl._find_matching_storyline(db, story) is None  # 'trump' is common → no merge


def test_exact_key_match_still_works(db):
    db.execute("INSERT INTO storylines (story_key, title, summary, status) "
               "VALUES ('fed-rate-path', 'Fed rate path', 'holding', 'active')")
    story = {"title": "Fed rate path", "key": "fed-rate-path",
             "monitors": ["Domain Study: Finance"], "developments": ["x"]}
    matched = sl._find_matching_storyline(db, story)
    assert matched is not None and matched["story_key"] == "fed-rate-path"


@pytest.mark.asyncio
async def test_track_storylines_returns_only_moved_threads(db):
    # Seed monitor_results so _collect_items has material.
    db.execute("INSERT INTO monitors (name, check_type, check_config, category) VALUES (?,?,?,?)",
               ("Domain Study: Geopolitics", "query", "{}", "content"))
    mid = db.fetchone("SELECT id FROM monitors WHERE name='Domain Study: Geopolitics'")["id"]
    val = ("1. Iran says it closed the Strait of Hormuz after Israeli strikes in Lebanon today.\n"
           "2. The United States disputes Iran's claim that the waterway is shut to shipping.\n"
           "3. Oil prices jumped as traders weighed the risk to Gulf shipping lanes worldwide.\n"
           "4. European leaders called for de-escalation between Israel and Iran in the region.")
    db.execute("INSERT INTO monitor_results (monitor_id, status, value) VALUES (?, 'ok', ?)", (mid, val))

    cluster_json = '[{"title": "Iran-Hormuz tensions", "items": [0, 1]}]'
    update_text = "Iran claims closure; US disputes.\nCHANGED: Iran announced a strait closure."
    calls = {"n": 0}
    async def fake_llm(messages, **kw):
        calls["n"] += 1
        return cluster_json if calls["n"] == 1 else update_text
    with patch.object(sl.llm, "invoke_nothink", side_effect=fake_llm):
        out = await sl.track_storylines(db)
    assert "STORYLINE UPDATES" in out
    assert "Iran-Hormuz tensions" in out
