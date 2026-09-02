"""Storyline re-attachment (2026-09-02, Phase 4.4).

`close_stale` closes a thread that has not moved in three weeks and its
docstring promised "the next development flips status back to active". Only
the exact-key path could keep that promise: the fuzzy matcher searched active
threads only, so a re-titled continuation of a quiet story forked instead of
reattaching. Live evidence the day this was written: two Strait of Hormuz
threads (one closed, one active) and two "global economic instability"
threads, each pair the same story under different titles.
"""
from __future__ import annotations

import app.core.storylines as sl


def _thread(db, key, title, summary, status, days_ago):
    db.execute(
        "INSERT INTO storylines (story_key, title, summary, monitors_csv, status, "
        "update_count, first_seen, last_updated) "
        "VALUES (?, ?, ?, 'World Awareness', ?, 3, datetime('now', ?), datetime('now', ?))",
        (key, title, summary, status, f"-{days_ago + 30} days", f"-{days_ago} days"),
    )


HORMUZ_STORY = {
    "title": "Strait of Hormuz blockade enters sixth month",
    "key": "strait-of-hormuz-blockade-enters-sixth-month",
    "monitors": ["World Awareness"],
    "developments": ["Iran keeps the Strait of Hormuz closed to tanker traffic.",
                     "Shipping insurers widen the Hormuz exclusion zone."],
}


def test_recently_closed_thread_is_reattached_not_forked(db):
    _thread(db, "mideast-hormuz", "Middle East Peace Talks & Strait of Hormuz Closure",
            "Iran threatens Hormuz closure; tanker traffic halted.", "closed", days_ago=25)
    matched = sl._find_matching_storyline(db, HORMUZ_STORY)
    assert matched is not None and matched["story_key"] == "mideast-hormuz"
    assert matched["status"] == "closed"


def test_reattaching_revives_the_thread_and_keeps_its_history(db):
    _thread(db, "mideast-hormuz", "Middle East Peace Talks & Strait of Hormuz Closure",
            "Iran threatens Hormuz closure; tanker traffic halted.", "closed", days_ago=25)
    row = sl._find_matching_storyline(db, HORMUZ_STORY)
    sid = sl._record(db, row, HORMUZ_STORY, HORMUZ_STORY["developments"], summary="Blockade holds.")
    assert sid == row["id"]                                   # same thread, not a new one
    after = db.fetchone("SELECT * FROM storylines WHERE id = ?", (sid,))
    assert after["status"] == "active"                        # revived
    assert after["update_count"] == 4                         # history preserved
    assert db.fetchone("SELECT COUNT(*) c FROM storylines")["c"] == 1


def test_a_thread_closed_longer_than_the_window_starts_a_new_story(db):
    _thread(db, "mideast-hormuz", "Middle East Peace Talks & Strait of Hormuz Closure",
            "Iran threatens Hormuz closure; tanker traffic halted.", "closed",
            days_ago=sl._REATTACH_DAYS + 5)
    assert sl._find_matching_storyline(db, HORMUZ_STORY) is None


def test_an_active_thread_wins_a_tie_with_a_closed_one(db):
    _thread(db, "hormuz-closed", "Strait of Hormuz Closure Threats",
            "Iran threatens Hormuz closure; tanker traffic halted.", "closed", days_ago=10)
    _thread(db, "hormuz-active", "Strait of Hormuz Closure Watch",
            "Iran threatens Hormuz closure; tanker traffic halted.", "active", days_ago=1)
    matched = sl._find_matching_storyline(db, HORMUZ_STORY)
    assert matched is not None and matched["story_key"] == "hormuz-active"


def test_closed_threads_do_not_loosen_the_discrimination_bar(db):
    # An unrelated closed thread must not attract a story that merely shares a
    # common actor — the rare-entity requirement is unchanged.
    _thread(db, "nvidia-export", "NVIDIA export controls on China chips",
            "NVIDIA cuts guidance citing China export limits.", "closed", days_ago=5)
    story = {"title": "Strait of Hormuz blockade enters sixth month",
             "key": "strait-of-hormuz-blockade-enters-sixth-month",
             "monitors": ["World Awareness"],
             "developments": ["Iran keeps the Strait of Hormuz closed to tanker traffic."]}
    assert sl._find_matching_storyline(db, story) is None


def test_closed_threads_count_toward_common_entity_downweighting(db):
    # Three closed 'trump' threads make 'trump' common, so a fourth distinct
    # trump story still does not merge into any of them.
    for i, (k, t, s) in enumerate([
        ("t-market", "Trump Administration Market Impact", "markets react to trump policy and inflation"),
        ("t-algae", "Trump Lincoln Memorial Algae Bloom", "trump memorial algae bloom potomac"),
        ("t-af1", "Qatar Air Force One for Trump", "trump qatar air force one gift"),
    ]):
        _thread(db, k, t, s, "closed", days_ago=5 + i)
    story = {"title": "Trump statue vandalized in DC", "key": "trump-statue-vandalized-in-dc",
             "monitors": ["World Awareness"], "developments": ["A Trump statue was vandalized downtown."]}
    assert sl._find_matching_storyline(db, story) is None
