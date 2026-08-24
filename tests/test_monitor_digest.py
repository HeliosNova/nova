"""Monitor alert batching into a per-cycle digest (2026-06-14).

80 monitors firing concurrently used to post 80 separate (interleaving)
messages. Alerts are now buffered per cycle and flushed as ONE digest per
channel-group. Pins the formatting, routing, buffering, and grouping.

2026-08-12: buffered alerts are additionally journaled to pending_deliveries
(at-least-once delivery — a restart between buffering and broadcast used to
silently destroy a completed digest). Buffer entries carry the journal row id
and a monotonic timestamp for the age flusher.
"""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.monitors.heartbeat_loop import HeartbeatLoop


def _loop(*, discord=True, telegram=True, whatsapp=False, signal=False):
    lp = HeartbeatLoop.__new__(HeartbeatLoop)
    lp._discord = AsyncMock() if discord else None
    lp._telegram = AsyncMock() if telegram else None
    lp._whatsapp = AsyncMock() if whatsapp else None
    lp._signal = AsyncMock() if signal else None
    lp._digest_enabled = True
    lp._digest_buffer = []
    lp._flush_lock = asyncio.Lock()
    lp._alert_retry_counts = {}
    return lp


def _entry(targets, name, msg, cat, row_id=None):
    return (frozenset(targets), name, msg, cat, row_id, 0.0)


def _mon(name, category="content", channels=None):
    return SimpleNamespace(name=name, category=category, channels=channels)


def test_format_single_is_plain():
    lp = _loop()
    out = lp._format_digest([("China Tech", "Big chip news today.")])
    assert out == "[China Tech]\nBig chip news today."
    assert "Monitor digest" not in out


def test_format_multiple_is_digest_with_sections_and_cap():
    lp = _loop()
    items = [("China Tech", "A" * 2000), ("Crypto", "BTC up.")]
    out = lp._format_digest(items)
    assert "Monitor digest — 2 updates" in out
    assert "## China Tech" in out and "## Crypto" in out
    assert "[…]" in out  # long body capped
    assert len([l for l in out.splitlines() if l.startswith("## ")]) == 2


def test_content_briefing_not_truncated_in_digest():
    # Best-possible delivery (2026-06-26): a full content briefing must NOT be
    # gutted to a 600-char preview when batched — only operational status lines get
    # the scannable cap. Pins the fix for "domain studies truncated when 2 land in
    # one cycle" found in the live monitor audit.
    lp = _loop()
    long_brief = "Lead Development: " + "x" * 3000
    items = [("Domain Study: Crypto", long_brief), ("System Health", "y" * 3000)]
    cats = {"Domain Study: Crypto": "content", "System Health": "system"}
    out = lp._format_digest(items, cats)
    assert long_brief in out                  # content posted in full, no '[…]'
    assert ("y" * 700) not in out             # operational line still capped
    assert out.count("[…]") == 1              # exactly the operational item truncated


def test_alert_targets_category_defaults():
    lp = _loop(discord=True, telegram=True)
    # content -> all configured channels
    assert lp._alert_targets(_mon("News", "content")) == {"discord", "telegram"}
    # system -> telegram only
    assert lp._alert_targets(_mon("Health", "system")) == {"telegram"}


def test_alert_targets_per_monitor_override():
    lp = _loop(discord=True, telegram=True)
    assert lp._alert_targets(_mon("X", "content", channels="discord")) == {"discord"}


@pytest.mark.asyncio
async def test_send_alert_buffers_when_digest_enabled():
    lp = _loop()
    await lp._send_alert(_mon("Health", "system"), "all good")  # system skips dedup
    assert len(lp._digest_buffer) == 1
    lp._telegram.send_alert.assert_not_called()  # buffered, not sent


@pytest.mark.asyncio
async def test_flush_groups_by_channel_and_sends_once():
    lp = _loop(discord=True, telegram=True)
    # two content alerts (both -> discord+telegram) and one system (-> telegram).
    lp._digest_buffer = [
        _entry({"discord", "telegram"}, "China Tech", "chips", "content"),
        _entry({"discord", "telegram"}, "Crypto", "btc", "content"),
        _entry({"telegram"}, "Health", "ok", "system"),
    ]
    await lp._flush_digest()
    # content group -> ONE send to each of discord+telegram (a 2-update digest)
    # system group -> ONE send to telegram. Telegram gets 2 sends (both groups).
    assert lp._discord.send_alert.await_count == 1
    assert lp._telegram.send_alert.await_count == 2
    # the content digest batched both monitors into one message
    content_msg = lp._discord.send_alert.await_args_list[0].args[0]
    assert "China Tech" in content_msg and "Crypto" in content_msg
    assert lp._digest_buffer == []  # drained


@pytest.mark.asyncio
async def test_two_concurrent_monitors_become_one_digest():
    import asyncio
    lp = _loop(discord=True, telegram=False)
    await asyncio.gather(
        lp._send_alert(_mon("A", "system", channels="discord"), "alpha"),
        lp._send_alert(_mon("B", "system", channels="discord"), "beta"),
    )
    assert lp._discord.send_alert.await_count == 0  # nothing sent yet (buffered)
    await lp._flush_digest()
    assert lp._discord.send_alert.await_count == 1  # ONE digest, not two posts
    msg = lp._discord.send_alert.await_args_list[0].args[0]
    assert "alpha" in msg and "beta" in msg


# ---------------------------------------------------------------------------
# Delivery journal (pending_deliveries) — at-least-once semantics
# ---------------------------------------------------------------------------

def _fake_db():
    db = MagicMock()
    cur = MagicMock()
    cur.lastrowid = 7
    db.execute.return_value = cur
    return db


@pytest.mark.asyncio
async def test_send_alert_journals_and_flush_deletes_row():
    lp = _loop(discord=True, telegram=False)
    db = _fake_db()
    with patch("app.database.get_db", return_value=db):
        await lp._send_alert(_mon("A", "system", channels="discord"), "alpha")
        assert lp._digest_buffer[0][4] == 7  # journal row id captured
        insert_sql = db.execute.call_args_list[0].args[0]
        assert "INSERT INTO pending_deliveries" in insert_sql
        await lp._flush_digest()
    assert lp._discord.send_alert.await_count == 1
    # confirmed broadcast cleans the journal row
    del_args = db.executemany.call_args
    assert "DELETE FROM pending_deliveries" in del_args.args[0]
    assert del_args.args[1] == [(7,)]


@pytest.mark.asyncio
async def test_journal_write_failure_still_buffers():
    # DB down must never block delivery — degrade to in-memory-only buffering.
    lp = _loop(discord=True, telegram=False)
    with patch("app.database.get_db", side_effect=RuntimeError("db down")):
        await lp._send_alert(_mon("A", "system", channels="discord"), "alpha")
    assert len(lp._digest_buffer) == 1
    assert lp._digest_buffer[0][4] is None


@pytest.mark.asyncio
async def test_recovery_rebuffers_and_delivers_undelivered_alert():
    # The 2026-08-12 failure mode: digest completed + buffered, restart wiped
    # the buffer, owner never saw it. Recovery must re-buffer from the journal
    # so the next flush actually posts it.
    lp = _loop(discord=True, telegram=False)
    db = _fake_db()
    db.fetchall.return_value = [
        {"id": 3, "targets": "discord", "monitor_name": "World Awareness",
         "message": "the briefing", "category": "content"},
    ]
    with patch("app.database.get_db", return_value=db):
        await lp._recover_pending_deliveries()
        assert len(lp._digest_buffer) == 1
        assert lp._digest_buffer[0][1] == "World Awareness"
        assert lp._digest_buffer[0][4] == 3
        await lp._flush_digest()
    assert lp._discord.send_alert.await_count == 1
    assert "the briefing" in lp._discord.send_alert.await_args_list[0].args[0]
    assert db.executemany.call_args.args[1] == [(3,)]


def test_inverted_leads_curation(db):
    # "Citadel leads Ken Griffin" alongside the correct direction: the
    # org-as-subject side is superseded; ambiguous pairs (both person-shaped
    # or both org-shaped) are left alone.
    from app.monitors.heartbeat_loop import _curate_inverted_leads
    from app.core.kg import KnowledgeGraph
    KnowledgeGraph(db)   # owns/upgrades kg_facts (adds superseded_at on fresh schemas)
    def _fact(s, o):
        db.execute("INSERT INTO kg_facts (subject, predicate, object, confidence) "
                   "VALUES (?, 'leads', ?, 0.9)", (s, o))
    _fact("Ken Griffin", "Citadel Capital")
    _fact("Citadel Capital", "Ken Griffin")           # wrong direction
    _fact("Acme Group", "Beta Fund")                  # org<->org: ambiguous
    _fact("Beta Fund", "Acme Group")
    n = _curate_inverted_leads(db)
    assert n == 1
    live = {r["subject"] for r in db.fetchall(
        "SELECT subject FROM kg_facts WHERE predicate='leads' AND superseded_at IS NULL")}
    assert "Ken Griffin" in live and "Citadel Capital" not in live
    assert {"Acme Group", "Beta Fund"} <= live        # ambiguity untouched


@pytest.mark.asyncio
async def test_send_alert_reports_delivery_outcome():
    # Deep pass 2026-08-14: the caller records result status from this return —
    # buffered/journaled alerts are True; a no-channel suppression is False, so
    # a suppressed result can never be recorded as a delivered alert.
    lp = _loop(discord=True, telegram=False)
    with patch("app.database.get_db", side_effect=RuntimeError("no journal in test")):
        assert await lp._send_alert(_mon("A", "system", channels="discord"), "alpha") is True
    lp2 = _loop(discord=False, telegram=False)
    assert await lp2._send_alert(_mon("B", "system", channels="discord"), "beta") is False


@pytest.mark.asyncio
async def test_failed_broadcast_keeps_journal_row_and_rebuffers():
    lp = _loop(discord=True, telegram=False)
    lp._discord.send_alert.return_value = False  # every channel fails
    lp._digest_buffer = [_entry({"discord"}, "A", "alpha", "content", row_id=9)]
    db = _fake_db()
    with patch("app.database.get_db", return_value=db):
        await lp._flush_digest()
    # re-buffered with its journal row intact; nothing deleted from the journal
    assert len(lp._digest_buffer) == 1
    assert lp._digest_buffer[0][4] == 9
    db.executemany.assert_not_called()
