"""Monitor alert batching into a per-cycle digest (2026-06-14).

80 monitors firing concurrently used to post 80 separate (interleaving)
messages. Alerts are now buffered per cycle and flushed as ONE digest per
channel-group. Pins the formatting, routing, buffering, and grouping.
"""
from types import SimpleNamespace
from unittest.mock import AsyncMock

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
    return lp


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
        (frozenset({"discord", "telegram"}), "China Tech", "chips", "content"),
        (frozenset({"discord", "telegram"}), "Crypto", "btc", "content"),
        (frozenset({"telegram"}), "Health", "ok", "system"),
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
