"""A digest-class monitor on on_change delivers its digest, not a rewrite
(2026-09-22).

World Awareness is the one query digest seeded with notify_condition
"on_change". Live 09:42 UTC it finished a 50-minute 27B chain, stored 9,011
characters, and the change path handed the result to _analyze_result — a
120-token rewrite on the default 9B — so Telegram received 310 characters
("**What changed:** ...") and the card swapped 27B→9B→27B to write them.
The rewrite exists to describe what changed in a short or numeric result;
a briefing describes itself.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from app.monitors.heartbeat_loop import HeartbeatLoop
from app.monitors.monitor_store import MonitorStore


@pytest.fixture
def no_kg(monkeypatch):
    import app.core.brain as brain
    monkeypatch.setattr(brain, "get_services", lambda: SimpleNamespace(kg=None))


def _on_change_monitor(db, store, name, check_type, old_result):
    mid = store.create(name, check_type, {"query": "what happened"} if check_type == "query"
                       else {"url": "https://example.com"},
                       schedule_seconds=3600, cooldown_minutes=0, notify_condition="on_change")
    db.execute("UPDATE monitors SET last_result = ? WHERE id = ?", (old_result, mid))
    return store.get(mid)


@pytest.mark.asyncio
async def test_digest_class_on_change_delivers_the_digest_itself(db, no_kg):
    store = MonitorStore(db)
    loop = HeartbeatLoop(store)
    monitor = _on_change_monitor(db, store, "Domain Study: Test Region", "query",
                                 "## briefing\n" + "Yesterday's grounded analysis. " * 60)
    new = "## briefing\n" + "A new paragraph of grounded analysis. " * 120
    with patch.object(loop, "_execute_check", new_callable=AsyncMock) as mock_exec, \
         patch.object(loop, "_send_alert", new_callable=AsyncMock) as mock_send, \
         patch.object(loop, "_analyze_result", new_callable=AsyncMock) as mock_analyze:
        mock_exec.return_value = new
        mock_send.return_value = True
        await loop._check_monitor(monitor)
    mock_analyze.assert_not_called()
    mock_send.assert_called_once()
    sent = mock_send.call_args.args[1]
    assert sent.strip() == new.strip()          # the briefing itself, whole
    latest = store.get_results(monitor.id)[0]
    assert latest.status == "changed"          # still recorded as a change


@pytest.mark.asyncio
async def test_short_monitor_on_change_still_gets_the_rewrite(db, no_kg):
    store = MonitorStore(db)
    loop = HeartbeatLoop(store)
    monitor = _on_change_monitor(db, store, "Price watch", "url", "price 40")
    with patch.object(loop, "_execute_check", new_callable=AsyncMock) as mock_exec, \
         patch.object(loop, "_send_alert", new_callable=AsyncMock) as mock_send, \
         patch.object(loop, "_analyze_result", new_callable=AsyncMock) as mock_analyze:
        mock_exec.return_value = "price 42 after the announcement"
        mock_analyze.return_value = "**What changed:** price rose. **Key detail:** 42"
        mock_send.return_value = True
        await loop._check_monitor(monitor)
    mock_analyze.assert_called_once()
    assert mock_send.call_args.args[1].startswith("**What changed:**")
