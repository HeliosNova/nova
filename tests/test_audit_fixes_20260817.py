"""Regression tests for the 2026-08-17 deep-audit fixes.

HIGH — duplicate ``"consolidation"`` key in ``_CHECK_DISPATCH`` made scheduled
Dream Consolidation silently run the dossier cycle instead (the later key won),
so the dream pipeline never ran on schedule. The dream handler now dispatches on
``"dream_consolidation"``. The AST test below fails on ANY future duplicate key,
not just this one.
"""
from __future__ import annotations

import ast
import asyncio
import pathlib
from unittest.mock import AsyncMock, MagicMock

_APP = pathlib.Path(__file__).resolve().parent.parent / "app"


def _dispatch_key_list() -> list[str]:
    """Every string key in the _CHECK_DISPATCH dict literal, WITH duplicates
    preserved (so we can detect shadowing)."""
    src = (_APP / "monitors" / "heartbeat_loop.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Dict):
            if any(isinstance(t, ast.Name) and t.id == "_CHECK_DISPATCH" for t in node.targets):
                return [k.value for k in node.value.keys if isinstance(k, ast.Constant)]
    raise AssertionError("_CHECK_DISPATCH dict literal not found")


def test_check_dispatch_has_no_duplicate_keys():
    keys = _dispatch_key_list()
    dupes = sorted({k for k in keys if keys.count(k) > 1})
    assert not dupes, f"duplicate dispatch keys silently shadow handlers: {dupes}"


def test_both_consolidation_handlers_are_reachable():
    from app.monitors.heartbeat_loop import HeartbeatLoop as H
    d = H._CHECK_DISPATCH
    assert "consolidation" in d, "knowledge/dossier consolidation key missing"
    assert "dream_consolidation" in d, "dream consolidation key missing (would be dead)"


def test_dream_monitor_seed_uses_distinct_check_type():
    """The seeded 'Dream Consolidation' monitor must carry check_type
    'dream_consolidation' so it reaches _execute_consolidation, not the dossier
    cycle."""
    src = (_APP / "monitors" / "monitor_store.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    seen = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            name = ctype = None
            for k, v in zip(node.keys, node.values):
                if isinstance(k, ast.Constant) and isinstance(v, ast.Constant):
                    if k.value == "name":
                        name = v.value
                    elif k.value == "check_type":
                        ctype = v.value
            if name in ("Dream Consolidation", "Knowledge Consolidation"):
                seen[name] = ctype
    assert seen.get("Dream Consolidation") == "dream_consolidation", seen
    assert seen.get("Knowledge Consolidation") == "consolidation", seen


# ---------------------------------------------------------------------------
# MED — decay_stale_skills was dead code, defined twice, never called.
# ---------------------------------------------------------------------------

def test_skills_methods_not_duplicated():
    src = (_APP / "core" / "skills.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    counts: dict[str, int] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            counts[node.name] = counts.get(node.name, 0) + 1
    for name in ("get_composed_steps", "get_skill_stats", "decay_stale_skills"):
        assert counts.get(name) == 1, f"{name} defined {counts.get(name)}x (duplicate shadowing)"


def test_decay_stale_skills_is_wired_into_maintenance():
    src = (_APP / "monitors" / "heartbeat_loop.py").read_text(encoding="utf-8")
    assert "svc.skills.decay_stale_skills" in src, \
        "decay_stale_skills must be called from _execute_maintenance (it was dead code)"


# ---------------------------------------------------------------------------
# LOW — Signal/WhatsApp send_alert must report a failed send as False so a lost
# digest is NOT purged from the at-least-once delivery journal.
# ---------------------------------------------------------------------------

def _signal_bot():
    from app.channels.signal import SignalBot
    bot = object.__new__(SignalBot)
    bot.api_url = "http://signal.local"
    bot.phone_number = "+10000000000"
    bot.default_recipient = "+19999999999"
    bot._send_lock = asyncio.Lock()
    bot._client = MagicMock()
    return bot


def _whatsapp_bot():
    from app.channels.whatsapp import WhatsAppBot
    bot = object.__new__(WhatsAppBot)
    bot.api_url = "http://wa.local/send"
    bot.api_token = "tok"
    bot.default_chat_id = "12345"
    bot._send_lock = asyncio.Lock()
    bot._client = MagicMock()
    return bot


def test_signal_send_message_reports_bool():
    bot = _signal_bot()
    bot._client.post = AsyncMock(return_value=MagicMock(status_code=200))
    assert asyncio.run(bot._send_message("+1999", "hi")) is True
    bot._client.post = AsyncMock(return_value=MagicMock(status_code=500, text="err"))
    assert asyncio.run(bot._send_message("+1999", "hi")) is False
    bot._client.post = AsyncMock(side_effect=RuntimeError("boom"))
    assert asyncio.run(bot._send_message("+1999", "hi")) is False


def test_whatsapp_send_message_reports_bool():
    bot = _whatsapp_bot()
    bot._client.post = AsyncMock(return_value=MagicMock(status_code=200))
    assert asyncio.run(bot._send_message("12345", "hi")) is True
    bot._client.post = AsyncMock(return_value=MagicMock(status_code=500, text="err"))
    assert asyncio.run(bot._send_message("12345", "hi")) is False
    bot._client.post = AsyncMock(side_effect=RuntimeError("boom"))
    assert asyncio.run(bot._send_message("12345", "hi")) is False


def test_signal_send_alert_propagates_failure():
    bot = _signal_bot()
    bot._split_message = lambda t: [t]
    bot._send_message = AsyncMock(return_value=False)
    assert asyncio.run(bot.send_alert("digest")) is False
    bot._send_message = AsyncMock(return_value=True)
    assert asyncio.run(bot.send_alert("digest")) is True


def test_whatsapp_send_alert_propagates_failure():
    bot = _whatsapp_bot()
    bot._split_message = lambda t: [t]
    bot._send_message = AsyncMock(return_value=False)
    assert asyncio.run(bot.send_alert("digest")) is False
    bot._send_message = AsyncMock(return_value=True)
    assert asyncio.run(bot.send_alert("digest")) is True
