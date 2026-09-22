"""Curiosity keeps the card on the model its item already holds (2026-09-22).

One hourly run on the live system: three evidence-first items on the 27B,
and the 9B loaded six seconds after each research finished — the KG-banking
call had no model (default 9B) — then once more after the run, because the
monitor is on_change and its status line went through the 120-token alert
rewrite. Four reloads for one run. The banking now names the path's model,
and a curiosity result is delivered as the line it already is.
"""
from __future__ import annotations

import ast
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

import app as _app_pkg
from app.monitors.heartbeat_loop import HeartbeatLoop
from app.monitors.monitor_store import Monitor

ROOT = Path(_app_pkg.__file__).resolve().parents[1]


def test_every_kg_banking_call_in_the_loop_names_its_model():
    """The residency scan looks for model-less invoke_nothink calls; a wrapper
    that forwards `model=model` hides a model-less CALLER from it."""
    src = (ROOT / "app" / "monitors" / "heartbeat_loop.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    bare = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", None)
            if name == "_extract_kg_triples" and not any(k.arg == "model" for k in node.keywords):
                bare.append(node.lineno)
    assert bare == [], f"_extract_kg_triples without model= at lines {bare}"


@pytest.mark.asyncio
async def test_curiosity_result_is_delivered_as_its_own_line(monkeypatch):
    assert "curiosity" in HeartbeatLoop._RAW_RESULT_CHECK_TYPES
    from app.core import llm
    monkeypatch.setattr(llm, "invoke_nothink", AsyncMock(side_effect=AssertionError("LLM must not run")))
    lp = object.__new__(HeartbeatLoop)
    mon = Monitor(id=1, name="Curiosity Research", check_type="curiosity", check_config={},
                  schedule_seconds=3600, enabled=True, cooldown_minutes=0,
                  notify_condition="on_change", last_check_at=None, last_alert_at=None,
                  last_result="CURIOSITY BATCH | 0/3 resolved this run", created_at="2026-01-01T00:00:00")
    line = "CURIOSITY BATCH | 1/3 resolved this run\nCURIOSITY RESOLVED | topic=x"
    out = await lp._analyze_result(mon, line, {"type": "text"})
    assert out == line
