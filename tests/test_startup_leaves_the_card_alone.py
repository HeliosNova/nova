"""Startup does no model work of its own; the heartbeat tick owns the card
(2026-09-23).

Measured on the 02:41 UTC restart: the 27B was resident for a digest batch,
startup's detached KG LLM curation asked for the 9B at 02:41:37, Ollama
evicted the 27B, the 9B cold-loaded for 4m09s (the warmup finished at 249 s),
and eight digests queued behind it before the 27B was loaded again. Both
startup callers are gone: the warmup steps aside while a heartbeat loop
runs, and the LLM curation pass runs inside the KG Health Monitor, which the
tick dispatches under the class gate on the model it already holds.
"""
from __future__ import annotations

import ast
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import app as _app_pkg
from app.monitors.heartbeat_loop import HeartbeatLoop

ROOT = Path(_app_pkg.__file__).resolve().parent
MAIN = ROOT / "main.py"


def _calls(tree: ast.AST, attr: str):
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == attr:
            yield node


def test_startup_runs_only_the_heuristic_curation():
    tree = ast.parse(MAIN.read_text(encoding="utf-8"))
    calls = list(_calls(tree, "curate"))
    assert calls, "startup still runs the heuristic curation pass"
    for call in calls:
        kws = {k.arg: k.value for k in call.keywords}
        llm_pass = "heuristic" in kws and isinstance(kws["heuristic"], ast.Constant) and kws["heuristic"].value is False
        assert not llm_pass, "the LLM curation pass is model work; it belongs under the tick's gate"
        sample = kws.get("sample_size")
        assert isinstance(sample, ast.Constant) and sample.value == 0, "startup curation must not sample facts for the LLM"


def test_the_warmup_steps_aside_for_the_heartbeat():
    tree = ast.parse(MAIN.read_text(encoding="utf-8"))
    warmups = [n for n in ast.walk(tree)
               if isinstance(n, ast.AsyncFunctionDef) and n.name == "_warmup"]
    assert len(warmups) == 1
    names = {n.id for n in ast.walk(warmups[0]) if isinstance(n, ast.Name)}
    assert "heartbeat_loop" in names, "the warmup must check for a running heartbeat loop before touching the card"
    # The guard comes before the generation: the first statement is the check.
    first = warmups[0].body[0]
    if isinstance(first, ast.Try):
        first = first.body[0]
    assert isinstance(first, ast.If), "the heartbeat guard is the warmup's first statement"


def _kg_mock():
    kg = MagicMock()
    kg.get_stats.return_value = {"total_facts": 100, "current_facts": 80, "superseded_facts": 20}
    kg._db.fetchone.side_effect = [{"c": 45}, {"c": 3}]
    return kg


@pytest.mark.asyncio
async def test_kg_health_runs_the_llm_curation_under_the_gate():
    loop = object.__new__(HeartbeatLoop)
    kg = _kg_mock()
    kg.curate = AsyncMock(return_value={"heuristic": 0, "llm": 2})
    svc = MagicMock()
    svc.kg = kg
    with patch("app.core.brain.get_services", return_value=svc):
        result = await loop._execute_kg_health_check()
    kg.curate.assert_awaited_once_with(sample_size=20, heuristic=False)
    assert "llm_retired: 2" in result
    assert "active: 80" in result


@pytest.mark.asyncio
async def test_a_failed_curation_does_not_change_the_health_verdict():
    loop = object.__new__(HeartbeatLoop)
    kg = _kg_mock()
    kg.curate = AsyncMock(side_effect=RuntimeError("model unreachable"))
    svc = MagicMock()
    svc.kg = kg
    with patch("app.core.brain.get_services", return_value=svc):
        result = await loop._execute_kg_health_check()
    assert "active: 80" in result
    assert "llm_retired" not in result
    assert "error" not in result.split("|")[0].lower()
