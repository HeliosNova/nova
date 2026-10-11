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





