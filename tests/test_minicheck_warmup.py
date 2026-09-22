"""The entailment sidecar warms itself at startup (2026-09-22).

Measured 07:40-07:48 UTC: the container idled for 6.5 minutes, then the
first /check_batch paid the 6 s model load and a 73 s first inference
(every later pair 1-2 s). nova-app's gate read the timeout as "sidecar
unavailable" and deferred the question. That happens after every restart.
The sidecar now loads and runs one inference on its startup, on the scoring
lock, and reports it in /health as "warm".
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

import app as _app_pkg

ROOT = Path(_app_pkg.__file__).resolve().parents[1]
SRC = ROOT / "minicheck_service" / "app.py"
pytestmark = pytest.mark.skipif(not SRC.exists(), reason="sidecar source is not shipped in the image")


def _load():
    spec = importlib.util.spec_from_file_location("minicheck_sidecar_under_test", SRC)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_warm_up_runs_one_inference_on_the_scoring_lock(monkeypatch):
    mod = _load()
    calls: list = []

    class _Scorer:
        def score(self, docs, claims):
            assert mod._score_lock.locked()        # a real request queues behind it
            calls.append((docs, claims))
            return ([1] * len(docs), [0.9] * len(docs))

    monkeypatch.setattr(mod, "_get_scorer", lambda: _Scorer())
    assert mod.health()["warm"] is False
    assert mod.warm_up() is True
    assert len(calls) == 1 and len(calls[0][0]) == 1
    assert mod.health()["warm"] is True
    assert not mod._score_lock.locked()            # released for real traffic


def test_warm_up_failure_leaves_lazy_loading_intact(monkeypatch):
    mod = _load()

    def _boom():
        raise RuntimeError("no model")

    monkeypatch.setattr(mod, "_get_scorer", _boom)
    assert mod.warm_up() is False
    assert mod.health()["warm"] is False


@pytest.mark.asyncio
async def test_startup_kicks_off_the_warm_up(monkeypatch):
    mod = _load()
    started: list = []
    monkeypatch.setattr(mod, "_start_warm_up", lambda: started.append(True))
    async with mod._lifespan(mod.app):
        assert started == [True]                    # before the app serves anything
