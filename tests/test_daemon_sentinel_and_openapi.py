"""Two one-line defects from the 2026-09-01 audit.

1. daemon.py used `0.0` as the never-ran sentinel for the curiosity cooldown and
   compared `time.monotonic() - 0.0 >= 1800`: on any process younger than 30
   minutes (every GitHub CI runner, and the first 30 minutes after each WSL
   boot) daemon curiosity research was silently suppressed. CI was 4/4 red on
   exactly this assertion.
2. /openapi.json was served without a key while /docs and /redoc were gated.
"""
from __future__ import annotations

import time

from app.config import config as _cfg
from app.monitors import daemon as daemon_mod


def _daemon():
    cls = daemon_mod.DaemonOrchestrator
    d = cls.__new__(cls)
    d._last_curiosity_research = None
    return d


def test_fresh_process_is_cold_and_may_research(monkeypatch):
    monkeypatch.setattr(time, "monotonic", lambda: 100.0)
    d = _daemon()
    assert d._curiosity_cooldown_elapsed() is True


def test_recent_research_is_on_cooldown(monkeypatch):
    monkeypatch.setattr(time, "monotonic", lambda: 100000.0)
    d = _daemon()
    d._last_curiosity_research = 100000.0 - 600
    assert d._curiosity_cooldown_elapsed() is False
    d._last_curiosity_research = 100000.0 - 1801
    assert d._curiosity_cooldown_elapsed() is True


def test_openapi_is_gated_like_docs():
    from app import main
    # gated exactly like /docs: both open, or both closed
    assert (main.app.openapi_url is None) == (main.app.docs_url is None)
    if main.app.docs_url is None:
        assert main.app.openapi_url is None
