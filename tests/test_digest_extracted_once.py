"""G1 (plan 2026-10-09): a digest's facts are banked once, not twice.

Measured 2026-10-09: a Domain Study digest was banked by
deep_research._learn_facts (source='researched') and AGAIN by the heartbeat's
post-digest `_extract_kg_triples` pass (source='extracted') on the same text,
and kg.py treated the second as independent corroboration (trust +0.2,
quarantine released). One 27B call per digest wasted, one false signal.

Fix: `_learn_facts` counts what it banked on a per-run ContextVar; the
heartbeat records that count for the monitor and skips the second pass when
the runner banked anything. Think-path query monitors (no runner) still get
the heartbeat extraction — it is their only banking path.
"""

from __future__ import annotations

import inspect
from unittest.mock import AsyncMock, patch

import pytest

from app.monitors import deep_research as dr
from app.monitors.heartbeat_loop import HeartbeatLoop


def _loop() -> HeartbeatLoop:
    loop = HeartbeatLoop.__new__(HeartbeatLoop)
    loop._runner_banked = {}
    return loop


class _Mon:
    def __init__(self, mid: int, name: str, check_type: str = "query"):
        self.id, self.name, self.check_type = mid, name, check_type


@pytest.mark.asyncio
async def test_learn_facts_counts_what_it_banked():
    kg = AsyncMock()
    kg.add_fact = AsyncMock(return_value=True)
    raw = '{"facts": [{"subject": "Reuters", "predicate": "reports_on", "object": "copper tariffs"}]}'
    findings = [("t", "https://reuters.com/x", "Reuters reports on copper tariffs today.")]
    token = dr.BANKED_THIS_RUN.set(0)
    try:
        with patch.object(dr, "_invoke_bg", AsyncMock(return_value=raw)):
            stored = await dr._learn_facts("copper", "Reuters reports on copper tariffs.", findings, kg)
        assert stored == 1
        assert dr.BANKED_THIS_RUN.get() == 1
    finally:
        dr.BANKED_THIS_RUN.reset(token)


def test_runner_banked_skips_the_second_extraction():
    loop = _loop()
    mon = _Mon(7, "Domain Study: Commodities")
    loop._runner_banked[mon.id] = 6
    assert loop._should_extract_after_digest(mon) is False
    # consumed: the next run of the same monitor decides afresh
    assert mon.id not in loop._runner_banked


def test_runner_banked_zero_still_extracts():
    loop = _loop()
    mon = _Mon(8, "Domain Study: Quantum")
    loop._runner_banked[mon.id] = 0
    assert loop._should_extract_after_digest(mon) is True


def test_think_path_monitor_still_extracts():
    loop = _loop()
    mon = _Mon(9, "Morning Check-in")
    assert loop._should_extract_after_digest(mon) is True


def test_query_runner_records_the_run_count():
    src = inspect.getsource(HeartbeatLoop._execute_query_monitor)
    assert "BANKED_THIS_RUN.set(0)" in src
    assert "self._runner_banked[monitor.id] = " in src
    check = inspect.getsource(HeartbeatLoop._check_monitor)
    assert "self._should_extract_after_digest(monitor)" in check
