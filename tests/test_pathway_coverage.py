"""Which writers are watched, and which are deliberately not (2026-09-04).

24 of 40 timestamped tables had no liveness pathway. Most of them rightly so —
a registry that watches everything is a registry nobody reads — but four were
real background writers that would have failed in silence, and one absence had
already cost three days of a broken canary.

The exclusions matter as much as the inclusions, which is why they are asserted
here rather than left as an absence somebody helpfully fills in. A pathway
pointed at a writer that cannot write reads DEAD forever, breaks the
"all pathways alive" marker permanently, and trains everyone to ignore it —
which is exactly what trust_ledger did after trust moved in-memory on
2026-09-01.
"""
from __future__ import annotations

import pytest

from app.monitors.pathways import PATHWAYS

BY_NAME = {p.name: p for p in PATHWAYS}
WATCHED_TABLES = {p.table for p in PATHWAYS if p.table}


def test_the_knowing_tier_has_both_halves_of_its_ledger():
    """question_ledger writing while belief_revisions is silent means the
    dossiers are being rewritten but no longer changing their mind — which
    looks healthy from every other angle."""
    assert "belief_revisions" in BY_NAME
    p = BY_NAME["belief_revisions"]
    assert p.table == "belief_revisions"
    assert p.flag == "ENABLE_DOSSIERS"
    assert p.monitor == "Knowledge Consolidation"


def test_procedural_memory_gets_a_window_wide_enough_for_its_own_cap():
    """The dream cycle caps at 3 clusters and refuses to re-consolidate the same
    family within 7 days. Live on 2026-09-04 it was 163 hours quiet and healthy;
    a one-week window would have been five hours from crying wolf."""
    p = BY_NAME["procedural_memory"]
    assert p.cadence_hours >= 336, "a fortnight, because a quiet week is normal"
    assert p.flag == "ENABLE_PROCEDURAL_CONSOLIDATION"


def test_a_writer_behind_a_disabled_monitor_reports_off_not_dead():
    """The registry should say WHY a writer is quiet."""
    p = BY_NAME["auto_tool_candidates"]
    assert p.monitor == "Auto-Tool Synthesis"


def test_deliberation_scratchpads_are_usage_gated():
    """They only write when the owner talks to Nova; silence is idle, not dead."""
    p = BY_NAME["agent_workspace"]
    assert p.usage_gated is True
    assert p.flag == "ENABLE_DELIBERATION"


@pytest.mark.parametrize("table,why", [
    ("capability_gaps",
     "written only when a query matches no skill AND uses no tools AND scores "
     "under 0.5 — supposed to be rare, has never held a row, so a probe cannot "
     "tell a dead writer from an unmet condition"),
    ("pending_deliveries",
     "an empty journal is the HEALTHY state; a pathway here would be inverted"),
    ("trust_scores",
     "trust moved in-memory 2026-09-01; the pathway outlived its writer and "
     "broke the canary for three days"),
    ("user_facts", "zero by design"),
    ("verifiable_signals", "RLVR archived 2026-09-01"),
    ("auth_lockouts", "a security event, not a background writer"),
    ("system_state", "key-value configuration"),
    ("goals", "event-driven"),
    ("monitor_dedup_log", "already covered by the dedup_decisions pathway"),
])
def test_these_tables_are_deliberately_unwatched(table, why):
    assert table not in WATCHED_TABLES, (
        f"{table} must NOT have a liveness pathway: {why}")


def test_every_pathway_names_something_it_can_probe():
    """A pathway with neither a table nor a file probes nothing and would
    report a verdict it never measured."""
    for p in PATHWAYS:
        assert p.table or p.path, f"{p.name} has nothing to probe"
        if p.table:
            assert p.time_col, f"{p.name} has no column to prove a write"


def test_pathway_names_are_unique():
    """_BY_NAME is a dict; a duplicate name silently drops a whole pathway —
    the same shape as the duplicate _CHECK_DISPATCH key that killed scheduled
    Dream Consolidation for weeks."""
    names = [p.name for p in PATHWAYS]
    assert len(names) == len(set(names)), \
        f"duplicate pathway name(s): {sorted({n for n in names if names.count(n) > 1})}"
