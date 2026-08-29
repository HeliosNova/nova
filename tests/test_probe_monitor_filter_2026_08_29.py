"""Operator sanity probes must never become auto-monitors.

Live 2026-08-29: the owner pinged Nova with liveness checks on Aug 26-28 —
"reply with exactly: operational", "in one word, are you operational",
"sanity probe: reply with the single word aligned and nothing else". brain
records general-intent chat queries as topics (topic_frequency), one reached
min_count=3, and the auto-monitor pass turned it into monitor 77
"Auto: reply with exactly: operational" — a 12-hourly deep-research job on a
non-topic, burning GPU and feeding junk into the KG.

_BAD_MONITOR_RE existed to stop exactly this but its imperative list ran
what/who/find/show/tell/give/list and never covered reply/respond/say/answer.

Both directions are asserted: probes rejected, real research topics kept. A
filter that swallowed genuine interests would be a worse bug than the one it
fixes.
"""

from __future__ import annotations

import pytest

# The filter is built inline inside _create_auto_monitors; import the module and
# rebuild the same pattern would drift. Instead assert against the live source so
# the test fails if the pattern regresses.
import inspect
import re

from app.monitors import heartbeat_loop


def _live_pattern() -> re.Pattern:
    """Extract the _BAD_MONITOR_RE literal from the running source."""
    src = inspect.getsource(heartbeat_loop)
    start = src.index("_BAD_MONITOR_RE = _re.compile(")
    end = src.index("_re.IGNORECASE,", start)
    body = src[start:end]
    parts = re.findall(r'r"([^"]*)"', body)
    assert parts, "could not extract _BAD_MONITOR_RE source"
    return re.compile("".join(parts), re.IGNORECASE)


PROBES = [
    "reply with exactly: operational",
    "in one word, are you operational",
    "sanity probe: reply with the single word aligned and nothing else",
    "respond with OK",
    "say hello",
    "are you alive",
    "confirm you are online",
    "echo this back",
]

REAL_TOPICS = [
    "nvidia vera rubin production ramp",
    "solana disinflation proposal SGP-0002",
    "uk fca crypto authorisation deadline",
    "samsung processing-in-memory LPDDR5X",
    "post-quantum cryptography migration",
    "eu ai act enforcement timeline",
    "taiwan semiconductor capex guidance",
]


@pytest.mark.parametrize("probe", PROBES)
def test_operator_probes_are_rejected(probe):
    assert _live_pattern().search(probe), (
        f"probe {probe!r} would spawn an auto-monitor (this is how monitor 77 "
        f"'Auto: reply with exactly: operational' was created)"
    )


@pytest.mark.parametrize("topic", REAL_TOPICS)
def test_real_research_topics_survive(topic):
    hit = _live_pattern().search(topic)
    assert not hit, f"real topic {topic!r} was filtered out by {hit.group(0)!r}"
