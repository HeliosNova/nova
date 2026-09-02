"""Dossier priming key resolution + priming excerpt (audit 2026-09-01).

Live finding: deep_research primed each digest with
`get_domain_dossier(db, label)` where `label` is the SHORT profile label from
domain_study_runner._profile_for ('AI/ML', 'open source', 'Europe + EU'), but
consolidation keys domain dossiers by slug(monitor name minus 'Domain Study: ')
('ai-and-ml', 'open-source-and-github', 'europe-and-eu'). Executed live: only
13 of 39 Domain Studies found their dossier; the other 26 ran unprimed and the
running priming A/B was comparing primed-vs-unprimed on 6 of 16 topics.
Also: the primed 2,500-char body slice ended before '## Open questions'.
"""
from __future__ import annotations

import pytest

from app.core.dossiers import (
    _slug,
    get_domain_dossier,
    priming_excerpt,
    resolve_domain_dkey,
)
from app.monitors.domain_study_runner import _DOMAIN_PROFILES


def _put(db, dkey: str, body: str, title: str = "T") -> None:
    db.execute(
        "INSERT INTO dossiers (kind, dkey, title, body, changed_note, update_count, "
        "created_at, updated_at) VALUES ('domain', ?, ?, ?, '', 1, datetime('now'), datetime('now'))",
        (dkey, title, body),
    )


def test_short_profile_label_resolves_to_monitor_slug(db):
    _put(db, "ai-and-ml", "## Current understanding\nAI body")
    row = get_domain_dossier(db, "AI/ML")
    assert row is not None and row["dkey"] == "ai-and-ml"


def test_monitor_name_resolves_directly(db):
    _put(db, "open-source-and-github", "## Current understanding\nOSS body")
    row = get_domain_dossier(db, "open source", monitor_name="Domain Study: Open Source and GitHub")
    assert row is not None and row["dkey"] == "open-source-and-github"


def test_direct_slug_still_wins_when_present(db):
    _put(db, "finance", "## Current understanding\nFinance body")
    row = get_domain_dossier(db, "finance")
    assert row is not None and row["dkey"] == "finance"


def test_every_profile_label_maps_to_its_monitor_slug():
    labels = [v[1].strip().lower() for v in _DOMAIN_PROFILES.values()]
    assert len(labels) == len(set(labels)), "profile labels must be unique for alias resolution"
    for key, (_emoji, label, _kw) in _DOMAIN_PROFILES.items():
        assert resolve_domain_dkey(label) == _slug(key), (key, label)
        assert resolve_domain_dkey(label, monitor_name=f"Domain Study: {key.title()}") == _slug(key)


def test_unknown_label_falls_back_to_its_own_slug():
    assert resolve_domain_dkey("Quantum Gravity Watch") == "quantum-gravity-watch"


def test_priming_excerpt_keeps_understanding_and_open_questions():
    cu = "Sentence about the domain. " * 200          # ~5,400 chars
    body = (
        "## Current understanding\n" + cu +
        "\n\n## How we got here\nA long history that should NOT be primed. " * 20 +
        "\n\n## Key facts & figures\n- 42\n" +
        "\n\n## Open questions\n- Will the widget ship in Q4?\n- Who funds the consortium?\n"
    )
    out = priming_excerpt(body)
    assert out.startswith("## Current understanding")
    assert "## Open questions" in out
    assert "Will the widget ship in Q4?" in out
    assert "How we got here" not in out
    assert len(out) <= 3900


def test_priming_excerpt_without_sections_uses_body_head():
    out = priming_excerpt("Plain prose without headings. " * 300)
    assert out.startswith("## Current understanding")
    assert len(out) <= 2700


@pytest.mark.asyncio
async def test_known_vs_new_counter_uses_schema_and_room(monkeypatch):
    from app.monitors import deep_research

    seen: dict = {}

    async def _fake_invoke(messages, **kwargs):
        seen.update(kwargs)
        return '{"new": 3, "updates": 2, "contradictions": 1}'

    monkeypatch.setattr(deep_research, "_invoke_bg", _fake_invoke)
    counts = await deep_research._count_known_vs_new(
        "PRIOR UNDERSTANDING: prior body\n\n", "TODAY: digest text", model=None, label="AI/ML")
    assert counts == {"new": 3, "updates": 2, "contradictions": 1}
    assert seen.get("max_tokens", 0) >= 160, "80 tokens truncated the counter to 'None' live"
    assert seen.get("json_schema"), "small-model JSON needs a schema, not a prefill"
