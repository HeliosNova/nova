"""Typed numeric observations (2026-10-19).

Finance/markets/macro news is quantities ("Brent 105.56 USD/bbl", "Fed held at
4.5%") the (s,p,o) KG can't hold -- they degraded to related_to and were
discarded ("0 facts learned" on those digests). _learn_observations captures
them into the observations table (entity|metric|value|unit|asof) so the knowing
tier can track levels/deltas with SQL. The triple KG is untouched. Flag-gated
(ENABLE_OBSERVATIONS); every value is verified VERBATIM against a source body at
write time -- the one real risk of generative extraction is invented numbers.
"""

from __future__ import annotations

import pytest

from app.monitors import deep_research as dr


# A briefing + the sources that back its numbers. "996 BTC" and "4.5%" appear in
# the source bodies; "$999 billion" is an invented figure the model added.
FINDINGS = [
    ("MARA moves BTC", "https://coindesk.com/a", "MARA Holdings transferred 996 BTC to a custody wallet."),
    ("Fed decision", "https://reuters.com/b", "The Federal Reserve held its policy rate at 4.5% on Wednesday."),
]
ARTICLES = FINDINGS
BRIEF = ("MARA Holdings transferred 996 BTC this week (coindesk.com). The Federal Reserve "
         "held its policy rate at 4.5% (reuters.com). Analysts floated a $999 billion "
         "stimulus that no source confirms.")


async def _run(monkeypatch, db, flag=True):
    from app.config import config
    monkeypatch.setattr(config, "ENABLE_OBSERVATIONS", flag)
    monkeypatch.setattr("app.database.get_db", lambda: db)

    async def _fake_extract(*a, **k):
        return [
            {"entity": "MARA Holdings", "metric": "BTC transfer", "value": "996", "unit": "BTC", "direction": "level"},
            {"entity": "Federal Reserve", "metric": "policy rate", "value": "4.5", "unit": "%", "direction": "level"},
            {"entity": "US Treasury", "metric": "stimulus", "value": "999", "unit": "billion USD", "direction": "level"},
        ]
    monkeypatch.setattr(dr, "_invoke_bg", _fake_extract)
    return await dr._learn_observations("crypto", BRIEF, FINDINGS, articles=ARTICLES)


@pytest.mark.asyncio
async def test_banks_verified_numbers_and_drops_the_invented_one(monkeypatch, db):
    n = await _run(monkeypatch, db, flag=True)
    assert n == 2                      # 996 and 4.5 verified; 999 absent from sources -> dropped
    rows = db.fetchall("SELECT entity, metric, value_text, unit, value_num FROM observations ORDER BY entity")
    got = {(r["entity"], r["value_text"], r["unit"]) for r in rows}
    assert ("Federal Reserve", "4.5", "%") in got
    assert ("MARA Holdings", "996", "BTC") in got
    assert not any(r["value_text"] == "999" for r in rows)     # invented number NOT banked
    assert any(r["value_num"] == 4.5 for r in rows)            # numeric value parsed


@pytest.mark.asyncio
async def test_flag_off_banks_nothing(monkeypatch, db):
    n = await _run(monkeypatch, db, flag=False)
    assert n == 0
    assert db.fetchall("SELECT 1 FROM observations") == []


def test_observations_table_and_pathway_exist(db):
    cols = {r[1] for r in db.fetchall("PRAGMA table_info(observations)")}
    assert {"entity", "metric", "value_num", "value_text", "unit", "asof"} <= cols
    from app.monitors.pathways import PATHWAYS
    names = {p.name for p in PATHWAYS}
    assert "observations" in names


def test_flag_defaults_off():
    import os
    from app.config import config
    if "ENABLE_OBSERVATIONS" not in os.environ:
        assert config.ENABLE_OBSERVATIONS is False
