"""Phase 4 (2026-10-12): fact spans kept at consolidation; superseded facts
filtered from chat injection; the tension resolver falls back to the spans."""

from __future__ import annotations

from app.core import fact_spans as fs

BODY = (
    "## Current understanding\n\nInflation cooled.\n\n"
    "## Key facts & figures\n\n"
    "* Headline inflation ran at 3.4% in August (reuters.com)\n"
    "* The ECB holds purchases at 2 billion euros (ecb.europa.eu)\n"
    "* Rates rose in 2024 (bbc.com)\n"
    "* GDP grew 1.5% in Q2 — superseded: Economics records 1.1% as of 2026-10-01\n"
    "## Open questions\n- x\n"
)
SOURCES = (
    "[digest 2026-09-02 08:00:00]\nMarkets were calm. Headline inflation ran at 3.4% in August, the agency said. Nothing else here.\n\n"
    "[digest 2026-10-05 08:00:00]\nThe ECB kept purchases at 2 billion euros this month. Headline inflation eased to 2.9% in September."
)


def test_spans_are_recorded_with_their_digest_date(db):
    n = fs.record_fact_spans(db, "domain", "finance", BODY, SOURCES)
    rows = db.fetchall("SELECT figure, sentence, source_date FROM fact_spans WHERE dkey='finance' ORDER BY id")
    assert n == len(rows) >= 2
    figs = {r["figure"] for r in rows}
    assert "3.4%" in figs and any("2 billion" in f for f in figs)
    assert {r["source_date"] for r in rows} == {"2026-09-02", "2026-10-05"}
    assert all("2024" not in r["figure"] for r in rows)        # years are not metrics
    assert not any("1.5%" in r["figure"] for r in rows)        # superseded bullets are skipped
    assert fs.record_fact_spans(db, "domain", "finance", BODY, SOURCES) == 0   # idempotent


def test_spans_for_finds_the_figure_newest_first(db):
    fs.record_fact_spans(db, "domain", "finance", BODY, SOURCES)
    got = fs.spans_for(db, "finance", 3.4, "%")
    assert got and got[0]["date"] == "2026-09-02" and "3.4%" in got[0]["sentence"]
    assert fs.spans_for(db, "finance", 9.9, "%") == []


def test_drop_superseded_removes_only_marked_lines():
    text = "* A fact (x.com)\n* Old fact — superseded: B records 2% as of 2026-10-01\n* Another (y.com)"
    assert fs.drop_superseded(text) == "* A fact (x.com)\n* Another (y.com)"
    assert fs.drop_superseded("plain") == "plain"


def test_chat_excerpt_and_priming_drop_superseded_facts(db):
    from app.core.dossiers import get_relevant_dossiers, priming_excerpt
    body = ("## Current understanding\n\nGDP grew 1.5% in Q2 — superseded: Economics records 1.1%\n"
            "Inflation story continues for the economy.\n\n## Key facts & figures\n\n* x\n")
    db.execute("INSERT INTO dossiers (kind, dkey, title, body) VALUES ('domain', 'economy', 'Economy', ?)", (body,))
    out = get_relevant_dossiers(db, "what is happening in the economy and inflation", limit=1)
    assert out and "superseded" not in out[0]["excerpt"] and "Inflation story" in out[0]["excerpt"]
    assert "superseded" not in priming_excerpt(body)


def test_resolver_falls_back_to_fact_spans():
    import inspect
    from app.core import tensions
    assert "spans_for(" in inspect.getsource(tensions.resolve_tension)
