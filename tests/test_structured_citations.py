"""Phase 1b (2026-10-13): structured citations, flag-gated for the replay."""

from __future__ import annotations

from app.monitors import deep_research as dr

FINDINGS = [("Copper rises", "https://www.reuters.com/a", "Copper rose 3%."),
            ("Peru exports", "https://apnews.com/b", "Peru exports fell 7%.")]


def test_evidence_carries_ids_only_when_on(monkeypatch):
    from app.config import config
    monkeypatch.setattr(config, "ENABLE_STRUCTURED_CITATIONS", False)
    assert "[F1]" not in dr._annotated_evidence(FINDINGS)
    monkeypatch.setattr(config, "ENABLE_STRUCTURED_CITATIONS", True)
    ev = dr._annotated_evidence(FINDINGS)
    assert ev.startswith("[F1] [Copper rises] (reuters.com") and "[F2] [Peru exports] (apnews.com" in ev


def test_ids_map_to_hosts_and_unknown_ids_vanish():
    text = "Copper rose 3% on Monday [F1]. Peru's exports fell 7% [F2]. A claim with no source [F9]."
    out, n = dr._resolve_structured_citations(text, FINDINGS)
    assert n == 2
    assert out == "Copper rose 3% on Monday (reuters.com). Peru's exports fell 7% (apnews.com). A claim with no source."


def test_resolved_citations_pass_the_gate_regex():
    out, _ = dr._resolve_structured_citations("Copper rose [F1].", FINDINGS)
    assert dr._PAREN_CITE_RE.search(out)


def test_rule_and_wiring(monkeypatch):
    import inspect
    from app.config import config
    monkeypatch.setattr(config, "ENABLE_STRUCTURED_CITATIONS", True)
    assert "[F3]" in dr._structured_citation_rule()
    src = inspect.getsource(dr._synthesize_from_evidence)
    assert "_resolve_structured_citations(draft, findings)" in src and "_structured_citation_rule()" in src
