"""Item 4 (2026-10-12): the citation-discipline rule, flag-gated for the replay."""

from __future__ import annotations

from app.monitors import deep_research as dr


def test_rule_is_empty_while_the_flag_is_off(monkeypatch):
    from app.config import config
    monkeypatch.setattr(config, "ENABLE_CITATION_DISCIPLINE", False)
    assert dr._citation_discipline_rule() == ""


def test_rule_names_the_attribution_the_gate_checks(monkeypatch):
    from app.config import config
    monkeypatch.setattr(config, "ENABLE_CITATION_DISCIPLINE", True)
    r = dr._citation_discipline_rule()
    assert "CITATION DISCIPLINE" in r and "never cited to another" in r


def test_rule_is_appended_to_the_synthesis_prompt():
    import inspect
    src = inspect.getsource(dr._synthesize_from_evidence)
    assert "_citation_discipline_rule()" in src
