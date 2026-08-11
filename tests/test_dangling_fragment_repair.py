"""Dangling-fragment repair (2026-07-08): recover digest sentences left broken
when a grounding/strip pass excised a named entity but not its scaffolding.

Real cases from a posted Current Events digest:
  - "cross-border security implications for, requiring careful monitoring"
    (contamination backstop removed a country after "for")
  - "'s small-molecule oral GLP-1 agonist" (company name stripped before "'s")
"""

from __future__ import annotations

from app.monitors.deep_research import _repair_dangling_fragments


class TestDanglingPreposition:
    def test_real_case_implications_for_comma(self):
        text = ("a potential destabilizing event with cross-border security "
                "implications for, requiring careful monitoring")
        out = _repair_dangling_fragments(text)
        assert "for," not in out
        assert "implications, requiring" in out or "implications requiring" in out
        assert "careful monitoring" in out   # rest of the sentence preserved

    def test_various_prepositions(self):
        for prep in ("in", "to", "with", "against", "including", "between"):
            out = _repair_dangling_fragments(f"the impact {prep}, and then more")
            assert f"{prep}," not in out
            assert "and then more" in out

    def test_object_bearing_preposition_untouched(self):
        # "for X, Y" has a real object → must NOT be altered
        text = "This matters for Rwanda, Uganda, and the wider region."
        assert _repair_dangling_fragments(text) == text

    def test_normal_comma_list_untouched(self):
        text = "It read apnews.com, cnbc.com, and bbc.com."
        assert _repair_dangling_fragments(text) == text


class TestOrphanPossessive:
    def test_clause_initial_orphan_possessive(self):
        text = "The trial concluded. 's small-molecule agonist beat the competitor."
        out = _repair_dangling_fragments(text)
        assert "'s small-molecule" not in out
        assert "small-molecule agonist beat" in out

    def test_real_possessive_untouched(self):
        text = "Novo Nordisk's oral drug advanced to Phase 3."
        assert _repair_dangling_fragments(text) == text


class TestSafety:
    def test_empty(self):
        assert _repair_dangling_fragments("") == ""

    def test_clean_text_untouched(self):
        text = "A perfectly normal briefing sentence with no defects at all."
        assert _repair_dangling_fragments(text) == text
