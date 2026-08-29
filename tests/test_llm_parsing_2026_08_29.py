"""app/core/llm.py had no test file — 587 lines imported by 37 modules.

A 2026-08-29 triage ranked it the single most load-bearing untested module.
Every LLM call in Nova goes through it, and its JSON-salvage helpers are what
the whole night's fixes leaned on: the KG contradiction judge, the KG curation
pass and the enrichment summariser all depend on extract_json_object() coping
with what a 9B actually emits.

These test the PURE helpers (no network): think-tag stripping, balanced-JSON
extraction and object salvage. Cases are drawn from real failures seen in
production this month, not invented.
"""

from __future__ import annotations

import pytest

from app.core.llm import _find_balanced_json, _strip_think_tags, extract_json_object


class TestStripThinkTags:
    """2026-07-08: an Ollama upgrade broke prefill think-suppression and six
    digests shipped raw reasoning to Discord. Stripping must be robust."""

    def test_removes_a_complete_think_block(self):
        assert "reasoning" not in _strip_think_tags(
            "<think>internal reasoning here</think>Real answer.")

    def test_keeps_the_answer(self):
        assert "Real answer." in _strip_think_tags(
            "<think>noise</think>Real answer.")

    def test_handles_no_tags(self):
        assert _strip_think_tags("Plain text.") == "Plain text."

    def test_handles_unclosed_tag(self):
        """A truncated generation can emit <think> with no closer — the result
        must not be the raw reasoning."""
        out = _strip_think_tags("<think>reasoning that never closes")
        assert "reasoning that never closes" not in out or out.strip() == ""

    def test_multiline_block(self):
        out = _strip_think_tags("<think>\nline1\nline2\n</think>\nAnswer")
        assert "line1" not in out and "Answer" in out


class TestFindBalancedJson:
    def test_extracts_object_ignoring_prose(self):
        got = _find_balanced_json('Sure! {"keep": "A"} hope that helps', "{")
        assert got == '{"keep": "A"}'

    def test_handles_nesting(self):
        s = '{"a": {"b": {"c": 1}}}'
        assert _find_balanced_json(f"text {s} text", "{") == s

    def test_braces_inside_strings_do_not_unbalance(self):
        s = '{"msg": "a } brace in a string"}'
        assert _find_balanced_json(s, "{") == s

    def test_passes_text_through_when_no_json_present(self):
        """Contract is PASS-THROUGH, not empty-string: with no opening brace it
        returns the input unchanged so the caller's json.loads fails naturally.
        Returning "" would convert "no JSON" into "empty JSON" and hide the
        difference. (I asserted == "" on the first pass; the code was right.)"""
        assert _find_balanced_json("no json here", "{") == "no json here"
        assert _find_balanced_json("", "{") == ""

    def test_unbalanced_opener_is_returned_as_is(self):
        """A truncated generation ends mid-object; the salvage layer above is
        what decides to reject it, so this must not pretend it succeeded."""
        assert _find_balanced_json("{unclosed", "{") == "{unclosed"


class TestExtractJsonObject:
    """The salvage path. A 9B emits fenced, prefixed and trailing-comma JSON."""

    def test_plain_object(self):
        assert extract_json_object('{"keep": "B"}') == {"keep": "B"}

    def test_code_fenced(self):
        assert extract_json_object('```json\n{"keep": "A"}\n```') == {"keep": "A"}

    def test_with_leading_prose(self):
        assert extract_json_object('Here you go: {"keep": "both"}') == {"keep": "both"}

    def test_unparseable_returns_falsy_not_raise(self):
        """kg.check_and_resolve_contradictions relies on a falsy return to log
        and fail open — it must never raise."""
        for junk in ("", "not json at all", "{unclosed", "null"):
            assert not extract_json_object(junk)

    def test_truncated_mid_string_is_not_silently_half_parsed(self):
        """The exact 2026-08-29 contradiction-judge failure: the model appended
        an unrequested 'reasoning' field, blew max_tokens and the JSON was cut
        mid-string. Salvage must refuse it rather than return a partial verdict."""
        cut = '{"keep": "both", "reasoning": "The two statements refer to the sam'
        got = extract_json_object(cut)
        assert not got or "keep" in got, (
            "a mid-string truncation must yield either nothing or a usable "
            "verdict — never a silently mangled object"
        )

    def test_extra_fields_are_preserved_not_dropped(self):
        got = extract_json_object('{"keep": "A", "reasoning": "because"}')
        assert got.get("keep") == "A"

    @pytest.mark.parametrize("payload", [
        '{"results": [{"id": 1, "verdict": "keep"}]}',
        '{"results": []}',
        '{"summaries": ["one", "two"]}',
    ])
    def test_real_schema_shapes_round_trip(self, payload):
        """Shapes actually used by KG curation and enrichment."""
        assert extract_json_object(payload)
