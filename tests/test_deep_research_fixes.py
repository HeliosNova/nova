"""Deep-research regressions from the 2026-08-24 A-Z audit.

1. `_findings` sent (now-12k-char) article bodies through invoke_nothink
   with NO num_ctx — Ollama silently truncated at the 4096 default
   (log-proven: prompt_eval 3914 + eval 182 = 4096 exactly) and dangling
   fragments re-entered the evidence pool. ~95 truncations/day.
2. `_weights` tokenized with [a-z0-9]{4,}, so short numbers ("68.5",
   "$44") carried ZERO weight in entail-evidence window selection —
   number-bearing claims missed their window and dropped (~51% of
   checked cited sentences per day).
"""

from __future__ import annotations

import asyncio

import pytest


class TestFindingsNumCtx:
    def test_findings_passes_explicit_num_ctx(self, monkeypatch):
        from app.monitors import deep_research

        captured: list[dict] = []

        async def _fake_invoke(messages, **kwargs):
            captured.append(kwargs)
            return "Finding: something concrete."

        monkeypatch.setattr(deep_research, "_invoke_bg", _fake_invoke)

        body = "word " * 2400  # ~12k chars — the post-08-22 body size
        arts = [("Title", "https://example.com/a", body)]
        asyncio.run(deep_research._findings(arts, "test subject"))

        assert captured, "findings did not invoke the LLM"
        num_ctx = captured[0].get("num_ctx")
        assert num_ctx is not None and num_ctx >= 8192, (
            f"_findings must size num_ctx for 12k-char bodies, got {num_ctx!r}"
        )


class TestClaimTokens:
    def test_short_numbers_participate_in_selection(self):
        from app.monitors.deep_research import _claim_tokens

        toks = _claim_tokens("revenue surging 68.5% to $44 million")
        assert "68.5" in toks
        assert "44" in toks
        assert "revenue" in toks and "million" in toks

    def test_single_digits_stay_excluded(self):
        from app.monitors.deep_research import _claim_tokens

        toks = _claim_tokens("a rise of 7% among 5 vendors")
        assert "7" not in toks
        assert "5" not in toks

    def test_long_numbers_still_present(self):
        from app.monitors.deep_research import _claim_tokens

        assert "246971" in _claim_tokens("the total reached 246971 units")
