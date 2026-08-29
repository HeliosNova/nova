"""The 2026-08-29 "monitors coming back as hyperlinks" fixes.

Diagnosis (from live data, not inference):

1. Enrichment summaries were CONFABULATED, and the entailment gate was
   correctly deleting them — leaving title+link. Live repro on
   inventati.org/who/manifesto: the page fetched fine (3469 chars) and the 9B
   wrote "founded in 2001 by autonomous anticapitalist activists" when neither
   "2001" nor "anticapitalist" occurs anywhere in the body (MiniCheck p=0.013
   at every premise width). Cause: the model was shown only body[:900] — often
   masthead/nav — while the prompt demanded concrete dates/quantities and the
   result had to clear a >=60-char floor. So the fix widens the window, forbids
   outside facts, and retries dropped items EXTRACTIVELY rather than relaxing
   the gate.

2. The chat gate posted 12 pairs x the whole evidence blob in one 60s request.
   Measured against the live sidecar that shape takes 132s, so 28 of 37 runs in
   48h timed out and fail-opened (~76% of tool-backed answers ungrounded). Now:
   per-claim best window, chunked posts, per-claim fail-open.
"""

from __future__ import annotations

import pytest

from app.core.brain import (
    _CHAT_ENTAIL_MAX_CLAIMS,
    _CHAT_ENTAIL_WINDOW,
    _best_evidence_window,
)
from app.monitors.domain_study_runner import _ENRICH_BODY_CHARS, _enrich_num_ctx


class TestBestEvidenceWindow:
    """The chat gate's premise selector."""

    def test_short_doc_returned_whole(self):
        doc = "The Bank of Japan raised its policy rate to 1.00%."
        assert _best_evidence_window(doc, "BoJ raised rates") == doc

    def test_selects_window_containing_the_claim_tokens(self):
        # The needle sits far past any fixed truncation point.
        filler = "Unrelated market commentary about equities. " * 120
        needle = "The Bank of Japan raised its policy rate to 1.00% in July."
        doc = filler + needle + filler
        got = _best_evidence_window(doc, "Bank of Japan policy rate 1.00%")
        assert "1.00%" in got, "selector missed the only window with the evidence"
        assert len(got) <= _CHAT_ENTAIL_WINDOW

    def test_window_is_bounded(self):
        doc = "x" * 50_000
        assert len(_best_evidence_window(doc, "anything")) <= _CHAT_ENTAIL_WINDOW

    def test_no_usable_tokens_falls_back_to_head(self):
        doc = "abc " * 2000
        got = _best_evidence_window(doc, "!!! ??")
        assert got == doc[:_CHAT_ENTAIL_WINDOW]

    def test_claim_cap_reduced_from_twelve(self):
        # 12 pairs measured at 132s against a 60s budget; 8 is the new cap.
        assert _CHAT_ENTAIL_MAX_CLAIMS < 12


class TestEnrichNumCtx:
    """num_ctx must follow the prompt, or the wider window silently truncates."""

    def test_small_prompt_keeps_floor(self):
        assert _enrich_num_ctx("short prompt", 512) == 8192

    def test_grows_past_8192_for_a_full_batch(self):
        # 15 items x 1800 chars is a real Hacker News batch; at a fixed 8192
        # this prompt would be cut and whole item bodies would vanish — which
        # is precisely what makes the model invent their summaries.
        prompt = "x" * (15 * _ENRICH_BODY_CHARS)
        assert _enrich_num_ctx(prompt, 1430) > 8192

    def test_is_clamped(self):
        assert _enrich_num_ctx("x" * 5_000_000, 4096) <= 32768

    @pytest.mark.parametrize("n_items", [1, 4, 8, 15])
    def test_never_below_what_the_prompt_needs(self, n_items):
        prompt = "x" * (n_items * _ENRICH_BODY_CHARS)
        max_tokens = 90 * n_items + 80
        assert _enrich_num_ctx(prompt, max_tokens) >= len(prompt) // 3 + max_tokens

    def test_window_widened_from_900(self):
        assert _ENRICH_BODY_CHARS > 900


class TestExtractiveRetryContract:
    """The retry must re-gate, never bypass."""

    @pytest.mark.asyncio
    async def test_empty_input_is_a_noop(self):
        from app.monitors.domain_study_runner import _extractive_retry
        assert await _extractive_retry("Some Monitor", "news", []) == []
