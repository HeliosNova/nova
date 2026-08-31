"""Findings cut by max_tokens must not leak dangling fragments into evidence.

Live tripwire, 2026-08-31 04:40:27 (seconds after a digest send):
    [truncation] invoke_nothink hit max_tokens (512) — output cut
    mid-generation (model=nova-ft, 2339 chars, eval=512, prompt_eval=2653)

The caller is deep_research._findings — the per-article evidence extractor.
Its cap history is 240 -> 320 -> 512 across three prior fixes and the tail
never fully dies: raising the cap shrinks truncations (~20/day at 320 to
~1/6h at 512) but a dense article can always overrun. A mid-sentence cut puts
a dangling fragment INTO THE EVIDENCE POOL, where synthesis may quote it.

Fix under test: deterministic repair (_trim_dangling_tail) instead of a
fourth cap raise — drop only a final line that plainly lacks a sentence
boundary; complete bullets, numeric endings and short texts survive.
"""

from __future__ import annotations

import pytest

from app.monitors.deep_research import _trim_dangling_tail


class TestTrimsRealTruncations:
    def test_cut_third_bullet_is_dropped(self):
        txt = ("- Nvidia shipped 40k GPUs in Q2.\n"
               "- Revenue rose 12% year over year.\n"
               "- The company also announced that it wo")
        out = _trim_dangling_tail(txt)
        assert out.endswith("year over year.")
        assert "announced that it wo" not in out

    def test_single_paragraph_cut_midword(self):
        txt = ("TSMC confirmed the fab expansion. Construction begins in "
               "October and the projected capacity increase would allo")
        out = _trim_dangling_tail(txt)
        assert out.endswith("begins in October and the projected capacity increase would allo") is False
        assert out.endswith("Construction begins in") is False  # not over-trimmed
        # the complete first sentence must survive
        assert "TSMC confirmed the fab expansion." in out


class TestLeavesCompleteTextAlone:
    @pytest.mark.parametrize("txt", [
        "- Complete finding one.\n- Complete finding two.",
        "The chip performs at 4.2 GHz under load.",
        "- TSMC yield reached 78%",           # numeric ending = complete thought
        "- Funding round closed at $1.2B)",    # citation-paren ending
        "Was it approved? Yes.",
        'He said "we are done."',
    ])
    def test_untouched(self, txt):
        assert _trim_dangling_tail(txt) == txt

    def test_never_guts_short_text(self):
        """If trimming would leave <40 chars, keep the original — the caller's
        own length gate decides its fate, not the trimmer."""
        txt = "Short fragment that would gut the text and"
        assert _trim_dangling_tail(txt) == txt

    def test_empty_and_none_safe(self):
        assert _trim_dangling_tail("") == ""
