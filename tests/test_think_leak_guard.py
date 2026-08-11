"""Thinking-leak guards (Science digest incident, 2026-07-08).

Root cause: Ollama 0.30's qwen renderer broke the "<think></think>" assistant-
prefill suppression trick, and an unterminated <think> block (model spent its
whole token budget reasoning) leaked a raw monologue into a POSTED digest.
Three independent layers now prevent it; these pin layers 2 and 3.
"""

from __future__ import annotations

from app.core.llm import _strip_think_tags
from app.monitors.deep_research import _accept_correction


class TestStripThinkTags:
    def test_matched_block_stripped(self):
        assert _strip_think_tags("<think>reasoning</think>\nThe answer.") == "The answer."

    def test_unterminated_block_stripped_to_end(self):
        # The exact failure shape: open tag, real reasoning, no close.
        out = _strip_think_tags("<think>\nHere's a thinking process:\n1. Analyze...")
        assert out.strip() == ""

    def test_unterminated_after_good_content_still_dropped(self):
        # Anything after an unterminated <think> is incomplete reasoning, not content.
        out = _strip_think_tags("Real answer.\n<think>then it started rambling")
        assert "rambling" not in out
        assert "Real answer." in out

    def test_clean_text_untouched(self):
        assert _strip_think_tags("Just a normal answer.") == "Just a normal answer."


class TestAcceptCorrectionRejectsThinkLeak:
    def _briefing(self, n=800):
        return ("## Lead\n**Development:** " + "x " * (n // 2)).strip()

    def test_rejects_open_think_tag(self):
        orig = self._briefing()
        leaked = "<think>\n**Analyze User Input:**\nHere's a thinking process: " + "y " * 400
        assert _accept_correction(orig, leaked) is False

    def test_rejects_thinking_preamble_without_tag(self):
        orig = self._briefing()
        leaked = "**Here's a thinking process:** first I will analyze user input " + "z " * 400
        assert _accept_correction(orig, leaked) is False

    def test_accepts_faithful_edit(self):
        orig = self._briefing()
        # a real correction: same structure, similar length, no reasoning
        fixed = orig.replace("x x", "x")
        assert _accept_correction(orig, fixed) is True

    def test_still_rejects_over_strip(self):
        orig = self._briefing()
        assert _accept_correction(orig, "## Lead\n**Development:** tiny") is False
