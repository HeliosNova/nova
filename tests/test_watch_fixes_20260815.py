"""Regression tests for the 2026-08-15 overnight-watch fixes.

The watch caught a cross-monitor synthesis preview shipping a mid-word cut
('…Leinweber Foundat…'). _preview must trim on a word boundary.
"""

from __future__ import annotations


class TestCrossMonitorPreview:
    def test_short_text_unchanged(self):
        from app.core.cross_monitor import _preview
        assert _preview("a short synthesis", 280) == "a short synthesis"

    def test_long_text_cuts_on_word_boundary(self):
        from app.core.cross_monitor import _preview
        text = ("This trend manifests through massive capital realignments like the "
                "Leinweber Foundation and other structural shifts " * 5)
        out = _preview(text, 280)
        assert out.endswith("…")
        body = out[:-1]
        # exact prefix of the source (no mangling) …
        assert text.startswith(body)
        # … and the cut landed at a word boundary, never mid-word
        nxt = text[len(body):len(body) + 1]
        assert nxt == "" or not nxt.isalnum(), f"cut mid-word before {nxt!r}"

    def test_the_leinweber_case(self):
        from app.core.cross_monitor import _preview
        # the exact 2026-08-15 shape: 'Leinweber Foundat...' must not survive
        text = "realignments like the Leinweber Foundation reshaping the sector " + "z" * 300
        out = _preview(text, 40)
        assert "Foundat…" not in out
        assert not out[:-1].rstrip().endswith(("Foundat", "Founda", "Foundati"))
