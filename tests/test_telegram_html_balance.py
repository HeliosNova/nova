"""Telegram HTML tag-balance repair (2026-08-26).

Telegram rejects a whole message on any tag imbalance ("Can't parse
entities: unmatched end tag"), and the plain-text fallback then strips all
formatting (~4x/48h live). Imbalance came from _split_message cutting across
an open tag and from model-emitted stray closers.
"""
from app.channels.telegram import TelegramBot


def _balance(chunks):
    return TelegramBot._balance_html_chunks(chunks)


class TestBalanceHtmlChunks:
    def test_balanced_chunk_untouched(self):
        assert _balance(["<b>hello</b> world"]) == ["<b>hello</b> world"]

    def test_split_across_bold_carries_tag(self):
        out = _balance(["<b>part one", "part two</b> done"])
        assert out[0] == "<b>part one</b>"
        assert out[1] == "<b>part two</b> done"

    def test_orphan_closer_stripped(self):
        out = _balance(["no opener</b> here"])
        assert out == ["no opener here"]

    def test_unclosed_tag_closed_at_end(self):
        out = _balance(["<i>drifting off"])
        assert out == ["<i>drifting off</i>"]

    def test_improper_nesting_fixed(self):
        # <b><i></b></i> → inner i closed before b, then reopened/closed
        out = _balance(["<b>x<i>y</b>z</i>"])
        assert out == ["<b>x<i>y</i></b><i>z</i>"]

    def test_anchor_href_preserved_across_split(self):
        out = _balance(['<a href="https://x.example/p">link text', "more</a> tail"])
        assert out[0] == '<a href="https://x.example/p">link text</a>'
        assert out[1] == '<a href="https://x.example/p">more</a> tail'

    def test_multi_chunk_stack_carry(self):
        out = _balance(["<b>a<i>b", "c", "d</i>e</b>"])
        assert out[0] == "<b>a<i>b</i></b>"
        assert out[1] == "<b><i>c</i></b>"
        assert out[2] == "<b><i>d</i>e</b>"

    def test_plain_text_untouched(self):
        chunks = ["plain text", "1 < 2 but not a tag"]
        assert _balance(chunks) == chunks
