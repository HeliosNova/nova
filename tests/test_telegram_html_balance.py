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


class TestCompactAnchors:
    """Owner-reported 'bare links' on Telegram (2026-09-01): rich digests
    (GitHub advisories 15/15 substantive) still READ as link lists because
    every anchor's visible text was the full URL — 2-3 wrapped blue lines per
    item on a phone. Visible text is now the domain; the URL stays in href."""

    def test_bracketed_url_shows_domain_only(self):
        from app.channels.format_for_channel import to_telegram_html
        line = ("↳ **github.com**  ·  📅 August 31, 2026  ·  "
                "<https://github.com/advisories/GHSA-fm3f-ch8h-qw8q>")
        out = to_telegram_html(line)
        assert '<a href="https://github.com/advisories/GHSA-fm3f-ch8h-qw8q">' in out
        assert ">github.com ↗</a>" in out
        # the long path must not be user-visible outside the href attribute
        assert out.count("GHSA-fm3f-ch8h-qw8q") == 1

    def test_bare_url_in_prose_compacted(self):
        from app.channels.format_for_channel import to_telegram_html
        out = to_telegram_html(
            "Filed at https://www.sec.gov/Archives/edgar/data/760498/000076049826000055/x-index.htm today")
        assert ">sec.gov ↗</a>" in out  # www. stripped
        assert out.count("edgar/data") == 1  # only inside href

    def test_unparseable_url_falls_back_to_link_label(self):
        from app.channels.format_for_channel import _compact_anchor
        a = _compact_anchor("https://")
        assert ">link ↗</a>" in a

    def test_fullline_underscore_italics_converted(self):
        from app.channels.format_for_channel import to_telegram_html
        out = to_telegram_html(
            "_read 28 sources: abc.net.au, apnews.com · 0 facts learned_")
        assert out.startswith("<i>read 28 sources")
        assert out.endswith("</i>")

    def test_emoji_prefixed_summary_line_converted(self):
        from app.channels.format_for_channel import to_telegram_html
        out = to_telegram_html("💡 _Multiple MariaDB connectors leak credentials_")
        assert "<i>Multiple MariaDB connectors leak credentials</i>" in out

    def test_inline_snake_case_never_italicized(self):
        from app.channels.format_for_channel import to_telegram_html
        out = to_telegram_html("set check_config and anchor_hour before quiz_engine runs")
        assert "<i>" not in out
        assert "check_config" in out
