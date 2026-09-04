"""Discord got the raw URL as its own link text (owner report, 2026-09-04).

Domain Studies are AUTHORED in Discord markdown, so Discord was the one channel
with no converter of its own: Telegram, WhatsApp and Signal each translate the
canonical text, and Discord shipped it verbatim. Discord renders a bare
`<https://...>` as a link whose visible text is the entire URL. That is fine
under a paragraph and ruinous under a feed row: an EDGAR filing URL is 96
characters and wraps to three blue lines above a one-line body.

Measured on the stored digests the owner was reading, the share of visible
message that was raw URL:

    SEC Insider Trading          32%
    FDA Drug Approvals           22%
    Product Hunt Trending        13%
    Hacker News Top Stories      11%
    GitHub Security Advisories    8%
    Research Frontiers            5%   (arxiv links are short)

which is the order they complained in. Telegram had exactly this defect and was
fixed on 2026-08-31; the fix was never carried across.
"""
from __future__ import annotations

import re

import pytest

from app.channels.format_for_channel import to_discord

EDGAR = ("https://www.sec.gov/Archives/edgar/data/1674910/000192461326000041/"
         "0001924613-26-000041-index.htm")
SEC_ROW = (
    "**`1.`** \U0001f575️  **VALVOLINE INC — insider: Flees Lori Ann (Form 4)**\n"
    "   ↳ **sec.gov**  ·  September 04, 2026  ·  <" + EDGAR + ">\n"
    "   ⚪ grant/award\n"
)


def _visible(text: str) -> str:
    """What a reader actually sees: a masked link shows only its label."""
    text = re.sub(r"\[([^\]]*)\]\((https?://[^)]+)\)", r"\1", text)
    return re.sub(r"<(https?://[^\s>]+)>", r"\1", text)


def test_a_long_url_is_labelled_by_its_domain():
    out = to_discord(SEC_ROW)
    assert f"[sec.gov ↗]({EDGAR})" in out
    assert "<" + EDGAR + ">" not in out


def test_the_reader_sees_a_third_less_of_a_filing_row():
    before, after = len(_visible(SEC_ROW)), len(_visible(to_discord(SEC_ROW)))
    assert after < before
    assert (before - after) / before > 0.3, "the URL was most of what was on screen"


def test_the_link_still_points_at_the_full_url():
    """A shorter label must not become a shorter link."""
    out = to_discord(SEC_ROW)
    assert EDGAR in out


def test_a_url_containing_parentheses_is_left_suppressed():
    """Discord ends a masked link at the first ')' and spills the tail as text,
    which is worse than a long label. Those keep the <> form."""
    row = "see <https://en.wikipedia.org/wiki/Foo_(bar)> for detail"
    assert to_discord(row) == row


def test_links_that_are_already_labelled_are_untouched():
    row = "**CVE:** corresponds to [CVE-2026-72800](https://nvd.nist.gov/vuln/detail/CVE-2026-72800)."
    assert to_discord(row) == row


def test_nothing_else_about_the_message_changes():
    row = ("## \U0001f4f0 **HACKER NEWS**\n\n"
           "\U0001f4a1 _Themes include AI model performance._\n"
           "**`1.`** **Carbon-aware electricity pricing**\n")
    assert to_discord(row) == row


def test_an_empty_message_survives():
    assert to_discord("") == ""
    assert to_discord(None) is None


@pytest.mark.asyncio
async def test_the_discord_adapter_actually_applies_it():
    """The converter is only worth anything if the send path calls it."""
    from app.channels.discord import DiscordBot

    sent: list[str] = []

    class _Chan:
        async def send(self, text):
            sent.append(text)

    class _Client:
        def is_ready(self):
            return True

        def get_channel(self, _id):
            return _Chan()

    bot = object.__new__(DiscordBot)
    bot._client = _Client()
    bot.default_channel_id = "123"
    import asyncio
    bot._send_lock = asyncio.Lock()

    assert await bot.send_alert(SEC_ROW) is True
    assert sent and "[sec.gov ↗]" in sent[0]
    assert "<" + EDGAR + ">" not in sent[0]
