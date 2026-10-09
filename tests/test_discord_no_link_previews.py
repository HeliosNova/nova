"""Discord feed posts must not unfurl a preview card per link (2026-10-09, the
seventh "feeds show hyperlinks instead of the report").

Domain Studies author links as `<URL>`; in Discord the angle brackets do two
things: hide the URL as link text AND suppress the link preview. The 2026-09-04
fix rewrote them to `[domain ↗](URL)` to shorten the visible text — and lost
the suppression. Measured on the bot's own channel: the FDA digest posted at
17:02 carried THREE unfurled article cards per chunk, six for one digest; the
Product Hunt, Hacker News, SEC and GitHub feeds post 15 links each over three
chunks. A reader sees a stack of preview cards with the prose squeezed between
them — "hyperlinks instead of what it is supposed to report". Domain overviews
have no links and were never affected.

Two layers: the converter keeps the suppression inside the masked link,
`[domain ↗](<URL>)`, and the adapter sends every chunk with
`suppress_embeds=True` so a bare URL the model writes into prose cannot unfurl
either.
"""
from __future__ import annotations

import asyncio
import re

import pytest

from app.channels.format_for_channel import to_discord

ARXIV = "https://arxiv.org/abs/2610.10541"
ROW = (
    "**`1.`** 🧠  **An Explainable Header-Centric Framework**\n"
    "   ↳ **export.arxiv.org**  ·  📅 October 09, 2026  ·  <" + ARXIV + ">\n"
    "   Knowledge Graph quality depends on tabular metadata.\n"
)

UNSUPPRESSED = re.compile(r"\]\((https?://)")          # ](http  — a masked link that will unfurl
BARE_UNSUPPRESSED = re.compile(r"(?<![<(])https?://")  # a URL not wrapped by <> or (<


def test_the_masked_link_keeps_the_preview_suppressed():
    out = to_discord(ROW)
    assert f"[arxiv.org ↗](<{ARXIV}>)" in out, out
    assert not UNSUPPRESSED.search(out), out


def test_no_url_in_a_converted_digest_can_unfurl():
    text = ROW + "   ↳ **sec.gov**  ·  <https://www.sec.gov/Archives/edgar/data/1/x-index.htm>\n"
    out = to_discord(text)
    assert not BARE_UNSUPPRESSED.search(out), out
    assert not UNSUPPRESSED.search(out), out


def test_the_label_is_still_what_the_reader_sees():
    out = to_discord(ROW)
    visible = re.sub(r"\[([^\]]*)\]\(<?(https?://[^)>]+)>?\)", r"\1", out)
    assert "arxiv.org ↗" in visible
    assert ARXIV not in visible


@pytest.mark.asyncio
async def test_the_adapter_sends_every_chunk_with_embeds_suppressed():
    from app.channels.discord import DiscordBot

    calls: list[dict] = []

    class _Chan:
        async def send(self, text, **kw):
            calls.append({"text": text, **kw})

    class _Client:
        def is_ready(self):
            return True

        def get_channel(self, _id):
            return _Chan()

    bot = object.__new__(DiscordBot)
    bot._client = _Client()
    bot.default_channel_id = "123"
    bot._send_lock = asyncio.Lock()

    long = "\n".join(ROW.replace("`1.`", f"`{i}.`") for i in range(1, 25))  # ~4,000 chars: splits at 1960
    assert await bot.send_alert(long) is True
    assert len(calls) >= 2, calls
    assert all(c.get("suppress_embeds") is True for c in calls), calls
    assert all(not UNSUPPRESSED.search(c["text"]) for c in calls), calls
