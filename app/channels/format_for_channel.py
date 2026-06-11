"""Per-channel markdown converters.

Domain Studies are formatted in Discord-style markdown (**bold**, *italic*,
`code`, <URL>). Each channel has its own dialect — Telegram uses HTML or
MarkdownV2, WhatsApp uses *single-asterisk* + _underscore_, Signal supports
limited markdown. This module converts the canonical Discord output to the
right target syntax so Domain Study posts look right everywhere, not just
on Discord.
"""

from __future__ import annotations

import re
from html import escape as html_escape


# ---------------------------------------------------------------------------
# Shared regexes
# ---------------------------------------------------------------------------

_DISCORD_BOLD = re.compile(r"\*\*(.+?)\*\*", re.DOTALL)
_DISCORD_ITALIC = re.compile(r"(?<!\*)\*(?!\*)([^*\n]+?)\*(?!\*)")
_DISCORD_CODE_INLINE = re.compile(r"`([^`\n]+?)`")
_DISCORD_URL_BRACKETED = re.compile(r"<(https?://[^\s>]+)>")
_DISCORD_URL_SQ_BRACKETED = re.compile(r"\[(https?://[^\s\]]+)\]")
_DISCORD_BARE_URL = re.compile(r"(?<![(\"\[<])(https?://[^\s\)>\]\"'<]+)")
_DISCORD_DIVIDER = re.compile(r"^[━─=]{6,}\s*$", re.MULTILINE)
_HEADING_HASH = re.compile(r"^(#{1,3})\s+(.+)$", re.MULTILINE)


# ---------------------------------------------------------------------------
# Telegram (HTML mode — caller must pass parse_mode="HTML")
# ---------------------------------------------------------------------------

def to_telegram_html(text: str) -> str:
    """Convert Discord-flavored markdown to Telegram HTML.

    Order of operations matters because Telegram HTML is strict:
      1. Stash code blocks (placeholders) so their content is not touched
      2. Stash URLs as ready-to-render <a> tags so they survive html_escape
      3. html_escape the rest of the body
      4. Apply bold / italic / divider / heading
      5. Restore placeholders
      6. Collapse any nested <b><b>…</b></b> that result from `## **Title**`
    """
    if not text:
        return text

    placeholders: dict[str, str] = {}

    def _stash(html: str, prefix: str) -> str:
        token = f"\x00{prefix}{len(placeholders)}\x00"
        placeholders[token] = html
        return token

    # 1a. Fenced code blocks ```lang\ncontent```
    text = re.sub(
        r"```(?:\w*)\n(.*?)\n```",
        lambda m: _stash(f"<pre>{html_escape(m.group(1))}</pre>", "FENCED"),
        text,
        flags=re.DOTALL,
    )
    # 1b. Inline code `code`
    text = _DISCORD_CODE_INLINE.sub(
        lambda m: _stash(f"<code>{html_escape(m.group(1))}</code>", "CODE"),
        text,
    )

    # 2. URLs — pull them out BEFORE html_escape so the angle/square brackets
    # around them are removed instead of escaped to &lt;/&gt;.
    text = _DISCORD_URL_BRACKETED.sub(
        lambda m: _stash(f'<a href="{html_escape(m.group(1), quote=True)}">{html_escape(m.group(1))}</a>', "URL"),
        text,
    )
    text = _DISCORD_URL_SQ_BRACKETED.sub(
        lambda m: _stash(f'<a href="{html_escape(m.group(1), quote=True)}">{html_escape(m.group(1))}</a>', "URL"),
        text,
    )
    text = _DISCORD_BARE_URL.sub(
        lambda m: _stash(f'<a href="{html_escape(m.group(1), quote=True)}">{html_escape(m.group(1))}</a>', "URL"),
        text,
    )

    # 3. Escape the rest. Placeholders are pure ASCII so they pass through
    # unchanged.
    text = html_escape(text)

    # 4. Inline tags + dividers + headings
    text = _DISCORD_BOLD.sub(r"<b>\1</b>", text)
    text = _DISCORD_ITALIC.sub(r"<i>\1</i>", text)
    text = _DISCORD_DIVIDER.sub("━━━━━━━━━━━━━━━━━━━━━━━━━━", text)
    text = _HEADING_HASH.sub(lambda m: f"<b>{m.group(2)}</b>", text)

    # 5. Restore placeholders
    for token, original in placeholders.items():
        text = text.replace(token, original)

    # 6. Collapse nested <b><b>…</b></b> (happens when a heading body itself
    # had **bold**). Telegram's parser tolerates it but the second tag is
    # redundant and visually noisy.
    text = re.sub(r"<b>(\s*)<b>(.+?)</b>(\s*)</b>", r"<b>\1\2\3</b>", text, flags=re.DOTALL)

    return text


# ---------------------------------------------------------------------------
# WhatsApp (uses *bold*, _italic_, ~strike~, no links)
# ---------------------------------------------------------------------------

def to_whatsapp(text: str) -> str:
    """Convert Discord markdown to WhatsApp's variant."""
    if not text:
        return text
    text = re.sub(r"\*\*(.+?)\*\*", r"*\1*", text, flags=re.DOTALL)
    text = _DISCORD_URL_BRACKETED.sub(r"\1", text)
    text = _DISCORD_URL_SQ_BRACKETED.sub(r"\1", text)
    text = _HEADING_HASH.sub(lambda m: f"*{m.group(2)}*", text)
    return text


# ---------------------------------------------------------------------------
# Signal (limited markdown — strip formatting, preserve URLs)
# ---------------------------------------------------------------------------

def to_signal(text: str) -> str:
    """Strip Discord markdown for Signal — Signal doesn't reliably render it."""
    if not text:
        return text
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text, flags=re.DOTALL)
    text = _DISCORD_ITALIC.sub(r"\1", text)
    text = _DISCORD_CODE_INLINE.sub(r"\1", text)
    text = _DISCORD_URL_BRACKETED.sub(r"\1", text)
    text = _DISCORD_URL_SQ_BRACKETED.sub(r"\1", text)
    text = _HEADING_HASH.sub(lambda m: m.group(2).upper(), text)
    return text
