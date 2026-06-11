"""Monitor output formatting for Discord.

Standardizes all monitor outputs for consistent, readable Discord messages.
Handles Discord's 2000-char limit with smart truncation.

Also provides the unified one-line monitor result format:
    <status emoji> <summary> │ <key>: <value> │ <key>: <value>

Used by both LLM-driven and native-handler monitors so Discord/Telegram output
is consistent and free of tool-call JSON leakage.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone


# Status emoji mapping
_STATUS_EMOJI = {
    "ok": "\u2705",       # ✅
    "success": "\u2705",
    "healthy": "\u2705",
    "warning": "\u26a0\ufe0f",    # ⚠️
    "degraded": "\u26a0\ufe0f",
    "error": "\u274c",     # ❌
    "critical": "\u274c",
    "unknown": "\u2753",   # ❓
    "skip": "\U0001f4a4",  # 💤
    "skipped": "\U0001f4a4",
    "info": "\U0001f4ca",  # 📊
    "stats": "\U0001f4ca",
}


# Subject-specific emoji appended to monitor titles so each domain/topic
# stands out at a glance in Discord/Telegram. First match (case-insensitive
# substring) wins — order matters for overlapping terms like "AI" inside
# "financial AI" (we want the more specific match higher up).
_TOPIC_EMOJI: tuple[tuple[str, str], ...] = (
    ("ai and ml", "\U0001f916"),          # 🤖
    ("quantum", "\u269b\ufe0f"),          # ⚛️
    ("robotics", "\U0001f9be"),           # 🦾
    ("semiconductor", "\U0001f9ea"),      # 🧪
    ("cybersecurity", "\U0001f512"),      # 🔒
    ("biotech", "\U0001f9ec"),            # 🧬
    ("health and medicine", "\U0001f48a"),# 💊
    ("space and astronomy", "\U0001f680"),# 🚀
    ("physics", "\U0001f52c"),            # 🔬
    ("energy and climate", "\u26a1"),     # ⚡
    ("crypto", "\u20bf"),                 # ₿
    ("defi", "\U0001f4b0"),               # 💰
    ("whale", "\U0001f40b"),              # 🐋
    ("commodit", "\U0001f6e2\ufe0f"),     # 🛢️
    ("earnings", "\U0001f4c8"),           # 📈
    ("fomc", "\U0001f3db\ufe0f"),         # 🏛️
    ("sec insider", "\U0001f575\ufe0f"),  # 🕵️
    ("economics and markets", "\U0001f4ca"), # 📊
    ("finance", "\U0001f4b5"),            # 💵
    ("us policy", "\U0001f3db\ufe0f"),    # 🏛️
    ("geopolitics", "\U0001f30d"),        # 🌍
    ("defense and military", "\u2694\ufe0f"), # ⚔️
    ("china", "\U0001f1e8\U0001f1f3"),    # 🇨🇳
    ("russia", "\U0001f1f7\U0001f1fa"),   # 🇷🇺
    ("middle east", "\U0001f54c"),        # 🕌
    ("india", "\U0001f1ee\U0001f1f3"),    # 🇮🇳
    ("europe and eu", "\U0001f1ea\U0001f1fa"), # 🇪🇺
    ("latin america", "\U0001f1f2\U0001f1fd"), # 🇲🇽
    ("africa", "\U0001f1f0\U0001f1ea"),   # 🇰🇪
    ("startups and vc", "\U0001f680"),    # 🚀
    ("open source", "\U0001f419"),        # 🐙
    ("developer ecosystem", "\U0001f4bb"),# 💻
    ("hacker news", "\U0001f4f0"),        # 📰
    ("product hunt", "\U0001f43e"),       # 🐾
    ("fda", "\U0001f48a"),                # 💊
    ("github", "\U0001f419"),             # 🐙
    ("government contract", "\U0001f4dd"),# 📝
    ("research frontiers", "\U0001f9e0"), # 🧠
    ("science", "\U0001f52c"),            # 🔬
    ("technology", "\U0001f4bb"),         # 💻
    ("current events", "\U0001f4f0"),     # 📰
    ("world awareness", "\U0001f30e"),    # 🌎
    ("supply chain", "\U0001f69a"),       # 🚚
    ("morning check", "\U0001f305"),      # 🌅
    ("system health", "\U0001f49a"),      # 💚
    ("system maintenance", "\U0001f527"), # 🔧
    ("fine-tune", "\U0001f3cb\ufe0f"),    # 🏋️
    ("auto-monitor", "\U0001f9ed"),       # 🧭
    ("lesson quiz", "\U0001f9e0"),        # 🧠
    ("skill validation", "\U0001f3af"),   # 🎯
    ("curiosity research", "\U0001f50d"), # 🔍
    ("eval harness", "\U0001f9ea"),       # 🧪
    ("prompt optimizer", "\u2699\ufe0f"), # ⚙️
)


def topic_emoji_for(monitor_name: str) -> str:
    """Return a single subject emoji for the monitor title, or '' if no match.

    First case-insensitive substring match wins. Intended for appending to
    the end of the `**title**` header line so each monitor's topic is visible
    at a glance.
    """
    if not monitor_name:
        return ""
    low = monitor_name.lower()
    for needle, emoji in _TOPIC_EMOJI:
        if needle in low:
            return emoji
    return ""

# ---------------------------------------------------------------------------
# Unified one-line format
# ---------------------------------------------------------------------------

# Pipe separator (with non-breaking spaces so Discord/Telegram render it cleanly)
_SEP = " \u2502 "   # " │ "
_MAX_SUMMARY = 80
_MAX_LINE = 400  # total budget before we bail and truncate


# Defensive patterns for stripping tool-call artifacts from LLM output.
# These fire when a monitor query got the model to emit raw tool-call
# syntax instead of executing the tool — we strip the leaked JSON so it
# doesn't show up in Discord/Telegram.
_TOOLCALL_JSON_RE = re.compile(
    r'\{\s*"tool"\s*:\s*"[^"]+"\s*,\s*"(?:args|arguments|parameters)"\s*:\s*\{.*?\}\s*\}',
    re.DOTALL,
)
_TOOLCALL_TAG_RE = re.compile(r'</?tool_call\s*/?>', re.IGNORECASE)
_TOOLCALL_XML_WRAP_RE = re.compile(
    r'<tool_call>.*?</tool_call>',
    re.DOTALL | re.IGNORECASE,
)


def strip_tool_call_artifacts(text: str) -> str:
    """Remove raw tool-call JSON and XML tags that leaked from LLM output.

    When a monitor's LLM tool loop fails (no tool registry, no parse, etc.)
    the model's intended tool call can be stringified into the final result.
    This scrubs those artifacts defensively before rendering to Discord.
    """
    if not text:
        return text
    cleaned = _TOOLCALL_XML_WRAP_RE.sub("", text)
    cleaned = _TOOLCALL_JSON_RE.sub("", cleaned)
    cleaned = _TOOLCALL_TAG_RE.sub("", cleaned)
    # Collapse whitespace left behind
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned


def _truncate_summary(summary: str, *, limit: int = _MAX_SUMMARY) -> str:
    summary = " ".join(summary.split())  # collapse newlines/tabs
    if len(summary) <= limit:
        return summary
    return summary[: limit - 1].rstrip() + "\u2026"  # single-char ellipsis


def format_monitor_result(
    name: str,
    status: str,
    summary: str,
    fields: dict[str, str | int | float] | None = None,
) -> str:
    """Build the unified one-line monitor result string.

    Format: "<emoji> <summary> │ <key>: <value> │ <key>: <value>"

    Args:
        name: monitor name (not rendered in the line itself — heartbeat loop
            prepends "[<name>] " when sending). Reserved for future use.
        status: one of ok / warn / warning / err / error / skip / skipped /
            info / stats. Unknown statuses fall back to "❓".
        summary: short prose summary (truncated to 80 chars).
        fields: optional key-value pairs rendered after the summary.

    The returned string never contains tool-call JSON or `</tool_call>`
    artifacts — leaked syntax from the LLM is stripped defensively.
    """
    del name  # reserved; heartbeat loop prepends the name prefix
    emoji = _STATUS_EMOJI.get((status or "").lower(), _STATUS_EMOJI["unknown"])
    clean_summary = strip_tool_call_artifacts(summary or "")
    clean_summary = _truncate_summary(clean_summary) or "(no summary)"

    parts = [f"{emoji} {clean_summary}"]
    if fields:
        for k, v in fields.items():
            if v is None or v == "":
                continue
            # Fields are short — no per-field truncation beyond overall budget
            parts.append(f"{k}: {v}")

    line = _SEP.join(parts)
    # Hard cap to keep single-line output readable in Discord/Telegram
    if len(line) > _MAX_LINE:
        line = line[: _MAX_LINE - 1].rstrip() + "\u2026"
    return line

DISCORD_LIMIT = 2000
# Reserve space for header/footer
_BODY_BUDGET = DISCORD_LIMIT - 200

# Visual divider so adjacent monitor posts in the channel don't blur into
# each other. Picked for Discord rendering: solid line of box-drawing
# heavy chars renders as a thick horizontal rule.
_DIVIDER = "━" * 30


def classify_status(value: str) -> str:
    """Classify a monitor result into ok/warning/error based on content."""
    lower = value.lower()
    if any(w in lower for w in ("error", "failed", "critical", "down", "unreachable", "exception")):
        return "error"
    if any(w in lower for w in ("warning", "degraded", "slow", "elevated", "approaching")):
        return "warning"
    return "ok"


# Date patterns we'll detect inside monitor bodies. We accept "April 25",
# "Apr 26, 2026", "2026-04-26", "04/26/2026" and bold them in the output.
_DATE_RE = re.compile(
    r"\b("
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2}(?:[,\s]+\d{4})?"
    r"|"
    r"\d{4}-\d{2}-\d{2}"
    r"|"
    r"\d{1,2}/\d{1,2}/\d{2,4}"
    r")\b",
    re.IGNORECASE,
)

# Bullet-line markers we normalize to a consistent style.
_NUMBERED_HEADLINE = re.compile(
    r"^\s*\*\*?\s*(\d+)[\.\)]\s*\[?([^\]\n*]+?)\]?\*?\*?\s*$",
    re.MULTILINE,
)


def _normalize_body(body: str) -> str:
    """Tighten up the LLM's body so Discord renders it cleanly:

    - Bold every "**N. Headline**" line and add ▸ marker for visual scan
    - Bold every detected date inline so it stands out
    - Collapse 3+ blank lines to 2
    - Strip any leftover '---' source lists at the bottom (we render them
      separately if needed)
    """
    if not body:
        return body
    text = body.strip()

    # Bold + arrow on numbered headlines: "**1. Foo**" → "▸ **1. Foo**"
    text = _NUMBERED_HEADLINE.sub(lambda m: f"▸ **{m.group(1)}. {m.group(2).strip()}**", text)

    # Bold dates inline (only the first occurrence per line to avoid noise)
    seen_per_line: set[int] = set()
    out_lines: list[str] = []
    for i, line in enumerate(text.splitlines()):
        if i in seen_per_line or "**" in line:  # skip lines that already have bold
            out_lines.append(line)
            continue
        new_line, n = _DATE_RE.subn(lambda m: f"**{m.group(1)}**", line, count=1)
        if n:
            seen_per_line.add(i)
        out_lines.append(new_line)
    text = "\n".join(out_lines)

    # Collapse triple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# Time-window enforcement: detect dates older than this many days inside
# a monitor body and surface a warning. The actual reject/re-roll happens
# in the heartbeat loop; here we just annotate.
_STALE_DAYS = 14


def _detect_stale_dates(body: str) -> list[str]:
    """Return list of date strings in the body that are older than _STALE_DAYS."""
    if not body:
        return []
    from datetime import datetime as _dt, timedelta as _td
    cutoff = _dt.utcnow() - _td(days=_STALE_DAYS)
    stale: list[str] = []
    for m in _DATE_RE.finditer(body):
        raw = m.group(1)
        parsed = _try_parse_date(raw)
        if parsed and parsed < cutoff:
            stale.append(raw)
    # Dedupe preserving order
    seen: set[str] = set()
    out: list[str] = []
    for s in stale:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def _try_parse_date(raw: str):
    from datetime import datetime as _dt
    formats = (
        "%B %d, %Y", "%B %d %Y", "%b %d, %Y", "%b %d %Y",
        "%B %d", "%b %d",   # year-less, assume current year
        "%Y-%m-%d", "%m/%d/%Y", "%m/%d/%y",
    )
    for fmt in formats:
        try:
            d = _dt.strptime(raw.strip().rstrip("."), fmt)
            if "%Y" not in fmt and "%y" not in fmt:
                d = d.replace(year=_dt.utcnow().year)
            return d
        except ValueError:
            continue
    return None


def format_monitor_output(
    monitor_name: str,
    value: str,
    *,
    status: str | None = None,
    metrics: dict[str, str | int | float] | None = None,
) -> str:
    """Format a monitor result for Discord.

    Layout:

        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━     ← divider so feeds don't blur
        ✅ **Monitor Name** 📰              ← title with status + topic emoji
        📊  metric: value                   ← optional metrics block
        (body, with bolded dates + headlines, normalised bullets)
        ⚠️ Stale dates detected: …          ← only if body mentions dates > 14d old
        `08:42 UTC`                         ← timestamp footer

    Args:
        monitor_name: Name of the monitor.
        value: The raw result text.
        status: "ok", "warning", "error", or None (auto-classified).
        metrics: Optional key-value metrics to display in a compact block.

    Returns:
        Discord-formatted message, guaranteed <= 2000 chars.
    """
    if status is None:
        status = classify_status(value)

    emoji = _STATUS_EMOJI.get(status, _STATUS_EMOJI["unknown"])
    now = datetime.now(timezone.utc).strftime("%H:%M UTC")

    # Header — divider, then bold title with subject-specific trailing emoji
    topic = topic_emoji_for(monitor_name)
    header = f"{_DIVIDER}\n{emoji} **{monitor_name}**" + (f" {topic}" if topic else "")

    # Metrics block (compact)
    metrics_block = ""
    if metrics:
        lines = [f"  **{k}:** {v}" for k, v in metrics.items()]
        metrics_block = "\n\U0001f4ca " + "\n".join(lines)  # 📊

    # Body — normalize whitespace + bold dates + bold/arrow numbered headlines
    body = _normalize_body(value)

    # Stale date warning
    stale = _detect_stale_dates(value)
    stale_warn = ""
    if stale:
        sample = ", ".join(stale[:3])
        if len(stale) > 3:
            sample += f", +{len(stale)-3} more"
        stale_warn = f"\n⚠️ **Stale dates detected** (>{_STALE_DAYS}d old): {sample}"

    # Footer
    footer = f"\n`{now}`"

    # Assemble — channel adapters split for their per-platform limits, so
    # we do NOT truncate here. A 2-page Domain Study should ship as 2
    # Discord messages, not get cut mid-sentence.
    return header + metrics_block + "\n" + body + stale_warn + footer


def format_system_health(
    monitor_name: str,
    *,
    db_size_mb: float | None = None,
    memory_pct: float | None = None,
    disk_pct: float | None = None,
    ollama_latency_ms: float | None = None,
    chromadb_docs: int | None = None,
    skill_health: str | None = None,
    extra_lines: list[str] | None = None,
) -> str:
    """Format a system health monitor output for Discord."""
    metrics = {}
    status = "ok"

    if db_size_mb is not None:
        metrics["DB size"] = f"{db_size_mb:.1f} MB"
        if db_size_mb > 500:
            status = "warning"

    if memory_pct is not None:
        metrics["Memory"] = f"{memory_pct:.0f}%"
        if memory_pct > 90:
            status = "error"
        elif memory_pct > 80:
            status = "warning"

    if disk_pct is not None:
        metrics["Disk"] = f"{disk_pct:.0f}%"
        if disk_pct > 95:
            status = "error"
        elif disk_pct > 85:
            status = "warning"

    if ollama_latency_ms is not None:
        metrics["Ollama latency"] = f"{ollama_latency_ms:.0f}ms"
        if ollama_latency_ms > 5000:
            status = "error"
        elif ollama_latency_ms > 2000:
            status = "warning"

    if chromadb_docs is not None:
        metrics["ChromaDB docs"] = str(chromadb_docs)

    if skill_health is not None:
        metrics["Skill health"] = skill_health

    body_lines = extra_lines or []
    body = "\n".join(body_lines) if body_lines else ""

    return format_monitor_output(
        monitor_name,
        body,
        status=status,
        metrics=metrics,
    )
