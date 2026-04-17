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


def classify_status(value: str) -> str:
    """Classify a monitor result into ok/warning/error based on content."""
    lower = value.lower()
    if any(w in lower for w in ("error", "failed", "critical", "down", "unreachable", "exception")):
        return "error"
    if any(w in lower for w in ("warning", "degraded", "slow", "elevated", "approaching")):
        return "warning"
    return "ok"


def format_monitor_output(
    monitor_name: str,
    value: str,
    *,
    status: str | None = None,
    metrics: dict[str, str | int | float] | None = None,
) -> str:
    """Format a monitor result for Discord.

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

    # Header
    header = f"{emoji} **{monitor_name}**"

    # Metrics block (compact)
    metrics_block = ""
    if metrics:
        lines = [f"  **{k}:** {v}" for k, v in metrics.items()]
        metrics_block = "\n\U0001f4ca " + "\n".join(lines)  # 📊

    # Body
    body = value.strip()

    # Footer
    footer = f"\n`{now}`"

    # Assemble and truncate if needed
    full = header + metrics_block + "\n" + body + footer
    if len(full) <= DISCORD_LIMIT:
        return full

    # Truncate body to fit
    suffix = "...\n*[truncated for Discord limit]*"
    overhead = len(header) + len(metrics_block) + len(footer) + len(suffix) + 5  # 5 for separators
    body_budget = DISCORD_LIMIT - overhead
    if body_budget < 100:
        body_budget = 100

    # Smart truncation: find last sentence boundary
    truncated = body[:body_budget]
    for sep in (". ", ".\n", "\n\n", "\n"):
        last = truncated.rfind(sep)
        if last > body_budget // 2:
            truncated = truncated[:last + len(sep)]
            break

    truncated += suffix

    return header + metrics_block + "\n" + truncated + footer


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
