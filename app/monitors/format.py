"""Monitor output formatting for Discord.

Standardizes all monitor outputs for consistent, readable Discord messages.
Handles Discord's 2000-char limit with smart truncation.
"""

from __future__ import annotations

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
}

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
