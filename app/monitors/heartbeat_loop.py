"""HeartbeatLoop — the background scheduling engine.

Checks monitors on schedule, executes them, and delivers alerts via
Discord, Telegram, WhatsApp, and Signal channel bots.

Extracted from heartbeat.py for maintainability.
"""

from __future__ import annotations

import asyncio
import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Any

from app.config import config
from app.monitors.format import (
    format_monitor_result,
    strip_tool_call_artifacts,
)
from app.monitors.monitor_store import (
    Monitor,
    MonitorResult,  # noqa: F401 — available for callers
    MonitorStore,
    detect_change,
)

logger = logging.getLogger(__name__)

_MAX_CONCURRENT_LLM_MONITORS = 2

# Monitors whose output is non-factual — skip KG extraction for these
_NO_KG_MONITORS = frozenset({"Morning Check-in", "Self-Reflection"})

# ---------------------------------------------------------------------------
# Deliberation scrubber — strip untagged model deliberation from monitor output
# ---------------------------------------------------------------------------

_DELIBERATION_PATTERNS = [
    re.compile(r"^(?:wait|okay|ok|hmm|let me|actually)[,\s].*?(?:let me|I(?:'ll| will| should)|re-?read|revis|re-?think|reconsider|check).*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^(?:Okay |OK )?(?:final|revised) (?:version|answer|response).*?:?\s*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^(?:Let me )?(?:re-?(?:read|think|consider|examine)|rephrase).*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^Actually (?:re-?reading|looking|checking).*$", re.IGNORECASE | re.MULTILINE),
]


def _strip_deliberation(text: str) -> str:
    """Remove untagged deliberation lines from monitor output."""
    for pat in _DELIBERATION_PATTERNS:
        text = pat.sub("", text)
    while "\n\n\n" in text:
        text = text.replace("\n\n\n", "\n\n")
    return text.strip()


# ---------------------------------------------------------------------------
# HeartbeatLoop — the background engine
# ---------------------------------------------------------------------------

class HeartbeatLoop:
    """Background loop that checks monitors on schedule and sends alerts."""

    def __init__(
        self,
        store: MonitorStore,
        *,
        discord_bot: Any = None,
        telegram_bot: Any = None,
        whatsapp_bot: Any = None,
        signal_bot: Any = None,
    ):
        self.store = store
        self._discord = discord_bot
        self._telegram = telegram_bot
        self._whatsapp = whatsapp_bot
        self._signal = signal_bot
        self._task: asyncio.Task | None = None
        self._running = False

    def start(self) -> asyncio.Task:
        """Start the heartbeat loop as a background task."""
        self._running = True
        self._task = asyncio.create_task(self._loop())
        logger.info("[Heartbeat] Started (interval=%ds)", config.HEARTBEAT_INTERVAL)
        return self._task

    def stop(self) -> None:
        """Stop the heartbeat loop."""
        self._running = False
        if self._task:
            self._task.cancel()
            logger.info("[Heartbeat] Stopped")

    async def _loop(self) -> None:
        """Main loop — check due monitors every HEARTBEAT_INTERVAL seconds."""
        try:
            # Small delay on startup to let services initialize
            await asyncio.sleep(10)

            while self._running:
                try:
                    due = self.store.get_due()
                    if due:
                        logger.info("[Heartbeat] %d monitor(s) due", len(due))

                        _FAST_TYPES = {"system_health", "maintenance"}
                        fast = [m for m in due if m.check_type in _FAST_TYPES]
                        slow = [m for m in due if m.check_type not in _FAST_TYPES]

                        # Fast monitors first (no LLM, sub-second)
                        for monitor in fast:
                            try:
                                await self._check_monitor(monitor)
                            except Exception as e:
                                logger.error("[Heartbeat] Monitor '%s' failed: %s", monitor.name, e)
                                self.store.record_check(monitor.id, f"error: {e}")
                                self.store.add_result(monitor.id, "error", message=str(e))

                        # LLM monitors with bounded concurrency
                        if slow:
                            sem = asyncio.Semaphore(_MAX_CONCURRENT_LLM_MONITORS)

                            async def _limited_check(monitor):
                                async with sem:
                                    try:
                                        await self._check_monitor(monitor)
                                    except Exception as e:
                                        logger.error("[Heartbeat] Monitor '%s' failed: %s", monitor.name, e)
                                        # Exponential backoff: count recent consecutive errors
                                        _recent_errors = 0
                                        try:
                                            _rows = self.store._db.fetchall(
                                                "SELECT status FROM monitor_results WHERE monitor_id = ? "
                                                "ORDER BY id DESC LIMIT 5",
                                                (monitor.id,),
                                            )
                                            for _row in _rows:
                                                if _row["status"] == "error":
                                                    _recent_errors += 1
                                                else:
                                                    break
                                        except Exception:
                                            _recent_errors = 0
                                        _BASE = 300  # 5 min
                                        _retry_delay = min(
                                            _BASE * (3 ** _recent_errors),
                                            monitor.schedule_seconds,
                                        )
                                        retry_at = datetime.now(timezone.utc) - timedelta(
                                            seconds=max(0, monitor.schedule_seconds - _retry_delay)
                                        )
                                        self.store.update(
                                            monitor.id,
                                            last_check_at=retry_at.strftime("%Y-%m-%d %H:%M:%S"),
                                        )
                                        self.store.add_result(
                                            monitor.id, "error",
                                            message=f"Exception — retry in ~{_retry_delay // 60} min: {e}",
                                        )

                            await asyncio.gather(*[_limited_check(m) for m in slow], return_exceptions=True)

                    # Execute due heartbeat instructions
                    due_instructions = self.store.get_due_instructions()
                    for inst in due_instructions:
                        try:
                            await self._execute_instruction(inst)
                        except Exception as e:
                            logger.error("[Heartbeat] Instruction #%d failed: %s", inst.id, e)
                except Exception as e:
                    logger.error("[Heartbeat] Loop iteration failed: %s", e)

                await asyncio.sleep(config.HEARTBEAT_INTERVAL)
        except asyncio.CancelledError:
            logger.info("[Heartbeat] Loop cancelled")
        except Exception as e:
            logger.error("[Heartbeat] Loop terminated unexpectedly: %s", e)

    async def _check_monitor(self, monitor: Monitor) -> None:
        """Execute a single monitor check."""
        logger.info("[Heartbeat] Checking '%s' (type=%s)", monitor.name, monitor.check_type)

        # Execute the check
        new_value = await self._execute_check(monitor)
        # Defensive: strip any tool-call artifacts the LLM may have emitted
        # instead of executing the tool. Keeps Discord/Telegram output clean.
        if new_value:
            new_value = strip_tool_call_artifacts(new_value)

        # Categorize the result BEFORE recording
        _lower = (new_value or "").lower()

        # LLM failures that warrant a retry (Ollama down, timeout, etc.)
        # Only match messages that indicate the LLM itself is down, not general errors.
        _is_llm_failure = new_value and (
            new_value.startswith("I can't reach the language model")
            or new_value.startswith("I attempted to use tools but couldn't complete")
            or "provide your answer" in _lower[:200]
            or "do NOT say you cannot" in new_value[:300]
            or (new_value.startswith("[") and "failed" in _lower
                and ("generation failed" in _lower or "grading failed" in _lower))
            or "llm failure" in _lower
            or "ollama" in _lower and ("timeout" in _lower or "timed out" in _lower)
        )

        # Legitimate skips — system working, just nothing to do
        _is_skip = new_value and (
            new_value.startswith("[No pending")
            or new_value.startswith("[No monitor candidates")
            or (new_value.startswith("[") and "skipped]" in new_value
                and "failed" not in _lower)
        )

        if _is_llm_failure:
            # Exponential backoff: 5min → 15min → 45min, capped at schedule interval.
            # Count recent consecutive errors to determine backoff level.
            recent_errors = 0
            try:
                rows = self.store._db.fetchall(
                    "SELECT status FROM monitor_results WHERE monitor_id = ? "
                    "ORDER BY id DESC LIMIT 5",
                    (monitor.id,),
                )
                for row in rows:
                    if row["status"] == "error":
                        recent_errors += 1
                    else:
                        break
            except Exception:
                recent_errors = 0

            _BASE_RETRY = 300  # 5 minutes
            _retry_delay = min(
                _BASE_RETRY * (3 ** recent_errors),  # 5m, 15m, 45m, 135m...
                monitor.schedule_seconds,              # cap at normal schedule
            )
            retry_at = datetime.now(timezone.utc) - timedelta(
                seconds=max(0, monitor.schedule_seconds - _retry_delay)
            )
            self.store.update(
                monitor.id,
                last_check_at=retry_at.strftime("%Y-%m-%d %H:%M:%S"),
            )
            self.store.add_result(monitor.id, "error", value=new_value[:4000] if new_value else "",
                                 message=f"LLM failure — retry in ~{_retry_delay // 60} min")
            logger.warning("[Heartbeat] '%s' LLM failure (streak=%d), retry in ~%d min: %s",
                           monitor.name, recent_errors + 1, _retry_delay // 60, (new_value or "")[:100])
            return

        if _is_skip:
            # Record normally — this is expected behavior, not an error
            self.store.record_check(monitor.id, new_value)
            self.store.add_result(monitor.id, "ok", value=new_value[:4000] if new_value else "")
            return

        # Only record check (update last_check_at) on successful results
        self.store.record_check(monitor.id, new_value)

        # Extract KG triples from all factual query monitors (skip non-factual ones)
        if monitor.check_type == "query" and monitor.name not in _NO_KG_MONITORS and new_value and len(new_value) > 100:
            try:
                from app.core.brain import get_services, _extract_kg_triples
                svc = get_services()
                if svc.kg:
                    asyncio.create_task(_extract_kg_triples(svc.kg, monitor.name, new_value[:2000], source_name=monitor.name))
            except Exception:
                pass

        # Determine if we should alert (non-results already returned above)
        should_alert = False
        change_info = None

        if monitor.notify_condition == "always":
            should_alert = True
        elif monitor.notify_condition in ("on_change", "on_alert"):
            if monitor.last_result:
                threshold = monitor.check_config.get("threshold_pct", 5.0)
                # Quiz/skill_test values contain topic text with incidental numbers
                # (years, percentages) — skip numeric comparison, use text-only
                if monitor.check_type in ("quiz", "skill_test"):
                    threshold = 999999  # Force text-only comparison
                change_info = detect_change(monitor.last_result, new_value, threshold)
                should_alert = change_info is not None
            else:
                # First check — always alert
                should_alert = True
        elif monitor.notify_condition == "on_error":
            # Check for error indicators in the result value (status is computed later)
            _val_lower = (new_value or "").lower()
            should_alert = any(w in _val_lower for w in ("error", "fail", "exception", "timeout"))
        elif monitor.notify_condition == "on_threshold":
            if new_value and monitor.check_config.get("threshold_value"):
                try:
                    val = float(new_value.split()[0]) if new_value else 0
                    threshold = float(monitor.check_config["threshold_value"])
                    should_alert = val > threshold
                except (ValueError, IndexError):
                    should_alert = False

        if not should_alert:
            self.store.add_result(monitor.id, "ok", value=new_value[:4000] if new_value else "")
            return

        # Check cooldown
        if monitor.last_alert_at:
            last_alert = datetime.fromisoformat(monitor.last_alert_at).replace(tzinfo=None)
            now_naive = datetime.now(timezone.utc).replace(tzinfo=None)
            if (now_naive - last_alert).total_seconds() < monitor.cooldown_minutes * 60:
                logger.info("[Heartbeat] '%s' in cooldown, skipping alert", monitor.name)
                self.store.add_result(monitor.id, "ok", value=new_value[:4000] if new_value else "",
                                      message="in cooldown")
                return

        # For "always" monitors (domain studies etc), the result IS the alert —
        # no LLM re-summarization needed (it only mangles good content).
        # Only use LLM analysis for change-detected alerts where we need to
        # describe what changed.
        if change_info:
            analysis = await self._analyze_result(monitor, new_value, change_info)
        else:
            # Send the raw result directly — channel adapters handle their own
            # message splitting (Discord splits at 2000, Telegram at 4096)
            analysis = new_value[:4000] if new_value else ""

        # Send alert
        await self._send_alert(monitor, analysis)

        # Auto-disable one-shot reminders after first alert
        if monitor.name.startswith("[Reminder]"):
            self.store.update(monitor.id, enabled=False)
            logger.info("[Heartbeat] Reminder '%s' auto-disabled after alert", monitor.name)

        # Record
        status = "changed" if change_info else "ok"
        if change_info and change_info.get("type") == "numeric":
            status = "alert"
        self.store.record_alert(monitor.id)
        self.store.add_result(monitor.id, status, value=new_value[:4000] if new_value else "",
                              message=analysis[:500] if analysis else "")

    async def _execute_check(self, monitor: Monitor) -> str:
        """Run the actual check based on monitor type."""
        from app.core.brain import get_services

        svc = get_services()
        cfg = monitor.check_config

        if monitor.check_type == "url":
            url = cfg.get("url", "")
            if svc.tool_registry:
                return await svc.tool_registry.execute("http_fetch", {"url": url})
            return f"[No tool registry — cannot fetch {url}]"

        elif monitor.check_type == "search":
            query = cfg.get("query", "")
            if svc.tool_registry:
                return await svc.tool_registry.execute("web_search", {"query": query})
            return "[No tool registry — cannot search]"

        elif monitor.check_type == "command":
            command = cfg.get("command", "")
            if svc.tool_registry:
                return await svc.tool_registry.execute("shell_exec", {"command": command})
            return "[No tool registry — cannot exec]"

        elif monitor.check_type == "system_health":
            return await self._execute_system_health()

        elif monitor.check_type == "query":
            # Use brain.think() directly — collect tokens
            query = cfg.get("query", "")
            return await self._think_query(query)

        elif monitor.check_type == "quiz":
            return await self._execute_quiz(cfg)

        elif monitor.check_type == "skill_test":
            return await self._execute_skill_test(cfg)

        elif monitor.check_type == "curiosity":
            return await self._execute_curiosity_research(cfg)

        elif monitor.check_type == "auto_monitor":
            return await self._execute_auto_monitor_detection(cfg)

        elif monitor.check_type == "maintenance":
            return await self._execute_maintenance(cfg)

        elif monitor.check_type == "finetune":
            return await self._execute_finetune_check(cfg)

        elif monitor.check_type == "consolidation":
            return await self._execute_consolidation(cfg)

        elif monitor.check_type == "capability_review":
            return await self._execute_capability_review(cfg)

        elif monitor.check_type == "eval":
            return await self._execute_eval_harness(cfg)

        elif monitor.check_type == "prompt_analyzer":
            return await self._execute_prompt_analyzer(cfg)

        elif monitor.check_type == "db_size":
            return await self._execute_db_size_check()

        elif monitor.check_type == "ollama_latency":
            return await self._execute_ollama_latency_check()

        elif monitor.check_type == "skill_quality":
            return await self._execute_skill_quality_check()

        elif monitor.check_type == "chromadb_integrity":
            return await self._execute_chromadb_integrity_check()

        elif monitor.check_type == "kg_health":
            return await self._execute_kg_health_check()

        elif monitor.check_type == "training_job":
            return await self._execute_training_job_check()

        elif monitor.check_type == "kg_growth":
            return await self._execute_kg_growth_check(monitor)

        elif monitor.check_type == "ollama_model":
            return await self._execute_ollama_model_check()

        return f"[Unknown check_type: {monitor.check_type}]"

    async def _execute_system_health(self) -> str:
        """Gather system health using Python stdlib — cross-platform (Linux + Windows)."""
        import os
        import platform
        import shutil

        lines: list[str] = []
        is_windows = platform.system() == "Windows"

        # Disk usage — shutil.disk_usage is cross-platform
        try:
            disk_path = "C:\\" if is_windows else "/"
            usage = shutil.disk_usage(disk_path)
            total_gb = usage.total / (1024 ** 3)
            used_gb = usage.used / (1024 ** 3)
            free_gb = usage.free / (1024 ** 3)
            used_pct = (used_gb / total_gb * 100) if total_gb else 0
            lines.append(f"Disk: {used_gb:.1f}G / {total_gb:.1f}G ({used_pct:.0f}% used, {free_gb:.1f}G free)")
        except OSError:
            lines.append("Disk: unavailable")

        # Load average — no Windows stdlib equivalent
        try:
            load1, load5, load15 = os.getloadavg()
            lines.append(f"Load: {load1:.2f} {load5:.2f} {load15:.2f}")
        except (OSError, AttributeError):
            lines.append("Load: unavailable")

        # Memory usage via psutil (graceful fallback chain)
        try:
            import psutil
            mem = psutil.virtual_memory()
            total_gb = mem.total / (1024 ** 3)
            used_gb = mem.used / (1024 ** 3)
            lines.append(f"Memory: {used_gb:.1f}G / {total_gb:.1f}G ({mem.percent}% used)")
        except ImportError:
            if is_windows:
                # Windows ctypes fallback via kernel32.GlobalMemoryStatusEx
                try:
                    import ctypes
                    import ctypes.wintypes

                    class MEMORYSTATUSEX(ctypes.Structure):
                        _fields_ = [
                            ("dwLength", ctypes.wintypes.DWORD),
                            ("dwMemoryLoad", ctypes.wintypes.DWORD),
                            ("ullTotalPhys", ctypes.c_uint64),
                            ("ullAvailPhys", ctypes.c_uint64),
                            ("ullTotalPageFile", ctypes.c_uint64),
                            ("ullAvailPageFile", ctypes.c_uint64),
                            ("ullTotalVirtual", ctypes.c_uint64),
                            ("ullAvailVirtual", ctypes.c_uint64),
                            ("ullAvailExtendedVirtual", ctypes.c_uint64),
                        ]

                    stat = MEMORYSTATUSEX()
                    stat.dwLength = ctypes.sizeof(stat)
                    if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                        total_gb = stat.ullTotalPhys / (1024 ** 3)
                        avail_gb = stat.ullAvailPhys / (1024 ** 3)
                        used_gb = total_gb - avail_gb
                        used_pct = (used_gb / total_gb * 100) if total_gb else 0
                        lines.append(f"Memory: {used_gb:.1f}G / {total_gb:.1f}G ({used_pct:.0f}% used)")
                    else:
                        lines.append("Memory: unavailable")
                except (OSError, AttributeError):
                    lines.append("Memory: unavailable")
            else:
                # Linux fallback via /proc/meminfo
                try:
                    with open("/proc/meminfo") as f:
                        info = {}
                        for line in f:
                            parts = line.split()
                            if len(parts) >= 2:
                                info[parts[0].rstrip(":")] = int(parts[1])
                    total_kb = info.get("MemTotal", 0)
                    avail_kb = info.get("MemAvailable", info.get("MemFree", 0))
                    if total_kb:
                        used_kb = total_kb - avail_kb
                        lines.append(
                            f"Memory: {used_kb / 1048576:.1f}G / {total_kb / 1048576:.1f}G "
                            f"({used_kb / total_kb * 100:.0f}% used)"
                        )
                    else:
                        lines.append("Memory: unavailable")
                except (OSError, KeyError):
                    lines.append("Memory: unavailable")

        # Uptime — cross-platform
        if is_windows:
            try:
                import ctypes
                uptime_ms = ctypes.windll.kernel32.GetTickCount64()
                uptime_secs = uptime_ms / 1000
                days = int(uptime_secs // 86400)
                hours = int((uptime_secs % 86400) // 3600)
                mins = int((uptime_secs % 3600) // 60)
                lines.append(f"Uptime: {days}d {hours}h {mins}m")
            except (OSError, AttributeError):
                lines.append(f"Platform: {platform.system()} {platform.release()}")
        else:
            try:
                with open("/proc/uptime") as f:
                    uptime_secs = float(f.read().split()[0])
                days = int(uptime_secs // 86400)
                hours = int((uptime_secs % 86400) // 3600)
                mins = int((uptime_secs % 3600) // 60)
                lines.append(f"Uptime: {days}d {hours}h {mins}m")
            except (OSError, ValueError):
                lines.append(f"Platform: {platform.system()} {platform.release()}")

        return "\n".join(lines)

    async def _think_query(self, query: str) -> str:
        """Run a query through brain.think() and collect the text response.

        Prepends live system context so the LLM knows about monitors,
        conversations, and learning activity.  Uses ephemeral=True to
        avoid polluting conversation history.
        """
        from app.core.brain import think, get_services
        from app.schema import EventType

        # --- Build system context ---
        ctx_lines: list[str] = []
        try:
            svc = get_services()

            # Monitors
            monitors = self.store.list_all()
            enabled = [m for m in monitors if m.enabled]
            ctx_lines.append(
                f"Monitors: {len(monitors)} total, {len(enabled)} enabled — "
                + ", ".join(m.name for m in monitors)
            )

            # Recent alerts (24h)
            recent = self.store.get_recent_results(hours=24, limit=20)
            if recent:
                alerts = [r for r in recent if r.status in ("alert", "changed", "error")]
                ctx_lines.append(f"Last 24h: {len(recent)} results, {len(alerts)} alerts/changes")
            else:
                ctx_lines.append("Last 24h: no monitor results yet")

            # Recent conversations
            if svc.conversations:
                convos = svc.conversations.list_conversations(limit=10)
                if convos:
                    titles = [c.get("title") or "(untitled)" for c in convos]
                    ctx_lines.append(f"Recent conversations ({len(convos)}): " + ", ".join(titles))
                else:
                    ctx_lines.append("Recent conversations: none")

            # Learning summary with actual content
            if svc.learning:
                summary = svc.learning.get_learning_summary(hours=24)
                parts = []
                if summary.get("new_lessons"):
                    parts.append(f"{len(summary['new_lessons'])} new lesson(s)")
                    for les in summary["new_lessons"][:5]:
                        topic = les.get("topic", "?")[:60]
                        lesson_text = (les.get("lesson_text") or les.get("correct_answer", ""))[:100]
                        ctx_lines.append(f"  Lesson: {topic} — {lesson_text}")
                if summary.get("new_skills"):
                    parts.append(f"{len(summary['new_skills'])} new skill(s)")
                if summary.get("degraded_skills"):
                    parts.append(f"{len(summary['degraded_skills'])} degraded skill(s)")
                if summary.get("new_reflexions"):
                    parts.append(f"{len(summary['new_reflexions'])} new reflexion(s)")
                    for ref in summary["new_reflexions"][:5]:
                        task = ref.get("task_summary", "?")[:60]
                        score = ref.get("quality_score", 0)
                        ctx_lines.append(f"  Reflexion (quality={score:.1f}): {task}")
                ctx_lines.append("Learning (24h): " + (", ".join(parts) if parts else "no activity"))

            # Owner facts
            if svc.user_facts:
                facts = svc.user_facts.get_all()
                if facts:
                    ctx_lines.append(
                        f"Known owner facts ({len(facts)}): "
                        + ", ".join(f"{f.key}={f.value}" for f in facts[:10])
                    )
        except Exception as e:
            logger.warning("[Heartbeat] Failed to build system context: %s", e)

        # Temporal grounding — inject current date so monitors never produce stale content
        _now = datetime.now(timezone.utc)
        ctx_lines.insert(0,
            f"TODAY IS: {_now.strftime('%A, %B %d, %Y')} (UTC). "
            "All searches and answers MUST be about events from TODAY or the past 24-48 hours. "
            "Do NOT report old news. Include specific dates in your findings."
        )

        # Strict output contract — stops the LLM from offering suggestions,
        # asking clarifying questions, or emitting raw tool-call JSON in the
        # final answer. Tool calls themselves still fire normally during the
        # tool loop (they're not final output).
        output_contract = (
            "=== OUTPUT CONTRACT ===\n"
            "This is a monitor report, NOT a conversation. Produce a snapshot, "
            "not a suggestion.\n"
            "- Do NOT ask the user questions.\n"
            "- Do NOT offer to set up, continue, or expand monitoring.\n"
            "- Do NOT narrate your reasoning.\n"
            "- Do NOT include raw tool-call JSON or </tool_call> in the answer.\n"
            "- If nothing notable changed, reply exactly: "
            "'no change | last: <UTC timestamp>'.\n"
            "- Otherwise produce 2-3 compact bullets with specific facts and dates.\n"
            "=== END CONTRACT ===\n\n"
        )

        # Prepend context to query
        if ctx_lines:
            context_block = "=== System Context ===\n" + "\n".join(ctx_lines) + "\n=== End Context ===\n\n"
            enriched_query = context_block + output_contract + query
        else:
            enriched_query = output_contract + query

        tokens = []
        try:
            async with asyncio.timeout(config.GENERATION_TIMEOUT):
                async for event in think(query=enriched_query, ephemeral=True):
                    if event.type == EventType.TOKEN:
                        text = event.data.get("text", "")
                        if text:
                            tokens.append(text)
        except asyncio.TimeoutError:
            logger.warning("[Heartbeat] _think_query timed out for: %s", query[:80])
            return "[Query timed out]"
        except Exception as e:
            logger.error("[Heartbeat] think() failed: %s", e)
            return f"[Query failed: {e}]"

        result = "".join(tokens).strip()
        result = _strip_deliberation(result)
        result = strip_tool_call_artifacts(result)
        return result

    async def _execute_instruction(self, inst) -> None:
        """Execute a user-defined heartbeat instruction via brain.think()."""
        from app.core.brain import think, get_services  # noqa: F401
        from app.schema import EventType

        logger.info("[Heartbeat] Running instruction #%d: '%s'", inst.id, inst.instruction[:80])

        tokens: list[str] = []
        try:
            async with asyncio.timeout(float(config.GENERATION_TIMEOUT)):
                async for event in think(inst.instruction, ephemeral=True):
                    if event.type == EventType.TOKEN:
                        text = event.data.get("text", "")
                        if text:
                            tokens.append(text)
        except (TimeoutError, asyncio.TimeoutError):
            logger.warning("[Heartbeat] Instruction #%d timed out after %ds", inst.id, config.GENERATION_TIMEOUT)
            self.store.record_instruction_run(inst.id)
            return
        except Exception as e:
            logger.error("[Heartbeat] Instruction #%d failed: %s", inst.id, e)
            self.store.record_instruction_run(inst.id)
            return

        result = "".join(tokens).strip()
        self.store.record_instruction_run(inst.id)

        if not result:
            return

        # Send via configured channels
        channels = {c.strip() for c in inst.notify_channels.split(",") if c.strip()}
        message = f"**Standing Instruction**\n{inst.instruction[:100]}\n\n{result[:1500]}"

        sent = False
        if "discord" in channels and self._discord:
            try:
                await self._discord.send_alert(message)
                sent = True
            except Exception as e:
                logger.warning("[Heartbeat] Instruction Discord send failed: %s", e)
        if "telegram" in channels and self._telegram:
            try:
                await self._telegram.send_alert(message)
                sent = True
            except Exception as e:
                logger.warning("[Heartbeat] Instruction Telegram send failed: %s", e)
        if "whatsapp" in channels and self._whatsapp:
            try:
                await self._whatsapp.send_alert(message)
                sent = True
            except Exception as e:
                logger.warning("[Heartbeat] Instruction WhatsApp send failed: %s", e)
        if "signal" in channels and self._signal:
            try:
                await self._signal.send_alert(message)
                sent = True
            except Exception as e:
                logger.warning("[Heartbeat] Instruction Signal send failed: %s", e)
        if sent:
            logger.info("[Heartbeat] Instruction #%d result delivered", inst.id)

    async def _execute_quiz(self, cfg: dict) -> str:
        """Pick a lesson using spaced repetition, quiz self, grade, and learn from failure.

        Prioritizes lessons with most quiz failures + oldest quiz date.
        """
        import random
        from app.core.brain import get_services
        from app.core import llm

        svc = get_services()
        if not svc.learning:
            return "[No learning engine — quiz skipped]"

        lessons = svc.learning.get_all_lessons(limit=200)
        if not lessons:
            return "[No lessons to quiz on — skipped]"

        # Spaced repetition: skip lessons stuck in failure loops (5+ failures, quizzed < 7 days ago)
        db = svc.learning._db
        lesson = None
        row = db.fetchone(
            "SELECT id FROM lessons "
            "WHERE (quiz_failures < 5 "
            "   OR last_quizzed_at < datetime('now', '-7 days') "
            "   OR last_quizzed_at IS NULL) "
            "AND correct_answer IS NOT NULL AND correct_answer != '' "
            "ORDER BY last_quizzed_at ASC NULLS FIRST, quiz_failures DESC "
            "LIMIT 1"
        )
        if row:
            lesson = next((l for l in lessons if l.id == row["id"]), None)
        if not lesson:
            # Fallback: pick a random lesson that has usable content
            usable = [l for l in lessons if l.correct_answer and len(l.correct_answer) > 20]
            if not usable:
                return "[No lessons with sufficient content to quiz on — skipped]"
            lesson = random.choice(usable)

        # Step 1: Generate a question from the lesson
        # Pick the longest available text source for context
        context_candidates = [lesson.context or '', lesson.lesson_text or '', lesson.correct_answer or '']
        context_text = max(context_candidates, key=len)
        if len(context_text.strip()) < 20:
            return f"[Lesson '{lesson.topic}' has insufficient context for quiz — skipped]"
        gen_prompt = (
            f"Topic: {lesson.topic}\n"
            f"Context: {context_text}\n\n"
            "Write a single, specific quiz question that tests knowledge of this topic. "
            "Just the question, nothing else."
        )
        try:
            question = await llm.invoke_nothink(
                [{"role": "user", "content": gen_prompt}],
                max_tokens=100, temperature=0.5,
            )
            question = question.strip()
        except Exception as e:
            return f"[Quiz question generation failed: {e}]"

        # Step 2: Answer WITH lesson topic as context (the model may not know
        # recent events from web searches, so provide grounding context)
        answer_prompt = (
            f"Topic context: {lesson.topic}. "
            f"Key information: {(lesson.lesson_text or lesson.correct_answer or '')[:300]}\n\n"
            f"Question: {question}\n\n"
            "Answer based on the context provided."
        )
        try:
            answer = await llm.invoke_nothink(
                [{"role": "user", "content": answer_prompt}],
                max_tokens=600, temperature=0.3,
            )
            answer = answer.strip()
        except Exception as e:
            return f"[Quiz answer generation failed: {e}]"

        # Step 3: Grade the answer against the correct answer.
        # IMPORTANT: The expected answer is ground truth (may contain data
        # beyond the model's training cutoff from web searches). The grader
        # must compare factual alignment, NOT question whether the expected
        # answer's facts are plausible.
        grade_prompt = (
            f"Question: {question}\n"
            f"Reference answer (GROUND TRUTH — treat as authoritative): {lesson.correct_answer}\n"
            f"Student answer: {answer}\n\n"
            "Does the student answer align with the key facts in the reference answer? "
            "The reference answer is verified and authoritative — do NOT question its accuracy. "
            'Respond with JSON: {{"pass": true}} or {{"pass": false, "reason": "brief explanation"}}. Keep the reason under 20 words.'
        )
        try:
            grade_raw = await llm.invoke_nothink(
                [{"role": "user", "content": grade_prompt}],
                max_tokens=200, temperature=0.1,
                json_mode=True,
            )
            grade = llm.extract_json_object(grade_raw)
            if not grade or not isinstance(grade, dict):
                grade = {"pass": False, "reason": "Could not parse grade"}
        except Exception as e:
            logger.warning("[Heartbeat] Quiz grading failed: %s", e)
            grade = {"pass": False, "reason": str(e)}

        passed = grade.get("pass", False)

        # Update quiz tracking
        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        try:
            db.execute(
                "UPDATE lessons SET last_quizzed_at = ? WHERE id = ?",
                (now_str, lesson.id),
            )
        except Exception as e:
            logger.warning("[Heartbeat] Quiz tracking update failed: %s", e)

        if passed:
            # Reinforce the lesson
            try:
                svc.learning.mark_lesson_helpful(lesson.id)
            except Exception as e:
                logger.warning("[Heartbeat] mark_lesson_helpful failed: %s", e)
            return f"QUIZ PASSED | topic={lesson.topic} | q={question[:80]} | a={answer[:80]}"

        # Failed — increment quiz_failures counter
        try:
            db.execute(
                "UPDATE lessons SET quiz_failures = COALESCE(quiz_failures, 0) + 1 WHERE id = ?",
                (lesson.id,),
            )
        except Exception as e:
            logger.warning("[Heartbeat] Quiz failure increment failed: %s", e)

        # Failed — reduce lesson confidence, create training pair, reflexion
        fail_reason = grade.get("reason", "incorrect")

        try:
            svc.learning.mark_lesson_unhelpful(lesson.id)
        except Exception as e:
            logger.warning("[Heartbeat] Quiz mark_lesson_unhelpful failed: %s", e)

        # NOTE: Quiz failures no longer generate DPO training pairs.
        # Quiz questions are synthetic (not real user queries) and training on them
        # teaches the model to respond to quiz-format prompts, not real conversations.
        # DPO pairs should only come from real user corrections.

        if svc.reflexions:
            try:
                svc.reflexions.store(
                    task_summary=f"Quiz on '{lesson.topic}': {question[:100]}",
                    outcome="failure",
                    reflection=f"Answered incorrectly. Expected: {lesson.correct_answer[:200]}. Got: {answer[:200]}. Reason: {fail_reason}",
                    quality_score=0.2,
                )
            except Exception as e:
                logger.warning("[Heartbeat] Quiz reflexion failed: %s", e)

        return f"QUIZ FAILED | topic={lesson.topic} | q={question[:80]} | reason={fail_reason[:80]}"

    async def _execute_skill_test(self, cfg: dict) -> str:
        """Pick a random active skill, generate a test query, run through brain, assess quality."""
        import random
        from app.core.brain import get_services
        from app.core import llm
        from app.core.reflexion import assess_quality

        svc = get_services()
        if not svc.skills:
            return "[No skill store — skill test skipped]"

        skills = svc.skills.get_active_skills()
        if not skills:
            return "[No active skills — skipped]"

        skill = random.choice(skills)

        # Generate a test query that matches the skill's trigger pattern.
        # Strategy 1: Ask LLM with explicit keyword groups extracted from regex
        # Strategy 2: Extract literal words from regex and build a query
        # Extract keyword groups from regex alternations for the LLM prompt
        _alt_groups = re.findall(r'\(\?[i:]*([:!])?([^)]+)\)', skill.trigger_pattern)
        keyword_groups = []
        for _flag, content in _alt_groups:
            # Skip flags-only groups like (?i)
            if "|" in content or re.match(r'^[a-zA-Z_\s]+$', content):
                words_in_group = [re.sub(r'\\[bBdDwWsS]', '', w).strip() for w in content.split("|")]
                words_in_group = [w for w in words_in_group if w]
                if words_in_group:
                    keyword_groups.append(words_in_group)

        if keyword_groups:
            keywords_desc = "\n".join(
                f"  Group {i+1}: use one of: {', '.join(grp)}"
                for i, grp in enumerate(keyword_groups)
            )
            example_words = [grp[0] for grp in keyword_groups]
            example_query = "What is the " + " of ".join(example_words) + "?"
        else:
            keywords_desc = f"  (raw regex: {skill.trigger_pattern})"
            example_query = skill.name.replace("_", " ") + "?"

        gen_prompt = (
            f"Skill: {skill.name}\n"
            f"The query MUST contain at least one word from EACH of these groups:\n"
            f"{keywords_desc}\n\n"
            f"Example matching query: '{example_query}'\n\n"
            "Write a SHORT, natural user query that includes the required keywords. "
            "Just the query, nothing else:"
        )
        test_query = None
        temperatures = [0.3, 0.5, 0.7, 0.9]
        for attempt, temp in enumerate(temperatures):
            try:
                candidate = await llm.invoke_nothink(
                    [{"role": "user", "content": gen_prompt}],
                    max_tokens=80, temperature=temp,
                )
                # Clean up: strip quotes, whitespace, leading "Query:" etc.
                candidate = candidate.strip().strip('"\'').strip()
                for prefix in ("Query:", "query:", "User:", "user:"):
                    if candidate.startswith(prefix):
                        candidate = candidate[len(prefix):].strip()
            except Exception as e:
                return f"[Skill test query generation failed: {e}]"
            if re.search(skill.trigger_pattern, candidate, re.IGNORECASE):
                test_query = candidate
                break
            logger.debug(
                "[Heartbeat] Skill test query attempt %d didn't match: '%s' vs '%s'",
                attempt + 1, candidate[:80], skill.trigger_pattern[:60],
            )
        if not test_query:
            # Fallback: extract literal words from the regex and build a test query.
            # Find alternation groups like (?:word1|word2|word3) and pick one from each.
            groups = re.findall(r'\(\?:([^)]+)\)', skill.trigger_pattern)
            if len(groups) >= 2:
                import random as _rand
                # Use re.sub to strip \b markers — str.strip("\\b ") is wrong
                # because it strips individual chars including 'b' from words.
                words = [re.sub(r'\\[bBdDwWsS]', '', _rand.choice(g.split("|"))).strip() for g in groups]
                fallback = "What is the " + " of ".join(words) + "?"
                if re.search(skill.trigger_pattern, fallback, re.IGNORECASE):
                    test_query = fallback
            if not test_query:
                # Try skill name directly
                fallback = skill.name.replace("_", " ")
                if re.search(skill.trigger_pattern, fallback, re.IGNORECASE):
                    test_query = fallback
            if not test_query:
                logger.warning(
                    "[Heartbeat] Skill '%s' — 4 attempts + fallback failed to match trigger '%s'",
                    skill.name, skill.trigger_pattern,
                )
                return f"[Skill test skipped — generated queries didn't match trigger for '{skill.name}']"

        # Run through brain pipeline
        response = await self._think_query(test_query)

        # Assess quality
        score, reason = assess_quality(
            answer=response,
            tool_results=[],
            max_tool_rounds=3,
            query=test_query,
        )

        passed = score >= 0.6
        svc.skills.record_use(skill.id, passed)
        status = "PASSED" if passed else "FAILED"
        return (
            f"SKILL TEST {status} | skill={skill.name} | "
            f"success_rate={skill.success_rate:.0%} | "
            f"quality={score:.2f} | q={test_query[:60]}"
        )

    async def _execute_curiosity_research(self, cfg: dict) -> str:
        """Pick the top curiosity item, research it, store findings."""
        from app.core.brain import get_services

        svc = get_services()
        if not svc.curiosity:
            return "[Curiosity engine not initialized — skipped]"

        item = svc.curiosity.get_next()
        if not item:
            return "[No pending curiosity items — skipped]"

        # Research via think() with web search
        research_query = (
            f"Research this topic thoroughly using web_search: {item.topic}\n"
            f"Provide a concise, factual summary of what you find."
        )
        try:
            result = await self._think_query(research_query)

            # LLM failures should NOT count toward attempt limit — they'll resolve when LLM recovers
            _is_llm_down = result and (
                result.startswith("I can't reach the language model")
                or result.startswith("I attempted to use tools but couldn't complete")
            )
            if _is_llm_down:
                # Don't call fail() — leave attempts unchanged so it retries next cycle
                return f"[Curiosity skipped — LLM unavailable, will retry]"

            if result and not result.startswith("["):
                # Store findings in KG if possible
                if svc.kg and len(result) > 50:
                    from app.core.brain import _extract_kg_triples
                    try:
                        await _extract_kg_triples(svc.kg, item.topic, result)
                    except Exception:
                        pass

                svc.curiosity.resolve(item.id, result[:2000])

                # --- Convert research findings into a lesson ---
                if svc.learning:
                    try:
                        from app.core import llm as llm_mod
                        extract_prompt = (
                            f"Topic researched: {item.topic}\n\n"
                            f"Findings:\n{result[:1000]}\n\n"
                            f"Write a concise lesson (1-2 sentences) that captures the key takeaway. "
                            f'Return JSON: {{"topic": "...", "lesson": "..."}}'
                        )
                        raw = await llm_mod.invoke_nothink(
                            [{"role": "user", "content": extract_prompt}],
                            json_mode=True, json_prefix="{",
                            max_tokens=200, model=config.FAST_MODEL,
                        )
                        obj = llm_mod.extract_json_object(raw)
                        lesson_text = (obj.get("lesson", "") if obj else "").strip()
                        if obj and lesson_text and len(lesson_text) >= 20:
                            svc.learning.add_knowledge_lesson(
                                topic=obj.get("topic", item.topic[:100]),
                                correct_answer=lesson_text,
                                lesson_text=lesson_text,
                                context=f"Curiosity research on: {item.topic[:100]}",
                            )
                    except Exception as e:
                        logger.warning("[Heartbeat] Curiosity lesson extraction failed: %s", e)

                # --- Proactive follow-up: tell the user what we learned ---
                await self._send_curiosity_followup(item.topic, result)

                return f"CURIOSITY RESOLVED | topic={item.topic[:80]} | findings={result[:200]}"
            else:
                svc.curiosity.fail(item.id)
                return f"CURIOSITY FAILED | topic={item.topic[:80]} | result={result[:100]}"
        except Exception as e:
            svc.curiosity.fail(item.id)
            return f"CURIOSITY ERROR | topic={item.topic[:80]} | error={e}"

    async def _send_curiosity_followup(self, topic: str, findings: str) -> None:
        """Send a proactive message when curiosity resolves a topic the user asked about."""
        from app.core import llm

        try:
            prompt = (
                f"You previously couldn't fully answer a question about: {topic}\n\n"
                f"You just researched it and found:\n{findings[:800]}\n\n"
                f"Write a short, natural follow-up message (2-4 sentences) to the user. "
                f"Start with something like 'I looked into...' or 'I did some research on...' "
                f"Be specific about what you learned. Sound like a helpful friend who went "
                f"and found the answer, not a robot reporting data."
            )
            followup = await llm.invoke_nothink(
                [{"role": "user", "content": prompt}],
                max_tokens=250,
                temperature=0.5,
            )
            followup = followup.strip()
            followup = _strip_deliberation(followup)
        except Exception as e:
            logger.warning("[Heartbeat] Curiosity follow-up generation failed: %s", e)
            followup = f"I did some research on '{topic[:60]}' and here's what I found: {findings[:200]}"

        # Send via all available channels
        sent = False
        if self._discord:
            try:
                await self._discord.send_alert(followup)
                sent = True
            except Exception as e:
                logger.error("[Heartbeat] Curiosity follow-up Discord failed: %s", e)
        if self._telegram:
            try:
                await self._telegram.send_alert(followup)
                sent = True
            except Exception as e:
                logger.error("[Heartbeat] Curiosity follow-up Telegram failed: %s", e)
        if self._whatsapp:
            try:
                await self._whatsapp.send_alert(followup)
                sent = True
            except Exception as e:
                logger.error("[Heartbeat] Curiosity follow-up WhatsApp failed: %s", e)
        if self._signal:
            try:
                await self._signal.send_alert(followup)
                sent = True
            except Exception as e:
                logger.error("[Heartbeat] Curiosity follow-up Signal failed: %s", e)

        if sent:
            logger.info("[Heartbeat] Curiosity follow-up sent for '%s'", topic[:60])
        else:
            logger.info("[Heartbeat] Curiosity resolved '%s' (no channels for follow-up)", topic[:60])

    async def _execute_auto_monitor_detection(self, cfg: dict) -> str:
        """Detect frequently-asked topics and create monitors for them."""
        from app.core.brain import get_services

        svc = get_services()
        if not svc.topic_tracker:
            return "[Topic tracker not initialized — skipped]"

        candidates = svc.topic_tracker.get_monitor_candidates(min_count=3, days=7)
        if not candidates:
            return "[No monitor candidates found — skipped]"

        # Filter out invalid/low-quality topics
        from app.core.curiosity import CuriosityQueue
        import re as _re
        _BAD_MONITOR_RE = _re.compile(
            r"(?i)^(?:what|who|where|when|how|is|are|was|were|do|does|did|can|could|will|would|should)\b"  # questions
            r"|(?i)\b(?:price|cost|worth|trading at|how much)\b"  # price queries
            r"|(?i)\b(?:dont search|don.t search|just tell|from memory)\b"  # test queries
            r"|(?i)\b(?:time is it|what time|current time)\b"  # time queries
            r"|(?i)\b(?:calculate|compute|solve|equation)\b"  # math
            r"|(?i)\b(?:write|generate|create|make me)\b",  # generation requests
            _re.IGNORECASE,
        )
        candidates = [
            c for c in candidates
            if CuriosityQueue._is_valid_topic(c["topic"])
            and not _BAD_MONITOR_RE.search(c["topic"])
        ]
        if not candidates:
            return "[No valid monitor candidates — skipped]"

        # Filter out topics that already have monitors
        existing_monitors = {m.name.lower() for m in self.store.list_all()}
        auto_count = sum(1 for name in existing_monitors if name.startswith("auto:"))

        created = []
        for candidate in candidates:
            if auto_count >= 5:
                break

            topic = candidate["topic"]
            monitor_name = f"Auto: {topic[:50]}"

            if monitor_name.lower() in existing_monitors:
                continue

            query_prompt = (
                f"Use web_search to research the latest developments on: {topic}\n"
                f"Find 2-3 notable updates from the past few days. For each, give "
                f"one bullet: what happened and why it matters. Use this format:\n"
                f"• Update 1: ...\n• Update 2: ...\n• Update 3: ..."
            )
            mid = self.store.create(
                name=monitor_name,
                check_type="query",
                check_config={"query": query_prompt},
                schedule_seconds=43200,  # 12h
                cooldown_minutes=660,
                notify_condition="on_change",
            )
            if mid > 0:
                created.append(topic)
                auto_count += 1

        if created:
            return f"AUTO-MONITORS CREATED | count={len(created)} | topics={', '.join(t[:40] for t in created)}"
        return "[No new monitors needed — all candidates already covered]"

    async def _execute_maintenance(self, cfg: dict) -> str:
        """Run periodic maintenance: decay stale lessons, KG facts, reflexions, prune curiosity."""
        from app.core.brain import get_services

        svc = get_services()
        parts = []
        if svc.learning:
            try:
                decayed = svc.learning.decay_stale_lessons(days=30)
                if decayed:
                    parts.append(f"lessons decayed: {decayed}")
            except Exception as e:
                parts.append(f"lesson decay failed: {e}")
                logger.warning("[Heartbeat] Lesson decay failed: %s", e)
        if svc.kg:
            try:
                decayed = await svc.kg.decay_stale(days=60)
                if decayed:
                    parts.append(f"KG facts decayed: {decayed}")
            except Exception as e:
                parts.append(f"KG decay failed: {e}")
                logger.warning("[Heartbeat] KG decay failed: %s", e)
        if svc.reflexions:
            try:
                decayed = svc.reflexions.decay_stale(days=90)
                if decayed:
                    parts.append(f"reflexions decayed: {decayed}")
            except Exception as e:
                parts.append(f"reflexion decay failed: {e}")
                logger.warning("[Heartbeat] Reflexion decay failed: %s", e)
        if svc.curiosity:
            try:
                pruned = svc.curiosity.prune(days=30)
                if pruned:
                    parts.append(f"curiosity items pruned: {pruned}")
            except Exception as e:
                parts.append(f"curiosity prune failed: {e}")
                logger.warning("[Heartbeat] Curiosity prune failed: %s", e)
        # Cross-monitor feedback loops
        try:
            loop_parts = await self._check_feedback_loops(svc)
            parts.extend(loop_parts)
        except Exception as e:
            logger.warning("[Heartbeat] Feedback loops failed: %s", e)

        return f"MAINTENANCE | {', '.join(parts)}" if parts else "[No maintenance needed]"

    async def _check_feedback_loops(self, svc) -> list[str]:
        """Cross-monitor intelligence: quiz→curiosity, skill degradation→early test, curiosity→quiz log."""
        from app.database import SafeDB

        parts: list[str] = []

        # Guard: feedback loops need real DB access via learning._db
        has_db = (
            svc.learning
            and hasattr(svc.learning, "_db")
            and isinstance(svc.learning._db, SafeDB)
        )

        # Loop A — Quiz failures → Curiosity re-research
        # Lessons with 3+ quiz failures in last 7 days → queue for curiosity re-research
        if has_db and svc.curiosity:
            try:
                db = svc.learning._db
                failing = db.fetchall(
                    "SELECT id, topic FROM lessons "
                    "WHERE quiz_failures >= 3 "
                    "AND last_quizzed_at > datetime('now', '-7 days')"
                )
                requeued = 0
                for row in failing:
                    topic = row["topic"]
                    # Prefix to pass CuriosityQueue validation (15+ chars, 4+ words)
                    padded = f"Re-research and verify: {topic}"
                    cid = svc.curiosity.add(padded, source="quiz_feedback", urgency=0.7)
                    if cid > 0:
                        requeued += 1
                if requeued:
                    parts.append(f"quiz→curiosity: {requeued} topics re-queued")
            except Exception as e:
                logger.warning("[Heartbeat] Loop A (quiz→curiosity) failed: %s", e)

        # Loop B — Skill degradation → Early validation
        # Skills with 0.3 ≤ success_rate < 0.5 and 5+ uses → force Skill Validation next cycle
        if svc.skills:
            try:
                degrading = [
                    s for s in svc.skills.get_active_skills()
                    if 0.3 <= s.success_rate < 0.5 and s.times_used >= 5
                ]
                if degrading:
                    sv_monitor = self.store.get_by_name("Skill Validation")
                    if sv_monitor:
                        self.store.update(sv_monitor.id, last_check_at=None)
                        parts.append(f"skill→validation: {len(degrading)} degrading skills, forced early test")
            except Exception as e:
                logger.warning("[Heartbeat] Loop B (skill→validation) failed: %s", e)

        # Loop C — Curiosity → Quiz logging
        # Lessons from curiosity in last 24h that haven't been quizzed yet
        if has_db:
            try:
                db = svc.learning._db
                row = db.fetchone(
                    "SELECT COUNT(*) AS c FROM lessons "
                    "WHERE last_quizzed_at IS NULL "
                    "AND created_at > datetime('now', '-1 day')"
                )
                unquizzed = row["c"] if row else 0
                if unquizzed:
                    parts.append(f"new lessons awaiting quiz: {unquizzed}")
            except Exception as e:
                logger.warning("[Heartbeat] Loop C (curiosity→quiz) failed: %s", e)

        return parts

    async def _execute_finetune_check(self, cfg: dict) -> str:
        """Check if enough new training pairs exist for fine-tuning.

        Reports readiness status — does NOT auto-trigger training.
        The heartbeat alert notifies the user so they can trigger manually.
        """
        import json as _json
        from pathlib import Path

        data_path = config.TRAINING_DATA_PATH
        output_dir = config.FINETUNE_OUTPUT_DIR
        min_pairs = config.FINETUNE_MIN_NEW_PAIRS

        # Count total valid training pairs
        path = Path(data_path)
        total = 0
        if path.exists():
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = _json.loads(line)
                        if entry.get("query", "").strip() and entry.get("chosen", "").strip():
                            total += 1
                    except _json.JSONDecodeError:
                        continue

        # Check last training run count
        history_path = Path(output_dir) / "run_history.json"
        last_count = 0
        if history_path.exists():
            try:
                with open(history_path, encoding="utf-8") as f:
                    history = _json.load(f)
                if history:
                    last_count = history[-1].get("training_pairs", 0)
            except (_json.JSONDecodeError, OSError):
                pass

        new_pairs = total - last_count

        if new_pairs >= min_pairs:
            return (
                f"FINETUNE READY | {new_pairs} new training pairs available "
                f"(total: {total}, threshold: {min_pairs}). "
                f"Run: python scripts/finetune_auto.py"
            )

        return (
            f"FINETUNE NOT READY | {new_pairs} new pairs "
            f"(need {min_pairs}, total: {total})"
        )

    async def _execute_consolidation(self, cfg: dict) -> str:
        """Run a Dream Consolidation cycle — compacts memory, resolves contradictions, mines DPO pairs.

        Uses the DreamConsolidator 4-phase pipeline:
          Phase 1 ORIENT  — inventory all memory stores
          Phase 2 GATHER  — scan for stale/overlapping/broken items
          Phase 3 CONSOLIDATE — dedup, contradiction resolution, promotions, DPO mining
          Phase 4 REPORT  — prune low-value items, generate digest
        """
        from app.database import AsyncSafeDB, SafeDB, get_db
        from app.core.dream import DreamConsolidator

        # Respect a per-monitor cooldown beyond the normal cooldown_minutes so we
        # don't pound the LLM if the monitor runs too frequently.
        try:
            db = get_db()
            row = db.fetchone("SELECT value FROM system_state WHERE key='last_dream_at'")
            if row and row["value"]:
                last = datetime.fromisoformat(row["value"])
                elapsed_hours = (datetime.now(timezone.utc).replace(tzinfo=None) - last).total_seconds() / 3600
                min_hours = float(cfg.get("min_hours_between", 1.0))
                if elapsed_hours < min_hours:
                    return format_monitor_result(
                        "Dream Consolidation", "skip", "cooldown",
                        {"cooldown": f"{elapsed_hours:.1f}h/{min_hours}h"},
                    )
        except Exception:
            pass  # If we can't check, proceed

        try:
            db = get_db()
            async_db = AsyncSafeDB(db) if isinstance(db, SafeDB) else db
            consolidator = DreamConsolidator(async_db)
            digest = await consolidator.run()
            return format_monitor_result(
                "Dream Consolidation", "ok", "consolidation complete",
                {"digest": str(digest)[:120]},
            )
        except Exception as e:
            logger.error("[Heartbeat] Dream consolidation failed: %s", e)
            return format_monitor_result(
                "Dream Consolidation", "error", f"dream failed: {e}",
            )

    async def _execute_capability_review(self, cfg: dict) -> str:
        """Review accumulated capability gaps and suggest new tools/skills.

        Reads unreviewed gaps from the capability_gaps table, groups them by
        semantic similarity, and asks Nova to identify patterns and suggest
        what tools or skills could be created to address them. Marks gaps as
        reviewed after processing.
        """
        from app.database import get_db
        from app.core import llm

        db = get_db()
        try:
            rows = db.fetchall(
                "SELECT id, query, reason, quality_score FROM capability_gaps "
                "WHERE reviewed = 0 ORDER BY created_at DESC LIMIT 50"
            )
        except Exception as e:
            return f"[Capability review failed: could not read gaps — {e}]"

        if not rows:
            return "[Capability review: no unreviewed gaps found]"

        gap_count = len(rows)
        gap_summaries = "\n".join(
            f"- [{row['id']}] quality={row['quality_score']:.2f}: {row['query'][:120]}"
            for row in rows
        )

        try:
            suggestion = await llm.invoke_nothink(
                [
                    {
                        "role": "system",
                        "content": (
                            "You are analyzing capability gaps in an AI assistant. "
                            "You will be shown queries where the assistant failed "
                            "(no matching skill, no tool used, low quality score). "
                            "Identify patterns and suggest 2-3 specific tools or skills "
                            "that could be created to address these gaps. "
                            "Be concrete: name the tool/skill, describe what it does, "
                            "and list which gap queries it would address."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Review these {gap_count} capability gaps:\n\n"
                            f"{gap_summaries}\n\n"
                            "What tools or skills should be created to address these? "
                            "Focus on the most common patterns."
                        ),
                    },
                ],
                max_tokens=600,
                temperature=0.3,
            )
        except Exception as e:
            suggestion = f"[LLM review failed: {e}]"

        # Mark all reviewed gaps as reviewed
        try:
            gap_ids = [row["id"] for row in rows]
            db.execute(
                f"UPDATE capability_gaps SET reviewed = 1 WHERE id IN ({','.join('?' * len(gap_ids))})",
                tuple(gap_ids),
            )
        except Exception as e:
            logger.warning("[Heartbeat] Failed to mark gaps reviewed: %s", e)

        return (
            f"CAPABILITY REVIEW | gaps_reviewed={gap_count}\n\n"
            f"Suggestions:\n{suggestion}"
        )

    async def _analyze_result(
        self,
        monitor: Monitor,
        new_value: str,
        change_info: dict | None,
    ) -> str:
        """Ask Nova to analyze a monitor result intelligently."""
        from app.core import llm

        # Build a concise analysis prompt
        parts = [f"Monitor '{monitor.name}' ({monitor.check_type}) just ran."]

        if change_info:
            if change_info.get("type") == "numeric":
                parts.append(
                    f"Value changed {change_info['direction']} by {change_info['pct_change']}% "
                    f"(from {change_info['old']} to {change_info['new']})."
                )
            else:
                parts.append("The result changed since last check.")

        parts.append(f"Result:\n{new_value[:800]}")

        if change_info and monitor.last_result:
            parts.append(f"Previous result:\n{monitor.last_result[:400]}")
            parts.append(
                "Write a short, structured alert in this EXACT format:\n"
                "**What changed:** <one sentence>\n"
                "**Key detail:** <the most important number, name, or fact>\n"
                "No other text. No preamble. No filler. No repetition."
            )
        else:
            parts.append(
                "Write a short, structured summary in this EXACT format:\n"
                "**Summary:** <one sentence describing the result>\n"
                "**Key detail:** <the most important number, name, or fact>\n"
                "No other text. No preamble. No filler. No repetition."
            )

        # Fallback: first 250 chars of the raw result, cleaned up
        _raw_fallback = new_value[:250].rsplit(".", 1)[0] + "." if new_value else ""

        try:
            analysis = await llm.invoke_nothink(
                [{"role": "user", "content": "\n\n".join(parts)}],
                max_tokens=120,
                temperature=0.2,
            )
            # Truncate any runaway generation at first obvious repetition
            result = analysis.strip()
            if len(result) > 300:
                result = result[:300].rsplit(".", 1)[0] + "."

            # If the LLM ignored the format or generated refusals, use the raw result
            _has_format = "**" in result
            _is_refusal = any(p in result.lower() for p in (
                "i cannot", "i can't", "i don't have", "as an ai",
                "i'm unable", "no such", "in the future",
            ))
            if _is_refusal or (not _has_format and len(result) > 100):
                logger.info("[Heartbeat] LLM alert was off-format, using raw fallback")
                return _raw_fallback

            return result
        except Exception as e:
            logger.warning("[Heartbeat] Analysis generation failed: %s", e)
            # Fallback to raw summary
            if change_info and change_info.get("type") == "numeric":
                return (
                    f"Monitor '{monitor.name}': value moved {change_info['direction']} "
                    f"by {change_info['pct_change']}%"
                )
            return f"Monitor '{monitor.name}' update: {new_value[:200]}"

    async def _send_alert(self, monitor: Monitor, message: str) -> None:
        """Send an alert via available channel bots.

        Channel routing by category:
        - system : Telegram ONLY. Internal health/meta output (DB size,
                   ollama latency, consolidation, eval harness, etc.). Discord
                   has users who get confused by this output so it stays
                   behind the operator channel.
        - content: Discord + Telegram + WhatsApp + Signal (all configured).
                   News feeds, domain studies, public-facing monitor results.
        """
        prefix = f"[{monitor.name}] "
        full_message = prefix + message

        is_system = monitor.category == "system"

        sent = False
        if self._discord and not is_system:
            try:
                await self._discord.send_alert(full_message)
                sent = True
            except Exception as e:
                logger.error("[Heartbeat] Discord alert failed: %s", e)

        if self._telegram:
            try:
                await self._telegram.send_alert(full_message)
                sent = True
            except Exception as e:
                logger.error("[Heartbeat] Telegram alert failed: %s", e)

        if self._whatsapp and not is_system:
            try:
                await self._whatsapp.send_alert(full_message)
                sent = True
            except Exception as e:
                logger.error("[Heartbeat] WhatsApp alert failed: %s", e)

        if self._signal and not is_system:
            try:
                await self._signal.send_alert(full_message)
                sent = True
            except Exception as e:
                logger.error("[Heartbeat] Signal alert failed: %s", e)

        if sent:
            logger.info(
                "[Heartbeat] Alert sent for '%s' (category=%s)",
                monitor.name, monitor.category,
            )
            try:
                from app.tools.action_logging import log_action
                log_action("alert", {"monitor": monitor.name}, message[:500], True)
            except Exception:
                pass
        elif is_system and not self._telegram:
            logger.warning(
                "[Heartbeat] system-category monitor '%s' has no Telegram channel — suppressed",
                monitor.name,
            )
        elif self._discord or self._telegram or self._whatsapp or self._signal:
            logger.error("[Heartbeat] ALL notification channels failed for '%s'", monitor.name)
        else:
            logger.warning("[Heartbeat] No channels configured for alert '%s'", monitor.name)

    async def _execute_eval_harness(self, cfg: dict) -> str:
        """Run the automated eval suite and return a summary string for the monitor result."""
        if not config.ENABLE_EVAL_HARNESS:
            return "[Eval harness disabled -- set ENABLE_EVAL_HARNESS=true to enable]"

        try:
            from app.monitors.eval_harness import EvalHarness
        except ImportError as e:
            return f"[Eval harness import failed: {e}]"

        suite_path = cfg.get("suite_path") or config.EVAL_SUITE_PATH
        report_dir = cfg.get("report_dir") or config.EVAL_REPORT_PATH

        harness = EvalHarness(suite_path=suite_path, report_dir=report_dir)

        # Verify suite file exists before attempting to run
        import pathlib
        if not pathlib.Path(suite_path).exists():
            return f"[Eval suite not found: {suite_path}]"

        try:
            report, json_path, md_path = await harness.run_and_persist()
        except Exception as e:
            logger.error("[Heartbeat] Eval harness run failed: %s", e, exc_info=True)
            return f"[Eval harness run failed: {e}]"

        flagged = [r for r in report.regressions if r.flagged]
        status = "REGRESSION" if flagged else "OK"
        reg_str = ""
        if flagged:
            reg_str = " | regressions: " + ", ".join(
                f"{r.metric}({r.baseline:.2f}->{r.current:.2f})" for r in flagged
            )

        cat_summary = " | ".join(
            f"{cat}:{cm.pass_rate:.0%}"
            for cat, cm in report.categories.items()
        )

        return (
            f"EVAL {status} | "
            f"pass={report.passed}/{report.total_tasks} ({report.pass_rate:.0%}) | "
            f"duration={report.duration_seconds:.0f}s | "
            f"{cat_summary}"
            f"{reg_str} | "
            f"report={json_path.name}"
        )

    async def _execute_prompt_analyzer(self, cfg: dict) -> str:
        """Run the PromptOptimizerAnalyzer: drift detection + candidate proposals."""
        from app.monitors.prompt_optimizer_monitor import run_prompt_analyzer
        try:
            return await run_prompt_analyzer(cfg)
        except Exception as e:
            logger.error("[Heartbeat] Prompt analyzer failed: %s", e, exc_info=True)
            return f"[Prompt analyzer failed: {e}]"

    async def _execute_db_size_check(self) -> str:
        """Check SQLite database file size and table row counts."""
        from app.database import get_db
        import os

        fields: dict[str, str | int | float] = {}
        summary = "db healthy"
        status = "info"

        try:
            db_path = config.DB_PATH if hasattr(config, "DB_PATH") else "/data/nova.db"
            if os.path.exists(db_path):
                size_mb = os.path.getsize(db_path) / (1024 * 1024)
                fields["size"] = f"{size_mb:.1f}MB"
                wal_path = db_path + "-wal"
                if os.path.exists(wal_path):
                    wal_mb = os.path.getsize(wal_path) / (1024 * 1024)
                    fields["wal"] = f"{wal_mb:.1f}MB"
                if size_mb > 500:
                    status = "warning"
                    summary = f"db size elevated ({size_mb:.1f}MB)"
                else:
                    summary = f"db {size_mb:.1f}MB"
            else:
                status = "error"
                summary = f"db missing: {db_path}"
        except Exception as e:
            return format_monitor_result(
                "DB Size Monitor", "error", f"db size error: {e}",
            )

        db = get_db()
        for table in ("conversations", "messages", "lessons", "reflexions",
                      "skills", "kg_facts", "monitors"):
            try:
                row = db.fetchone(f"SELECT count(*) as c FROM {table}")
                fields[table] = row["c"]
            except Exception:
                pass

        return format_monitor_result("DB Size Monitor", status, summary, fields)

    async def _execute_ollama_latency_check(self) -> str:
        """Measure Ollama response latency with a trivial prompt."""
        import time
        try:
            from app.core import llm
            provider = llm.get_provider()
            start = time.monotonic()
            healthy = await provider.check_health()
            elapsed_ms = (time.monotonic() - start) * 1000
            if not healthy:
                status, summary = "error", f"ollama unhealthy ({elapsed_ms:.0f}ms)"
            elif elapsed_ms > 5000:
                status, summary = "error", f"ollama very slow ({elapsed_ms:.0f}ms)"
            elif elapsed_ms > 2000:
                status, summary = "warning", f"ollama slow ({elapsed_ms:.0f}ms)"
            else:
                status, summary = "ok", f"ollama healthy ({elapsed_ms:.0f}ms)"
            return format_monitor_result(
                "Ollama Latency Monitor", status, summary,
                {"latency": f"{elapsed_ms:.0f}ms"},
            )
        except Exception as e:
            return format_monitor_result(
                "Ollama Latency Monitor", "error", f"ollama error: {e}",
            )

    async def _execute_skill_quality_check(self) -> str:
        """Check skill corpus quality: success rates, disabled skills, dedup guard rate."""
        from app.core.brain import get_services

        svc = get_services()
        if not svc.skills:
            return format_monitor_result(
                "Skill Quality Monitor", "error", "skill store unavailable",
            )

        try:
            db = svc.skills._db
            total = db.fetchone("SELECT count(*) as c FROM skills")["c"]
            enabled = db.fetchone("SELECT count(*) as c FROM skills WHERE enabled = 1")["c"]
            disabled = total - enabled
            avg_row = db.fetchone("SELECT avg(success_rate) as avg_sr FROM skills WHERE enabled = 1")
            avg_sr = avg_row["avg_sr"] if avg_row and avg_row["avg_sr"] is not None else 0.0
            degrading = db.fetchone(
                "SELECT count(*) as c FROM skills WHERE enabled = 1 AND success_rate < 0.5 AND times_used >= 3"
            )["c"]
            if degrading > 5 or avg_sr < 0.4:
                status = "warning"
                summary = f"{degrading} degrading, avg {avg_sr:.2f}"
            else:
                status = "info"
                summary = f"{enabled}/{total} skills healthy"
            return format_monitor_result(
                "Skill Quality Monitor", status, summary,
                {
                    "total": total,
                    "enabled": enabled,
                    "disabled": disabled,
                    "avg_sr": f"{avg_sr:.2f}",
                    "degrading": degrading,
                },
            )
        except Exception as e:
            return format_monitor_result(
                "Skill Quality Monitor", "error", f"skill quality error: {e}",
            )

    async def _execute_chromadb_integrity_check(self) -> str:
        """Check ChromaDB collection health: doc count, collection status."""
        from app.core.brain import get_services
        from app.database import get_db

        svc = get_services()
        fields: dict[str, str | int | float] = {}
        status = "info"
        summary = "chromadb healthy"
        if svc.retriever:
            try:
                collection = svc.retriever._get_collection()
                doc_count = collection.count()
                fields["docs"] = doc_count
                summary = f"{doc_count} docs indexed"
            except Exception as e:
                status = "error"
                summary = f"chromadb error: {e}"
        else:
            status = "error"
            summary = "retriever unavailable"

        try:
            db = get_db()
            fts_row = db.fetchone("SELECT count(*) as c FROM chunks_fts")
            fields["fts5"] = fts_row["c"]
        except Exception:
            pass

        return format_monitor_result("ChromaDB Integrity", status, summary, fields)

    async def _execute_kg_health_check(self) -> str:
        """Check Knowledge Graph health: node count, edge count, fragmentation."""
        from app.core.brain import get_services

        svc = get_services()
        if not svc.kg:
            return format_monitor_result("KG Health Monitor", "error", "kg unavailable")

        try:
            stats = svc.kg.get_stats()
            fields: dict[str, str | int | float] = {
                "facts": stats.get("total_facts", 0),
                "active": stats.get("current_facts", 0),
                "superseded": stats.get("superseded_facts", 0),
            }
            db = svc.kg._db
            entities_row = db.fetchone(
                "SELECT count(DISTINCT subject) + count(DISTINCT object) as c FROM kg_facts WHERE valid_to IS NULL"
            )
            if entities_row:
                fields["entities"] = entities_row["c"]
            orphans_row = db.fetchone("""
                SELECT count(*) as c FROM (
                    SELECT subject as entity FROM kg_facts WHERE valid_to IS NULL
                    GROUP BY subject HAVING count(*) = 1
                    EXCEPT
                    SELECT object as entity FROM kg_facts WHERE valid_to IS NULL
                )
            """)
            if orphans_row:
                fields["orphans"] = orphans_row["c"]
            active = fields.get("active", 0)
            orphans = fields.get("orphans", 0)
            status = "warning" if isinstance(active, int) and active and isinstance(orphans, int) and orphans / max(active, 1) > 0.6 else "info"
            summary = f"{active} active facts"
            return format_monitor_result("KG Health Monitor", status, summary, fields)
        except Exception as e:
            return format_monitor_result(
                "KG Health Monitor", "error", f"kg health error: {e}",
            )

    async def _execute_training_job_check(self) -> str:
        """Detect a failed or stale fine-tune run.

        Reads the last entry from scripts/run_history.json (written by
        finetune_auto.py). Flags runs with status='failed' or 'rejected'.
        """
        import json as _json
        from pathlib import Path

        history_path = Path(config.FINETUNE_OUTPUT_DIR) / "run_history.json"
        if not history_path.exists():
            return format_monitor_result(
                "Training Job Watch", "info", "no training history yet",
            )

        try:
            with open(history_path, encoding="utf-8") as f:
                history = _json.load(f)
        except Exception as e:
            return format_monitor_result(
                "Training Job Watch", "error", f"history unreadable: {e}",
            )

        if not history:
            return format_monitor_result(
                "Training Job Watch", "info", "no training runs",
            )

        last = history[-1]
        status_field = (last.get("status") or "").lower()
        started = last.get("started_at") or last.get("timestamp") or ""
        pairs = last.get("training_pairs", 0)
        fields = {"last_run": started[:19], "pairs": pairs}

        if status_field in ("failed", "error"):
            return format_monitor_result(
                "Training Job Watch", "error",
                f"last fine-tune failed ({last.get('reason', 'unknown')})",
                fields,
            )
        if status_field in ("rejected",):
            return format_monitor_result(
                "Training Job Watch", "warning",
                "candidate rejected by A/B eval", fields,
            )
        return format_monitor_result(
            "Training Job Watch", "ok",
            f"last run {status_field or 'ok'}", fields,
        )

    async def _execute_kg_growth_check(self, monitor: Monitor) -> str:
        """Detect unusual spikes in KG growth over the last 6 hours."""
        from app.core.brain import get_services

        svc = get_services()
        if not svc.kg:
            return format_monitor_result(
                "KG Growth Rate", "error", "kg unavailable",
            )

        db = svc.kg._db
        try:
            last_6h = db.fetchone(
                "SELECT count(*) as c FROM kg_facts WHERE created_at > datetime('now', '-6 hours')"
            )
            prev_6h = db.fetchone(
                "SELECT count(*) as c FROM kg_facts "
                "WHERE created_at > datetime('now', '-12 hours') "
                "AND created_at <= datetime('now', '-6 hours')"
            )
        except Exception as e:
            return format_monitor_result(
                "KG Growth Rate", "error", f"kg query failed: {e}",
            )

        now_count = last_6h["c"] if last_6h else 0
        prev_count = prev_6h["c"] if prev_6h else 0
        threshold = float(monitor.check_config.get("spike_threshold_pct", 25.0))

        if prev_count == 0:
            pct = 0.0
        else:
            pct = ((now_count - prev_count) / prev_count) * 100.0

        fields = {
            "last_6h": now_count,
            "prev_6h": prev_count,
            "delta_pct": f"{pct:+.1f}%",
        }
        if abs(pct) >= threshold and prev_count >= 10:
            direction = "spike" if pct > 0 else "drop"
            return format_monitor_result(
                "KG Growth Rate", "warning",
                f"kg growth {direction} ({pct:+.1f}% over prev 6h)",
                fields,
            )
        return format_monitor_result(
            "KG Growth Rate", "info",
            f"kg growth normal ({pct:+.1f}%)", fields,
        )

    async def _execute_ollama_model_check(self) -> str:
        """Verify the configured LLM model is actually loaded in Ollama."""
        import httpx

        model_name = getattr(config, "LLM_MODEL", None) or "qwen3.5:27b"
        ollama_url = getattr(config, "OLLAMA_URL", None) or "http://localhost:11434"
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{ollama_url}/api/tags")
                resp.raise_for_status()
                payload = resp.json()
        except Exception as e:
            return format_monitor_result(
                "Ollama Model Loaded", "error", f"ollama unreachable: {e}",
            )

        names = {m.get("name", "") for m in payload.get("models", [])}
        base = model_name.split(":")[0]
        found = any(n == model_name or n.startswith(base + ":") for n in names)
        fields = {"expected": model_name, "total_models": len(names)}
        if not found:
            return format_monitor_result(
                "Ollama Model Loaded", "error",
                f"model {model_name} not loaded", fields,
            )
        return format_monitor_result(
            "Ollama Model Loaded", "ok",
            f"model {model_name} loaded", fields,
        )

    async def trigger_monitor(self, monitor_id: int) -> dict:
        """Manually trigger a monitor check. Returns result info."""
        monitor = self.store.get(monitor_id)
        if not monitor:
            return {"error": "Monitor not found"}

        try:
            await self._check_monitor(monitor)
            # Get the latest result
            results = self.store.get_results(monitor_id, limit=1)
            if results:
                r = results[0]
                return {"status": r.status, "value": r.value, "message": r.message}
            return {"status": "ok", "message": "Check completed"}
        except Exception as e:
            return {"error": str(e)}
