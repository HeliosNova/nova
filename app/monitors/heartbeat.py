"""Heartbeat system — monitors, change detection, and proactive alerting.

Runs a background loop that checks monitors on schedule,
detects changes, asks brain.think() to analyze them, and
delivers alerts via channel bots.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from app.config import config
from app.database import SafeDB

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class Monitor:
    id: int
    name: str
    check_type: str          # 'url', 'search', 'command', 'system_health', 'query', 'quiz', 'skill_test', 'curiosity', 'auto_monitor', 'maintenance', 'finetune'
    check_config: dict       # JSON parsed: {url, query, command, threshold_pct}
    schedule_seconds: int
    enabled: bool
    cooldown_minutes: int
    notify_condition: str    # 'always', 'on_change', 'on_alert'
    last_check_at: str | None
    last_alert_at: str | None
    last_result: str | None
    created_at: str


@dataclass
class MonitorResult:
    id: int
    monitor_id: int
    status: str              # 'ok', 'changed', 'alert', 'error'
    value: str | None
    message: str | None
    created_at: str
    user_rating: int = 0     # -1 (bad), 0 (neutral), 1 (good)


@dataclass
class HeartbeatInstruction:
    id: int
    instruction: str
    schedule_seconds: int
    enabled: bool
    last_run_at: str | None
    notify_channels: str
    created_at: str


# ---------------------------------------------------------------------------
# MonitorStore — CRUD for monitors + results + heartbeat instructions
# ---------------------------------------------------------------------------

class MonitorStore:
    def __init__(self, db: SafeDB):
        self._db = db

    def create(
        self,
        name: str,
        check_type: str,
        check_config: dict,
        schedule_seconds: int = 300,
        cooldown_minutes: int = 60,
        notify_condition: str = "on_change",
    ) -> int:
        """Create a monitor. Returns its ID, or -1 if name exists."""
        try:
            cursor = self._db.execute(
                """INSERT INTO monitors (name, check_type, check_config, schedule_seconds,
                   cooldown_minutes, notify_condition)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (name, check_type, json.dumps(check_config), schedule_seconds,
                 cooldown_minutes, notify_condition),
            )
            return cursor.lastrowid
        except Exception as e:
            logger.warning("Monitor create failed: %s", e)
            return -1

    def get(self, monitor_id: int) -> Monitor | None:
        row = self._db.fetchone("SELECT * FROM monitors WHERE id = ?", (monitor_id,))
        return self._row_to_monitor(row) if row else None

    def get_by_name(self, name: str) -> Monitor | None:
        row = self._db.fetchone("SELECT * FROM monitors WHERE name = ?", (name,))
        return self._row_to_monitor(row) if row else None

    def list_all(self) -> list[Monitor]:
        rows = self._db.fetchall("SELECT * FROM monitors ORDER BY id")
        return [self._row_to_monitor(r) for r in rows]

    def update(self, monitor_id: int, **kwargs) -> bool:
        allowed = {"name", "check_type", "check_config", "schedule_seconds",
                    "enabled", "cooldown_minutes", "notify_condition"}
        updates = {k: v for k, v in kwargs.items() if k in allowed}
        if not updates:
            return False
        if "check_config" in updates and isinstance(updates["check_config"], dict):
            updates["check_config"] = json.dumps(updates["check_config"])
        if "enabled" in updates:
            updates["enabled"] = 1 if updates["enabled"] else 0
        sets = ", ".join(f"{k} = ?" for k in updates)
        vals = list(updates.values()) + [monitor_id]
        self._db.execute(f"UPDATE monitors SET {sets} WHERE id = ?", tuple(vals))
        return True

    def delete(self, monitor_id: int) -> bool:
        self._db.execute("DELETE FROM monitors WHERE id = ?", (monitor_id,))
        return True

    def get_due(self) -> list[Monitor]:
        """Return enabled monitors that are due for a check."""
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        monitors = self.list_all()
        due = []
        for m in monitors:
            if not m.enabled:
                continue
            if m.last_check_at:
                last = datetime.fromisoformat(m.last_check_at).replace(tzinfo=None)
                if (now - last).total_seconds() < m.schedule_seconds:
                    continue
            due.append(m)
        return due

    def record_check(self, monitor_id: int, result: str) -> None:
        """Update last_check_at and last_result."""
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        self._db.execute(
            "UPDATE monitors SET last_check_at = ?, last_result = ? WHERE id = ?",
            (now, result[:2000] if result else "", monitor_id),
        )

    def record_alert(self, monitor_id: int) -> None:
        """Update last_alert_at."""
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        self._db.execute(
            "UPDATE monitors SET last_alert_at = ? WHERE id = ?",
            (now, monitor_id),
        )

    def add_result(self, monitor_id: int, status: str, value: str = "", message: str = "") -> int:
        """Store a monitor result. Returns its ID."""
        cursor = self._db.execute(
            "INSERT INTO monitor_results (monitor_id, status, value, message) VALUES (?, ?, ?, ?)",
            (monitor_id, status, value[:2000] if value else "", message[:2000] if message else ""),
        )
        return cursor.lastrowid

    def get_results(self, monitor_id: int, limit: int = 20) -> list[MonitorResult]:
        rows = self._db.fetchall(
            "SELECT * FROM monitor_results WHERE monitor_id = ? ORDER BY created_at DESC LIMIT ?",
            (monitor_id, limit),
        )
        return [self._row_to_result(r) for r in rows]

    def get_recent_results(self, hours: int = 24, limit: int = 50) -> list[MonitorResult]:
        # Use SQLite's datetime for consistent comparison with datetime('now') defaults
        rows = self._db.fetchall(
            "SELECT * FROM monitor_results WHERE created_at > datetime('now', ?) ORDER BY created_at DESC LIMIT ?",
            (f"-{hours} hours", limit),
        )
        return [self._row_to_result(r) for r in rows]

    def _row_to_monitor(self, row) -> Monitor:
        cfg = row["check_config"]
        try:
            parsed = json.loads(cfg) if isinstance(cfg, str) else cfg
        except (json.JSONDecodeError, TypeError):
            parsed = {}
        return Monitor(
            id=row["id"],
            name=row["name"],
            check_type=row["check_type"],
            check_config=parsed,
            schedule_seconds=row["schedule_seconds"],
            enabled=bool(row["enabled"]),
            cooldown_minutes=row["cooldown_minutes"],
            notify_condition=row["notify_condition"],
            last_check_at=row["last_check_at"],
            last_alert_at=row["last_alert_at"],
            last_result=row["last_result"],
            created_at=row["created_at"],
        )

    def _row_to_result(self, row) -> MonitorResult:
        # user_rating may not exist in old databases before migration
        try:
            rating = row["user_rating"] if "user_rating" in row.keys() else 0
        except (KeyError, TypeError):
            rating = 0
        return MonitorResult(
            id=row["id"],
            monitor_id=row["monitor_id"],
            status=row["status"],
            value=row["value"],
            message=row["message"],
            created_at=row["created_at"],
            user_rating=rating,
        )

    def rate_result(self, result_id: int, rating: int) -> bool:
        """Rate a monitor result (-1, 0, or 1). Returns True on success."""
        if rating not in (-1, 0, 1):
            return False
        cursor = self._db.execute(
            "UPDATE monitor_results SET user_rating = ? WHERE id = ?",
            (rating, result_id),
        )
        return cursor.rowcount > 0

    def adapt_cooldown(self, monitor_id: int) -> int | None:
        """Auto-adjust cooldown based on recent ratings.

        3+ negative ratings on recent results → double cooldown.
        3+ positive ratings → halve cooldown.
        Returns new cooldown or None if no change.
        """
        recent = self._db.fetchall(
            "SELECT user_rating FROM monitor_results "
            "WHERE monitor_id = ? AND user_rating != 0 "
            "ORDER BY created_at DESC LIMIT 10",
            (monitor_id,),
        )
        if len(recent) < 3:
            return None

        negatives = sum(1 for r in recent if r["user_rating"] == -1)
        positives = sum(1 for r in recent if r["user_rating"] == 1)

        monitor = self.get(monitor_id)
        if not monitor:
            return None

        new_cooldown = monitor.cooldown_minutes
        if negatives >= 3 and negatives > positives:
            new_cooldown = min(monitor.cooldown_minutes * 2, 1440)  # Max 24h
        elif positives >= 3 and positives > negatives:
            new_cooldown = max(monitor.cooldown_minutes // 2, 5)  # Min 5min

        if new_cooldown != monitor.cooldown_minutes:
            self.update(monitor_id, cooldown_minutes=new_cooldown)
            logger.info(
                "Auto-adapted monitor %d cooldown: %d → %d min (neg=%d, pos=%d)",
                monitor_id, monitor.cooldown_minutes, new_cooldown, negatives, positives,
            )
            return new_cooldown
        return None

    # --- Seed monitors ---

    def seed_defaults(self) -> int:
        """Create default seed monitors, skipping any that already exist by name."""
        existing_names = {m.name for m in self.list_all()}

        seeds = [
            {
                "name": "Morning Check-in",
                "check_type": "query",
                "check_config": {
                    "query": (
                        "Good morning. Using the system context above, give a brief status: "
                        "monitor health, any notable alerts from overnight, recent learning "
                        "activity, and one interesting thing about today's date."
                    ),
                },
                "schedule_seconds": 86400,  # daily
                "cooldown_minutes": 1380,   # 23 hours
                "notify_condition": "always",
            },
            {
                "name": "System Health",
                "check_type": "system_health",
                "check_config": {
                    "threshold_pct": 10,
                },
                "schedule_seconds": 7200,   # every 2 hours
                "cooldown_minutes": 120,
                "notify_condition": "on_change",
            },
            {
                "name": "World Awareness",
                "check_type": "query",
                "check_config": {
                    "query": (
                        "Use web_search to find major global news today (politics, "
                        "environment, health, culture — NOT technology/AI, that's covered "
                        "by Domain Study: Technology). Summarize the top 2-3 developments. "
                        "Don't just list links — explain why each matters."
                    ),
                },
                "schedule_seconds": 14400,  # every 4 hours
                "cooldown_minutes": 240,
                "notify_condition": "on_change",
            },
            {
                "name": "Self-Reflection",
                "check_type": "query",
                "check_config": {
                    "query": (
                        "Review the conversations and learning activity shown in the system "
                        "context. What did you learn from any corrections? What new facts did "
                        "you discover about your owner? Are any skills degrading? "
                        "Summarize in 2-3 sentences."
                    ),
                },
                "schedule_seconds": 86400,  # daily
                "cooldown_minutes": 1380,
                "notify_condition": "always",
            },
            # --- Teaching monitors ---
            {
                "name": "Domain Study: Science",
                "check_type": "query",
                "check_config": {
                    "query": (
                        "Use web_search to find a recent science discovery from the past week. "
                        "Summarize what was discovered, why it matters, and one key takeaway."
                    ),
                },
                "schedule_seconds": 43200,  # 12h
                "cooldown_minutes": 660,
                "notify_condition": "always",
            },
            {
                "name": "Domain Study: Technology",
                "check_type": "query",
                "check_config": {
                    "query": (
                        "Use web_search to learn about a new programming tool, framework, or "
                        "AI model released recently. Summarize what it does and why it's notable."
                    ),
                },
                "schedule_seconds": 43200,  # 12h
                "cooldown_minutes": 660,
                "notify_condition": "always",
            },
            {
                "name": "Domain Study: Current Events",
                "check_type": "query",
                "check_config": {
                    "query": (
                        "Use web_search to find and summarize one significant world event "
                        "from today. Include who, what, where, and why it matters."
                    ),
                },
                "schedule_seconds": 28800,  # 8h
                "cooldown_minutes": 420,
                "notify_condition": "always",
            },
            {
                "name": "Domain Study: Finance",
                "check_type": "query",
                "check_config": {
                    "query": (
                        "Use web_search to check current market trends, notable crypto movements, "
                        "or economic news. Summarize the most important development."
                    ),
                },
                "schedule_seconds": 43200,  # 12h
                "cooldown_minutes": 660,
                "notify_condition": "always",
            },
            {
                "name": "Lesson Quiz",
                "check_type": "quiz",
                "check_config": {},
                "schedule_seconds": 21600,  # 6h
                "cooldown_minutes": 300,
                "notify_condition": "always",
            },
            {
                "name": "Skill Validation",
                "check_type": "skill_test",
                "check_config": {},
                "schedule_seconds": 43200,  # 12h
                "cooldown_minutes": 660,
                "notify_condition": "always",
            },
            {
                "name": "Curiosity Research",
                "check_type": "curiosity",
                "check_config": {},
                "schedule_seconds": 3600,  # 1h
                "cooldown_minutes": 55,
                "notify_condition": "on_change",
            },
            {
                "name": "Auto-Monitor Detector",
                "check_type": "auto_monitor",
                "check_config": {},
                "schedule_seconds": 86400,  # daily
                "cooldown_minutes": 1380,
                "notify_condition": "on_change",
            },
            {
                "name": "System Maintenance",
                "check_type": "maintenance",
                "check_config": {},
                "schedule_seconds": 86400,  # daily
                "cooldown_minutes": 1380,
                "notify_condition": "on_change",
            },
            {
                "name": "Fine-Tune Check",
                "check_type": "finetune",
                "check_config": {},
                "schedule_seconds": 604800,  # weekly
                "cooldown_minutes": 10000,   # ~7 days
                "notify_condition": "on_change",
            },
        ]

        count = 0
        for seed in seeds:
            if seed["name"] in existing_names:
                continue
            mid = self.create(**seed)
            if mid > 0:
                count += 1
        return count

    # --- Heartbeat Instructions CRUD ---

    def create_instruction(self, instruction: str, schedule_seconds: int = 3600,
                           notify_channels: str = "discord,telegram") -> int:
        cursor = self._db.execute(
            "INSERT INTO heartbeat_instructions (instruction, schedule_seconds, notify_channels) VALUES (?, ?, ?)",
            (instruction, schedule_seconds, notify_channels),
        )
        return cursor.lastrowid

    def get_instruction(self, instruction_id: int) -> HeartbeatInstruction | None:
        row = self._db.fetchone("SELECT * FROM heartbeat_instructions WHERE id = ?", (instruction_id,))
        return self._row_to_instruction(row) if row else None

    def list_instructions(self) -> list[HeartbeatInstruction]:
        rows = self._db.fetchall("SELECT * FROM heartbeat_instructions ORDER BY id")
        return [self._row_to_instruction(r) for r in rows]

    def update_instruction(self, instruction_id: int, **kwargs) -> bool:
        allowed = {"instruction", "schedule_seconds", "enabled", "notify_channels"}
        updates = {k: v for k, v in kwargs.items() if k in allowed}
        if not updates:
            return False
        if "enabled" in updates:
            updates["enabled"] = 1 if updates["enabled"] else 0
        sets = ", ".join(f"{k} = ?" for k in updates)
        vals = list(updates.values()) + [instruction_id]
        self._db.execute(f"UPDATE heartbeat_instructions SET {sets} WHERE id = ?", tuple(vals))
        return True

    def delete_instruction(self, instruction_id: int) -> bool:
        self._db.execute("DELETE FROM heartbeat_instructions WHERE id = ?", (instruction_id,))
        return True

    def get_due_instructions(self) -> list[HeartbeatInstruction]:
        """Return enabled instructions that are due for execution."""
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        instructions = self.list_instructions()
        due = []
        for inst in instructions:
            if not inst.enabled:
                continue
            if inst.last_run_at:
                last = datetime.fromisoformat(inst.last_run_at).replace(tzinfo=None)
                if (now - last).total_seconds() < inst.schedule_seconds:
                    continue
            due.append(inst)
        return due

    def record_instruction_run(self, instruction_id: int) -> None:
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        self._db.execute(
            "UPDATE heartbeat_instructions SET last_run_at = ? WHERE id = ?",
            (now, instruction_id),
        )

    def _row_to_instruction(self, row) -> HeartbeatInstruction:
        return HeartbeatInstruction(
            id=row["id"],
            instruction=row["instruction"],
            schedule_seconds=row["schedule_seconds"],
            enabled=bool(row["enabled"]),
            last_run_at=row["last_run_at"],
            notify_channels=row["notify_channels"],
            created_at=row["created_at"],
        )


# ---------------------------------------------------------------------------
# Change detection
# ---------------------------------------------------------------------------

_NUMBER_RE = re.compile(r"[\$€£¥]?\s*(-?\d[\d,]*\.?\d*)\s*(%|[KMBTkmbt](?![a-zA-Z]))?")


def extract_numbers(text: str) -> list[float]:
    """Extract significant numbers from text (prices, percentages, etc.)."""
    numbers = []
    for match in _NUMBER_RE.finditer(text):
        raw = match.group(1).replace(",", "")
        try:
            val = float(raw)
            suffix = match.group(2)
            if suffix:
                s = suffix.upper()
                if s == "K": val *= 1000
                elif s == "M": val *= 1_000_000
                elif s == "B": val *= 1_000_000_000
                elif s == "T": val *= 1_000_000_000_000
            numbers.append(val)
        except ValueError:
            continue
    return numbers


def detect_change(old_value: str, new_value: str, threshold_pct: float = 5.0) -> dict | None:
    """Compare old and new values. Returns change info or None if no significant change.

    Tries numeric comparison first, falls back to text equality.
    """
    if not old_value or not new_value:
        return None

    old_value = old_value.strip()
    new_value = new_value.strip()

    # Numeric comparison
    old_nums = extract_numbers(old_value)
    new_nums = extract_numbers(new_value)

    if old_nums and new_nums:
        old_n = old_nums[0]
        new_n = new_nums[0]
        if old_n == 0:
            # Zero-crossing: report absolute change instead of percentage
            if new_n != 0:
                direction = "up" if new_n > 0 else "down"
                return {
                    "type": "numeric",
                    "old": old_n,
                    "new": new_n,
                    "pct_change": 100.0,
                    "direction": direction,
                }
        else:
            pct_change = abs(new_n - old_n) / abs(old_n) * 100
            if pct_change >= threshold_pct:
                direction = "up" if new_n > old_n else "down"
                return {
                    "type": "numeric",
                    "old": old_n,
                    "new": new_n,
                    "pct_change": round(pct_change, 2),
                    "direction": direction,
                }
        return None  # Numbers present but didn't change enough

    # Text comparison — Jaccard similarity on normalized words
    from app.core.text_utils import normalize_words
    old_words = normalize_words(old_value, min_length=2)
    new_words = normalize_words(new_value, min_length=2)

    if not old_words and not new_words:
        return None  # Both empty after normalization
    if not old_words or not new_words:
        # One is empty — treat as major change
        return {
            "type": "text",
            "changed": True,
            "severity": "major",
            "old_len": len(old_value),
            "new_len": len(new_value),
        }

    intersection = len(old_words & new_words)
    union = len(old_words | new_words)
    similarity = intersection / union if union > 0 else 1.0

    if similarity > 0.8:
        return None  # Same content, just reworded
    severity = "minor" if similarity >= 0.3 else "major"
    return {
        "type": "text",
        "changed": True,
        "severity": severity,
        "similarity": round(similarity, 2),
        "old_len": len(old_value),
        "new_len": len(new_value),
    }


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
        # Small delay on startup to let services initialize
        await asyncio.sleep(10)

        while self._running:
            try:
                due = self.store.get_due()
                if due:
                    logger.info("[Heartbeat] %d monitor(s) due", len(due))
                    for monitor in due:
                        try:
                            await self._check_monitor(monitor)
                        except Exception as e:
                            logger.error("[Heartbeat] Monitor '%s' failed: %s", monitor.name, e)
                            self.store.record_check(monitor.id, f"error: {e}")
                            self.store.add_result(monitor.id, "error", message=str(e))

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

    async def _check_monitor(self, monitor: Monitor) -> None:
        """Execute a single monitor check."""
        logger.info("[Heartbeat] Checking '%s' (type=%s)", monitor.name, monitor.check_type)

        # Execute the check
        new_value = await self._execute_check(monitor)

        # Record the check
        self.store.record_check(monitor.id, new_value)

        # Extract KG triples from Domain Study monitors only (not all query types)
        if monitor.check_type == "query" and monitor.name.startswith("Domain Study") and new_value and len(new_value) > 100:
            try:
                from app.core.brain import get_services, _extract_kg_triples
                svc = get_services()
                if svc.kg:
                    asyncio.create_task(_extract_kg_triples(svc.kg, monitor.name, new_value[:2000]))
            except Exception:
                pass

        # Determine if we should alert
        should_alert = False
        change_info = None

        if monitor.notify_condition == "always":
            should_alert = True
        elif monitor.notify_condition in ("on_change", "on_alert"):
            if monitor.last_result:
                threshold = monitor.check_config.get("threshold_pct", 5.0)
                change_info = detect_change(monitor.last_result, new_value, threshold)
                should_alert = change_info is not None
            else:
                # First check — always alert
                should_alert = True

        if not should_alert:
            self.store.add_result(monitor.id, "ok", value=new_value[:500] if new_value else "")
            return

        # Check cooldown
        if monitor.last_alert_at:
            last_alert = datetime.fromisoformat(monitor.last_alert_at).replace(tzinfo=None)
            now_naive = datetime.now(timezone.utc).replace(tzinfo=None)
            if (now_naive - last_alert).total_seconds() < monitor.cooldown_minutes * 60:
                logger.info("[Heartbeat] '%s' in cooldown, skipping alert", monitor.name)
                self.store.add_result(monitor.id, "ok", value=new_value[:500] if new_value else "",
                                      message="in cooldown")
                return

        # Generate intelligent analysis via brain.think()
        analysis = await self._analyze_result(monitor, new_value, change_info)

        # Send alert
        await self._send_alert(monitor, analysis)

        # Record
        status = "changed" if change_info else "ok"
        if change_info and change_info.get("type") == "numeric":
            status = "alert"
        self.store.record_alert(monitor.id)
        self.store.add_result(monitor.id, status, value=new_value[:500] if new_value else "",
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

        return f"[Unknown check_type: {monitor.check_type}]"

    async def _execute_system_health(self) -> str:
        """Gather system health using Python stdlib — no shell commands needed."""
        import os
        import platform

        lines: list[str] = []

        # Disk usage
        try:
            stat = os.statvfs("/")
            total_gb = (stat.f_frsize * stat.f_blocks) / (1024 ** 3)
            avail_gb = (stat.f_frsize * stat.f_bavail) / (1024 ** 3)
            used_gb = total_gb - avail_gb
            used_pct = (used_gb / total_gb * 100) if total_gb else 0
            lines.append(f"Disk: {used_gb:.1f}G / {total_gb:.1f}G ({used_pct:.0f}% used, {avail_gb:.1f}G free)")
        except (OSError, AttributeError):
            # os.statvfs not available on Windows
            lines.append("Disk: unavailable")

        # Load average
        try:
            load1, load5, load15 = os.getloadavg()
            lines.append(f"Load: {load1:.2f} {load5:.2f} {load15:.2f}")
        except (OSError, AttributeError):
            lines.append("Load: unavailable")

        # Memory usage via psutil (graceful fallback)
        try:
            import psutil
            mem = psutil.virtual_memory()
            total_gb = mem.total / (1024 ** 3)
            used_gb = mem.used / (1024 ** 3)
            lines.append(f"Memory: {used_gb:.1f}G / {total_gb:.1f}G ({mem.percent}% used)")
        except ImportError:
            # psutil not installed — try /proc/meminfo (Linux)
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

        # Uptime
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

            # Learning summary
            if svc.learning:
                summary = svc.learning.get_learning_summary(hours=24)
                parts = []
                if summary.get("new_lessons"):
                    parts.append(f"{len(summary['new_lessons'])} new lesson(s)")
                if summary.get("new_skills"):
                    parts.append(f"{len(summary['new_skills'])} new skill(s)")
                if summary.get("degraded_skills"):
                    parts.append(f"{len(summary['degraded_skills'])} degraded skill(s)")
                if summary.get("new_reflexions"):
                    parts.append(f"{len(summary['new_reflexions'])} new reflexion(s)")
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

        # Prepend context to query
        if ctx_lines:
            context_block = "=== System Context ===\n" + "\n".join(ctx_lines) + "\n=== End Context ===\n\n"
            enriched_query = context_block + query
        else:
            enriched_query = query

        tokens = []
        try:
            async for event in think(query=enriched_query, ephemeral=True):
                if event.type == EventType.TOKEN:
                    text = event.data.get("text", "")
                    if text:
                        tokens.append(text)
        except Exception as e:
            logger.error("[Heartbeat] think() failed: %s", e)
            return f"[Query failed: {e}]"

        return "".join(tokens).strip()

    async def _execute_instruction(self, inst: HeartbeatInstruction) -> None:
        """Execute a user-defined heartbeat instruction via brain.think()."""
        from app.core.brain import think, get_services
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
        from app.core.learning import Correction

        svc = get_services()
        if not svc.learning:
            return "[No learning engine — quiz skipped]"

        lessons = svc.learning.get_all_lessons(limit=200)
        if not lessons:
            return "[No lessons to quiz on — skipped]"

        # Spaced repetition: prioritize by failures (desc) then last_quizzed_at (asc/null first)
        db = svc.learning._db
        lesson = None
        row = db.fetchone(
            "SELECT id FROM lessons "
            "ORDER BY quiz_failures DESC, last_quizzed_at ASC NULLS FIRST "
            "LIMIT 1"
        )
        if row:
            lesson = next((l for l in lessons if l.id == row["id"]), None)
        if not lesson:
            lesson = random.choice(lessons)

        # Step 1: Generate a question from the lesson
        gen_prompt = (
            f"Topic: {lesson.topic}\n"
            f"Context: {lesson.context or lesson.lesson_text or ''}\n\n"
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

        # Step 2: Answer WITHOUT the lesson injected
        try:
            answer = await llm.invoke_nothink(
                [{"role": "user", "content": question}],
                max_tokens=300, temperature=0.3,
            )
            answer = answer.strip()
        except Exception as e:
            return f"[Quiz answer generation failed: {e}]"

        # Step 3: Grade the answer against the correct answer
        grade_prompt = (
            f"Question: {question}\n"
            f"Expected answer: {lesson.correct_answer}\n"
            f"Given answer: {answer}\n\n"
            'Is the given answer correct? Respond with a single JSON object: {{"pass": true}} or {{"pass": false, "reason": "brief explanation"}}. Keep the reason under 20 words.'
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

        # Failed — create correction, training pair, reflexion
        fail_reason = grade.get("reason", "incorrect")
        correction = Correction(
            user_message=f"Quiz self-test on: {lesson.topic}",
            previous_answer=answer,
            topic=lesson.topic,
            correct_answer=lesson.correct_answer,
            wrong_answer=answer,
            original_query=question,
            lesson_text=f"Quiz failure: {fail_reason}",
        )

        try:
            svc.learning.save_lesson(correction)
        except Exception as e:
            logger.warning("[Heartbeat] Quiz save_lesson failed: %s", e)

        try:
            await svc.learning.save_training_pair(
                query=question,
                bad_answer=answer,
                good_answer=lesson.correct_answer,
            )
        except Exception as e:
            logger.warning("[Heartbeat] Quiz save_training_pair failed: %s", e)

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

        # Generate a test query that matches the skill's trigger pattern
        gen_prompt = (
            f"Skill name: {skill.name}\n"
            f"Trigger pattern: {skill.trigger_pattern}\n\n"
            "Write a single, natural user query that would match this trigger pattern. "
            "Just the query, nothing else."
        )
        try:
            test_query = await llm.invoke_nothink(
                [{"role": "user", "content": gen_prompt}],
                max_tokens=100, temperature=0.5,
            )
            test_query = test_query.strip()
        except Exception as e:
            return f"[Skill test query generation failed: {e}]"

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
                        if obj and obj.get("lesson"):
                            from app.core.learning import Correction
                            correction = Correction(
                                user_message=f"Curiosity research on: {item.topic[:100]}",
                                previous_answer="",
                                topic=obj.get("topic", item.topic[:100]),
                                wrong_answer="",
                                correct_answer=obj["lesson"],
                                lesson_text=obj["lesson"],
                            )
                            svc.learning.save_lesson(correction)
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

            mid = self.store.create(
                name=monitor_name,
                check_type="search",
                check_config={"query": topic},
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
            decayed = svc.learning.decay_stale_lessons(days=30)
            if decayed:
                parts.append(f"lessons decayed: {decayed}")
        if svc.kg:
            decayed = svc.kg.decay_stale(days=60)
            if decayed:
                parts.append(f"KG facts decayed: {decayed}")
        if svc.reflexions:
            decayed = svc.reflexions.decay_stale(days=90)
            if decayed:
                parts.append(f"reflexions decayed: {decayed}")
        if svc.curiosity:
            pruned = svc.curiosity.prune(days=30)
            if pruned:
                parts.append(f"curiosity items pruned: {pruned}")
        return f"MAINTENANCE | {', '.join(parts)}" if parts else "[No maintenance needed]"

    async def _execute_finetune_check(self, cfg: dict) -> str:
        """Check if enough new training pairs exist for fine-tuning.

        Reports readiness status — does NOT auto-trigger training.
        The heartbeat alert notifies the user so they can trigger manually.
        """
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
                        entry = json.loads(line)
                        if entry.get("query", "").strip() and entry.get("chosen", "").strip():
                            total += 1
                    except json.JSONDecodeError:
                        continue

        # Check last training run count
        history_path = Path(output_dir) / "run_history.json"
        last_count = 0
        if history_path.exists():
            try:
                with open(history_path, encoding="utf-8") as f:
                    history = json.load(f)
                if history:
                    last_count = history[-1].get("training_pairs", 0)
            except (json.JSONDecodeError, OSError):
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

        if monitor.last_result:
            parts.append(f"Previous result:\n{monitor.last_result[:400]}")

        parts.append(
            "Write a concise, friendly 1-3 sentence alert message for the user. "
            "Be specific about what changed and why it matters. "
            "Sound natural, like a thoughtful assistant noticing something."
        )

        try:
            analysis = await llm.invoke_nothink(
                [{"role": "user", "content": "\n\n".join(parts)}],
                max_tokens=200,
                temperature=0.4,
            )
            return analysis.strip()
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
        """Send an alert via available channel bots."""
        prefix = f"[{monitor.name}] "
        full_message = prefix + message

        sent = False
        if self._discord:
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

        if self._whatsapp:
            try:
                await self._whatsapp.send_alert(full_message)
                sent = True
            except Exception as e:
                logger.error("[Heartbeat] WhatsApp alert failed: %s", e)

        if self._signal:
            try:
                await self._signal.send_alert(full_message)
                sent = True
            except Exception as e:
                logger.error("[Heartbeat] Signal alert failed: %s", e)

        if sent:
            logger.info("[Heartbeat] Alert sent for '%s'", monitor.name)
        else:
            logger.warning("[Heartbeat] No channels available for alert '%s'", monitor.name)

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
