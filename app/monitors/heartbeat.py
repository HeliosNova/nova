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

_MAX_CONCURRENT_LLM_MONITORS = 2

# Defense-in-depth: validate column names before SQL interpolation
_VALID_COLUMN_RE = re.compile(r'^[a-z_][a-z0-9_]*$')

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
                    "enabled", "cooldown_minutes", "notify_condition", "last_check_at"}
        updates = {k: v for k, v in kwargs.items() if k in allowed}
        if not updates:
            return False
        for col in updates:
            if not _VALID_COLUMN_RE.match(col):
                raise ValueError(f"Invalid column name: {col!r}")
        if "check_config" in updates and isinstance(updates["check_config"], dict):
            updates["check_config"] = json.dumps(updates["check_config"])
        if "enabled" in updates:
            updates["enabled"] = 1 if updates["enabled"] else 0
        sets = ", ".join(f"{k} = ?" for k in updates)
        vals = list(updates.values()) + [monitor_id]
        self._db.execute(f"UPDATE monitors SET {sets} WHERE id = ?", tuple(vals))
        return True

    def delete(self, monitor_id: int) -> bool:
        cursor = self._db.execute("DELETE FROM monitors WHERE id = ?", (monitor_id,))
        return cursor.rowcount > 0

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
            (now, result[:4000] if result else "", monitor_id),
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
            (monitor_id, status, value[:4000] if value else "", message[:4000] if message else ""),
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
                        "Use web_search to find major global news from TODAY (politics, "
                        "environment, health, culture — NOT technology/AI, that's covered "
                        "by Domain Study: Technology). Summarize the top 2-3 developments "
                        "from the past 24 hours. Include specific dates. "
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
                        "Use web_search to find 3 science discoveries or developments "
                        "from the past 24-48 hours. For each, give one bullet: what was discovered, "
                        "the date it was reported, and why it matters. Use this format:\n"
                        "• Discovery 1: ...\n• Discovery 2: ...\n• Discovery 3: ..."
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
                        "Use web_search to find 3 notable new programming tools, frameworks, "
                        "or AI models released in the past 24-48 hours. For each, give one bullet: "
                        "what it does, when it was released, and why it's notable. Use this format:\n"
                        "• Tool 1: ...\n• Tool 2: ...\n• Tool 3: ..."
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
                        "Use web_search to find and summarize 3 significant world events from TODAY. "
                        "Only report events from the past 24 hours with specific dates. "
                        "For each: who, what, where, when, why it matters. Use this format:\n"
                        "• Event 1: ...\n• Event 2: ...\n• Event 3: ..."
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
                        "Use web_search to check TODAY's market trends, notable crypto movements, "
                        "and economic news from the past 24 hours. Include specific prices and dates. "
                        "Summarize the top 3 developments. Use this format:\n"
                        "• Market 1: ...\n• Market 2: ...\n• Market 3: ..."
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
                "notify_condition": "on_change",
            },
            {
                "name": "Skill Validation",
                "check_type": "skill_test",
                "check_config": {},
                "schedule_seconds": 43200,  # 12h
                "cooldown_minutes": 660,
                "notify_condition": "on_change",
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
            # --- Expanded Domain Studies (all prompts anchored to TODAY) ---
            {"name": "Domain Study: AI and ML", "check_type": "query", "schedule_seconds": 28800, "cooldown_minutes": 420, "notify_condition": "always",
             "check_config": {"query": "Use web_search to find 3 notable AI/ML developments from TODAY or the past 24-48 hours: new model releases, research breakthroughs, benchmark results, or major company announcements. For each: what happened, who did it, the date, and why it matters.\n• Development 1: ...\n• Development 2: ...\n• Development 3: ..."}},
            {"name": "Domain Study: Space and Astronomy", "check_type": "query", "schedule_seconds": 43200, "cooldown_minutes": 660, "notify_condition": "always",
             "check_config": {"query": "Use web_search to find 2-3 space and astronomy developments from the past 24-48 hours: rocket launches, satellite deployments, exoplanet discoveries, NASA/ESA/SpaceX missions. Include dates.\n• Development 1: ...\n• Development 2: ...\n• Development 3: ..."}},
            {"name": "Domain Study: Health and Medicine", "check_type": "query", "schedule_seconds": 43200, "cooldown_minutes": 660, "notify_condition": "always",
             "check_config": {"query": "Use web_search to find 2-3 notable health and medical developments from the past 24-48 hours: drug approvals, clinical trial results, disease outbreaks, public health policy, or medical technology breakthroughs. Include dates.\n• Development 1: ...\n• Development 2: ...\n• Development 3: ..."}},
            {"name": "Domain Study: Energy and Climate", "check_type": "query", "schedule_seconds": 43200, "cooldown_minutes": 660, "notify_condition": "always",
             "check_config": {"query": "Use web_search to find 2-3 energy and climate developments from the past 24-48 hours: renewable energy milestones, climate policy changes, emissions data, battery technology, nuclear energy. Include dates.\n• Development 1: ...\n• Development 2: ...\n• Development 3: ..."}},
            {"name": "Domain Study: Cybersecurity", "check_type": "query", "schedule_seconds": 43200, "cooldown_minutes": 660, "notify_condition": "always",
             "check_config": {"query": "Use web_search to find 2-3 cybersecurity developments from the past 24-48 hours: major breaches, new CVEs, ransomware attacks, security tool releases, or policy changes. Include dates and affected entities.\n• Incident 1: ...\n• Incident 2: ...\n• Incident 3: ..."}},
            {"name": "Domain Study: Geopolitics", "check_type": "query", "schedule_seconds": 28800, "cooldown_minutes": 420, "notify_condition": "always",
             "check_config": {"query": "Use web_search to find 2-3 significant geopolitical developments from TODAY: international conflicts, diplomatic negotiations, sanctions, military movements, trade disputes, or elections. Include dates and key actors.\n• Development 1: ...\n• Development 2: ...\n• Development 3: ..."}},
            {"name": "Domain Study: Crypto and Web3", "check_type": "query", "schedule_seconds": 21600, "cooldown_minutes": 300, "notify_condition": "always",
             "check_config": {"query": "Use web_search to find 3 notable cryptocurrency and blockchain developments from TODAY: major price movements, protocol upgrades, DeFi events, regulatory actions, ETF developments. Include specific prices, numbers, and dates.\n• Development 1: ...\n• Development 2: ...\n• Development 3: ..."}},
            {"name": "Domain Study: Quantum Computing", "check_type": "query", "schedule_seconds": 86400, "cooldown_minutes": 1380, "notify_condition": "always",
             "check_config": {"query": "Use web_search to find quantum computing developments from the past 48 hours: qubit milestones, error correction, new processors, or company announcements from IBM/Google/IonQ. Include dates.\n• Update 1: ...\n• Update 2: ...\n• Update 3: ..."}},
            {"name": "Domain Study: Robotics and Autonomy", "check_type": "query", "schedule_seconds": 86400, "cooldown_minutes": 1380, "notify_condition": "always",
             "check_config": {"query": "Use web_search to find 2-3 robotics and autonomous systems developments from the past 48 hours: humanoid robots, self-driving vehicles, industrial automation, drones, embodied AI. Include dates.\n• Development 1: ...\n• Development 2: ...\n• Development 3: ..."}},
            {"name": "Domain Study: US Policy and Regulation", "check_type": "query", "schedule_seconds": 43200, "cooldown_minutes": 660, "notify_condition": "always",
             "check_config": {"query": "Use web_search to find 2-3 US policy and regulatory developments from the past 24-48 hours: tech regulation, AI governance, trade policy, Supreme Court rulings, executive orders, Congressional actions. Include dates.\n• Policy 1: ...\n• Policy 2: ...\n• Policy 3: ..."}},
            {"name": "Domain Study: Startups and VC", "check_type": "query", "schedule_seconds": 43200, "cooldown_minutes": 660, "notify_condition": "always",
             "check_config": {"query": "Use web_search to find 2-3 notable startup and venture capital developments from the past 24-48 hours: major funding rounds, IPOs, acquisitions, unicorn valuations. Include company names, amounts, investors, and dates.\n• Deal 1: ...\n• Deal 2: ...\n• Deal 3: ..."}},
            {"name": "Domain Study: Physics and Mathematics", "check_type": "query", "schedule_seconds": 86400, "cooldown_minutes": 1380, "notify_condition": "always",
             "check_config": {"query": "Use web_search to find physics and mathematics developments from the past 48 hours: theoretical results, experimental confirmations, major papers, breakthrough proofs. Include dates.\n• Result 1: ...\n• Result 2: ...\n• Result 3: ..."}},
            {"name": "Domain Study: Biotech and Genetics", "check_type": "query", "schedule_seconds": 86400, "cooldown_minutes": 1380, "notify_condition": "always",
             "check_config": {"query": "Use web_search to find 2-3 biotechnology and genetics developments from the past 48 hours: CRISPR advances, gene therapy trials, synthetic biology, longevity research, biotech milestones. Include dates.\n• Advance 1: ...\n• Advance 2: ...\n• Advance 3: ..."}},
            {"name": "Local: Los Angeles", "check_type": "query", "schedule_seconds": 43200, "cooldown_minutes": 660, "notify_condition": "always",
             "check_config": {"query": "Use web_search to find 2-3 significant Los Angeles area news from TODAY: local politics, infrastructure, weather/fires, sports, culture, tech scene. Include specific dates.\n• News 1: ...\n• News 2: ...\n• News 3: ..."}},
            {"name": "Domain Study: Economics and Markets", "check_type": "query", "schedule_seconds": 43200, "cooldown_minutes": 660, "notify_condition": "always",
             "check_config": {"query": "Use web_search to find 2-3 macroeconomic developments from TODAY: GDP data, unemployment, inflation reports, central bank decisions, housing market. Include specific numbers and dates.\n• Data 1: ...\n• Data 2: ...\n• Data 3: ..."}},
            # --- Tier 1: Financial/Trading Intelligence + International ---
            {"name": "Domain Study: Whale Watch", "check_type": "query", "schedule_seconds": 21600, "cooldown_minutes": 300, "notify_condition": "always",
             "check_config": {"query": "Use web_search to find crypto whale movements and large on-chain transactions from the past 6-12 hours. Search for 'crypto whale alert today' and 'large bitcoin ethereum transfers'. Report transfers over $10M between wallets/exchanges, whale accumulation patterns, and notable wallet activity. Include asset, amount in coins and USD, from/to, and significance.\n• Whale 1: [asset] [amount] from [source] to [destination] - [significance]\n• Whale 2: ...\n• Whale 3: ..."}},
            {"name": "Domain Study: Top Trades and Positioning", "check_type": "query", "schedule_seconds": 28800, "cooldown_minutes": 420, "notify_condition": "always",
             "check_config": {"query": "Use web_search to find what notable traders and funds are positioning in TODAY. Search for 'top trades today', 'hedge fund positioning', 'institutional crypto trades', 'most traded assets today'. Report notable large trades, most actively traded assets, publicized trades from known investors, unusual options activity. Include who traded, what asset, direction, size, and platform.\n• Trade 1: [trader/fund] [action] [asset] on [platform] - [details]\n• Trade 2: ...\n• Trade 3: ..."}},
            {"name": "Domain Study: China Tech and Economy", "check_type": "query", "schedule_seconds": 28800, "cooldown_minutes": 420, "notify_condition": "always",
             "check_config": {"query": "Use web_search to find 3 significant developments from China TODAY in tech, economy, or policy. Search 'China tech news today', 'China economy latest', 'China AI developments'. Cover: Chinese tech companies (Baidu, Alibaba, Tencent, Huawei, ByteDance, BYD), Chinese AI models (DeepSeek, Qwen), economic data, government tech policy, US-China competition. Include dates.\n• Development 1: ...\n• Development 2: ...\n• Development 3: ..."}},
            {"name": "Domain Study: Russia and Eastern Europe", "check_type": "query", "schedule_seconds": 43200, "cooldown_minutes": 660, "notify_condition": "always",
             "check_config": {"query": "Use web_search to find 2-3 significant developments from Russia and Eastern Europe from the past 24-48 hours. Cover: Russia-Ukraine conflict updates, Russian economic developments, Eastern European politics, NATO developments. Include dates and key actors.\n• Development 1: ...\n• Development 2: ...\n• Development 3: ..."}},
            {"name": "Domain Study: Middle East", "check_type": "query", "schedule_seconds": 43200, "cooldown_minutes": 660, "notify_condition": "always",
             "check_config": {"query": "Use web_search to find 2-3 significant Middle East developments from the past 24-48 hours. Cover: regional conflicts, OPEC decisions, Gulf state diversification (Saudi Vision 2030, UAE tech), Iran developments, Israel-Palestine. Include dates.\n• Development 1: ...\n• Development 2: ...\n• Development 3: ..."}},
            {"name": "Domain Study: India", "check_type": "query", "schedule_seconds": 43200, "cooldown_minutes": 660, "notify_condition": "always",
             "check_config": {"query": "Use web_search to find 2-3 significant developments from India from the past 24-48 hours. Cover: tech sector (Infosys, TCS, Reliance Jio), startup ecosystem, economic data (GDP, rupee), digital policy (UPI, Aadhaar), semiconductor ambitions. Include dates.\n• Development 1: ...\n• Development 2: ...\n• Development 3: ..."}},
            {"name": "Domain Study: Europe and EU", "check_type": "query", "schedule_seconds": 43200, "cooldown_minutes": 660, "notify_condition": "always",
             "check_config": {"query": "Use web_search to find 2-3 significant European and EU developments from the past 24-48 hours. Cover: EU regulatory actions (AI Act, DMA, antitrust), ECB decisions, European tech (SAP, ASML, ARM), defense policy, Brexit. Include dates.\n• Development 1: ...\n• Development 2: ...\n• Development 3: ..."}},
            {"name": "Domain Study: Semiconductors", "check_type": "query", "schedule_seconds": 28800, "cooldown_minutes": 420, "notify_condition": "always",
             "check_config": {"query": "Use web_search to find 2-3 semiconductor and chip industry developments from the past 24-48 hours. Cover: NVIDIA, AMD, Intel, TSMC, Qualcomm chip announcements, AI chip developments, fab construction, export controls, market data. Include specific specs and dates.\n• Development 1: ...\n• Development 2: ...\n• Development 3: ..."}},
            {"name": "Domain Study: Commodities and Forex", "check_type": "query", "schedule_seconds": 21600, "cooldown_minutes": 300, "notify_condition": "always",
             "check_config": {"query": "Use web_search to find TODAY's notable commodities and forex movements. Report on: oil (WTI, Brent), gold/silver, major forex pairs (EUR/USD, USD/JPY, GBP/USD), agricultural commodities, industrial metals (copper, lithium). Include current prices, percent changes, and driving factors.\n• Movement 1: [commodity/pair] at [price] ([change]) - [driver]\n• Movement 2: ...\n• Movement 3: ..."}},
            # --- Tier 2: High KG Value ---
            {"name": "Domain Study: Earnings and Corporate Events", "check_type": "query", "schedule_seconds": 28800, "cooldown_minutes": 420, "notify_condition": "always",
             "check_config": {"query": "Use web_search to find notable corporate earnings reports, M&A activity, and major corporate events from TODAY. Cover: companies reporting earnings (revenue, EPS, guidance), mergers/acquisitions, IPOs, CEO changes, layoffs, major product launches. Include company names, numbers, and market reaction.\n• Event 1: [company] [event type] - [details]\n• Event 2: ...\n• Event 3: ..."}},
            {"name": "Domain Study: Open Source and GitHub", "check_type": "query", "schedule_seconds": 43200, "cooldown_minutes": 660, "notify_condition": "always",
             "check_config": {"query": "Use web_search to find trending open source projects and notable GitHub activity from the past 24-48 hours. Search 'GitHub trending today', 'new open source projects', 'popular repositories this week'. Cover: trending repos gaining stars, notable tool releases, major version releases, license changes. Include project names, languages, star counts.\n• Project 1: [name] ([language]) - [description] - [stars/growth]\n• Project 2: ...\n• Project 3: ..."}},
            {"name": "Domain Study: Defense and Military Tech", "check_type": "query", "schedule_seconds": 43200, "cooldown_minutes": 660, "notify_condition": "always",
             "check_config": {"query": "Use web_search to find 2-3 defense and military technology developments from the past 24-48 hours. Cover: new weapons systems, drones, autonomous military platforms, AI in defense, defense contracts (Lockheed Martin, Raytheon, Northrop), space militarization, hypersonic weapons. Include dates.\n• Development 1: ...\n• Development 2: ...\n• Development 3: ..."}},
            {"name": "Domain Study: Social Media Platforms", "check_type": "query", "schedule_seconds": 43200, "cooldown_minutes": 660, "notify_condition": "always",
             "check_config": {"query": "Use web_search to find 2-3 notable social media and platform developments from the past 24-48 hours. Cover: feature launches or algorithm changes (X/Twitter, Meta, TikTok, YouTube, Reddit, Bluesky), content moderation shifts, user metrics, creator economy, regulatory threats. Include dates.\n• Change 1: [platform] - [what changed] - [impact]\n• Change 2: ...\n• Change 3: ..."}},
            {"name": "Domain Study: Sports", "check_type": "query", "schedule_seconds": 21600, "cooldown_minutes": 300, "notify_condition": "always",
             "check_config": {"query": "Use web_search to find notable sports developments from TODAY. If web_search only returns portal links without actual scores, use browser to navigate to https://www.espn.com/nba/scoreboard or https://www.espn.com/nfl/scoreboard to get today's game results. Cover: major game results (NBA, NFL, MLB, NHL, Premier League, Champions League, F1, UFC), trades, records broken, star injuries. Include scores, stats, player names, and dates.\n• Result 1: [league] [teams/players] - [score/outcome] - [notable detail]\n• Result 2: ...\n• Result 3: ..."}},
            {"name": "Domain Study: Entertainment and Gaming", "check_type": "query", "schedule_seconds": 43200, "cooldown_minutes": 660, "notify_condition": "always",
             "check_config": {"query": "Use web_search to find 2-3 notable entertainment, streaming, or gaming developments from the past 24-48 hours. Cover: movie/show releases and box office, game releases, streaming platform competition (Netflix, Disney+, Apple TV+), music industry, gaming M&A. Include dates.\n• Development 1: ...\n• Development 2: ...\n• Development 3: ..."}},
            {"name": "Domain Study: DeFi and Protocols", "check_type": "query", "schedule_seconds": 28800, "cooldown_minutes": 420, "notify_condition": "always",
             "check_config": {"query": "Use web_search to find 2-3 notable DeFi and blockchain protocol developments from the past 24 hours. Cover: protocol upgrades, governance decisions, TVL changes, bridge hacks/exploits, airdrop announcements, L2/rollup developments (Arbitrum, Optimism, Base, zkSync). Include protocol names, TVL/volume impact, and dates.\n• Update 1: [protocol] - [change] - [impact]\n• Update 2: ...\n• Update 3: ..."}},
            {"name": "Domain Study: Developer Ecosystem", "check_type": "query", "schedule_seconds": 43200, "cooldown_minutes": 660, "notify_condition": "always",
             "check_config": {"query": "Use web_search to find 2-3 notable developer ecosystem changes from the past 24-48 hours. Cover: programming language updates (Python, Rust, Go, TypeScript), framework releases (React, Next.js, Django, FastAPI), package manager changes, IDE updates (VS Code, JetBrains, Cursor). Include versions and dates.\n• Update 1: [tool/language] [version] - [key change]\n• Update 2: ...\n• Update 3: ..."}},
            # --- Tier 3: Geographic/Domain Gaps ---
            {"name": "Domain Study: Latin America", "check_type": "query", "schedule_seconds": 86400, "cooldown_minutes": 1380, "notify_condition": "always",
             "check_config": {"query": "Use web_search to find 2-3 significant developments from Latin America from the past 48 hours. Cover: Brazilian economy/politics (Petrobras, real), Mexican economy and US-Mexico relations, Argentine reforms, regional tech (Mercado Libre, Nubank), lithium/resources. Include dates.\n• Development 1: ...\n• Development 2: ...\n• Development 3: ..."}},
            {"name": "Domain Study: Africa and Emerging Markets", "check_type": "query", "schedule_seconds": 86400, "cooldown_minutes": 1380, "notify_condition": "always",
             "check_config": {"query": "Use web_search to find 2-3 significant developments from Africa and emerging markets from the past 48 hours. Cover: African fintech/mobile money, emerging market currencies, natural resources, startup ecosystems (Nigeria, Kenya, South Africa), Southeast Asia (ASEAN, Vietnam). Include dates.\n• Development 1: ...\n• Development 2: ...\n• Development 3: ..."}},
            {"name": "Domain Study: Supply Chain and Trade", "check_type": "query", "schedule_seconds": 43200, "cooldown_minutes": 660, "notify_condition": "always",
             "check_config": {"query": "Use web_search to find 2-3 supply chain and global trade developments from the past 24-48 hours. Cover: shipping disruptions (Red Sea, Panama Canal), tariff changes, reshoring/nearshoring, container rates, critical minerals (rare earths, lithium). Include dates.\n• Development 1: ...\n• Development 2: ...\n• Development 3: ..."}},
            {"name": "Domain Study: Research Frontiers", "check_type": "query", "schedule_seconds": 86400, "cooldown_minutes": 1380, "notify_condition": "always",
             "check_config": {"query": "Use web_search to find 2-3 notable research papers or preprints gaining attention in the past 48 hours. Search 'trending arxiv papers', 'notable research papers this week', 'science paper viral'. Cover: AI/ML papers, biology/medicine papers, physics/materials breakthroughs. Include paper title, authors/institution, and key finding.\n• Paper 1: [title] by [authors] - [key finding]\n• Paper 2: ...\n• Paper 3: ..."}},
            {"name": "Domain Study: Climate and Weather", "check_type": "query", "schedule_seconds": 43200, "cooldown_minutes": 660, "notify_condition": "always",
             "check_config": {"query": "Use web_search to find 2-3 significant climate and extreme weather events from the past 24-48 hours. Cover: hurricanes/typhoons, wildfires (especially California), record temperatures, droughts/flooding, climate policy milestones, CO2/ice sheet data. Include locations, severity, and dates.\n• Event 1: [type] in [location] - [severity] - [impact]\n• Event 2: ...\n• Event 3: ..."}},
        ]

        count = 0
        for seed in seeds:
            if seed["name"] in existing_names:
                continue
            mid = self.create(**seed)
            if mid > 0:
                count += 1

        # Migrate existing monitors: update domain study queries + fix check_types
        self._migrate_existing_monitors()

        return count

    def _migrate_existing_monitors(self) -> None:
        """Update existing domain study queries to multi-topic format and fix check_types."""
        # Check if migration already applied
        _MIGRATION_VERSION = 3
        self._db.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)")
        row = self._db.fetchone("SELECT value FROM meta WHERE key = 'monitor_migration_version'")
        if row and int(row["value"]) >= _MIGRATION_VERSION:
            return

        # V3: Update ALL query-type monitor prompts for temporal freshness
        # Match all seeds' updated prompts with "from TODAY" / "past 24-48 hours" anchoring
        _freshness_updates = {
            "Domain Study: Science": "Use web_search to find 3 science discoveries or developments from the past 24-48 hours. For each, give one bullet: what was discovered, the date it was reported, and why it matters. Use this format:\n• Discovery 1: ...\n• Discovery 2: ...\n• Discovery 3: ...",
            "Domain Study: Technology": "Use web_search to find 3 notable new programming tools, frameworks, or AI models released in the past 24-48 hours. For each, give one bullet: what it does, when it was released, and why it's notable. Use this format:\n• Tool 1: ...\n• Tool 2: ...\n• Tool 3: ...",
            "Domain Study: Current Events": "Use web_search to find and summarize 3 significant world events from TODAY. Only report events from the past 24 hours with specific dates. For each: who, what, where, when, why it matters. Use this format:\n• Event 1: ...\n• Event 2: ...\n• Event 3: ...",
            "Domain Study: Finance": "Use web_search to check TODAY's market trends, notable crypto movements, and economic news from the past 24 hours. Include specific prices and dates. Summarize the top 3 developments. Use this format:\n• Market 1: ...\n• Market 2: ...\n• Market 3: ...",
            "World Awareness": "Use web_search to find major global news from TODAY (politics, environment, health, culture — NOT technology/AI, that's covered by Domain Study: Technology). Summarize the top 2-3 developments from the past 24 hours. Include specific dates. Don't just list links — explain why each matters.",
        }
        for name, query in _freshness_updates.items():
            monitor = self.get_by_name(name)
            if monitor:
                cfg = monitor.check_config.copy()
                cfg["query"] = query
                self.update(monitor.id, check_config=cfg)
                logger.info("[MonitorStore] V3 freshness update: '%s'", name)

        # Also update any existing expanded monitors that were added before V3
        for m in self.list_all():
            if m.check_type == "query" and m.name.startswith("Domain Study:"):
                cfg = m.check_config.copy()
                q = cfg.get("query", "")
                # Replace vague temporal language
                if "from the past few days" in q or ("recent" in q.lower() and "past 24" not in q):
                    q = q.replace("from the past few days", "from the past 24-48 hours")
                    q = q.replace("recently", "in the past 24-48 hours")
                    if "Include dates" not in q:
                        q = q.rstrip(".") + ". Include specific dates."
                    cfg["query"] = q
                    self.update(m.id, check_config=cfg)
                    logger.info("[MonitorStore] V3 freshness fix for: '%s'", m.name)

        # Fix System Health check_type if corrupted to 'command'
        health = self.get_by_name("System Health")
        if health and health.check_type != "system_health":
            self.update(health.id, check_type="system_health")
            logger.info("[MonitorStore] Fixed System Health check_type: %s -> system_health",
                        health.check_type)

        # Migrate quiz/skill monitors from "always" to "on_change"
        for name in ("Lesson Quiz", "Skill Validation"):
            monitor = self.get_by_name(name)
            if monitor and monitor.notify_condition == "always":
                self.update(monitor.id, notify_condition="on_change")
                logger.info("[MonitorStore] Migrated '%s' notify_condition: always -> on_change", name)

        # Migrate auto-monitors from search → query type
        for m in self.list_all():
            if m.name.startswith("Auto:") and m.check_type == "search":
                topic = m.name[len("Auto:"):].strip()
                query_prompt = (
                    f"Use web_search to research the latest developments on: {topic}\n"
                    f"Find 2-3 notable updates from the past few days. For each, give "
                    f"one bullet: what happened and why it matters. Use this format:\n"
                    f"• Update 1: ...\n• Update 2: ...\n• Update 3: ..."
                )
                self.update(m.id, check_type="query", check_config={"query": query_prompt})
                logger.info("[MonitorStore] Migrated auto-monitor '%s': search -> query", m.name)

        # Delete garbage auto-monitors whose topics fail validation
        from app.core.curiosity import CuriosityQueue
        for m in self.list_all():
            if m.name.startswith("Auto:"):
                topic = m.name[len("Auto:"):].strip()
                if not CuriosityQueue._is_valid_topic(topic):
                    self.delete(m.id)
                    logger.info("[MonitorStore] Deleted garbage auto-monitor: %s", m.name)

        # Mark migration as applied
        self._db.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
            ("monitor_migration_version", str(_MIGRATION_VERSION)),
        )

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
        for col in updates:
            if not _VALID_COLUMN_RE.match(col):
                raise ValueError(f"Invalid column name: {col!r}")
        if "enabled" in updates:
            updates["enabled"] = 1 if updates["enabled"] else 0
        sets = ", ".join(f"{k} = ?" for k in updates)
        vals = list(updates.values()) + [instruction_id]
        self._db.execute(f"UPDATE heartbeat_instructions SET {sets} WHERE id = ?", tuple(vals))
        return True

    def delete_instruction(self, instruction_id: int) -> bool:
        cursor = self._db.execute("DELETE FROM heartbeat_instructions WHERE id = ?", (instruction_id,))
        return cursor.rowcount > 0

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

        # Prepend context to query
        if ctx_lines:
            context_block = "=== System Context ===\n" + "\n".join(ctx_lines) + "\n=== End Context ===\n\n"
            enriched_query = context_block + query
        else:
            enriched_query = query

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
        return result

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
        candidates = [c for c in candidates if CuriosityQueue._is_valid_topic(c["topic"])]
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
            try:
                from app.tools.action_logging import log_action
                log_action("alert", {"monitor": monitor.name}, message[:500], True)
            except Exception:
                pass
        elif self._discord or self._telegram or self._whatsapp or self._signal:
            logger.error("[Heartbeat] ALL notification channels failed for '%s'", monitor.name)
        else:
            logger.warning("[Heartbeat] No channels configured for alert '%s'", monitor.name)

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
