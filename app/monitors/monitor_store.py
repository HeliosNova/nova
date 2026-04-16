"""Monitor data types, store, and change detection utilities.

Extracted from heartbeat.py for maintainability.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any  # noqa: F401 — re-exported for type annotations

from app.database import SafeDB

logger = logging.getLogger(__name__)

# Defense-in-depth: validate column names before SQL interpolation
_VALID_COLUMN_RE = re.compile(r'^[a-z_][a-z0-9_]*$')

# Regex for numeric extraction (used in change detection)
_NUMBER_RE = re.compile(r"[\$€£¥]?\s*(-?\d[\d,]*\.?\d*)\s*(%|[KMBTkmbt](?![a-zA-Z]))?")


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class Monitor:
    id: int
    name: str
    check_type: str          # 'url', 'search', 'command', 'system_health', 'query', 'quiz', 'skill_test', 'curiosity', 'auto_monitor', 'maintenance', 'finetune', 'consolidation'
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

    @property
    def db(self) -> SafeDB:
        return self._db

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
            {
                "name": "Capability Review",
                "check_type": "capability_review",
                "check_config": {},
                "schedule_seconds": 86400,   # daily
                "cooldown_minutes": 1380,    # 23h
                "notify_condition": "on_change",
            },
            {
                "name": "Dream Consolidation",
                "check_type": "consolidation",
                "check_config": {
                    "min_hours_between": 1.0,  # minimum 1h between dream cycles
                },
                "schedule_seconds": 21600,   # every 6h (actual run gated by min_hours_between)
                "cooldown_minutes": 60,
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
            # --- High-Value Intelligence ---
            {"name": "Hacker News Top Stories", "check_type": "query", "schedule_seconds": 28800, "cooldown_minutes": 420, "notify_condition": "always",
             "check_config": {"query": "Use web_search to search for \"site:news.ycombinator.com\" to find current top Hacker News stories. Also search for \"hacker news front page today top stories\". Report the top 5 trending stories with title and why they are notable. Focus on AI, programming, open source, and startup stories.\n• Story 1: ...\n• Story 2: ...\n• Story 3: ...\n• Story 4: ...\n• Story 5: ..."}},
            {"name": "Product Hunt Trending", "check_type": "query", "schedule_seconds": 86400, "cooldown_minutes": 1380, "notify_condition": "always",
             "check_config": {"query": "Use web_search to find trending products on Product Hunt from TODAY. Search for \"product hunt today trending\". Report: product name, tagline, category, and upvote count for the top 3-5 products. Focus on AI, developer tools, and productivity.\n• Product 1: [name] - [tagline] - [category] - [upvotes]\n• Product 2: ...\n• Product 3: ..."}},
            {"name": "FDA Drug Approvals", "check_type": "query", "schedule_seconds": 86400, "cooldown_minutes": 1380, "notify_condition": "always",
             "check_config": {"query": "Use web_search to find recent FDA drug approvals and notable clinical trial results from the past 48 hours. Search for \"FDA approval today\" and \"clinical trial results today\". Report: drug name, company, condition treated, and significance. Relevant for biotech investing.\n• Approval 1: ...\n• Approval 2: ..."}},
            {"name": "FOMC and Fed Watch", "check_type": "query", "schedule_seconds": 86400, "cooldown_minutes": 1380, "notify_condition": "always",
             "check_config": {"query": "Use web_search to find the current Federal Reserve stance and next FOMC meeting date. Report: next meeting date, current fed funds rate, market expectations for rate change, and any recent Fed official statements. This affects all markets.\n• Rate: ...\n• Next meeting: ...\n• Expectations: ..."}},
            {"name": "SEC Insider Trading", "check_type": "query", "schedule_seconds": 43200, "cooldown_minutes": 660, "notify_condition": "always",
             "check_config": {"query": "Use web_search to find notable SEC insider trading filings from TODAY. Search for \"SEC insider trading filings today\" and \"notable insider buys sells\". Report: company name, insider name/title, buy or sell, number of shares, dollar amount. Focus on large transactions over $1M. Include dates.\n• Filing 1: [company] [insider] [buy/sell] [shares] [$amount]\n• Filing 2: ...\n• Filing 3: ..."}},
            {"name": "GitHub Security Advisories", "check_type": "query", "schedule_seconds": 43200, "cooldown_minutes": 660, "notify_condition": "always",
             "check_config": {"query": "Use web_search to find recent GitHub security advisories and critical CVEs from the past 24-48 hours. Search for \"github security advisory critical\" and \"CVE critical\". Report: CVE ID, affected software, severity, and description. Focus on widely-used packages.\n• CVE 1: [ID] [software] [severity] - [description]\n• CVE 2: ...\n• CVE 3: ..."}},
            {"name": "Government Contract Awards", "check_type": "query", "schedule_seconds": 86400, "cooldown_minutes": 1380, "notify_condition": "always",
             "check_config": {"query": "Use web_search to find major US government contract awards from the past 48 hours. Search for \"government contract award today\" and \"defense contract awarded\". Report: contractor, agency, dollar amount, and purpose. Focus on tech, defense, and AI contracts over $10M.\n• Contract 1: [contractor] [agency] [$amount] - [purpose]\n• Contract 2: ...\n• Contract 3: ..."}},
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
