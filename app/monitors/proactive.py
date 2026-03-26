"""Proactive engine — makes Nova feel alive.

Components:
- DailyDigest: evening summary of monitor findings
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

from app.config import config
from app.core import llm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DailyDigest — evening summary
# ---------------------------------------------------------------------------

class DailyDigest:
    """Compiles and sends a daily digest of monitor findings."""

    def __init__(
        self,
        monitor_store: Any,
        *,
        discord_bot: Any = None,
        telegram_bot: Any = None,
        whatsapp_bot: Any = None,
        signal_bot: Any = None,
        learning_engine: Any = None,
        db: Any = None,
    ):
        self._store = monitor_store
        self._discord = discord_bot
        self._telegram = telegram_bot
        self._whatsapp = whatsapp_bot
        self._signal = signal_bot
        self._learning = learning_engine
        self._db = db
        self._task: asyncio.Task | None = None
        self._running = False
        self._last_digest: str | None = self._load_last_digest()

    def _load_last_digest(self) -> str | None:
        """Load last_digest_date from the system_state table."""
        if self._db is None:
            return None
        try:
            row = self._db.fetchone(
                "SELECT value FROM system_state WHERE key = ?",
                ("last_digest_date",),
            )
            return row["value"] if row else None
        except Exception as e:
            logger.warning("[Digest] Failed to load last_digest_date: %s", e)
            return None

    def _save_last_digest(self, date_str: str) -> None:
        """Save last_digest_date to the system_state table."""
        if self._db is None:
            return
        try:
            self._db.execute(
                "INSERT OR REPLACE INTO system_state (key, value, updated_at) "
                "VALUES (?, ?, CURRENT_TIMESTAMP)",
                ("last_digest_date", date_str),
            )
        except Exception as e:
            logger.warning("[Digest] Failed to save last_digest_date: %s", e)

    def start(self) -> asyncio.Task:
        self._running = True
        self._task = asyncio.create_task(self._loop())
        logger.info("[Digest] Started (hour=%d)", config.DIGEST_HOUR)
        return self._task

    def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()

    async def _loop(self) -> None:
        """Check every 15 minutes if it's digest time."""
        try:
            await asyncio.sleep(30)  # Let things initialize

            while self._running:
                try:
                    now_utc = datetime.now(timezone.utc)
                    # Convert to user's timezone for hour comparison
                    try:
                        from zoneinfo import ZoneInfo
                        user_tz = ZoneInfo(config.USER_TIMEZONE) if config.USER_TIMEZONE else timezone.utc
                        now_local = now_utc.astimezone(user_tz)
                    except (KeyError, ImportError):
                        now_local = now_utc
                    today = now_local.strftime("%Y-%m-%d")

                    # Send digest if it's the right hour in user's timezone
                    if now_local.hour == config.DIGEST_HOUR and self._last_digest != today:
                        await self.send_digest()
                        self._last_digest = today
                        self._save_last_digest(today)
                except Exception as e:
                    logger.error("[Digest] Loop failed: %s", e)

                await asyncio.sleep(900)  # Check every 15 min
        except asyncio.CancelledError:
            logger.info("[Digest] Loop cancelled")
        except Exception as e:
            logger.error("[Digest] Loop terminated unexpectedly: %s", e)

    async def send_digest(self) -> str | None:
        """Compile and send the daily digest. Returns the digest text."""
        results = self._store.get_recent_results(hours=24)
        if not results:
            return None

        monitors = {m.id: m for m in self._store.list_all()}

        # --- Monitor section ---
        monitor_lines = []
        for r in results[:30]:  # Cap at 30 results
            m = monitors.get(r.monitor_id)
            name = m.name if m else f"monitor#{r.monitor_id}"
            monitor_lines.append(f"- [{r.status}] {name}: {r.message or r.value or 'no details'}"[:150])

        summary_input = "## Monitors\n" + "\n".join(monitor_lines)

        # --- Learning section ---
        learning_input = ""
        if self._learning:
            try:
                ls = self._learning.get_learning_summary(hours=24)
                learning_lines = []

                if ls["new_lessons"]:
                    learning_lines.append(f"New lessons learned: {len(ls['new_lessons'])}")
                    for item in ls["new_lessons"][:3]:
                        learning_lines.append(f"  - {item['topic']}: {item['lesson'][:80]}")

                if ls["new_skills"]:
                    learning_lines.append(f"New skills created: {len(ls['new_skills'])}")
                    for item in ls["new_skills"][:3]:
                        learning_lines.append(f"  - {item['name']} (trigger: {item['trigger'][:60]})")

                if ls["new_tools"]:
                    learning_lines.append(f"New custom tools: {len(ls['new_tools'])}")
                    for item in ls["new_tools"][:3]:
                        learning_lines.append(f"  - {item['name']}: {item['description'][:60]}")

                if ls["new_reflexions"]:
                    learning_lines.append(f"New reflexions: {len(ls['new_reflexions'])}")
                    for item in ls["new_reflexions"][:3]:
                        learning_lines.append(f"  - {item['task'][:60]} (score: {item['score']:.1f})")

                if ls["new_training_pairs"]:
                    learning_lines.append(f"New training pairs: {ls['new_training_pairs']}")

                if ls["degraded_skills"]:
                    learning_lines.append(f"Degraded skills (need attention): {len(ls['degraded_skills'])}")
                    for item in ls["degraded_skills"][:3]:
                        learning_lines.append(
                            f"  - {item['name']}: {item['success_rate']:.0%} success over {item['uses']} uses"
                        )

                totals = ls["totals"]
                learning_lines.append(
                    f"Totals: {totals['total_lessons']} lessons, "
                    f"{totals['total_skills']} skills, "
                    f"{totals['training_examples']} training pairs"
                )

                if learning_lines:
                    learning_input = "\n\n## Learning Activity\n" + "\n".join(learning_lines)
            except Exception as e:
                logger.warning("[Digest] Learning summary failed: %s", e)

        # --- Teaching activity section ---
        teaching_input = ""
        quiz_pass = quiz_fail = skill_pass = skill_fail = domain_count = 0
        for r in results:
            m = monitors.get(r.monitor_id)
            if not m:
                continue
            val = r.value or r.message or ""
            if m.check_type == "quiz":
                if "QUIZ PASSED" in val:
                    quiz_pass += 1
                elif "QUIZ FAILED" in val:
                    quiz_fail += 1
            elif m.check_type == "skill_test":
                if "SKILL TEST PASSED" in val:
                    skill_pass += 1
                elif "SKILL TEST FAILED" in val:
                    skill_fail += 1
            elif m.name.startswith("Domain Study:"):
                if val and not val.startswith("["):
                    domain_count += 1

        teaching_lines = []
        if quiz_pass or quiz_fail:
            teaching_lines.append(f"Lesson quizzes: {quiz_pass}/{quiz_pass + quiz_fail} passed")
        if skill_pass or skill_fail:
            teaching_lines.append(f"Skill tests: {skill_pass}/{skill_pass + skill_fail} passed")
        if domain_count:
            teaching_lines.append(f"Domain studies completed: {domain_count}")
        if teaching_lines:
            teaching_input = "\n\n## Teaching Activity\n" + "\n".join(teaching_lines)

        # --- Curiosity research section ---
        curiosity_input = ""
        try:
            from app.core.brain import get_services
            svc = get_services()
            if svc.curiosity:
                resolved = svc.curiosity.get_resolved_since(hours=24)
                failed = svc.curiosity.get_failed_exhausted(hours=24)
                stats = svc.curiosity.get_stats()
                curiosity_lines = []
                if resolved:
                    curiosity_lines.append(f"Topics researched: {len(resolved)}")
                    for item in resolved[:3]:
                        curiosity_lines.append(f"  - {item.topic[:60]}: {(item.resolution or '')[:80]}")
                if failed:
                    curiosity_lines.append(f"Research failed (exhausted attempts): {len(failed)}")
                    for item in failed[:2]:
                        curiosity_lines.append(f"  - {item.topic[:60]}")
                if stats["pending"]:
                    curiosity_lines.append(f"Pending research topics: {stats['pending']}")
                if curiosity_lines:
                    curiosity_input = "\n\n## Curiosity Research\n" + "\n".join(curiosity_lines)
        except Exception as e:
            logger.debug("[Digest] Curiosity section failed: %s", e)

        full_input = summary_input + learning_input + teaching_input + curiosity_input

        system_content = (
            "You are Nova composing your evening digest. Use this format:\n\n"
            "**Monitors:** X passed, Y failed, Z changed\n"
            "**Learning:** (new lessons, skills, training pairs — numbers + highlights)\n"
            "**Teaching:** (quiz results, skill test results)\n"
            "**Issues:** (anything degraded or failing — be specific)\n\n"
            "Keep each section to 1-2 lines. Be specific with numbers. "
            "Skip sections with no activity. No filler."
        )

        try:
            digest = await llm.invoke_nothink(
                [
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": f"Today's activity:\n{full_input}"},
                ],
                max_tokens=400,
                temperature=0.5,
            )
        except Exception as e:
            logger.error("[Digest] LLM generation failed: %s", e)
            digest = ""

        # Fallback to raw results if LLM returned empty
        if not digest or not digest.strip():
            logger.warning("[Digest] LLM returned empty, using raw results")
            digest = f"{len(results)} monitor result(s) in the last 24 hours:\n{full_input}"

        message = f"**Evening Digest**\n\n{digest.strip()}"
        await self._send(message)
        logger.info("[Digest] Sent (%d chars)", len(message))
        return message

    async def _send(self, message: str) -> None:
        if self._discord:
            try:
                await self._discord.send_alert(message)
            except Exception as e:
                logger.error("[Digest] Discord send failed: %s", e)
        if self._telegram:
            try:
                await self._telegram.send_alert(message)
            except Exception as e:
                logger.error("[Digest] Telegram send failed: %s", e)
        if self._whatsapp:
            try:
                await self._whatsapp.send_alert(message)
            except Exception as e:
                logger.error("[Digest] WhatsApp send failed: %s", e)
        if self._signal:
            try:
                await self._signal.send_alert(message)
            except Exception as e:
                logger.error("[Digest] Signal send failed: %s", e)
