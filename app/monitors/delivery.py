"""Alert delivery: routing, batching, broadcast and the durable journal.

Split out of heartbeat_loop.py 2026-09-04, which had grown to 4,251 lines. Every
defect fixed in that file this week lived in it, and the reason several went
unnoticed for months is that nobody can hold four thousand lines in view.

A MIXIN, deliberately: these methods keep the same `self` as the loop that owns
`_digest_buffer`, `_alert_retry_counts` and the channel bots, so behaviour and
every existing import are unchanged. This is a move, not a rewrite.

What lives here: which channels an alert is routed to, the per-cycle digest
buffer, the broadcast itself, the `pending_deliveries` journal that makes a
buffered alert survive a restart, and the age flusher that guarantees a
buffered alert is never held forever.
"""
from __future__ import annotations

import asyncio
import logging
import re
import time
from datetime import datetime

from app.config import config
from app.monitors.monitor_store import Monitor

logger = logging.getLogger(__name__)

# Flush the per-cycle digest buffer once this many alerts are queued, so a long
# tick delivers progressively instead of holding everything to the end.
_DIGEST_FLUSH_EVERY = 3


class DeliveryMixin:
    """Alert routing, digest batching and the delivery journal."""

    async def _send_alert(self, monitor: Monitor, message: str) -> bool:
        """Send an alert via available channel bots. Returns True when the
        alert was delivered or buffered/journaled for delivery, False when it
        was SUPPRESSED (dedup, no channels) — the caller records status from
        this, so a suppressed result can never masquerade as a delivered one
        (deep pass 2026-08-14).

        Routing precedence (per owner directive 2026-04-25):
          1. Per-monitor `channels` column — if set (CSV like "discord,signal"),
             routes ONLY to those channels. Overrides everything else.
          2. Category default fallback if `channels` is NULL/empty:
             - system  → Telegram ONLY (internal health/meta)
             - content → Discord + Telegram + WhatsApp + Signal (all configured)

        Cross-monitor dedup: content monitors that produce the same salient
        claims as another recent monitor get suppressed. Prevents the same
        Iran-Israel ceasefire showing up in 3 different domain studies.
        """
        # Cross-monitor dedup for content monitors (system/health monitors
        # always post — they're about Nova's own state and shouldn't dedupe).
        if monitor.category != "system":
            try:
                from app.monitors.dedup import is_duplicate
                from app.database import get_db
                # to_thread (2026-08-29): is_duplicate runs FIVE sync DB ops —
                # two CREATE IF NOT EXISTS, a DELETE prune, a SELECT and an
                # INSERT — on what is an async path, so all five landed on the
                # event-loop thread (5 of the tripwire's daily warnings came
                # from dedup.py alone). The three WRITES take the write lock on
                # the loop thread, which is the 54h-freeze bug class.
                if await asyncio.to_thread(is_duplicate, get_db(), monitor.name, message):
                    logger.info(
                        "[Heartbeat] '%s' suppressed by cross-monitor dedup",
                        monitor.name,
                    )
                    return False
            except Exception as e:
                logger.warning("[Heartbeat] dedup check failed: %s", e)

        targets = self._alert_targets(monitor)
        if not targets:
            if monitor.category == "system" and not self._telegram:
                logger.warning(
                    "[Heartbeat] system-category monitor '%s' has no Telegram channel — suppressed",
                    monitor.name,
                )
            elif not (self._discord or self._telegram or self._whatsapp or self._signal):
                logger.warning("[Heartbeat] No channels configured for alert '%s'", monitor.name)
            return False

        # Batch this cycle's alerts into one digest per channel-group (the loop
        # flushes after the tick), unless digest mode is disabled. The alert is
        # simultaneously journaled to pending_deliveries so a restart between
        # buffering and broadcast can't destroy it (at-least-once delivery).
        if self._digest_enabled:
            row_id: int | None = None
            try:
                from app.database import get_db

                def _journal() -> int:
                    cur = get_db().execute(
                        "INSERT INTO pending_deliveries (targets, monitor_name, message, category) "
                        "VALUES (?, ?, ?, ?)",
                        (",".join(sorted(targets)), monitor.name, message, monitor.category or ""),
                    )
                    return cur.lastrowid

                row_id = await asyncio.to_thread(_journal)
            except Exception as e:
                logger.warning("[Heartbeat] delivery journal write failed for '%s': %s", monitor.name, e)
            self._digest_buffer.append(
                (frozenset(targets), monitor.name, message, monitor.category or "", row_id, time.monotonic())
            )
            return True

        full_message = f"[{monitor.name}]\n" + message.lstrip("\n")
        sent = await self._broadcast(full_message, targets)
        if sent:
            logger.info("[Heartbeat] Alert sent for '%s' (category=%s)", monitor.name, monitor.category)
            try:
                from app.tools.action_logging import log_action
                await asyncio.to_thread(log_action, "alert", {"monitor": monitor.name}, message[:500], True)
            except Exception:
                pass
        else:
            logger.error("[Heartbeat] ALL notification channels failed for '%s'", monitor.name)
        return bool(sent)

    def _alert_targets(self, monitor: Monitor) -> set[str]:
        """Channels that should receive this alert: a per-monitor `channels`
        override, else the category default (system → Telegram only; content →
        every configured channel). Only returns channels that have a bot."""
        if monitor.channels:
            allowed = {c.strip().lower() for c in monitor.channels.split(",") if c.strip()}
        else:
            allowed = None
        is_system = monitor.category == "system"
        out: set[str] = set()
        for ch, bot in (("discord", self._discord), ("telegram", self._telegram),
                        ("whatsapp", self._whatsapp), ("signal", self._signal)):
            if not bot:
                continue
            ok = (ch in allowed) if allowed is not None else (ch == "telegram" or not is_system)
            if ok:
                out.add(ch)
        return out

    async def _broadcast(self, text: str, targets: set[str]) -> bool:
        """Send one message to each target channel. Returns True if any sent."""
        bots = {"discord": self._discord, "telegram": self._telegram,
                "whatsapp": self._whatsapp, "signal": self._signal}
        sent = False
        for ch in ("discord", "telegram", "whatsapp", "signal"):
            if ch in targets and bots[ch]:
                try:
                    # send_alert returns True only on actual delivery; adapters
                    # used to swallow failures, so `sent` lied and the digest
                    # was recorded delivered while nothing reached the owner.
                    ok = await bots[ch].send_alert(text)
                    sent = sent or bool(ok)
                    if not ok:
                        logger.error("[Heartbeat] %s alert NOT delivered", ch)
                    elif ch == "telegram":
                        # Talk-back anchor (2026-09-01): a delivered digest becomes
                        # an assistant turn in the owner's channel conversation, so
                        # "that's wrong, X is Y" typed after it has an answer to
                        # correct and "what did you send me about X" has history.
                        await asyncio.to_thread(
                            self._record_channel_turn, "telegram",
                            getattr(bots[ch], "default_chat_id", None), text)
                except Exception as e:
                    logger.error("[Heartbeat] %s alert failed: %s", ch, e)
        return sent

    def _record_channel_turn(self, channel: str, user_id, text: str) -> None:
        """Append a delivered digest (bounded) as an assistant message in the
        channel user's persistent conversation, creating the mapping if needed."""
        if not user_id or not text:
            return
        try:
            from app.core.brain import get_services
            from app.database import ChannelConversationStore, get_db
            svc = get_services()
            if not svc or not getattr(svc, "conversations", None):
                return
            store = ChannelConversationStore(get_db())
            conv_id = store.get(channel, str(user_id))
            if not conv_id:
                conv_id = svc.conversations.create_conversation()
                store.set(channel, str(user_id), conv_id)
            body = text.strip()
            if len(body) > 1200:
                body = body[:1200].rstrip() + " […]"
            svc.conversations.add_message(conv_id, "assistant", body)
        except Exception as e:
            logger.debug("[Heartbeat] channel turn not recorded: %s", e)

    def _format_digest(self, items: list[tuple[str, str]], categories: dict | None = None) -> str:
        """One message from a cycle's alerts. Single item keeps its plain form;
        multiple get a digest with each monitor as a section.

        CONTENT briefings (the deep-research product) post in FULL — gutting a
        ~4000-char briefing down to a preview throws away the secondary developments
        and bottom line the engine produced, which is the whole point of the feed.
        Only operational/system status lines get the scannable per-item cap (they're
        short, so it rarely bites — it's just a runaway guard). The channel adapters
        already split over-long messages, so a full briefing delivers fine.
        `categories` maps monitor name → category; absent → cap applies to all."""
        if len(items) == 1:
            name, msg = items[0]
            return f"[{name}]\n" + msg.lstrip("\n")
        categories = categories or {}
        cap = int(getattr(config, "MONITOR_DIGEST_ITEM_MAX_CHARS", 600))
        parts = [f"🛰 **Monitor digest — {len(items)} updates**", ""]
        for name, msg in items:
            body = (msg or "").strip()
            if categories.get(name) != "content" and len(body) > cap:
                body = body[:cap].rstrip() + " […]"
            parts.append(f"## {name}")
            parts.append(body)
            parts.append("")
        return "\n".join(parts).strip()

    async def _flush_digest(self) -> None:
        """Send the buffered cycle alerts as one digest per channel-group.

        Confirmed sends delete their pending_deliveries journal rows; failures
        keep the row and re-buffer (so neither a channel outage nor a restart
        loses the alert). The lock keeps the progressive, end-of-tick, and
        age-flusher call sites from interleaving digests.
        """
        async with self._flush_lock:
            buf = self._digest_buffer
            self._digest_buffer = []
            if not buf:
                return
            # Group by channel-set, carrying each entry's journal row_id WITH the
            # entry. The old (tgt, name)-keyed dict collided when the same monitor
            # produced two alerts in one flush (a re-buffered/recovered entry plus
            # a fresh one): the second overwrote the first's row_id, so on success
            # one journal row leaked (→ duplicate repost after restart) and on
            # salience suppression the wrong row could be deleted.
            groups: dict[frozenset, list[tuple[str, str, int | None]]] = {}
            cats: dict[str, str] = {}
            for tgt, name, msg, cat, row_id, _ts in buf:
                groups.setdefault(tgt, []).append((name, msg, row_id))
                cats[name] = cat
            for tgt, entries in groups.items():
                try:
                    kept = entries
                    # Salience: lead with what matters to the owner, drop sub-floor
                    # noise (only kicks in on larger digests; never thins a tiny one).
                    if getattr(config, "ENABLE_SALIENCE_FILTER", True) and len(entries) > 2:
                        try:
                            from app.core.salience import rank_digest_items
                            from app.database import get_db
                            items = [(n, m) for n, m, _ in entries]
                            # to_thread: rank_digest_items reads dossier bodies —
                            # a heavy sync DB read that must not block the loop.
                            ranked = await asyncio.to_thread(rank_digest_items, get_db(), items)
                            # Reconstruct concrete entries (with row_ids) in ranked
                            # order — duplicates consumed positionally, leftovers
                            # are the suppressed set.
                            pool: dict[tuple[str, str], list] = {}
                            for e in entries:
                                pool.setdefault((e[0], e[1]), []).append(e)
                            kept = []
                            for n, m in ranked:
                                lst = pool.get((n, m))
                                if lst:
                                    kept.append(lst.pop(0))
                            # A salience drop is a final suppression — clear its
                            # journal row or recovery would repost it forever.
                            suppressed_rows = [e[2] for lst in pool.values() for e in lst]
                            if suppressed_rows:
                                await self._delete_journal_rows(suppressed_rows)
                        except Exception as e:
                            logger.warning("[Heartbeat] salience ranking skipped: %s", e)
                            kept = entries
                    items = [(n, m) for n, m, _ in kept]
                    digest = self._format_digest(items, cats)
                    if await self._broadcast(digest, set(tgt)):
                        logger.info("[Heartbeat] digest sent: %d update(s) → %s",
                                    len(items), ",".join(sorted(tgt)))
                        for n, _m, _r in kept:
                            self._alert_retry_counts.pop(n, None)
                        await self._delete_journal_rows([e[2] for e in kept])
                    else:
                        # Every channel failed — re-buffer for the next flush so
                        # the intelligence isn't silently lost (record_check has
                        # already advanced, so the monitor won't refire on its own).
                        kept_count = 0
                        dropped_rows: list[int | None] = []
                        for n, m, row_id in kept:
                            cnt = self._alert_retry_counts.get(n, 0) + 1
                            self._alert_retry_counts[n] = cnt
                            if cnt <= 3:
                                self._digest_buffer.append(
                                    (tgt, n, m, cats.get(n, ""), row_id, time.monotonic())
                                )
                                kept_count += 1
                            else:
                                logger.error(
                                    "[Heartbeat] digest for '%s' undelivered after %d attempts — dropped",
                                    n, cnt,
                                )
                                dropped_rows.append(row_id)
                        await self._delete_journal_rows(dropped_rows)
                        if kept_count:
                            logger.warning(
                                "[Heartbeat] delivery failed on all channels — re-buffered %d item(s)",
                                kept_count,
                            )
                except Exception as e:
                    logger.error("[Heartbeat] digest flush failed: %s", e)

    async def _delete_journal_rows(self, row_ids: list[int | None]) -> None:
        """Remove delivered (or terminally dropped) alerts from the journal."""
        ids = [r for r in row_ids if r is not None]
        if not ids:
            return
        try:
            from app.database import get_db
            await asyncio.to_thread(
                get_db().executemany,
                "DELETE FROM pending_deliveries WHERE id = ?",
                [(i,) for i in ids],
            )
        except Exception as e:
            logger.warning("[Heartbeat] delivery journal cleanup failed: %s", e)

    async def _recover_pending_deliveries(self) -> None:
        """Reload journaled-but-never-broadcast alerts into the digest buffer.

        A crash after broadcast but before the journal delete re-posts that
        digest once (at-least-once semantics) — acceptable; the failure mode
        this kills is silent loss. Stale rows (> _DELIVERY_RECOVERY_MAX_AGE_H)
        are purged instead of posting day-old briefings after a long outage.
        """
        try:
            from app.database import get_db
            db = get_db()

            def _load():
                db.execute(
                    "DELETE FROM pending_deliveries WHERE created_at < "
                    f"datetime('now', '-{_DELIVERY_RECOVERY_MAX_AGE_H} hours')"
                )
                return db.fetchall(
                    "SELECT id, targets, monitor_name, message, category "
                    "FROM pending_deliveries ORDER BY id"
                )

            rows = await asyncio.to_thread(_load)
            for row in rows:
                targets = frozenset(t for t in (row["targets"] or "").split(",") if t)
                if not targets:
                    await self._delete_journal_rows([row["id"]])
                    continue
                self._digest_buffer.append(
                    (targets, row["monitor_name"], row["message"],
                     row["category"] or "", row["id"], time.monotonic())
                )
            if rows:
                logger.info(
                    "[Heartbeat] recovered %d undelivered alert(s) from the journal — "
                    "flushing within ~%ds", len(rows), _DIGEST_MAX_BUFFER_AGE,
                )
        except Exception as e:
            logger.warning("[Heartbeat] delivery journal recovery failed: %s", e)

    async def _age_flusher(self) -> None:
        """Post any buffer that has been waiting past _DIGEST_MAX_BUFFER_AGE.

        The progressive flush needs _DIGEST_FLUSH_EVERY items and the end-of-tick
        flush sits behind every remaining slow monitor, so without this a 1-2
        item buffer (a completed digest) could wait 30+ minutes — and a restart
        in that window used to destroy it. Runs as its own task because the main
        loop blocks inside asyncio.gather for the whole slow-monitor batch.
        """
        while self._running:
            try:
                await asyncio.sleep(60)
                if not self._digest_buffer:
                    continue
                oldest = min(entry[5] for entry in self._digest_buffer)
                if time.monotonic() - oldest >= _DIGEST_MAX_BUFFER_AGE:
                    logger.info(
                        "[Heartbeat] age flush: %d buffered alert(s) waited %ds",
                        len(self._digest_buffer), int(time.monotonic() - oldest),
                    )
                    await self._flush_digest()
            except asyncio.CancelledError:
                break
            except Exception as e:
                # One bad iteration must NOT kill the guard against silent digest
                # loss — log and keep looping (the old outer except exited the
                # task forever, so the guard could vanish until restart).
                logger.error("[Heartbeat] age flush iteration failed (continuing): %s", e)
