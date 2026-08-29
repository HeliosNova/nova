"""HeartbeatLoop — the background scheduling engine.

Checks monitors on schedule, executes them, and delivers alerts via
Discord, Telegram, WhatsApp, and Signal channel bots.

Extracted from heartbeat.py for maintainability.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
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

_MAX_CONCURRENT_LLM_MONITORS = 1  # one monitor's GPU work at a time — leaves the 3090 free for a side process (owner 2026-06-30)
# Digest-class (27B deep-research) monitors may overlap each other at width 2:
# a digest spends minutes in network gather and the CPU MiniCheck gate while
# the GPU idles — a second digest fills that idle time, and Ollama queues
# same-model calls so the 27B is never contended. Cross-CLASS overlap stays
# forbidden (see _ClassGate): brain-query monitors drive the 9B, and
# 27B+9B co-residency exceeds the 24GB card — the documented thrash ceiling.
_MAX_CONCURRENT_DIGEST_MONITORS = 2
# Post digests progressively as monitors finish, so a large due-batch (or a restart
# mid-batch) can't strand every update behind the slowest monitor. Small groups keep
# most of the digest-bundling benefit while making the feed timely + restart-resilient.
_DIGEST_FLUSH_EVERY = 3
# A sub-threshold buffer (1-2 items) used to wait for the END-OF-TICK flush, which
# sits behind every remaining slow monitor — a single 27B digest kept a completed
# briefing hostage for 30+ min, and a restart in that window destroyed it
# (2026-08-12). The age flusher posts any buffer older than this regardless of size.
_DIGEST_MAX_BUFFER_AGE = 300  # seconds
# Buffered alerts older than this at recovery are stale intelligence — drop them
# rather than posting yesterday's briefing after a long outage.
_DELIVERY_RECOVERY_MAX_AGE_H = 24

# Monitors whose output is non-factual — skip KG extraction for these
_NO_KG_MONITORS = frozenset({"Morning Check-in", "Self-Reflection"})

# Analytical/forecast curiosity topics ("<Domain>: Can X absorb Y?" — the
# dossier open-question style). Their best-possible research answer is a hedged
# synthesis, which the FACTUAL closure judge structurally rejects — these
# resolve as [provisional] instead of burning MAX_ATTEMPTS and discarding the
# work (2026-08-18). Factual topics ("What is X?", "Resolve contradiction: …")
# deliberately do NOT match.
_ANALYTICAL_TOPIC_RE = re.compile(
    r"(?i)^(?:[^:]{0,80}:\s*)?"                       # optional "Domain: " prefix
    r"(?:can|will|could|might|would|should|do(?:es)?\s+\w+\s+(?:have|stand|survive)"
    r"|how\s+(?:might|will|could|would|likely)"
    r"|what\s+(?:happens|if|would|will)"
    r"|is\s+it\s+likely|are\s+\w+\s+likely)\b")

# Session-summary / tool-log shaped "answers" (2026-08-25): the provisional
# path bypasses the closure judge by design, and one resolution stored
# "Based on the tools executed in this session, here is a summary of your
# recent activities: ### ✅ Completed" as the ANSWER to a Broadcom/Lumentum
# question. A provisional answer must at least be ABOUT the topic and must
# not be the assistant narrating its own tool session.
_SESSION_SUMMARY_RE = re.compile(
    r"(?i)(?:\btools?\s+(?:executed|used|invoked)\b|\bin\s+this\s+session\b"
    r"|\byour\s+recent\s+activit|\bhere\s+is\s+a\s+summary\s+of\s+your\b"
    r"|✅\s*Completed)")


# Digest Health Canary thresholds (weekly check_type="digest_health").
# The per-digest summary line the entail gate emits since 2026-08-25:
#   [entail-gate] <label>: N checked, ... M dropped
_ENTAIL_GATE_LINE_RE = re.compile(
    r"\[entail-gate\] .*?: (\d+) checked.*?(\d+) dropped")


def _digest_health_verdict(lengths: list[int], linkish: int,
                           checked: int, dropped: int) -> tuple[str, str]:
    """(status, summary) for the digest-health canary. Pure for tests.

    error  — pipeline broken: no digests in 7d, thin output (avg < 2000
             chars — healthy live average is ~8k), or >10% link-only.
    warning — degradation: avg < 4000 chars, or entail drop-rate > 55%
             (the pre-fix live rate was ~51%; digit-aware windows should
             push it DOWN, so exceeding 55% means a new regression).
    """
    if not lengths:
        return "error", "no content digests stored in 7 days — pipeline dead?"
    avg = sum(lengths) / len(lengths)
    link_share = linkish / len(lengths)
    drop_rate = (dropped / checked) if checked else 0.0
    stats = (f"{len(lengths)} digests, avg {avg:.0f} chars, "
             f"{link_share:.0%} link-only, entail drop {drop_rate:.0%}")
    if avg < 2000 or link_share > 0.10:
        return "error", f"digest substance degraded — {stats}"
    if avg < 4000 or drop_rate > 0.55:
        return "warning", f"digest quality drifting — {stats}"
    return "info", stats


# Stat-line canaries whose numbers drift every run by design (counts,
# latencies). Generic numeric change-detection re-delivered their HEALTHY
# readouts forever ("kg growth normal" 3x/day, "✅ ollama healthy (6ms)"
# whenever latency jittered a few ms — live 2026-08-26, both stored as
# status=alert). Maps check_type -> the marker that identifies a healthy
# verdict line.
_CANARY_NORMAL_MARKERS: dict[str, str] = {
    "kg_growth": "kg growth normal",
    "ollama_latency": "ollama healthy",
}


# Lessons whose "answer" is a process instruction, not a gradable fact —
# the Lesson Quiz skips these (see _execute_quiz).
_UNQUIZZABLE_ANSWER_RE = re.compile(
    r"(?i)\b(use (a |the |your )?(calculator|web[ _]?search|browser|tool)"
    r"|search the web|perform a (web )?search|look (it )?up"
    r"|consult (the|a|current)|verify (with|using|by))\b")


def _canary_should_alert(check_type: str, last_result: str | None,
                         new_value: str | None) -> bool:
    """Delivery gate for stat-line canary monitors.

    The only deliverable states are: any non-healthy verdict (spike/drop/
    FLATLINE/slow/error — active alarms must repeat), and the single
    recovery edge back to healthy. healthy → healthy is suppressed.
    """
    marker = _CANARY_NORMAL_MARKERS.get(check_type)
    if marker is None:
        return True
    was_normal = marker in (last_result or "")
    is_normal = marker in (new_value or "")
    return not (was_normal and is_normal)


def _provisional_acceptable(topic: str, result: str) -> bool:
    """Is `result` an acceptable PROVISIONAL answer for `topic`?

    Two cheap screens (the closure judge was already bypassed for
    analytical topics): reject session-summary-shaped text, and require at
    least one substantive topic token to appear in the answer.
    """
    if not result or _SESSION_SUMMARY_RE.search(result):
        return False
    topic_tokens = {t for t in re.findall(r"[a-z0-9]{4,}", (topic or "").lower())
                    if t not in ("what", "will", "could", "might", "would",
                                 "should", "does", "have", "likely", "happens")}
    if not topic_tokens:
        return True   # nothing substantive to anchor on — don't over-reject
    low = result.lower()
    return any(t in low for t in topic_tokens)


_PERSON_TITLE_RE = re.compile(r"(?i)^(?:dr|mr|mrs|ms|prof|gen|sen|rep|gov|amb)\.?\s")
_ORG_WORDS = frozenset({
    "inc", "corp", "llc", "ltd", "fund", "forum", "commission", "institute",
    "university", "bank", "group", "chase", "capital", "partners", "company",
    "committee", "council", "agency", "administration", "ministry",
    "department", "association", "foundation", "laboratory", "labs",
})


def _person_shaped(name: str) -> bool:
    """Conservative person detector for direction curation: title prefix, or
    2-3 title-case tokens with no org marker words."""
    name = (name or "").strip()
    if _PERSON_TITLE_RE.match(name):
        return True
    toks = name.split()
    if any(t.lower().strip(".,") in _ORG_WORDS for t in toks):
        return False
    return 2 <= len(toks) <= 3 and all(t[:1].isupper() for t in toks if t)


def _curate_inverted_leads(db) -> int:
    """Supersede the org-as-subject side of mutual A-leads-B / B-leads-A pairs.

    Extraction sometimes emits both directions ("Citadel leads Ken Griffin"
    alongside the correct one). Only acts when EXACTLY one side is
    person-shaped — ambiguous pairs are left alone. Supersession, not
    deletion: the losing row keeps its audit trail (found live 2026-08-14,
    4 pairs)."""
    pairs = db.fetchall(
        "SELECT a.id aid, a.subject asub, b.id bid, b.subject bsub "
        "FROM kg_facts a JOIN kg_facts b "
        "ON LOWER(a.subject)=LOWER(b.object) AND LOWER(a.object)=LOWER(b.subject) "
        "AND a.predicate=b.predicate AND a.id < b.id "
        "WHERE a.predicate='leads' AND a.superseded_at IS NULL "
        "AND b.superseded_at IS NULL LIMIT 20"
    )
    n = 0
    for p in pairs:
        a_person, b_person = _person_shaped(p["asub"]), _person_shaped(p["bsub"])
        if a_person == b_person:
            continue                      # ambiguous — leave both
        wrong_id = p["bid"] if a_person else p["aid"]
        n += db.execute(
            "UPDATE kg_facts SET superseded_at = datetime('now'), "
            "provenance = COALESCE(provenance,'') || "
            "' | superseded:inverted-direction-curation' "
            "WHERE id = ? AND superseded_at IS NULL", (wrong_id,)).rowcount
    return n


def _skeletal_digest(text: str, cap: int = 1200) -> str:
    """Deterministic skeleton of a digest for demoted retention: headings and
    bolded lead lines only — the structure and headline claims survive, the
    prose body (already consolidated into dossiers by now) is released."""
    keep = []
    for ln in (text or "").split("\n"):
        s = ln.strip()
        if s.startswith("#") or s.startswith(("* **", "- **", "**")):
            keep.append(s)
    out = "\n".join(keep) or (text or "")[:cap]
    return out[:cap]

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


_CITATION_RE = re.compile(r"(?i)\bsource\s*[:–]\s*\S")
_URL_RE = re.compile(r"https?://[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}(?:/[^\s)]*)?")
_DATE_RE_GATE = re.compile(
    r"\b("
    r"(?:January|February|March|April|May|June|July|August|September|October|November|December|"
    r"Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\.?\s+\d{1,2},\s+\d{4}"
    r"|"
    r"\d{4}-\d{2}-\d{2}"
    r")\b"
)
# Hedging patterns that violate the no-hedging rule
_HEDGE_RE = re.compile(
    r"(?i)("
    r"\bapr[/\-]may\b|\bmay[/\-]apr\b|"
    r"\bapril[–\-]+may\b|\bmay[–\-]+april\b|"
    r"\b~\s*[a-z]+\b|"
    r"\b(?:approximately|around|circa|roughly)\s+(?:apr|april|may|jun|june)\b|"
    r"\b(?:early|mid|late)[\s\-](?:april|may|june)\b|"
    r"\b\d+[–\-]+\d+\s*days?\s*ago\b|"
    r"\b\d+\s*days?\s*ago\b"
    r")"
)


def _domain_study_passes_citation_gate(result: str) -> bool:
    """A Domain Study output is acceptable if it either:
      - contains >= 2 'Source:' citations AND >= 2 well-formed dates within the
        last 48h AND >= 2 well-formed URLs AND no hedging-language matches, OR
      - is the explicit 'No significant ... in the past 48 hours' fallback,
      - OR is empty/error (those bypass since we can't fix them by re-rolling).
    """
    if not result:
        return True
    low = result.lower()
    if "no significant" in low and "past 48 hours" in low:
        return True
    if low.startswith("[query failed") or low.startswith("[query timed out"):
        return True

    # Hard rejects
    if _HEDGE_RE.search(result):
        logger.info("[Heartbeat] citation gate FAIL: hedging language detected")
        return False

    citations = len(_CITATION_RE.findall(result))
    if citations < 2:
        logger.info("[Heartbeat] citation gate FAIL: only %d Source: citations", citations)
        return False

    urls = [u for u in _URL_RE.findall(result) if "." in u]
    if len(urls) < 2:
        logger.info("[Heartbeat] citation gate FAIL: only %d well-formed URLs", len(urls))
        return False

    # Parse dates and require at least 2 within the last 48h
    from datetime import datetime as _dt, timedelta as _td
    cutoff = _dt.utcnow() - _td(hours=48)
    fresh_count = 0
    stale_count = 0
    for m in _DATE_RE_GATE.finditer(result):
        raw = m.group(1)
        parsed = None
        for fmt in ("%B %d, %Y", "%b %d, %Y", "%B %d %Y", "%b %d %Y", "%Y-%m-%d"):
            try:
                parsed = _dt.strptime(raw.strip().rstrip("."), fmt)
                break
            except ValueError:
                continue
        if not parsed:
            continue
        if parsed >= cutoff:
            fresh_count += 1
        else:
            stale_count += 1
    if fresh_count < 2:
        logger.info(
            "[Heartbeat] citation gate FAIL: %d fresh dates, %d stale (need ≥2 fresh)",
            fresh_count, stale_count,
        )
        return False

    return True


def _class_floor_order(slow: list, classify, now: datetime) -> list:
    """Promote starved non-digest monitors to the FRONT of the batch.

    Tick tails die on every restart, and 'other'-class monitors always LIVE in
    the tail: digest cadences (4-8h) produce overdue ratios that outrank a
    daily monitor's for most of its life, so eval/forecast-resolution ran 31h+
    late whenever restarts kept truncating batches (2026-08-14). Any
    other-class monitor ≥25% past schedule jumps the queue — digests wait
    behind at most a few starved dailies instead of the reverse, forever."""
    def _ratio(m) -> float:
        if not m.last_check_at:
            return float("inf")
        try:
            last = datetime.fromisoformat(m.last_check_at).replace(tzinfo=None)
        except Exception:
            return float("inf")
        return (now - last).total_seconds() / max(m.schedule_seconds, 1)

    starved = [m for m in slow if classify(m) == "other" and _ratio(m) >= 1.25]
    if not starved:
        return slow
    starved_ids = {id(m) for m in starved}
    return starved + [m for m in slow if id(m) not in starved_ids]


# ---------------------------------------------------------------------------
# _ClassGate — model-aware monitor concurrency
# ---------------------------------------------------------------------------

class _ClassGate:
    """Concurrency gate: monitors of the SAME class may overlap up to that
    class's width; different classes NEVER overlap.

    Classes map to the GPU model a monitor drives (27B digest chain vs 9B
    brain queries). The 24GB card cannot hold both resident, so cross-class
    concurrency churns model load/unload on every call — the documented
    thrash ceiling. Same-class width 2 lets one digest's network gather and
    CPU MiniCheck phases overlap another digest's GPU synthesis; Ollama
    queues same-model requests, so the GPU itself is never contended.

    Fairness is drain-and-switch: entrants are FIFO, and a waiter of a
    different class blocks LATER same-class entrants from jumping the queue —
    so an hourly 9B monitor can't starve behind a day-long digest backlog,
    and one 9B run can't be starved out by a stream of digests."""

    def __init__(self, widths: dict[str, int], default: int = 1):
        self._widths = widths
        self._default = default
        self._cond = asyncio.Condition()
        self._cls: str | None = None
        self._n = 0
        self._queue: list[tuple[str, object]] = []   # FIFO of (class, token)

    async def acquire(self, cls: str) -> None:
        token = object()
        async with self._cond:
            self._queue.append((cls, token))
            while True:
                pos = next(i for i, (_, t) in enumerate(self._queue) if t is token)
                width = self._widths.get(cls, self._default)
                same_prefix = all(c == cls for c, _ in self._queue[:pos])
                if ((self._n == 0 and pos == 0)
                        or (self._cls == cls and self._n < width and same_prefix)):
                    self._queue.pop(pos)
                    self._cls = cls
                    self._n += 1
                    return
                await self._cond.wait()

    async def release(self) -> None:
        async with self._cond:
            self._n -= 1
            if self._n == 0:
                self._cls = None
            self._cond.notify_all()


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
        # Strong-ref set for fire-and-forget background tasks (KG extraction)
        # so the GC can't cancel them mid-flight; the done_callback discards
        # the entry and surfaces any exception at WARNING.
        self._kg_bg_tasks: set[asyncio.Task] = set()
        # Per-cycle alert batching. When enabled, _send_alert buffers each
        # monitor's alert (after dedup/routing) and the loop flushes ONE digest
        # per channel-group at the end of the tick — so 80 due monitors post a
        # single briefing instead of 80 interleaving messages.
        self._digest_enabled = bool(getattr(config, "ENABLE_MONITOR_DIGEST", True))
        # Entries: (targets, monitor_name, message, category, ledger_row_id, buffered_at).
        # ledger_row_id points at the pending_deliveries row (migration 27) that makes
        # the buffered alert survive a restart; buffered_at (monotonic) drives the
        # age flusher. row_id None = ledger write failed, in-memory-only fallback.
        self._digest_buffer: list[tuple[frozenset, str, str, str, int | None, float]] = []
        self._flush_lock = asyncio.Lock()
        self._flusher_task: asyncio.Task | None = None
        # Per-monitor delivery-failure retry counts: a digest whose broadcast
        # failed on EVERY channel is re-buffered for the next flush instead of
        # being silently dropped (audit 2026-07-08); capped so a permanently
        # broken channel can't grow the buffer without bound.
        self._alert_retry_counts: dict[str, int] = {}

    def start(self) -> asyncio.Task:
        """Start the heartbeat loop as a background task."""
        self._running = True
        self._task = asyncio.create_task(self._loop())
        if self._digest_enabled:
            self._flusher_task = asyncio.create_task(self._age_flusher())
        logger.info("[Heartbeat] Started (interval=%ds)", config.HEARTBEAT_INTERVAL)
        return self._task

    def stop(self) -> None:
        """Stop the heartbeat loop."""
        self._running = False
        if self._task:
            self._task.cancel()
            logger.info("[Heartbeat] Stopped")
        if self._flusher_task:
            self._flusher_task.cancel()

    async def _loop(self) -> None:
        """Main loop — check due monitors every HEARTBEAT_INTERVAL seconds."""
        try:
            # Small delay on startup to let services initialize
            await asyncio.sleep(10)

            # Recover alerts that buffered but never broadcast before the last
            # shutdown (the 2026-08-12 lost-World-digest failure mode). They
            # re-enter the normal buffer; the age flusher posts them within
            # minutes, once the channel bots have connected.
            if self._digest_enabled:
                await self._recover_pending_deliveries()

            while self._running:
                try:
                    due = await asyncio.to_thread(self.store.get_due)
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
                                await asyncio.to_thread(self.store.record_check, monitor.id, f"error: {e}")
                                await asyncio.to_thread(self.store.add_result, monitor.id, "error", message=str(e))

                        # Interactive-priority: if the owner is actively chatting,
                        # defer the LLM-heavy monitors this cycle so chat keeps the
                        # GPU (measured 2026-06-11: monitor generations are what
                        # push interactive latency from ~13s to 60-85s). The fast,
                        # no-LLM monitors above still ran. Due monitors aren't lost
                        # — last_check_at isn't advanced, so they're picked up on
                        # the next tick once chat goes quiet.
                        from app.core import llm as _llm
                        if slow and _llm.interactive_active():
                            # Escape hatch against indefinite starvation: a
                            # never-run monitor (no baseline) or one already past
                            # 2x its schedule still runs even while chatting, so a
                            # continuously-active owner can't permanently block
                            # background intelligence. Everything else defers.
                            _now = datetime.now(timezone.utc).replace(tzinfo=None)

                            def _badly_overdue(m) -> bool:
                                if not m.last_check_at:
                                    return True
                                try:
                                    last = datetime.fromisoformat(m.last_check_at).replace(tzinfo=None)
                                except Exception:
                                    return True
                                return (_now - last).total_seconds() >= 2 * max(m.schedule_seconds, 1)

                            overdue = [m for m in slow if _badly_overdue(m)]
                            deferred = [m for m in slow if not _badly_overdue(m)]
                            # Post-restart, EVERY monitor is badly overdue (or has
                            # no last_check_at), so the escape hatch used to flood
                            # the GPU with the whole catch-up queue exactly while
                            # the owner was chatting (audit 2026-07-08). Cap the
                            # bypass; the rest catch up once chat goes quiet.
                            if len(overdue) > 2:
                                deferred.extend(overdue[2:])
                                overdue = overdue[:2]
                            if deferred:
                                logger.info(
                                    "[Heartbeat] owner is chatting — deferring %d LLM monitor(s); "
                                    "running %d badly-overdue to avoid starvation",
                                    len(deferred), len(overdue),
                                )
                            slow = overdue

                        # LLM monitors with bounded, model-aware concurrency:
                        # digest-class monitors overlap each other (width 2);
                        # everything else is exclusive, and classes never mix.
                        # Starved non-digest monitors jump the queue first.
                        slow = _class_floor_order(
                            slow, self._monitor_class,
                            datetime.now(timezone.utc).replace(tzinfo=None))
                        if slow:
                            gate = _ClassGate({"digest": _MAX_CONCURRENT_DIGEST_MONITORS},
                                              default=_MAX_CONCURRENT_LLM_MONITORS)

                            async def _limited_check(monitor):
                                await gate.acquire(self._monitor_class(monitor))
                                try:
                                    try:
                                        await self._check_monitor(monitor)
                                    except Exception as e:
                                        logger.error("[Heartbeat] Monitor '%s' failed: %s", monitor.name, e)
                                        # Exponential backoff: count recent consecutive errors
                                        _recent_errors = 0
                                        try:
                                            _rows = await asyncio.to_thread(
                                                self.store._db.fetchall,
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
                                        await asyncio.to_thread(
                                            self.store.update,
                                            monitor.id,
                                            last_check_at=retry_at.strftime("%Y-%m-%d %H:%M:%S"),
                                        )
                                        await asyncio.to_thread(
                                            self.store.add_result,
                                            monitor.id, "error",
                                            message=f"Exception — retry in ~{_retry_delay // 60} min: {e}",
                                        )
                                finally:
                                    await gate.release()
                                # Progressive flush (OUTSIDE the gate — Discord I/O must
                                # not hold a monitor concurrency slot): post as soon as a few
                                # updates buffer, so nothing waits behind the slowest run and a
                                # restart mid-batch keeps what already completed.
                                if len(self._digest_buffer) >= _DIGEST_FLUSH_EVERY:
                                    await self._flush_digest()

                            await asyncio.gather(*[_limited_check(m) for m in slow], return_exceptions=True)

                    # Flush this cycle's batched monitor alerts as ONE digest per
                    # channel-group (replaces N separate posts and their interleaving).
                    await self._flush_digest()

                    # Execute due heartbeat instructions
                    due_instructions = await asyncio.to_thread(self.store.get_due_instructions)
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

    def _monitor_class(self, monitor: Monitor) -> str:
        """Concurrency class for _ClassGate — keyed to the GPU model the
        monitor drives. Mirrors the _execute_query_monitor routing predicate:
        Domain Study:*/Auto:*/feed-backed query monitors run the 27B
        deep-research chain ("digest"); every other slow monitor (brain.think
        9B queries, quiz, consolidation, eval) stays exclusive ("other")."""
        if monitor.check_type != "query":
            return "other"
        try:
            from app.monitors.rss_feeds import feeds_for
            if monitor.name.startswith(("Domain Study:", "Auto:")) or feeds_for(monitor.name):
                return "digest"
        except Exception:
            pass
        return "other"

    async def _check_monitor(self, monitor: Monitor) -> None:
        """Execute a single monitor check."""
        logger.info("[Heartbeat] Checking '%s' (type=%s)", monitor.name, monitor.check_type)

        # Execute the check
        new_value = await self._execute_check(monitor)
        # Defensive: strip any tool-call artifacts the LLM may have emitted
        # instead of executing the tool. Keeps Discord/Telegram output clean.
        if new_value:
            new_value = strip_tool_call_artifacts(new_value)

        # Empty-result gate: query-type monitors that come back with <50 chars of
        # actual content (after artifact strip) are treated as soft failures —
        # log as info, don't alert, don't update last_check_at so we retry on
        # the next schedule. Without this gate, Domain Study monitors silently
        # log status=ok with empty value and never get flagged.
        _stripped = (new_value or "").strip()
        if monitor.check_type == "query" and len(_stripped) < 50:
            logger.warning(
                "[Heartbeat] '%s' returned empty/short result (%d chars) — soft retry on next tick",
                monitor.name, len(_stripped),
            )
            await asyncio.to_thread(
                self.store.add_result,
                monitor.id, "skip",
                value=_stripped,
                message=f"empty result ({len(_stripped)} chars) — will retry",
            )
            return

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
                rows = await asyncio.to_thread(
                    self.store._db.fetchall,
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
            await asyncio.to_thread(
                self.store.update,
                monitor.id,
                last_check_at=retry_at.strftime("%Y-%m-%d %H:%M:%S"),
            )
            await asyncio.to_thread(
                self.store.add_result,
                monitor.id, "error", value=new_value[:12000] if new_value else "",
                message=f"LLM failure — retry in ~{_retry_delay // 60} min")
            logger.warning("[Heartbeat] '%s' LLM failure (streak=%d), retry in ~%d min: %s",
                           monitor.name, recent_errors + 1, _retry_delay // 60, (new_value or "")[:100])
            return

        if _is_skip:
            # Record normally — this is expected behavior, not an error
            await asyncio.to_thread(self.store.record_check, monitor.id, new_value)
            await asyncio.to_thread(
                self.store.add_result,
                monitor.id, "ok", value=new_value[:12000] if new_value else "")
            return

        # Only record check (update last_check_at) on successful results
        await asyncio.to_thread(self.store.record_check, monitor.id, new_value)

        # Extract KG triples from all factual query monitors (skip non-factual ones).
        # We hold a strong reference to each create_task() result in `_kg_bg_tasks`
        # so the GC can't cancel the coroutine before it finishes (raw
        # asyncio.create_task without retention was the prior pattern, and
        # Python's docs flag that as unsafe). The done_callback logs at
        # WARNING when the extraction raised so failures surface in operator
        # logs instead of vanishing.
        if monitor.check_type == "query" and monitor.name not in _NO_KG_MONITORS and new_value and len(new_value) > 100:
            try:
                from app.core.brain import get_services, _extract_kg_triples
                svc = get_services()
                if svc.kg:
                    # Monitor digests are the main KG-growth pipe and are multi-
                    # paragraph: give the extractor the whole briefing and a higher
                    # triple budget (the lean 1000-char/5-triple chat default threw
                    # away the back half of every domain study). Route extraction to
                    # the synthesis model when set — a 4-arm A/B (2026-06-29) showed the
                    # 27B yields ~4.6× more grounded facts/digest at 100% grounding vs
                    # the 9B; the 27B is already loaded from this monitor's synthesis.
                    _syn = (getattr(config, "MONITOR_SYNTHESIS_MODEL", "") or "").strip() or None
                    _kg_task = asyncio.create_task(
                        _extract_kg_triples(svc.kg, monitor.name, new_value[:12000],
                                            source_name=monitor.name,
                                            max_answer_chars=12000, max_triples=22,
                                            model=_syn, trust=0.7)
                    )
                    self._kg_bg_tasks.add(_kg_task)

                    def _on_kg_done(t: asyncio.Task, _name: str = monitor.name) -> None:
                        self._kg_bg_tasks.discard(t)
                        if t.cancelled():
                            logger.warning("[KG bg] extraction for %r was cancelled", _name)
                            return
                        exc = t.exception()
                        if exc is not None:
                            logger.warning("[KG bg] extraction for %r raised: %s", _name, exc)

                    _kg_task.add_done_callback(_on_kg_done)
            except Exception:
                # This except used to be a bare `pass` — an import/services
                # error here silently killed the MAIN KG-growth pipe forever
                # (every digest banking zero facts, no log line). Audit
                # 2026-07-08: silent kill switches on the memory loop are the
                # same failure class as the 2026-05-30 postmortem.
                logger.error(
                    "[KG bg] failed to schedule KG extraction for %r — "
                    "digest facts NOT banked", getattr(monitor, "name", "?"),
                    exc_info=True,
                )

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
                if monitor.check_type in _CANARY_NORMAL_MARKERS:
                    # Stat-line canaries: numbers drift every run by design,
                    # so numeric detect_change re-delivered healthy readouts
                    # every cycle (stored status=alert), burying the canaries'
                    # real signal (live 2026-08-26: 3 "growth normal" + 3
                    # "ollama healthy" alerts/day). Deliver only warnings
                    # (which must repeat) and the recovery edge to healthy.
                    should_alert = _canary_should_alert(
                        monitor.check_type, monitor.last_result, new_value)
                    if should_alert:
                        change_info = detect_change(
                            monitor.last_result, new_value, threshold)
                else:
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
            await asyncio.to_thread(
                self.store.add_result,
                monitor.id, "ok", value=new_value[:12000] if new_value else "")
            return

        # Check cooldown
        if monitor.last_alert_at:
            last_alert = datetime.fromisoformat(monitor.last_alert_at).replace(tzinfo=None)
            now_naive = datetime.now(timezone.utc).replace(tzinfo=None)
            if (now_naive - last_alert).total_seconds() < monitor.cooldown_minutes * 60:
                logger.info("[Heartbeat] '%s' in cooldown, skipping alert", monitor.name)
                await asyncio.to_thread(
                    self.store.add_result,
                    monitor.id, "ok", value=new_value[:12000] if new_value else "",
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
            analysis = new_value[:12000] if new_value else ""

        # Empty-body gate: if Nova returned nothing meaningful, don't broadcast
        # a silent/placeholder message to the user's alert channels. Nothing
        # is better than noise in a monitor feed.
        if not analysis or len(analysis.strip()) < 20:
            logger.info(
                "[Heartbeat] '%s' produced empty/tiny body (%d chars); skipping alert",
                monitor.name, len(analysis.strip()) if analysis else 0,
            )
            await asyncio.to_thread(
                self.store.add_result,
                monitor.id, "ok", value=new_value[:12000] if new_value else "",
                message="empty_body_suppressed")
            return

        # Send alert
        delivered = await self._send_alert(monitor, analysis)

        # Auto-disable one-shot reminders after first alert (supports both
        # legacy "[Reminder]" and current "reminder:" prefixes). Recurring
        # reminders carry a `recurring` flag in check_config and stay enabled.
        is_reminder = (
            monitor.name.startswith("[Reminder]")
            or monitor.name.startswith("reminder:")
        )
        if is_reminder:
            cfg = monitor.check_config or {}
            if cfg.get("recurring"):
                # Recurring: monitor's schedule_seconds is the recurrence period;
                # leave enabled so it fires again next cycle.
                logger.info(
                    "[Heartbeat] Recurring reminder '%s' fired — staying enabled "
                    "(period=%ds)",
                    monitor.name, monitor.schedule_seconds,
                )
            else:
                await asyncio.to_thread(self.store.update, monitor.id, enabled=False)
                logger.info("[Heartbeat] Reminder '%s' auto-disabled after alert", monitor.name)

        # Record. A result that reached this point WAS delivered (or buffered
        # for delivery) — it is an alert. The old `"changed" if change_info
        # else "ok"` stored every always-notify digest (the bulk of the feed)
        # as 'ok': indistinguishable from a suppressed run in every metric,
        # AND on the hard-delete path of the demote-don't-delete retention,
        # which protects status='alert' only (deep pass 2026-08-14).
        status = "alert" if delivered else "ok"
        if delivered and change_info and change_info.get("type") != "numeric":
            status = "changed"
        if delivered:
            # Only advance cooldown / last_alert_at when something actually went
            # out (or was journaled for delivery). Recording a suppressed run
            # previously advanced the cooldown for an alert that never fired.
            await asyncio.to_thread(self.store.record_alert, monitor.id)
        await asyncio.to_thread(
            self.store.add_result,
            monitor.id, status, value=new_value[:12000] if new_value else "",
            message=((analysis[:500] if analysis else "") if delivered
                     else "not delivered (suppressed or channel failure)"))

    # Registry: check_type -> handler. Adding a new check type is one method
    # plus one entry here — _execute_check never changes. Lambdas adapt the
    # handlers' real signatures (cfg-only / no-arg / monitor-arg) to a uniform
    # (self, monitor, cfg) dispatch call.
    _CHECK_DISPATCH = {
        "url": lambda self, m, cfg: self._execute_url(cfg),
        "search": lambda self, m, cfg: self._execute_search(cfg),
        "command": lambda self, m, cfg: self._execute_command(cfg),
        "system_health": lambda self, m, cfg: self._execute_system_health(),
        "query": lambda self, m, cfg: self._execute_query_monitor(m, cfg),
        "quiz": lambda self, m, cfg: self._execute_quiz(cfg),
        "skill_test": lambda self, m, cfg: self._execute_skill_test(cfg),
        "curiosity": lambda self, m, cfg: self._execute_curiosity_research(cfg),
        "auto_monitor": lambda self, m, cfg: self._execute_auto_monitor_detection(cfg),
        "maintenance": lambda self, m, cfg: self._execute_maintenance(cfg),
        "finetune": lambda self, m, cfg: self._execute_finetune_check(cfg),
        "dream_consolidation": lambda self, m, cfg: self._execute_consolidation(cfg),
        "capability_review": lambda self, m, cfg: self._execute_capability_review(cfg),
        "eval": lambda self, m, cfg: self._execute_eval_harness(cfg),
        "prompt_analyzer": lambda self, m, cfg: self._execute_prompt_analyzer(cfg),
        "db_size": lambda self, m, cfg: self._execute_db_size_check(),
        "feed_health": lambda self, m, cfg: self._execute_feed_health(),
        "kg_consistency": lambda self, m, cfg: self._execute_kg_consistency(),
        "ollama_latency": lambda self, m, cfg: self._execute_ollama_latency_check(),
        "skill_quality": lambda self, m, cfg: self._execute_skill_quality_check(),
        "chromadb_integrity": lambda self, m, cfg: self._execute_chromadb_integrity_check(),
        "kg_health": lambda self, m, cfg: self._execute_kg_health_check(),
        "digest_health": lambda self, m, cfg: self._execute_digest_health(),
        "training_job": lambda self, m, cfg: self._execute_training_job_check(),
        "kg_growth": lambda self, m, cfg: self._execute_kg_growth_check(m),
        "ollama_model": lambda self, m, cfg: self._execute_ollama_model_check(),
        "goal_derivation": lambda self, m, cfg: self._execute_goal_derivation(),
        "synthesis": lambda self, m, cfg: self._execute_cross_synthesis(),
        "storyline": lambda self, m, cfg: self._execute_storyline_tracker(),
        "consolidation": lambda self, m, cfg: self._execute_knowledge_consolidation(),
        "forecast_resolve": lambda self, m, cfg: self._execute_forecast_resolve(),
        "auto_tool": lambda self, m, cfg: self._execute_auto_tool_synthesis(),
        "output_eval": lambda self, m, cfg: self._execute_output_eval(),
    }

    async def _execute_check(self, monitor: Monitor) -> str:
        """Run the actual check based on monitor type (registry dispatch)."""
        handler = self._CHECK_DISPATCH.get(monitor.check_type)
        if handler is None:
            return f"[Unknown check_type: {monitor.check_type}]"
        return await handler(self, monitor, monitor.check_config)

    async def _execute_url(self, cfg: dict) -> str:
        from app.core.brain import get_services
        svc = get_services()
        url = cfg.get("url", "")
        if svc.tool_registry:
            return await svc.tool_registry.execute("http_fetch", {"url": url})
        return f"[No tool registry — cannot fetch {url}]"

    async def _execute_search(self, cfg: dict) -> str:
        from app.core.brain import get_services
        svc = get_services()
        query = cfg.get("query", "")
        if svc.tool_registry:
            return await svc.tool_registry.execute("web_search", {"query": query})
        return "[No tool registry — cannot search]"

    async def _execute_command(self, cfg: dict) -> str:
        from app.core.brain import get_services
        svc = get_services()
        command = cfg.get("command", "")
        if svc.tool_registry:
            return await svc.tool_registry.execute("shell_exec", {"command": command})
        return "[No tool registry — cannot exec]"

    async def _execute_query_monitor(self, monitor: Monitor, cfg: dict) -> str:
        # For Domain Study:* monitors use the direct-fetch runner that
        # gets dates from the search engine (not from the LLM's belief
        # about what year it is). nova-ft hedges dates badly and the
        # citation gate then fails everything; the direct-fetch runner
        # sidesteps that by handing pre-verified items to the LLM only
        # for formatting.
        # Route through the direct-fetch runner if Domain Study:* OR
        # the monitor has curated RSS feeds (SEC Insider Trading, FOMC,
        # Hacker News, FDA, etc). brain.think() hallucinates fake
        # filings and dates for these niche topics — the runner pulls
        # real items from real RSS sources.
        from app.monitors.rss_feeds import feeds_for
        # "Auto:*" monitors (created by the Auto-Monitor Detector for frequently-asked
        # topics) route through the SAME rich deep-research pipeline as Domain Studies,
        # so every current AND future topic gets the full synthesized overview — not the
        # thin brain.think() bullet list. (owner: "on every topic and future topics too")
        if monitor.name.startswith(("Domain Study:", "Auto:")) or feeds_for(monitor.name):
            from app.monitors.domain_study_runner import run_domain_study
            try:
                result = await run_domain_study(monitor.name)
            except Exception as e:
                logger.exception("[Heartbeat] domain_study_runner failed")
                result = f"## ⚠️ {monitor.name} — runner error\n\n{e}"
            return result
        # Operator/internal queries (Morning Check-in, [Reminder]:* etc)
        # keep the brain.think() path.
        query = cfg.get("query", "")
        return await self._think_query(query)

    async def _execute_kg_consistency(self) -> str:
        from app.monitors.kg_consistency import run_kg_consistency_check
        return await run_kg_consistency_check()

    async def _execute_goal_derivation(self) -> str:
        """Derive new goals from operational state. The KAIROS executor
        picks them up on its next tick."""
        from app.database import get_db
        from app.core.goal_deriver import derive_and_log
        try:
            return await derive_and_log(get_db())
        except Exception as e:
            logger.exception("[Heartbeat] Goal derivation failed")
            return f"GOAL DERIVATION ERROR: {e}"

    async def _execute_cross_synthesis(self) -> str:
        """Read recent monitor outputs across categories, surface cross-cutting
        themes, write them to the KG as cross_synthesis facts."""
        from app.database import get_db
        from app.core.brain import get_services
        from app.core.cross_monitor import synthesize_and_log
        try:
            svc = get_services()
            kg = getattr(svc, "kg", None)
            return await synthesize_and_log(get_db(), kg)
        except Exception as e:
            logger.exception("[Heartbeat] Cross-monitor synthesis failed")
            return f"CROSS-SYNTHESIS ERROR: {e}"

    async def _execute_storyline_tracker(self) -> str:
        """Cluster recent monitor items into ongoing STORY THREADS, diff what's
        new vs each thread's prior state, and surface ONLY the threads that moved
        ("here's how your threads changed" — not raw headlines)."""
        if not getattr(config, "ENABLE_STORYLINES", True):
            return "STORYLINES | disabled (ENABLE_STORYLINES=false)"
        from app.database import get_db
        from app.core.brain import get_services
        from app.core.storylines import track_storylines
        try:
            kg = getattr(get_services(), "kg", None)
            return await track_storylines(get_db(), kg)
        except Exception as e:
            logger.exception("[Heartbeat] Storyline tracker failed")
            return f"STORYLINE ERROR: {e}"

    async def _execute_knowledge_consolidation(self) -> str:
        """Distill recent digests + moved storylines into standing DOSSIERS —
        the knowing tier (2026-08-12). Runs daily, before the 30-day
        monitor_results retention can shred the analytical content."""
        if not getattr(config, "ENABLE_DOSSIERS", True):
            return "KNOWING | disabled (ENABLE_DOSSIERS=false)"
        from app.database import get_db
        from app.core.dossiers import consolidate_dossiers
        try:
            return await consolidate_dossiers(get_db())
        except Exception as e:
            logger.exception("[Heartbeat] Knowledge consolidation failed")
            return f"KNOWING ERROR: {e}"

    async def _execute_forecast_resolve(self) -> str:
        """Grade falsifiable forecasts whose horizon has passed (hit/miss), and
        report the rolling track record — Nova scores its own calls."""
        if not getattr(config, "ENABLE_FORECASTS", True):
            return "FORECASTS | disabled (ENABLE_FORECASTS=false)"
        from app.database import get_db
        from app.core.forecasts import resolve_due
        try:
            return await resolve_due(get_db())
        except Exception as e:
            logger.exception("[Heartbeat] Forecast resolution failed")
            return f"FORECAST ERROR: {e}"

    async def _execute_auto_tool_synthesis(self) -> str:
        """Mine capability_gap clusters, ask the LLM to write a tool to fix
        each, and store passes in custom_tools — Nova literally writes its
        own tools without needing a code rebuild."""
        from app.database import get_db
        from app.core.auto_tools import synthesize_and_log
        try:
            return await synthesize_and_log(get_db())
        except Exception as e:
            logger.exception("[Heartbeat] Auto-tool synthesis failed")
            return f"AUTO-TOOL ERROR: {e}"

    async def _execute_output_eval(self) -> str:
        """Grade a sample of recent monitor outputs on relevance/facts/
        freshness/format. Tracks production-quality drift over time."""
        from app.database import get_db
        from app.core.output_eval import grade_and_log
        try:
            return await grade_and_log(get_db())
        except Exception as e:
            logger.exception("[Heartbeat] Output eval failed")
            return f"OUTPUT EVAL ERROR: {e}"

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
        # The whole block is sequential sync store reads; it runs via
        # to_thread so the event loop never waits on the SQLite lock here
        # (this exact path blocked the loop >60s in the 2026-06-11 incident).
        def _build_context_lines() -> list[str]:
            lines: list[str] = []
            svc = get_services()

            # Monitors
            monitors = self.store.list_all()
            enabled = [m for m in monitors if m.enabled]
            lines.append(
                f"Monitors: {len(monitors)} total, {len(enabled)} enabled — "
                + ", ".join(m.name for m in monitors)
            )

            # Recent alerts (24h)
            recent = self.store.get_recent_results(hours=24, limit=20)
            if recent:
                alerts = [r for r in recent if r.status in ("alert", "changed", "error")]
                lines.append(f"Last 24h: {len(recent)} results, {len(alerts)} alerts/changes")
            else:
                lines.append("Last 24h: no monitor results yet")

            # Recent conversations
            if svc.conversations:
                convos = svc.conversations.list_conversations(limit=10)
                if convos:
                    titles = [c.get("title") or "(untitled)" for c in convos]
                    lines.append(f"Recent conversations ({len(convos)}): " + ", ".join(titles))
                else:
                    lines.append("Recent conversations: none")

            # Learning summary with actual content
            if svc.learning:
                summary = svc.learning.get_learning_summary(hours=24)
                parts = []
                if summary.get("new_lessons"):
                    parts.append(f"{len(summary['new_lessons'])} new lesson(s)")
                    for les in summary["new_lessons"][:5]:
                        topic = les.get("topic", "?")[:60]
                        lesson_text = (les.get("lesson_text") or les.get("correct_answer", ""))[:100]
                        lines.append(f"  Lesson: {topic} — {lesson_text}")
                if summary.get("new_skills"):
                    parts.append(f"{len(summary['new_skills'])} new skill(s)")
                if summary.get("degraded_skills"):
                    parts.append(f"{len(summary['degraded_skills'])} degraded skill(s)")
                if summary.get("new_reflexions"):
                    parts.append(f"{len(summary['new_reflexions'])} new reflexion(s)")
                    for ref in summary["new_reflexions"][:5]:
                        task = ref.get("task_summary", "?")[:60]
                        score = ref.get("quality_score", 0)
                        lines.append(f"  Reflexion (quality={score:.1f}): {task}")
                lines.append("Learning (24h): " + (", ".join(parts) if parts else "no activity"))

            # Owner facts
            if svc.user_facts:
                facts = svc.user_facts.get_all()
                if facts:
                    lines.append(
                        f"Known owner facts ({len(facts)}): "
                        + ", ".join(f"{f.key}={f.value}" for f in facts[:10])
                    )
            return lines

        ctx_lines: list[str] = []
        try:
            ctx_lines = await asyncio.to_thread(_build_context_lines)
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
                async for event in think(query=enriched_query, ephemeral=True, channel="monitor"):
                    if event.type == EventType.TOKEN:
                        text = event.data.get("text", "")
                        if text:
                            tokens.append(text)
        except asyncio.TimeoutError:
            logger.warning("[Heartbeat] _think_query timed out for: %s", query[:80])
            return "[Query timed out]"
        except Exception as e:
            logger.error("[Heartbeat] think() failed: %s", e, exc_info=True)
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
                async for event in think(inst.instruction, ephemeral=True, channel="monitor"):
                    if event.type == EventType.TOKEN:
                        text = event.data.get("text", "")
                        if text:
                            tokens.append(text)
        except (TimeoutError, asyncio.TimeoutError):
            logger.warning("[Heartbeat] Instruction #%d timed out after %ds", inst.id, config.GENERATION_TIMEOUT)
            await asyncio.to_thread(self.store.record_instruction_run, inst.id)
            return
        except Exception as e:
            logger.error("[Heartbeat] Instruction #%d failed: %s", inst.id, e)
            await asyncio.to_thread(self.store.record_instruction_run, inst.id)
            return

        result = "".join(tokens).strip()
        await asyncio.to_thread(self.store.record_instruction_run, inst.id)

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

        lessons = await asyncio.to_thread(svc.learning.get_all_lessons, limit=200)
        if not lessons:
            return "[No lessons to quiz on — skipped]"

        # Spaced repetition: skip lessons stuck in failure loops (5+ failures, quizzed < 7 days ago).
        # Priority: lessons with PENDING failures from >7 days ago jump the queue —
        # they've been waiting for a re-test to either close (#167) or escalate.
        # Otherwise NULLS FIRST → unquizzed; then oldest-quizzed by failure count.
        db = svc.learning._db
        candidate_rows = await asyncio.to_thread(
            db.fetchall,
            "SELECT id FROM lessons "
            "WHERE (quiz_failures < 5 "
            "   OR last_quizzed_at < datetime('now', '-7 days') "
            "   OR last_quizzed_at IS NULL) "
            "AND correct_answer IS NOT NULL AND correct_answer != '' "
            "ORDER BY "
            "  (CASE WHEN quiz_failures > 0 "
            "        AND (last_quizzed_at IS NULL OR last_quizzed_at < datetime('now','-7 days')) "
            "        THEN 0 ELSE 1 END), "
            "  last_quizzed_at ASC NULLS FIRST, "
            "  quiz_failures DESC "
            "LIMIT 10"
        )
        # Technique lessons are unquizzable (2026-08-27): a correct_answer
        # that says "do a web search" / "use the calculator" is a PROCESS
        # instruction, not a gradable fact — the generator has nothing
        # factual to ask for, and the grader then fails whatever the student
        # says against it. The art-history lesson looped this way for weeks:
        # quiz fail → curiosity re-research → same lesson → fail again.
        # 29 of 44 current lessons are this shape, so stamp EVERY gated one
        # encountered (the spaced-repetition picker moves past them) and quiz
        # the first factual candidate — one cycle drains gated backlog AND
        # still runs a real quiz. Behavior-shaped lessons are covered by the
        # nightly eval's tool-use category instead.
        lesson = None
        gated = 0
        by_id = {l.id: l for l in lessons}
        for r in candidate_rows or []:
            cand = by_id.get(r["id"])
            if cand is None:
                continue
            if _UNQUIZZABLE_ANSWER_RE.search((cand.correct_answer or "")[:200]):
                await asyncio.to_thread(
                    db.execute,
                    "UPDATE lessons SET last_quizzed_at=datetime('now') WHERE id=?",
                    (cand.id,))
                gated += 1
                continue
            lesson = cand
            break
        if not lesson:
            # Fallback: pick a random lesson that has usable, gradable content
            usable = [l for l in lessons
                      if l.correct_answer and len(l.correct_answer) > 20
                      and not _UNQUIZZABLE_ANSWER_RE.search(l.correct_answer[:200])]
            if not usable:
                return (f"[No quizzable lessons — {gated} technique lesson(s) "
                        f"stamped this cycle]")
            lesson = random.choice(usable)
        if gated:
            logger.info("[Quiz] stamped %d unquizzable technique lesson(s) this cycle", gated)

        # Step 1: Generate a question from the lesson.
        # NOT lesson.context: that field holds provenance/bookkeeping text
        # ("Promoted from success reflexion (quality 0.82)"), and max-by-
        # length kept selecting it — a live quiz asked "what is the quality
        # score associated with 'Promoted from success reflexion'"
        # (2026-08-27). Ground questions in the knowledge fields only.
        context_candidates = [lesson.lesson_text or '', lesson.correct_answer or '']
        context_text = max(context_candidates, key=len)
        if len(context_text.strip()) < 20:
            return f"[Lesson '{lesson.topic}' has insufficient context for quiz — skipped]"
        # The question MUST be answerable by the stored correct_answer — that
        # is the only reference the grader has. The old free-form prompt let
        # the model invent NEW problems from technique lessons ("Use
        # calculators..." → "Calculate 12×45 + (80−67)÷9"), which the grader
        # then failed against the UNRELATED stored answer it is ordered to
        # treat as ground truth. Structural false failures on the whole
        # math/tool-technique lesson family (2 of the last 3 quizzes,
        # observed 2026-08-27), feeding false negatives into lesson
        # confidence and curiosity re-research.
        gen_prompt = (
            f"Topic: {lesson.topic}\n"
            f"The correct answer (the question must ask for THIS): {lesson.correct_answer}\n"
            f"Additional context: {context_text[:400]}\n\n"
            "Write a single quiz question whose correct answer is exactly the "
            "answer given above. Do NOT invent new calculations, numbers, or "
            "scenarios that the given answer does not cover. "
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
            # Concise: an unbounded ramble overran the 600-token cap and got cut
            # mid-answer, which the grader then scored as a knowledge FAILURE
            # (false-failure reflexion, 2026-08-15 watch). 2-3 sentences is ample
            # for a quiz answer and keeps grading clean.
            "Answer concisely in 2-3 sentences based on the context provided."
        )
        try:
            answer = await llm.invoke_nothink(
                [{"role": "user", "content": answer_prompt}],
                # 900, was 600: the concise instruction alone doesn't stop the
                # 9B from rambling past the cap (re-fired 2026-08-19), and a
                # mid-answer cut grades as a false knowledge FAILURE.
                max_tokens=900, temperature=0.3,
            )
            answer = answer.strip()
            # Belt-and-suspenders: grade only complete sentences — drop any
            # trailing fragment so a cut never reads as a wrong answer.
            m = re.search(r"^(.*[.!?])(?:[^.!?]*)$", answer, re.DOTALL)
            if m and len(m.group(1)) >= 60:
                answer = m.group(1)
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
            await asyncio.to_thread(
                db.execute,
                "UPDATE lessons SET last_quizzed_at = ? WHERE id = ?",
                (now_str, lesson.id),
            )
        except Exception as e:
            logger.warning("[Heartbeat] Quiz tracking update failed: %s", e)

        # RLVR — record quiz outcome as a verifiable signal regardless of pass/fail.
        try:
            from app.config import config as _cfg
            if getattr(_cfg, "ENABLE_RLVR_SIGNALS", False):
                from app.core import rlvr as _rlvr
                _rlvr.record_signal(
                    "quiz_correct",
                    1.0 if passed else 0.0,
                    query=str(question)[:500],
                    response=str(answer)[:500],
                    evidence=f"lesson_id={lesson.id} topic={(lesson.topic or '')[:80]}",
                )
        except Exception:
            pass

        if passed:
            # Reinforce the lesson
            try:
                await asyncio.to_thread(svc.learning.mark_lesson_helpful, lesson.id)
            except Exception as e:
                logger.warning("[Heartbeat] mark_lesson_helpful failed: %s", e)
            # CLOSURE: clear the quiz_failures counter — the lesson has been
            # re-validated. This is the closure signal for the
            # quiz-fail → curiosity-research → re-quiz feedback loop.
            try:
                cleared_row = await asyncio.to_thread(
                    db.fetchone,
                    "SELECT quiz_failures FROM lessons WHERE id = ?", (lesson.id,)
                )
                prior_failures = int(cleared_row["quiz_failures"]) if cleared_row else 0
                if prior_failures > 0:
                    await asyncio.to_thread(
                        db.execute,
                        "UPDATE lessons SET quiz_failures = 0 WHERE id = ?",
                        (lesson.id,),
                    )
                    logger.info(
                        "[Quiz] CLOSURE: cleared %d prior failures on lesson #%d (%s)",
                        prior_failures, lesson.id, lesson.topic[:60],
                    )
            except Exception as e:
                logger.warning("[Heartbeat] Quiz failure reset failed: %s", e)
            return f"QUIZ PASSED | topic={lesson.topic} | q={question[:80]} | a={answer[:80]}"

        # Failed — increment quiz_failures counter
        try:
            await asyncio.to_thread(
                db.execute,
                "UPDATE lessons SET quiz_failures = COALESCE(quiz_failures, 0) + 1 WHERE id = ?",
                (lesson.id,),
            )
        except Exception as e:
            logger.warning("[Heartbeat] Quiz failure increment failed: %s", e)

        # Failed — reduce lesson confidence, create training pair, reflexion
        fail_reason = grade.get("reason", "incorrect")

        try:
            await asyncio.to_thread(svc.learning.mark_lesson_unhelpful, lesson.id)
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

        skills = await asyncio.to_thread(svc.skills.get_active_skills)
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
                # Last-resort fallback: ask the LLM to invent ONE concrete string
                # that would match the regex. Works for skills whose regex needs
                # a digit ("\d+ days in seconds") or specific casing ("TVL")
                # that the keyword-group prompt above misses. We only ask for the
                # match; we don't run brain on a synthetic query unless it does
                # actually match the trigger.
                regex_prompt = (
                    "Here is a Python regular expression:\n"
                    f"  {skill.trigger_pattern}\n\n"
                    "Output ONE short example user query (under 10 words) that this "
                    "regex would match. No explanation, no quotes, just the query."
                )
                try:
                    candidate = await llm.invoke_nothink(
                        [{"role": "user", "content": regex_prompt}],
                        max_tokens=40, temperature=0.4,
                    )
                    candidate = candidate.strip().strip('"\'').strip()
                    if re.search(skill.trigger_pattern, candidate, re.IGNORECASE):
                        test_query = candidate
                except Exception as e:
                    logger.debug("[Heartbeat] regex-fallback skill query failed: %s", e)
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
        # to_thread (2026-08-29): record_use is a sync UPDATE on skills and this
        # is an async monitor path — it wrote from the event-loop thread.
        await asyncio.to_thread(svc.skills.record_use, skill.id, passed)
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

        item = await asyncio.to_thread(svc.curiosity.get_next)
        if not item:
            return "[No pending curiosity items — skipped]"

        # Research via think() — memory-first, web only for public/external facts.
        research_query = (
            f"Research question: {item.topic}\n\n"
            f"First consult personal memory (user facts, knowledge graph, conversation history). "
            f"If the question is about the user's own projects, preferences, or things they've said, "
            f"answer from memory only — do NOT web search. "
            f"Only use web_search for clearly public/external information (news, prices, public figures, public documentation). "
            f"If a name is ambiguous — could refer to a user project OR a commercial product with the same name — "
            f"assume user context and answer from memory. "
            f"Provide a concise, factual summary."
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
                # --- Semantic closure check ---
                # Verify the result ACTUALLY answers the original question.
                # Pattern check filters obvious deflections; LLM judge handles
                # the rest. Failed closure → requeue (fail() bumps attempts).
                _resolved_ok = await self._curiosity_closure_check(item.topic, result)
                if not _resolved_ok:
                    # ANALYTICAL/forecast questions ("Can India's markets absorb
                    # $50B?" — the dossier open-question style) structurally
                    # fail a FACTUAL closure judge: their best-possible answer
                    # is a hedged synthesis, which the judge rejects as not
                    # "concrete". Pre-2026-08-18 that meant 3 research passes
                    # per item, all DISCARDED (~16h of GPU across the backlog,
                    # zero knowledge stored). If the question is analytical and
                    # the research is substantive (stage-1 deflection heuristics
                    # already passed to reach the judge), keep the knowledge:
                    # resolve as PROVISIONAL and bank grounded facts below.
                    # Factual topics keep the strict requeue-on-fail path.
                    if (_ANALYTICAL_TOPIC_RE.search(item.topic or "")
                            and _provisional_acceptable(item.topic or "", result)):
                        if svc.kg and len(result) > 50:
                            from app.core.brain import _extract_kg_triples
                            try:
                                await _extract_kg_triples(svc.kg, item.topic, result, trust=0.5,
                                                          source_name="Curiosity Research")
                            except Exception:
                                pass
                        await asyncio.to_thread(
                            svc.curiosity.resolve, item.id,
                            "[provisional] " + result[:1985])
                        logger.info("[Curiosity] analytical question resolved as provisional: %s",
                                    item.topic[:80])
                        return (f"CURIOSITY PROVISIONAL | topic={item.topic[:80]} | "
                                f"analytical question — hedged synthesis stored")
                    # Degraded-search guard (2026-08-26): when most recent web
                    # searches are coming back empty (network/engine outage —
                    # e.g. the DDG IP block), a thin answer that fails closure
                    # is the NETWORK's fault, not the topic's. Mirror the
                    # LLM-down guard above: requeue without burning an attempt.
                    from app.tools.native_search import search_health
                    if search_health() < 0.25:
                        logger.info("[Curiosity] closure failed under degraded search "
                                    "(health=%.2f) — deferred without attempt burn: %s",
                                    search_health(), item.topic[:80])
                        return (f"CURIOSITY DEFERRED | topic={item.topic[:80]} | "
                                f"reason=search_degraded")
                    await asyncio.to_thread(svc.curiosity.fail, item.id)
                    logger.info("[Curiosity] closure check failed — requeued: %s", item.topic[:80])
                    return f"CURIOSITY UNRESOLVED | topic={item.topic[:80]} | reason=closure_check_failed"

                # Store findings in KG if possible
                if svc.kg and len(result) > 50:
                    from app.core.brain import _extract_kg_triples
                    try:
                        await _extract_kg_triples(svc.kg, item.topic, result, trust=0.55,
                                                  source_name="Curiosity Research")
                    except Exception:
                        pass

                await asyncio.to_thread(svc.curiosity.resolve, item.id, result[:2000])

                # --- Convert research findings into a lesson ---
                # Gate: only create a lesson when the research result LOOKS LIKE actual
                # findings, not a "I cannot do this" / "I don't have access" deflection.
                # Without this gate, failed-research outputs become poisoned lessons
                # ("Tooling lacks capability to retrieve lessons" — false).
                _result_lower = result.lower()
                _looks_like_failure = any(p in _result_lower[:300] for p in (
                    "i cannot", "i can't", "i don't have", "i dont have",
                    "unable to", "lacks the capability", "not able to",
                    "without access", "requires direct access",
                    "limitations", "no data", "no findings",
                ))
                if svc.learning and not _looks_like_failure:
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
                            await asyncio.to_thread(
                                svc.learning.add_knowledge_lesson,
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
                await asyncio.to_thread(svc.curiosity.fail, item.id)
                return f"CURIOSITY FAILED | topic={item.topic[:80]} | result={result[:100]}"
        except Exception as e:
            await asyncio.to_thread(svc.curiosity.fail, item.id)
            return f"CURIOSITY ERROR | topic={item.topic[:80]} | error={e}"

    async def _curiosity_closure_check(self, topic: str, result: str) -> bool:
        """Return True if `result` plausibly answers the curiosity `topic`.

        Two-stage: cheap heuristic (length + deflection patterns) → LLM judge.
        The LLM judge runs on FAST_MODEL with strict json output, single call.
        """
        # Stage 1: cheap heuristics
        if not result or len(result.strip()) < 80:
            return False
        rl = result.lower()[:600]
        deflection_markers = (
            "i cannot", "i can't", "i don't have", "i dont have",
            "unable to", "lacks the capability", "not able to",
            "without access", "requires direct access",
            "no findings", "no data available", "unclear from",
            "i'm not sure", "i am not sure", "uncertain about",
        )
        if any(m in rl for m in deflection_markers):
            return False

        # Stage 2: LLM judge — does this answer the question?
        try:
            from app.core import llm as llm_mod
            # Date-anchored: without this the FAST_MODEL's training-cutoff prior
            # rejects real post-cutoff events as "fictional future scenarios"
            # (seen live 2026-08-19: a researched answer about the sitting Fed
            # chair was requeued forever). Judge concreteness, not plausibility.
            _today = datetime.now(timezone.utc).strftime("%A, %B %d, %Y")
            judge_prompt = (
                f"TODAY IS {_today}. The proposed answer was produced from LIVE web research and "
                f"may describe events after your training cutoff — those events are real. Judge "
                f"ONLY whether the answer concretely addresses the question, NOT whether its "
                f"events match your training knowledge.\n\n"
                f"QUESTION: {topic[:300]}\n\n"
                f"PROPOSED ANSWER:\n{result[:1200]}\n\n"
                f"Did the proposed answer actually answer the question with concrete information? "
                f"Reply with JSON: {{\"answers\": true|false, \"reason\": \"<one short sentence>\"}}"
            )
            raw = await llm_mod.invoke_nothink(
                [{"role": "user", "content": judge_prompt}],
                json_mode=True, json_prefix="{",
                max_tokens=120, model=config.FAST_MODEL, temperature=0.0,
            )
            obj = llm_mod.extract_json_object(raw) or {}
            answered = bool(obj.get("answers"))
            if not answered:
                logger.info("[Curiosity] judge says no: %s", str(obj.get("reason", ""))[:120])
            return answered
        except Exception as e:
            # On judge failure, default to True so we don't loop forever
            logger.warning("[Curiosity] closure judge failed (defaulting to resolve): %s", e)
            return True

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

        candidates = await asyncio.to_thread(svc.topic_tracker.get_monitor_candidates, min_count=3, days=7)
        if not candidates:
            return "[No monitor candidates found — skipped]"

        # Filter out invalid/low-quality topics
        from app.core.curiosity import CuriosityQueue
        import re as _re
        # Python 3.12 disallows inline (?i) anywhere except position 0; rely on
        # the IGNORECASE flag below instead.
        _BAD_MONITOR_RE = _re.compile(
            r"^(?:what|who|where|when|how|is|are|was|were|do|does|did|can|could|will|would|should|find|search|look\s+up|show|tell|give|list)\b"  # questions + imperative verbs
            r"|\b(?:price|cost|worth|trading at|how much)\b"  # price queries
            r"|\b(?:dont search|don.t search|just tell|from memory)\b"  # test queries
            r"|\b(?:time is it|what time|current time)\b"  # time queries
            r"|\b(?:calculate|compute|solve|equation|multipl[yi](?:ed)?|divid(?:e|ed)|plus|minus|equals?)\b"  # math
            r"|\b(?:write|generate|create|make me)\b"  # generation requests
            r"|\bgreat\s+question\b"  # conversational filler
            r"|\bshow\s+work\b"  # math/homework
            r"|^\d+\s*[\+\-\*x×]\s*\d+"  # bare arithmetic ("847 x 193")
            r"|\bbefore\s+you\s+answer\b",  # adversarial framing
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
        existing_monitors = {m.name.lower() for m in await asyncio.to_thread(self.store.list_all)}
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
            mid = await asyncio.to_thread(
                self.store.create,
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
                decayed = await asyncio.to_thread(svc.learning.decay_stale_lessons, days=30)
                if decayed:
                    parts.append(f"lessons decayed: {decayed}")
            except Exception as e:
                parts.append(f"lesson decay failed: {e}")
                logger.warning("[Heartbeat] Lesson decay failed: %s", e)
            try:
                deleted = await asyncio.to_thread(svc.learning.prune_dead_lessons)
                if deleted:
                    parts.append(f"dead lessons pruned: {deleted}")
            except Exception as e:
                parts.append(f"dead-lesson prune failed: {e}")
                logger.warning("[Heartbeat] Dead-lesson prune failed: %s", e)
        if svc.kg:
            try:
                decayed = await svc.kg.decay_stale(days=60)
                if decayed:
                    parts.append(f"KG facts decayed: {decayed}")
            except Exception as e:
                parts.append(f"KG decay failed: {e}")
                logger.warning("[Heartbeat] KG decay failed: %s", e)
            # Hard-retire never-retrieved old facts. Runtime audit found 92%
            # of KG facts are never queried — they're dead weight diluting
            # retrieval quality. Soft retire (valid_to set), not delete, so
            # they're still recoverable.
            try:
                pruned = await svc.kg.hard_prune_dead_facts(days=120, max_count=500)
                if pruned:
                    parts.append(f"KG dead-fact retire: {pruned}")
            except Exception as e:
                parts.append(f"KG hard-prune failed: {e}")
                logger.warning("[Heartbeat] KG hard-prune failed: %s", e)
            # related_to junk (61% of the store, audit 2026-07-09): retire the
            # vague associations that a specific predicate already covers + the
            # stale never-retrieved ones.
            try:
                rel = await svc.kg.prune_related_to_junk(days=45, max_count=1000)
                if rel:
                    parts.append(f"KG related_to junk retired: {rel}")
            except Exception as e:
                logger.warning("[Heartbeat] KG related_to prune failed: %s", e)
            # Point-in-time research facts (prices/percentages) expire after a
            # week — without this the KG fills with stale "current" truths.
            try:
                # 21-day window + release to sub-authoritative trust (0.6): the
                # quarantine holds only low-credibility single-source claims, so
                # age-release must not hand a patient poisoner an authoritative
                # fact (full-system exploration 2026-07-09).
                promoted = await svc.kg.promote_aged_quarantine(days=21, max_count=500)
                if promoted:
                    parts.append(f"KG quarantine age-released (low-trust): {promoted}")
                snap = await svc.kg.retire_stale_snapshots(days=7, max_count=500)
                if snap:
                    parts.append(f"KG stale snapshots retired: {snap}")
            except Exception as e:
                parts.append(f"KG snapshot-retire failed: {e}")
                logger.warning("[Heartbeat] KG snapshot-retire failed: %s", e)
            # Aggressively decay speculative cross_synthesis facts that no
            # query ever retrieved — closes the loop on synthesis quality.
            try:
                cs_decayed = await svc.kg.decay_unused_speculative(
                    provenance="cross_synthesis", days=14, decay_amount=0.15
                )
                cs_stats = await asyncio.to_thread(svc.kg.get_provenance_usage_stats, "cross_synthesis")
                parts.append(
                    f"cross_synthesis: total={cs_stats['total']} used={cs_stats['used']} "
                    f"avg_retrievals={cs_stats['avg_retrievals']:.1f} decayed={cs_decayed}"
                )
            except Exception as e:
                parts.append(f"cross_synthesis decay failed: {e}")
                logger.warning("[Heartbeat] cross_synthesis decay failed: %s", e)
        if svc.skills:
            # Disuse-based skill retirement (audit 2026-08-17): decay_stale_skills
            # was defined but NEVER called — a skill created and rarely/never
            # triggered never aged out (dream only disables skills that RAN and
            # failed). Wired in next to the lesson/KG/reflexion decays.
            try:
                decayed = await asyncio.to_thread(svc.skills.decay_stale_skills)
                if decayed:
                    parts.append(f"stale skills decayed: {decayed}")
            except Exception as e:
                parts.append(f"skill decay failed: {e}")
                logger.warning("[Heartbeat] Skill staleness decay failed: %s", e)
        # Knowing-tier retention (audit 2026-08-17): dossier_revisions and
        # storyline_events are append-only; keep a generous window so they don't
        # grow unbounded (volumes are small — keep-last-N / age prune).
        try:
            from app.database import get_db
            _db = get_db()

            def _knowing_retention():
                a = _db.execute(
                    "DELETE FROM dossier_revisions WHERE id NOT IN ("
                    "  SELECT id FROM dossier_revisions dr WHERE ("
                    "    SELECT COUNT(*) FROM dossier_revisions d2 "
                    "    WHERE d2.dossier_id = dr.dossier_id AND d2.id >= dr.id) <= 30)").rowcount
                b = _db.execute(
                    "DELETE FROM storyline_events "
                    "WHERE created_at < datetime('now', '-180 days')").rowcount
                return a, b
            drev, sev = await asyncio.to_thread(_knowing_retention)
            if drev or sev:
                parts.append(f"knowing-tier retention: {drev} revisions + {sev} events pruned")
        except Exception as e:
            logger.warning("[Heartbeat] knowing-tier retention failed: %s", e)
        if svc.reflexions:
            try:
                decayed = await asyncio.to_thread(svc.reflexions.decay_stale, days=90)
                if decayed:
                    parts.append(f"reflexions decayed: {decayed}")
            except Exception as e:
                parts.append(f"reflexion decay failed: {e}")
                logger.warning("[Heartbeat] Reflexion decay failed: %s", e)
            # Demote success patterns whose injection correlates with low quality
            # (A/B closure — useless suggestions get filtered out over time).
            try:
                useless_ids = await asyncio.to_thread(
                    svc.reflexions.get_useless_success_patterns,
                    min_uses=5, max_avg_quality=0.5,
                )
                if useless_ids:
                    placeholders = ",".join("?" for _ in useless_ids)
                    await asyncio.to_thread(
                        svc.reflexions._db.execute,
                        f"UPDATE reflexions SET outcome='failure' WHERE id IN ({placeholders})",
                        tuple(useless_ids),
                    )
                    parts.append(f"useless success patterns demoted: {len(useless_ids)}")
            except Exception as e:
                parts.append(f"success pattern A/B demotion failed: {e}")
                logger.warning("[Heartbeat] Success pattern demotion failed: %s", e)
        if svc.curiosity:
            try:
                pruned = await asyncio.to_thread(svc.curiosity.prune, days=30)
                if pruned:
                    parts.append(f"curiosity items pruned: {pruned}")
            except Exception as e:
                parts.append(f"curiosity prune failed: {e}")
                logger.warning("[Heartbeat] Curiosity prune failed: %s", e)
        # Disable auto-tools that aren't earning their keep — unused or low success rate.
        try:
            from app.core.auto_tools import prune_unused_tools, get_auto_tool_health
            from app.database import get_db
            _db = get_db()
            res = await asyncio.to_thread(prune_unused_tools, _db, min_age_days=3)
            if res.get("disabled"):
                parts.append(
                    f"auto-tools disabled: {res['disabled']} "
                    f"(unused={res.get('unused', 0)} bad={res.get('bad', 0)})"
                )
            health = await asyncio.to_thread(get_auto_tool_health, _db)
            if health.get("total", 0) > 0:
                parts.append(
                    f"auto-tool health: total={health['total']} enabled={health['enabled']} "
                    f"used={health['used']} avg_uses={health['avg_uses']:.1f} "
                    f"avg_success={health['avg_success']:.2f}"
                )
        except Exception as e:
            parts.append(f"auto-tool prune failed: {e}")
            logger.warning("[Heartbeat] Auto-tool prune failed: %s", e)
        # Audit log retention — keep 30 days for action_log, trust_audit_log,
        # and monitor_results. Was unbounded; 20k+ rows accumulated over 6 weeks
        # (monitor_results hit 13.7k rows of multi-KB TEXT by 2026-06 and was
        # part of the event-loop blocking incident).
        try:
            from app.database import get_db
            db = get_db()
            action_deleted = (await asyncio.to_thread(
                db.execute,
                "DELETE FROM action_log WHERE created_at < datetime('now', '-30 days')",
            )).rowcount
            if action_deleted:
                parts.append(f"action_log pruned: {action_deleted}")
            trust_deleted = (await asyncio.to_thread(
                db.execute,
                "DELETE FROM trust_audit_log WHERE timestamp < datetime('now', '-30 days')",
            )).rowcount
            if trust_deleted:
                parts.append(f"trust_audit pruned: {trust_deleted}")
            # Demote-don't-delete (CrystalMem pattern, adopted 2026-08-13):
            # digests are the knowing tier's evidence base, and capability lost
            # to hard deletion never fully recovers ("memory hysteresis",
            # arXiv:2608.00303). Content rows demote full → skeletal (30d) →
            # trace (90d) and are never deleted; non-content rows (ok/skip/
            # error — no knowledge value) still purge at 30 days.
            nonalert_deleted = (await asyncio.to_thread(
                db.execute,
                "DELETE FROM monitor_results WHERE created_at < datetime('now', '-30 days') "
                "AND status NOT IN ('alert', 'changed')",
            )).rowcount
            if nonalert_deleted:
                parts.append(f"monitor_results non-content pruned: {nonalert_deleted}")

            def _demote_pass():
                rows = db.fetchall(
                    "SELECT id, value FROM monitor_results WHERE "
                    "created_at < datetime('now', '-30 days') AND status IN ('alert', 'changed') "
                    "AND COALESCE(message,'') NOT LIKE 'demoted:%' AND LENGTH(value) > 1200 "
                    "LIMIT 200"
                )
                for r in rows:
                    db.execute(
                        "UPDATE monitor_results SET value = ?, message = 'demoted:skeletal' WHERE id = ?",
                        (_skeletal_digest(r["value"]), r["id"]),
                    )
                n_skel = len(rows)
                n_trace = db.execute(
                    "UPDATE monitor_results SET value = substr(value, 1, 200), "
                    "message = 'demoted:trace' WHERE "
                    "created_at < datetime('now', '-90 days') AND message = 'demoted:skeletal'",
                ).rowcount
                return n_skel, n_trace

            n_skel, n_trace = await asyncio.to_thread(_demote_pass)
            if n_skel or n_trace:
                parts.append(f"digests demoted: {n_skel} skeletal, {n_trace} trace")

            # Quarantine disposition (2026-08-14): jailed facts had NO exit
            # path — 811 rows accumulated since Jul 8, no release and no
            # expiry. Quarantine is a 30-day audit window for suspected
            # poisoning, not a life sentence: rows still jailed after 30 days
            # expire. (Unlike digests these are UNTRUSTED accusations, already
            # excluded from all retrieval — deletion is the safe direction.)
            q_expired = (await asyncio.to_thread(
                db.execute,
                "DELETE FROM kg_facts WHERE quarantined=1 "
                "AND created_at < datetime('now','-30 days')",
            )).rowcount
            if q_expired:
                parts.append(f"quarantine expired: {q_expired}")
            # Vector-index lifecycle hygiene (2026-08-14): supersessions,
            # expiries, and the quarantine purge above never touched their
            # VECTORS — the kg_facts collection grew to 3× the live set and
            # diluted every semantic top-k with dead rows.
            try:
                from app.core.brain import get_services
                _svc = get_services()
                if _svc.kg:
                    n_vec = await _svc.kg.prune_stale_vectors()
                    if n_vec:
                        parts.append(f"stale KG vectors pruned: {n_vec}")
                # Same hygiene for the lessons index (2026-08-20 sweep): it had
                # NO vector lifecycle, so churn left ghosts + unindexed lessons.
                if _svc.learning:
                    l_del, l_idx = await asyncio.to_thread(
                        _svc.learning.prune_and_backfill_lesson_vectors)
                    if l_del:
                        parts.append(f"stale lesson vectors pruned: {l_del}")
            except Exception as e:
                logger.warning("[Heartbeat] vector hygiene failed: %s", e)
            # ROT SWEEP (2026-08-25): the prunes above keep the Chroma VIEW
            # clean, but every delete is only an hnswlib tombstone — never
            # compacted — and a churny index eventually fails all k>=8
            # queries (lessons died at ~9x tombstones on 2026-08-22 with no
            # self-heal path). Assess canary + churn-watermark and
            # drop+rebuild from SQL before queries start failing. The
            # documents store was originally excluded ("near-zero churn, the
            # in-request degrade + telemetry cover it") — WRONG on both
            # counts by 2026-08-26: the index rotted anyway and the k=5
            # degrade failed with the same hnsw error, so every retrieval
            # lost its vector arm. It sweeps canary-only (uuid ids can't
            # form a churn watermark).
            try:
                from app.core import vector_health as _vh

                def _rot_sweep() -> list[str]:
                    def _canary_for(col):
                        def _run():
                            if col is not None and col.count() > 0:
                                col.query(
                                    query_texts=["vector index health canary"],
                                    n_results=min(10, col.count()),
                                )
                        return _run

                    targets = []
                    if _svc.learning:
                        _le = _svc.learning
                        _lrow = db.fetchone(
                            "SELECT COALESCE(MAX(id),0) AS m, COUNT(*) AS c FROM lessons")
                        targets.append({
                            "name": "lessons",
                            "live": _lrow["c"], "ever": _lrow["m"],
                            "canary": _canary_for(_le._get_lessons_collection()),
                            "watermark": _vh.get_watermark(db, "lessons"),
                            "rebuild": lambda: _le.rebuild_lessons_vectors(
                                reason="maintenance rot sweep"),
                            "record_watermark": lambda: None,  # rebuild records it
                        })
                    if _svc.kg:
                        _kgr = _svc.kg
                        _krow = db.fetchone(
                            "SELECT COALESCE(MAX(id),0) AS m,"
                            " (SELECT COUNT(*) FROM kg_facts WHERE superseded_at IS NULL"
                            "  AND valid_to IS NULL AND quarantined = 0) AS c"
                            " FROM kg_facts")
                        targets.append({
                            "name": "kg_facts",
                            "live": _krow["c"], "ever": _krow["m"],
                            "canary": _canary_for(_kgr._get_collection()),
                            "watermark": _vh.get_watermark(db, "kg_facts"),
                            "rebuild": lambda: _kgr.rebuild_vectors(
                                reason="maintenance rot sweep"),
                            "record_watermark": lambda: None,  # rebuild records it
                        })
                    if _svc.retriever:
                        _ret = _svc.retriever
                        _drow = db.fetchone(
                            "SELECT COUNT(*) AS c FROM chunks_fts")
                        _dc = _drow["c"] if _drow else 0
                        targets.append({
                            "name": "documents",
                            # uuid chunk ids can't form an ever/churn
                            # watermark — ever=live keeps the churn arm
                            # inert; the canary is the trigger here.
                            "live": _dc, "ever": _dc,
                            "canary": _canary_for(_ret._get_collection()),
                            "watermark": None,
                            "rebuild": lambda: _ret.rebuild_vectors(
                                reason="maintenance rot sweep"),
                            "record_watermark": lambda: None,
                        })
                    return _vh.sweep(targets)

                _rot_lines = await asyncio.to_thread(_rot_sweep)
                _rot_events = [ln for ln in _rot_lines
                               if "REBUILT" in ln or "FAILED" in ln]
                if _rot_events:
                    parts.append("vector rot sweep: " + "; ".join(_rot_events))
            except Exception as e:
                logger.warning("[Heartbeat] vector rot sweep failed: %s", e)
            try:
                n_inv = await asyncio.to_thread(_curate_inverted_leads, db)
                if n_inv:
                    parts.append(f"inverted-direction facts superseded: {n_inv}")
            except Exception as e:
                logger.warning("[Heartbeat] inverted-leads curation failed: %s", e)
            # Lifecycle rot on the DELETE side (2026-08-14 audit): several stores
            # were insert-only and grew unbounded. Each prune is failure-isolated.
            try:
                from app.core.storylines import close_stale
                n_sl = await asyncio.to_thread(close_stale, db, 21)
                if n_sl:
                    parts.append(f"storylines auto-closed: {n_sl}")
            except Exception as e:
                logger.warning("[Heartbeat] storyline auto-close failed: %s", e)
            try:
                from app.core.brain import get_services as _get_svc
                _svc2 = _get_svc()
                if _svc2.kg:
                    n_al = await _svc2.kg.prune_dead_aliases()
                    if n_al:
                        parts.append(f"dead KG aliases pruned: {n_al}")
            except Exception as e:
                logger.warning("[Heartbeat] alias prune failed: %s", e)
            try:
                n_hp = (await asyncio.to_thread(
                    db.execute,
                    "DELETE FROM host_cooccurrence WHERE COALESCE(n_cooccur,0) <= 1 "
                    "AND last_seen < datetime('now', '-60 days')",
                )).rowcount
                # Also drop ANY pair (recurring included) not seen in 90 days: a
                # co-occurrence network gone quiet for 3 months is dead and rebuilds
                # if the hosts reappear. The singleton prune alone left recurring
                # pairs (n_cooccur>=2) unbounded; this bounds the table to a rolling
                # active window without harming live-network detection (audit
                # 2026-08-22).
                n_hp += (await asyncio.to_thread(
                    db.execute,
                    "DELETE FROM host_cooccurrence WHERE last_seen < datetime('now', '-90 days')",
                )).rowcount
                if n_hp:
                    parts.append(f"host pairs pruned: {n_hp}")
            except Exception as e:
                logger.warning("[Heartbeat] host_cooccurrence prune failed: %s", e)
            try:
                n_dd = (await asyncio.to_thread(
                    db.execute,
                    "DELETE FROM dedup_decisions WHERE created_at < datetime('now', '-90 days')",
                )).rowcount
                n_ql = (await asyncio.to_thread(
                    db.execute,
                    "DELETE FROM output_quality_log WHERE created_at < datetime('now', '-90 days')",
                )).rowcount
                if n_dd or n_ql:
                    parts.append(f"decision logs pruned: {n_dd + n_ql}")
            except Exception as e:
                logger.warning("[Heartbeat] decision-log prune failed: %s", e)
            # eval_reports: keep the newest 40 JSON+MD reports; never touch the
            # append-only history log or the regression baseline.
            try:
                from pathlib import Path as _Path
                _erd = _Path("/data/eval_reports")
                removed = 0
                if _erd.is_dir():
                    for _pat in ("eval_*.json", "eval_*.md"):
                        for _old in sorted(_erd.glob(_pat))[:-40]:
                            if _old.name in ("eval_baseline.json", "eval_history.jsonl"):
                                continue
                            try:
                                _old.unlink()
                                removed += 1
                            except OSError:
                                pass
                if removed:
                    parts.append(f"eval reports pruned: {removed}")
            except Exception as e:
                logger.warning("[Heartbeat] eval_reports retention failed: %s", e)
        except Exception as e:
            logger.warning("[Heartbeat] Audit prune failed: %s", e)
        # Periodic SQLite backup — daily snapshot, VERIFIED, kept in two
        # places: /data/backups (fast local restore) AND the off-volume
        # bind mount (survives loss of the nova_data volume itself — the
        # in-volume copies die with the volume they protect). Each new
        # snapshot is opened read-only and integrity-checked immediately:
        # an unverified backup is a hope, not a backup.
        try:
            import shutil
            from pathlib import Path
            from app.core.backup import verify_snapshot

            backup_dir = Path("/data/backups")
            backup_dir.mkdir(exist_ok=True)
            today = datetime.now(timezone.utc).strftime("%Y%m%d")
            target = backup_dir / f"nova-{today}.db"
            if not target.exists():
                # SQLite recommends VACUUM INTO for atomic snapshots.
                # MUST run off the event loop: it copies the whole DB while
                # holding the SafeDB lock (seconds-to-minutes) — running it
                # inline was a prime contributor to the 2026-06-11 incident
                # where the loop blocked >60s and the container went unhealthy.
                from app.database import get_db
                _db = get_db()
                await asyncio.to_thread(_db.execute, f"VACUUM INTO '{target}'")
                ok, detail = await asyncio.to_thread(verify_snapshot, target)
                if not ok:
                    # A failed verification is an alert-worthy event, not a
                    # log line — the monitor result carries it to channels.
                    logger.error("[Heartbeat] Backup verification FAILED: %s", detail)
                    parts.append(f"BACKUP VERIFY FAILED: {detail}")
                    try:
                        target.unlink()  # don't retain a snapshot proven bad
                    except OSError:
                        pass
                else:
                    parts.append(f"backup created+verified: {target.name}")
                    # Off-volume copy — plain file copy of the just-verified
                    # snapshot (cheaper than a second VACUUM INTO and
                    # byte-identical), then verify the COPY independently:
                    # bind-mount I/O has its own failure modes.
                    off_dir = Path(config.BACKUP_OFFVOLUME_DIR or "")
                    if str(off_dir) and off_dir.is_dir():
                        off_target = off_dir / target.name
                        await asyncio.to_thread(shutil.copyfile, target, off_target)
                        off_ok, off_detail = await asyncio.to_thread(verify_snapshot, off_target)
                        if off_ok:
                            parts.append(f"off-volume backup verified: {off_target.name}")
                            off_snaps = sorted(off_dir.glob("nova-*.db"))
                            for old in off_snaps[:-7]:
                                try:
                                    old.unlink()
                                except OSError:
                                    pass
                            # Disaster-recovery extras (2026-07-08): 30GB of
                            # model weights are re-pullable — a MANIFEST is the
                            # backup. Config overrides are tiny and essential.
                            try:
                                import httpx as _httpx
                                async with _httpx.AsyncClient(timeout=10) as _c:
                                    _tags = (await _c.get(f"{config.OLLAMA_URL}/api/tags")).json()
                                _names = [m.get("name", "?") for m in _tags.get("models", [])]
                                (off_dir / "models_manifest.txt").write_text(
                                    "\n".join(sorted(_names)) + "\n", encoding="utf-8")
                            except Exception as _e:
                                logger.warning("[Heartbeat] models manifest failed: %s", _e)
                            try:
                                _ov = Path("/data/config_overrides.json")
                                if _ov.exists():
                                    await asyncio.to_thread(
                                        shutil.copyfile, _ov, off_dir / "config_overrides.json")
                            except Exception as _e:
                                logger.warning("[Heartbeat] config override copy failed: %s", _e)
                        else:
                            logger.error(
                                "[Heartbeat] Off-volume backup verification FAILED: %s", off_detail)
                            parts.append(f"OFF-VOLUME BACKUP VERIFY FAILED: {off_detail}")
                    else:
                        logger.warning(
                            "[Heartbeat] Off-volume backup dir %r not mounted — "
                            "snapshots only exist inside the volume they protect",
                            str(off_dir),
                        )
                        parts.append("off-volume backup SKIPPED (dir not mounted)")
                    # True OFFSITE leg (2026-08-14): E:\nova-offsite was fed by
                    # a MANUAL robocopy that was never scheduled — the daily leg
                    # silently didn't exist (found one snapshot behind). The
                    # drive bind-mounts at /offsite; same copy+verify+retention.
                    offsite_dir = Path("/offsite")
                    if offsite_dir.is_dir():
                        try:
                            os_target = offsite_dir / target.name
                            await asyncio.to_thread(shutil.copyfile, target, os_target)
                            os_ok, os_detail = await asyncio.to_thread(verify_snapshot, os_target)
                            if os_ok:
                                parts.append(f"offsite backup verified: {os_target.name}")
                                for old in sorted(offsite_dir.glob("nova-*.db"))[:-7]:
                                    try:
                                        old.unlink()
                                    except OSError:
                                        pass
                            else:
                                logger.error("[Heartbeat] OFFSITE backup verify FAILED: %s", os_detail)
                                parts.append(f"OFFSITE BACKUP VERIFY FAILED: {os_detail}")
                        except Exception as e:
                            logger.error("[Heartbeat] offsite backup copy failed: %s", e)
                            parts.append(f"offsite backup copy failed: {e}")
                    else:
                        parts.append("offsite backup SKIPPED (E: not mounted)")
                # Retain last 7 in-volume backups
                snapshots = sorted(backup_dir.glob("nova-*.db"))
                for old in snapshots[:-7]:
                    try:
                        old.unlink()
                    except OSError:
                        pass
        except Exception as e:
            logger.warning("[Heartbeat] DB backup failed: %s", e)
        # Auto-disable garbage monitors — any whose last 3 results all match
        # known no-signal patterns. This used to require manual SQL from the
        # operator; now Nova prunes himself.
        try:
            disabled = await self._auto_disable_garbage_monitors()
            if disabled:
                parts.append(f"garbage monitors disabled: {disabled}")
        except Exception as e:
            logger.warning("[Heartbeat] Garbage monitor disable failed: %s", e)
        # Principle distillation — surface load-bearing facts from clusters of
        # high-confidence lessons. Survives lesson decay (provenance='principle').
        try:
            from app.core.principles import distill_principles
            if svc.kg:
                distilled = await distill_principles(get_db(), svc.kg)
                if distilled:
                    parts.append(f"principles distilled: {distilled}")
        except Exception as e:
            logger.warning("[Heartbeat] Principle distillation failed: %s", e)
        # Procedure-skill induction — distill NL procedure skills from proven
        # memory (success reflexions + repeatedly-helpful lessons). Scheduled
        # here, NOT in the chat path: chat-gated extraction produced 0 organic
        # skills in Nova's lifetime because tool-using chat barely exists
        # (audit 2026-08-23). Same decoupling that makes auto_tools work.
        try:
            from app.core.auto_skills import induce_procedure_skills
            if svc.skills:
                induced = await induce_procedure_skills(get_db(), svc.skills)
                if induced:
                    parts.append(f"procedure skills induced: {induced}")
        except Exception as e:
            logger.warning("[Heartbeat] Skill induction failed: %s", e)
        # Recurring-failure promotion sweep — the chat-path trigger
        # (check_recurring_failures) requires a NEW live-chat failure and so
        # never fired under monitor-driven usage; clusters sat unpromoted
        # (audit 2026-08-23: an n=9 quiz-failure cluster, 0 auto-lessons ever).
        try:
            from app.core.reflexion import sweep_recurring_failures
            if svc.reflexions and svc.learning:
                swept = await sweep_recurring_failures(svc.reflexions, svc.learning)
                if swept:
                    parts.append(f"failure clusters promoted: {swept}")
        except Exception as e:
            logger.warning("[Heartbeat] Failure sweep failed: %s", e)
        # Cross-monitor feedback loops
        try:
            loop_parts = await self._check_feedback_loops(svc)
            parts.extend(loop_parts)
        except Exception as e:
            logger.warning("[Heartbeat] Feedback loops failed: %s", e)

        return f"MAINTENANCE | {', '.join(parts)}" if parts else "[No maintenance needed]"

    async def _auto_disable_garbage_monitors(self) -> int:
        """Disable monitors whose last 3 results are all structurally garbage.

        Garbage patterns: 'No Significant Developments' filler, 'no change |'
        empty deltas, dictionary.com hits (search returning definition not
        signal), 'no results found' empty searches. The check only fires for
        monitors with 3+ results so we don't kill new ones.
        """
        import re
        from app.database import get_db

        garbage = re.compile(
            r"no significant developments|"
            r"no significant\b.*\bdevelopments|"
            r"no change \| last:|"
            r"dictionary\.com|"
            r"no results found|"
            r"\bno significant\b.*\bin the past|"
            r"completely irrelevant",
            re.IGNORECASE,
        )

        # Pure DB loop — one thread hop for the whole scan instead of
        # blocking the event loop per query.
        def _scan_and_disable() -> int:
            db = get_db()
            rows = db.fetchall(
                "SELECT id, name FROM monitors WHERE enabled = 1"
            )
            disabled = 0
            for row in rows:
                mid, name = row["id"], row["name"]
                results = db.fetchall(
                    "SELECT value FROM monitor_results "
                    "WHERE monitor_id = ? ORDER BY created_at DESC LIMIT 3",
                    (mid,),
                )
                if len(results) < 3:
                    continue
                if all(r["value"] and garbage.search(r["value"]) for r in results):
                    db.execute(
                        "UPDATE monitors SET enabled = 0 WHERE id = ?", (mid,)
                    )
                    disabled += 1
                    logger.info(
                        "[Heartbeat] Auto-disabled garbage monitor: [%d] %s "
                        "(3 consecutive no-signal results)",
                        mid, name,
                    )
            return disabled

        return await asyncio.to_thread(_scan_and_disable)

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
                failing = await asyncio.to_thread(
                    db.fetchall,
                    "SELECT id, topic FROM lessons "
                    "WHERE quiz_failures >= 3 "
                    "AND last_quizzed_at > datetime('now', '-7 days')"
                )
                requeued = 0
                for row in failing:
                    topic = row["topic"]
                    # Prefix to pass CuriosityQueue validation (15+ chars, 4+ words)
                    padded = f"Re-research and verify: {topic}"
                    cid = await asyncio.to_thread(
                        svc.curiosity.add, padded, source="quiz_feedback", urgency=0.7)
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
                    s for s in await asyncio.to_thread(svc.skills.get_active_skills)
                    if 0.3 <= s.success_rate < 0.5 and s.times_used >= 5
                ]
                if degrading:
                    sv_monitor = await asyncio.to_thread(self.store.get_by_name, "Skill Validation")
                    if sv_monitor:
                        await asyncio.to_thread(self.store.update, sv_monitor.id, last_check_at=None)
                        parts.append(f"skill→validation: {len(degrading)} degrading skills, forced early test")
            except Exception as e:
                logger.warning("[Heartbeat] Loop B (skill→validation) failed: %s", e)

        # Loop C — Curiosity → Quiz logging
        # Lessons from curiosity in last 24h that haven't been quizzed yet
        if has_db:
            try:
                db = svc.learning._db
                row = await asyncio.to_thread(
                    db.fetchone,
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
        """Inert since 2026-06-12 — the weight-training stack is archived.

        Fine-tuning + GRPO + RLVR-trainer (~21.7k LOC, 0 successful
        train→A/B→deploy, ties the base per the one honest A/B) were moved to
        `archive/training/` because the in-context memory loop is the actual
        product. This monitor used to emit "FINETUNE READY → run
        scripts/finetune_auto.py", but that script no longer ships. It now
        no-ops so the (possibly still-seeded) monitor can't surface a stale
        command. To revive training, restore `archive/training/` and remove
        this guard. See archive/training/README.md.
        """
        return "FINETUNE ARCHIVED | weight training is retired; the in-context memory loop is the product"

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
            row = await asyncio.to_thread(
                db.fetchone, "SELECT value FROM system_state WHERE key='last_dream_at'")
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
            rows = await asyncio.to_thread(
                db.fetchall,
                "SELECT id, query, reason, quality_score FROM capability_gaps "
                "WHERE reviewed = 0 ORDER BY created_at DESC LIMIT 50"
            )
        except Exception as e:
            return f"[Capability review failed: could not read gaps — {e}]"

        if not rows:
            return "[Capability review: no unreviewed gaps found]"

        gap_count = len(rows)
        gap_summaries = "\n".join(
            f"- [{row['id']}] quality={(row['quality_score'] or 0.0):.2f}: {(row['query'] or '')[:120]}"
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
            await asyncio.to_thread(
                db.execute,
                f"UPDATE capability_gaps SET reviewed = 1 WHERE id IN ({','.join('?' * len(gap_ids))})",
                tuple(gap_ids),
            )
        except Exception as e:
            logger.warning("[Heartbeat] Failed to mark gaps reviewed: %s", e)

        # Take ACTION on the suggestions: enqueue gaps as goals so KAIROS picks
        # them up. Without this hook the suggestions just sit in the alert
        # text and never drive any work.
        actions_taken = []
        try:
            from app.core.goal_deriver import derive_goals
            new_goals = await derive_goals(db, max_new_goals=3)
            actions_taken.extend(f"goal #{g['id']} ({g['source_kind']})" for g in new_goals)
        except Exception as e:
            logger.warning("[Heartbeat] Capability review goal-derivation failed: %s", e)

        action_summary = (
            "\n\nActions taken: " + "; ".join(actions_taken)
            if actions_taken
            else "\n\nActions taken: none (no goal patterns met threshold)"
        )

        return (
            f"CAPABILITY REVIEW | gaps_reviewed={gap_count}\n\n"
            f"Suggestions:\n{suggestion}{action_summary}"
        )

    # Numeric/health monitors produce structured key=value output (e.g. KG
    # Growth Rate's "kg growth drop (-33.0%) | last_6h: 65 | prev_6h: 97").
    # Asking the LLM to "summarize" them produces hallucinated math like
    # "$35tn occurred in 2024" or "-156.66%, shifting the metric value down
    # by 209 units" when neither figure is in the source. Trust the raw line
    # for these check types and skip the LLM rephrasing pass.
    _RAW_RESULT_CHECK_TYPES: frozenset[str] = frozenset({
        "kg_growth", "kg_health", "ollama_latency", "ollama_model",
        "system_health", "db_size", "chromadb_integrity", "skill_quality",
        "training_job",
        # capability_review's output already has its own structured "CAPABILITY
        # REVIEW | gaps=N\n\nSuggestions:\n..." shape; the alert summarizer
        # mis-detects its long-form suggestion as off-format and falls back to
        # a [:250]-char raw truncation that cuts mid-sentence (observed
        # 2026-05-08 — "...issues reg." dangling). Treat as raw to preserve
        # the full text.
        "capability_review",
        "consolidation",         # knowledge/dossier digests are already structured
        "dream_consolidation",   # dream digests are already concise
        "eval",           # eval reports are already structured
    })

    async def _analyze_result(
        self,
        monitor: Monitor,
        new_value: str,
        change_info: dict | None,
    ) -> str:
        """Ask Nova to analyze a monitor result intelligently."""
        from app.core import llm

        # Numeric/health monitors: skip LLM rephrasing — see comment above.
        if monitor.check_type in self._RAW_RESULT_CHECK_TYPES:
            return new_value[:600] if new_value else ""

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
                except Exception as e:
                    logger.error("[Heartbeat] %s alert failed: %s", ch, e)
        return sent

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
        # (multi-agent category skip removed 2026-08-25 — the capability and
        # its suite category are archived in archive/multi_agent/.)

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
        chronic = getattr(report, "chronic_failures", []) or []
        status = "REGRESSION" if flagged else ("CHRONIC" if chronic else "OK")
        reg_str = ""
        if flagged:
            reg_str = " | regressions: " + ", ".join(
                f"{r.metric}({r.baseline:.2f}->{r.current:.2f})" for r in flagged
            )
        if chronic:
            # Baseline-equality can never flag a task that was red at baseline
            # time — 3+ consecutive reds get their own escalation channel.
            reg_str += " | CHRONIC (3+ runs red): " + ", ".join(chronic)

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

        def _count_tables() -> None:
            for table in ("conversations", "messages", "lessons", "reflexions",
                          "skills", "kg_facts", "monitors"):
                try:
                    row = db.fetchone(f"SELECT count(*) as c FROM {table}")
                    fields[table] = row["c"]
                except Exception:
                    pass

        await asyncio.to_thread(_count_tables)

        # Dead-man's floor (2026-08-18): size-only thresholding had no LOWER bound,
        # so a wiped/wrong DB reported "healthy". `monitors`==0 is unambiguous (the
        # app seeds ~50 monitors on first start and can't run without them);
        # `kg_facts`==0 alongside a populated monitors table means the memory-loop
        # store — "the product" — was wiped. Escalate, never downgrade. Only when
        # the DB file actually exists ("size" was recorded) — a missing/inaccessible
        # DB is already reported above and must not be masked by table counts.
        if "size" in fields:
            if fields.get("monitors") == 0:
                status = "error"
                summary = "monitors table EMPTY — DB not seeded / wrong DB path"
            elif fields.get("kg_facts") == 0 and (fields.get("monitors") or 0) > 0:
                if status != "error":
                    status = "warning"
                summary = "kg_facts EMPTY on an established install — memory-loop store wiped?"

        return format_monitor_result("DB Size Monitor", status, summary, fields)

    async def _execute_feed_health(self) -> str:
        """Ping every curated RSS feed and report dead/unreachable ones.

        Catches the dead-feed class of bug (e.g. the Reuters 404s) automatically
        instead of by accident — a feed is 'dead' if it errors, returns non-200,
        isn't XML, or has no items. Read-only network probes; bounded concurrency.
        """
        import httpx
        from app.monitors.rss_feeds import _FEEDS, _USER_AGENT, _SEC_USER_AGENT

        urls = sorted({u for feeds in _FEEDS.values() for u in feeds})

        # Accept header so servers return the feed, not an HTML landing page.
        _accept = "application/rss+xml, application/atom+xml, application/xml, text/xml;q=0.9, */*;q=0.8"
        # Anti-bot / transient codes: the feed likely EXISTS but refused this probe
        # — classify as "blocked" (not actionable-dead) so we don't cry wolf.
        _blocked_codes = {401, 403, 406, 429, 503}

        async def _check(url: str) -> tuple[str, str, str] | None:
            ua = _SEC_USER_AGENT if "sec.gov" in url else _USER_AGENT
            try:
                async with httpx.AsyncClient(
                    timeout=15.0, follow_redirects=True,
                    headers={"User-Agent": ua, "Accept": _accept},
                ) as client:
                    r = await client.get(url)
                if r.status_code == 200:
                    low = r.text.lower()
                    if "<rss" not in low[:1000] and "<feed" not in low[:1000] and "<?xml" not in low[:1000] and "<rdf" not in low[:1000]:
                        # 200 but HTML (URL redirects to a landing page) = dead feed.
                        return (url, "dead", "not XML")
                    # Reachable + valid XML is healthy. A momentarily-empty feed
                    # (e.g. arxiv between updates) has no <item> but is NOT dead.
                    return None
                if r.status_code in _blocked_codes:
                    return (url, "blocked", f"HTTP {r.status_code}")
                return (url, "dead", f"HTTP {r.status_code}")
            except Exception as e:
                return (url, "dead", type(e).__name__)

        sem = asyncio.Semaphore(8)

        async def _limited(u: str) -> tuple[str, str, str] | None:
            async with sem:
                return await _check(u)

        results = await asyncio.gather(*[_limited(u) for u in urls], return_exceptions=True)
        problems = [r for r in results if isinstance(r, tuple)]
        dead = sorted([p for p in problems if p[1] == "dead"], key=lambda d: d[0])
        blocked = sorted([p for p in problems if p[1] == "blocked"], key=lambda d: d[0])

        total = len(urls)
        healthy = total - len(problems)
        # Only genuinely-dead feeds raise a warning; blocked are informational.
        status = "warning" if dead else "info"
        summary = (
            f"{healthy}/{total} live, {len(dead)} dead, {len(blocked)} bot-blocked"
            if (dead or blocked) else f"all {total} feeds live"
        )
        fields: dict[str, str | int | float] = {
            "checked": total, "dead": len(dead), "blocked": len(blocked),
        }
        for url, _kind, reason in dead[:25]:
            host = re.sub(r"^https?://(www\.)?", "", url).split("/")[0]
            fields[f"✗ {host}"] = reason
        return format_monitor_result("Source Health Monitor", status, summary, fields)

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

            def _skill_stats() -> tuple[int, int, float, int]:
                t = db.fetchone("SELECT count(*) as c FROM skills")["c"]
                en = db.fetchone("SELECT count(*) as c FROM skills WHERE enabled = 1")["c"]
                avg_row = db.fetchone("SELECT avg(success_rate) as avg_sr FROM skills WHERE enabled = 1")
                avg = avg_row["avg_sr"] if avg_row and avg_row["avg_sr"] is not None else 0.0
                deg = db.fetchone(
                    "SELECT count(*) as c FROM skills WHERE enabled = 1 AND success_rate < 0.5 AND times_used >= 3"
                )["c"]
                return t, en, avg, deg

            total, enabled, avg_sr, degrading = await asyncio.to_thread(_skill_stats)
            disabled = total - enabled
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
                if doc_count == 0:
                    # Not "healthy" (2026-08-18): a zero count is either a genuinely
                    # empty store OR the known stale-handle-after-reindex failure
                    # (known failure mode) where the app holds a dropped collection and every
                    # retrieval silently returns nothing. Surface it as a warning so a
                    # wiped index is visible instead of reading as normal.
                    status = "warning"
                    summary = "0 docs indexed — empty store or stale collection handle"
            except Exception as e:
                status = "error"
                summary = f"chromadb error: {e}"
        else:
            status = "error"
            summary = "retriever unavailable"

        try:
            db = get_db()
            fts_row = await asyncio.to_thread(
                db.fetchone, "SELECT count(*) as c FROM chunks_fts")
            fields["fts5"] = fts_row["c"]
        except Exception:
            pass

        # Dead-man's switch for the vector ARM (2026-08-25): the lessons
        # HNSW index was tombstone-dead for 3 days — 133 warnings, zero
        # alerts — because nothing watched query failures. Any store
        # failing repeatedly in 24h is an ERROR, not a log line.
        try:
            from app.core import vector_health as _vh
            _fails = _vh.failures_in_window(hours=24)
            _bad = {s: n for s, n in _fails.items() if n >= 5}
            for s, n in _fails.items():
                if n:
                    fields[f"vector_failures_{s}"] = n
            if _bad:
                status = "error"
                summary = (
                    "vector index failing: "
                    + ", ".join(f"{s} ({n}x/24h)" for s, n in sorted(_bad.items()))
                    + " — tombstone rot; maintenance rebuild pending"
                )
        except Exception:
            pass

        return format_monitor_result("ChromaDB Integrity", status, summary, fields)

    async def _execute_digest_health(self) -> str:
        """Weekly canary over the digest pipeline's output-quality signals.

        The "monitors deliver only hyperlinks" failure recurred TWICE with
        zero automated coverage (2026-08-19, 2026-08-21), and the entail
        gate silently dropped ~51% of cited sentences per day until log
        archaeology found it (audit 2026-08-24). Deterministic — no GPU,
        no network: 7d of stored content digests (substance + link-only
        share) plus the [entail-gate] per-digest summary lines from the
        persistent log (drop-rate trend).
        """
        from app.database import get_db

        db = get_db()

        def _stats() -> tuple[list[int], int]:
            rows = db.fetchall(
                "SELECT mr.value AS value FROM monitor_results mr "
                "JOIN monitors m ON m.id = mr.monitor_id "
                "WHERE m.category = 'content' "
                "AND mr.created_at > datetime('now', '-7 days') "
                "AND mr.status IN ('ok','changed','alert') "
                "AND mr.value IS NOT NULL AND length(mr.value) > 0")
            lengths = [len(r["value"]) for r in rows]
            linkish = sum(1 for r in rows
                          if len(r["value"]) < 600 and "http" in r["value"])
            return lengths, linkish

        lengths, linkish = await asyncio.to_thread(_stats)

        checked = dropped = 0
        try:
            import glob as _glob
            for lp in _glob.glob("/data/logs/nova-app.log*"):
                with open(lp, encoding="utf-8", errors="replace") as f:
                    for line in f:
                        m = _ENTAIL_GATE_LINE_RE.search(line)
                        if m:
                            checked += int(m.group(1))
                            dropped += int(m.group(2))
        except OSError:
            pass

        status, summary = _digest_health_verdict(lengths, linkish, checked, dropped)
        fields: dict[str, str | int | float] = {
            "digests_7d": len(lengths),
            "avg_chars": int(sum(lengths) / len(lengths)) if lengths else 0,
            "link_only": linkish,
            "entail_checked": checked,
            "entail_dropped": dropped,
        }
        return format_monitor_result("Digest Health Canary", status, summary, fields)

    async def _execute_kg_health_check(self) -> str:
        """Check Knowledge Graph health: node count, edge count, fragmentation."""
        from app.core.brain import get_services

        svc = get_services()
        if not svc.kg:
            return format_monitor_result("KG Health Monitor", "error", "kg unavailable")

        try:
            stats = await asyncio.to_thread(svc.kg.get_stats)
            fields: dict[str, str | int | float] = {
                "facts": stats.get("total_facts", 0),
                "active": stats.get("current_facts", 0),
                "superseded": stats.get("superseded_facts", 0),
            }
            db = svc.kg._db
            entities_row = await asyncio.to_thread(
                db.fetchone,
                "SELECT count(DISTINCT subject) + count(DISTINCT object) as c FROM kg_facts WHERE valid_to IS NULL"
            )
            if entities_row:
                fields["entities"] = entities_row["c"]
            orphans_row = await asyncio.to_thread(db.fetchone, """
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
            if isinstance(active, int) and active == 0:
                # Dead-man's switch (2026-08-18): an established KG reporting ZERO
                # active facts is broken (wiped store / stale handle / get_stats
                # returning zeros), not "healthy" — the old code fell through to
                # status="info" ("0 active facts" read as normal, same blind spot
                # that hid the extraction flatline).
                status = "error"
            elif (isinstance(active, int) and active and isinstance(orphans, int)
                    and orphans / max(active, 1) > 0.6):
                status = "warning"
            else:
                status = "info"
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

        # Check both the in-container data path AND the host-mounted finetune_output
        # path (where finetune_oneclick.py writes). One-click writes to the host
        # repo dir, so we need to fall back to it when the data-side file is missing.
        candidate_paths = [
            Path(config.FINETUNE_OUTPUT_DIR) / "run_history.json",
            Path("/repo/finetune_output/run_history.json"),  # host bind-mount, if present
            Path("/data/finetune_output/run_history.json"),  # alt data location
        ]
        history_path = next((p for p in candidate_paths if p.exists()), None)
        if history_path is None:
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
            last_6h = await asyncio.to_thread(
                db.fetchone,
                "SELECT count(*) as c FROM kg_facts WHERE created_at > datetime('now', '-6 hours')"
            )
            prev_6h = await asyncio.to_thread(
                db.fetchone,
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

        # Dead-man's switch on the EXTRACTION pipe specifically (2026-08-18). The
        # spike/drop logic above counts ALL kg_facts, so it reported "normal" while
        # source='extracted' SILENTLY FLATLINED FOR 3 DAYS: an Ollama-0.32.13 JSON
        # array-parse regression killed digest KG extraction, and steady non-
        # extraction sources (curiosity, storylines has_status, principles) masked
        # the total. Worse, a true zero-vs-zero window fell into `prev_count==0 →
        # pct=0.0 → "normal"`. A digest pipeline that RUNS but banks ZERO extracted
        # facts is broken — alarm on it directly (this would have caught R1 in hours).
        try:
            ex_row = await asyncio.to_thread(
                db.fetchone,
                "SELECT count(*) as c FROM kg_facts "
                "WHERE source='extracted' AND created_at > datetime('now', '-24 hours')")
            dg_row = await asyncio.to_thread(
                db.fetchone,
                "SELECT count(*) as c FROM monitor_results mr JOIN monitors m ON m.id=mr.monitor_id "
                "WHERE m.check_type='query' AND mr.created_at > datetime('now', '-24 hours')")
            ex_24h = ex_row["c"] if ex_row else 0
            dg_24h = dg_row["c"] if dg_row else 0
            fields["extracted_24h"] = ex_24h
            fields["digests_24h"] = dg_24h
            if dg_24h >= 3 and ex_24h == 0:
                return format_monitor_result(
                    "KG Growth Rate", "warning",
                    f"KG extraction FLATLINE — {dg_24h} digests ran in 24h but 0 facts "
                    f"extracted (likely a JSON parse/extraction regression)", fields)
            # Second dead-man's switch: the digest pipeline ITSELF stalled. If
            # enabled query monitors exist but produced ZERO digests in 24h, the
            # heartbeat/monitor loop is wedged — the old zero-vs-zero window still
            # reported "+0.0% normal" one level up (2026-08-18).
            if dg_24h == 0:
                enq_row = await asyncio.to_thread(
                    db.fetchone,
                    "SELECT count(*) as c FROM monitors WHERE check_type='query' AND enabled=1")
                if enq_row and enq_row["c"] > 0:
                    return format_monitor_result(
                        "KG Growth Rate", "warning",
                        f"DIGEST PIPELINE STALL — {enq_row['c']} query monitors enabled but "
                        f"0 digests produced in 24h (monitor loop wedged?)", fields)
        except Exception:
            pass

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
        monitor = await asyncio.to_thread(self.store.get, monitor_id)
        if not monitor:
            return {"error": "Monitor not found"}

        try:
            await self._check_monitor(monitor)
            # Get the latest result
            results = await asyncio.to_thread(self.store.get_results, monitor_id, limit=1)
            if results:
                r = results[0]
                return {"status": r.status, "value": r.value, "message": r.message}
            return {"status": "ok", "message": "Check completed"}
        except Exception as e:
            return {"error": str(e)}
