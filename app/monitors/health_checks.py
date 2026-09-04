"""Deterministic health checks: DB size, feeds, latency, indexes, liveness.

Split out of heartbeat_loop.py 2026-09-04, which had grown past four thousand
lines. A MIXIN, deliberately: these methods keep the same `self` as the loop
that owns the store and the channel bots, so behaviour and every existing
import are unchanged. This is a move, not a rewrite.
"""
from __future__ import annotations

import logging
import asyncio
import json
import re
import time
from datetime import datetime

from app.config import config
from app.monitors.monitor_store import Monitor
from app.monitors.format import format_monitor_result

logger = logging.getLogger(__name__)


class HealthChecksMixin:

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

    async def _execute_pathway_liveness(self) -> str:
        """Fast-lane liveness verdict over every optional background writer
        (app/monitors/pathways.py). Deterministic — DB reads and one file
        mtime; no LLM, no network. A pathway whose table stopped growing
        past its window is DEAD; the summary names it and the fields carry
        the silence, so the failure mode that hid storylines for five weeks
        (2026-08-11) surfaces within one cycle.
        """
        from app.database import get_db
        from app.monitors.pathways import liveness_report

        status, summary, fields = await asyncio.to_thread(liveness_report, get_db())
        return format_monitor_result("Pathway Liveness", status, summary, fields)

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
