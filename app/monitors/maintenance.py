"""Daily maintenance: decay, pruning, hygiene sweeps and the feedback loops.

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
from datetime import datetime, timezone

from app.config import config

logger = logging.getLogger(__name__)


# Moved here from heartbeat_loop.py 2026-09-04: the split left these
# module-level names behind, so every call site above resolved to
# nothing at runtime and raised NameError into an except.
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
    prose body (already consolidated into dossiers by now) is released.

    Whole-word bound (2026-08-31): the cap was a hard `out[:cap]` mid-word cut.
    All 515 stored length==1200 monitor_results rows (Jul 14-24 cohort) are
    THIS function's output — shape-tested 515/515 skeletal-heading lines, i.e.
    the cluster earlier attributed to cross_monitor's text[:1200] actually
    came from here, and every future demotion pass would have kept re-biting.
    Prefer the last sentence boundary in the tail; fall back to the last
    whole word + ellipsis (same policy as cross_monitor._synthesize)."""
    keep = []
    for ln in (text or "").split("\n"):
        s = ln.strip()
        if s.startswith("#") or s.startswith(("* **", "- **", "**")):
            keep.append(s)
    out = "\n".join(keep) or (text or "")
    if len(out) <= cap:
        return out
    cut = out[:cap]
    tail = max(cut.rfind("."), cut.rfind("!"), cut.rfind("?"))
    if tail >= int(cap * 0.83):
        return cut[:tail + 1]
    ws = cut.rfind(" ")
    return (cut[:ws].rstrip() + "…") if ws > 0 else cut

# ---------------------------------------------------------------------------
# Deliberation scrubber — strip untagged model deliberation from monitor output
# ---------------------------------------------------------------------------


class MaintenanceMixin:

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
