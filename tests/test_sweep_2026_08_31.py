"""Regression tests for the 2026-08-31 fresh-eyes sweep.

Every test is built from a REAL defective sample found live and asserts both
directions: the fix catches the defect AND does not over-block legitimate
cases. Fix inventory (F-numbers match the sweep report):

  F1  heartbeat_loop._skeletal_digest — whole-word/sentence bound (was
      out[:1200]; all 515 stored len==1200 monitor_results were its output)
  F2  storylines._event_summary — whole-word bound (948/1,761 events cut
      mid-word at exactly 300)
  F3  storylines NO-CHANGE guard — standalone-line verdict anywhere, not
      just prefix (storyline 31's summary was overwritten by a rationale
      that ENDED with "NO CHANGE")
  F4  dossiers._wtrim/_fmt_qty — word-safe caps + unit spacing
      ("2billion" → "2 billion"; tension topics opened mid-word)
  F5  agent_loop.sanitize_synthesis — leading-stub repair (messages rowid
      250 opened "and my standing knowledge dossiers, …")
  F6  http_fetch binary guard (38 action_log rows of raw %PDF byte soup
      fed to the model as tool output)
  F7  "===" scaffold-header guard on tool-demand mining (auto_tool_candidates
      rows 21/22/89 held internal orchestration prompts verbatim)
  F8  monitor_store.get_due anchored-due ranking floor (Morning Check-in
      missed two whole days behind a ratio-2+ digest backlog)
  F10 dream._prune_reflexions chroma cleanup (54 ghost vectors)
  F11 active_memory ephemeral gates (Prometheus-9 fiction persisted)
  F12 brain topic-tracking ephemeral gate (topic_frequency row 35)
  F13 fire-and-forget task refs held (extensions register, activity writes)
  F15 storyline QA meta-annotations excluded from dossier consolidation
      sources (was display-only filtering in api/monitors.py)
"""

from __future__ import annotations

import pathlib
import re
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.tools.base import EPHEMERAL_REQUEST

REPO = pathlib.Path(__file__).resolve().parent.parent
APP = REPO / "app"


@pytest.fixture(autouse=True)
def reset_ephemeral():
    token = EPHEMERAL_REQUEST.set(False)
    yield
    EPHEMERAL_REQUEST.reset(token)


# ── F1: skeletal digest whole-word bound ────────────────────────────────────

class TestSkeletalDigest:
    def _long_digest(self):
        lines = ["# Weekly Intelligence Digest"]
        for i in range(40):
            lines.append(f"* **Claim {i}:** the observed measurement number "
                         f"{i} moved substantially against expectations")
        return "\n".join(lines)

    def test_cap_never_bites_midword(self):
        from app.monitors.heartbeat_loop import _skeletal_digest
        out = _skeletal_digest(self._long_digest(), cap=1200)
        assert len(out) <= 1201  # cap + possible ellipsis char
        assert out.endswith(("…", ".", "!", "?")), f"mid-word tail: …{out[-40:]!r}"
        # The defective behavior: hard len==cap ending inside a word.
        if len(out) >= 1200:
            assert not out[-1].isalnum()

    def test_word_fragment_is_prefix_of_source(self):
        from app.monitors.heartbeat_loop import _skeletal_digest
        src = self._long_digest()
        out = _skeletal_digest(src, cap=1200)
        skeleton = "\n".join(s.strip() for s in src.split("\n")
                             if s.strip().startswith(("#", "* **", "- **", "**")))
        assert skeleton.startswith(out.rstrip("…"))

    def test_short_digest_untouched(self):
        from app.monitors.heartbeat_loop import _skeletal_digest
        text = "# Digest\n* **One claim.** body prose here\nplain body line dropped"
        assert _skeletal_digest(text, cap=1200) == \
            "# Digest\n* **One claim.** body prose here"

    def test_no_headings_falls_back_to_full_text(self):
        from app.monitors.heartbeat_loop import _skeletal_digest
        text = "Plain prose digest with no markdown structure at all."
        assert _skeletal_digest(text, cap=1200) == text


# ── F2: storyline event summary whole-word bound ────────────────────────────

class TestEventSummary:
    def test_no_midword_cut(self):
        from app.core.storylines import _event_summary
        dev = ("OPEC-plus announced a production increase of five hundred "
               "thousand barrels per day effective October citing stabilized "
               "global prices and resilient demand growth across all major "
               "importing economies while several member states signaled "
               "further quota flexibility heading into the northern winter "
               "heating season and beyond")
        assert len(dev) > 300
        out = _event_summary(dev, cap=300)
        assert out.endswith("…")
        frag = out[:-1]
        assert dev.startswith(frag)
        # the char after the kept fragment must not be mid-word
        assert not dev[len(frag)].isalnum() or not frag[-1].isalnum()

    def test_prefers_sentence_boundary(self):
        from app.core.storylines import _event_summary
        s = ("The Fed held rates steady in September as inflation cooled. " * 4
             + "Markets rallied broadly on the announcement with equities up.")
        out = _event_summary(s, cap=300)
        assert out.endswith(".")
        assert s.startswith(out)

    def test_short_dev_untouched(self):
        from app.core.storylines import _event_summary
        assert _event_summary("Fed held rates.", cap=300) == "Fed held rates."

    def test_collect_items_extraction_is_word_safe(self, db, monkeypatch):
        """2026-09-01: the first post-fix Morning Check-in still minted 6
        len-300 mid-word events — the ACTUAL cutter was _collect_items'
        extraction [:300], which delivered pre-truncated text that
        _event_summary passed through untouched."""
        import app.core.storylines as sl
        long_line = ("**US tariffs on Southeast Asian imports intensified as "
                     "Michigan, Ohio, and Kentucky manufacturers that rely on "
                     "imported components reported sharply higher input costs "
                     "while several state commissions opened formal reviews of "
                     "the resulting consumer price increases across household "
                     "categories including appliances electronics and repair " * 2)
        monkeypatch.setattr(sl, "_gather_recent_outputs",
                            lambda db_, hours, max_per_monitor: {"Test Monitor": [long_line]})
        items = sl._collect_items(db)
        assert items, "extraction dropped the line entirely"
        for it in items:
            t = it["text"]
            assert len(t) <= 301
            if len(t) >= 300:
                assert not t[-1].isalnum(), f"mid-word extraction cut: …{t[-30:]!r}"


# ── F3: NO-CHANGE verdict as standalone line anywhere ───────────────────────

class TestNoChangeGuard:
    def _seed(self, db):
        cur = db.execute(
            "INSERT INTO storylines (story_key, title, summary, monitors_csv, "
            "update_count, last_updated) VALUES (?,?,?,?,1,datetime('now'))",
            ("fed-rate-policy", "Fed rate policy",
             "Fed held rates in July; two cuts priced for Q4.", "Finance Watch"))
        return cur.lastrowid

    def _story(self, dev):
        return {"key": "fed-rate-policy", "title": "Fed rate policy",
                "developments": [dev], "monitors": ["Finance Watch"]}

    @pytest.mark.asyncio
    async def test_trailing_verdict_does_not_overwrite_summary(self, db, monkeypatch):
        """The storyline-31 defect: rationale first, verdict last — the whole
        rationale used to be stored as the thread summary."""
        import app.core.storylines as sl
        sid = self._seed(db)
        rambling = ("The provided new developments contain no information "
                    "regarding the Federal Reserve's rate path.\n\nNO CHANGE")
        monkeypatch.setattr(sl, "_bg_invoke", AsyncMock(return_value=rambling))
        out = await sl._update_story(db, self._story("September FOMC minutes released"))
        assert out is None
        row = db.fetchone("SELECT summary FROM storylines WHERE id = ?", (sid,))
        assert row["summary"] == "Fed held rates in July; two cuts priced for Q4."

    @pytest.mark.asyncio
    async def test_midsentence_no_change_is_not_a_verdict(self, db, monkeypatch):
        """'Fed made no change to rates' inside prose is CONTENT, not the
        verdict — must still update the thread."""
        import app.core.storylines as sl
        sid = self._seed(db)
        real = ("The Fed made no change to rates but shifted guidance toward "
                "a December cut.\nCHANGED: guidance shifted dovish\nFORECAST: none")
        monkeypatch.setattr(sl, "_bg_invoke", AsyncMock(return_value=real))
        out = await sl._update_story(db, self._story("Powell press conference signals"))
        assert out is not None
        assert out["changed"] == "guidance shifted dovish"
        row = db.fetchone("SELECT summary FROM storylines WHERE id = ?", (sid,))
        assert "shifted guidance" in row["summary"]

    @pytest.mark.asyncio
    async def test_prefix_verdict_still_caught(self, db, monkeypatch):
        import app.core.storylines as sl
        self._seed(db)
        monkeypatch.setattr(sl, "_bg_invoke", AsyncMock(return_value="NO CHANGE"))
        out = await sl._update_story(db, self._story("minor headline repeat"))
        assert out is None


# ── F4: dossier word-safe trim + quantity formatting ────────────────────────

class TestDossierTrims:
    def test_wtrim_no_midword_open_or_close(self):
        from app.core.dossiers import _wtrim
        s = ("Total revenue was $22.8 billion for the quarter, up 14 percent "
             "year over year according to the company filing")
        out = _wtrim(s, 80)
        assert len(out) <= 81
        assert out.endswith("…")
        assert s.startswith(out[:-1])
        assert not s[len(out) - 1].isalnum()

    def test_wtrim_short_untouched(self):
        from app.core.dossiers import _wtrim
        assert _wtrim("CPI at 2.4%", 80) == "CPI at 2.4%"

    def test_fmt_qty_word_units_get_space(self):
        from app.core.dossiers import _fmt_qty
        assert _fmt_qty(2.0, "billion") == "2 billion"
        assert _fmt_qty(500.0, "thousand") == "500 thousand"

    def test_fmt_qty_symbol_units_attach(self):
        from app.core.dossiers import _fmt_qty
        assert _fmt_qty(3.4, "%") == "3.4%"
        assert _fmt_qty(25.0, "bp") == "25bp"
        assert _fmt_qty(2.0, "") == "2"


# ── F5: sanitize_synthesis leading-stub repair ──────────────────────────────

class TestSanitizeLeadingStub:
    def test_msg250_reconstruction(self):
        """The live victim: opening scaffold phrase redacted, answer shipped
        to the owner starting 'and my standing knowledge dossiers, …'."""
        from app.core.agent_loop import sanitize_synthesis
        raw = ("From the search results and my standing knowledge dossiers, "
               "here are the key developments on the topic.")
        out, changes = sanitize_synthesis(raw)
        assert changes >= 1
        assert not out.lower().startswith(("and ", "but ", "or ", ", "))
        assert out[0].isupper()
        assert out.startswith("My standing knowledge dossiers")

    def test_no_redaction_no_repair(self):
        """A legit answer that opens with a conjunction must survive when
        nothing was redacted."""
        from app.core.agent_loop import sanitize_synthesis
        raw = "And yet the market rallied. Prices rose 4% on the day."
        out, changes = sanitize_synthesis(raw)
        assert changes == 0
        assert out == raw

    def test_midtext_redaction_leaves_clean_opening_alone(self):
        from app.core.agent_loop import sanitize_synthesis
        raw = ("The rollout succeeded (based on the completed analysis plan) "
               "and adoption doubled.")
        out, changes = sanitize_synthesis(raw)
        assert changes >= 1
        assert out.startswith("The rollout succeeded")
        assert "analysis plan" not in out


# ── F6: http_fetch binary-content guard ─────────────────────────────────────

class TestHttpFetchBinaryGuard:
    def _resp(self, text, ctype):
        r = MagicMock()
        r.text = text
        r.status_code = 200
        r.url = "https://example.com/doc"
        r.headers = {"content-type": ctype}
        r.is_redirect = False
        return r

    async def _fetch(self, resp):
        from app.tools.http_fetch import HttpFetchTool
        client = MagicMock()
        client.request = AsyncMock(return_value=resp)
        with patch("app.tools.http_fetch._get_client", return_value=client):
            return await HttpFetchTool().execute(url="https://example.com/doc")

    @pytest.mark.asyncio
    async def test_pdf_content_type_refused(self):
        r = await self._fetch(self._resp("%PDF-1.4\x00\x01 byte soup " * 50,
                                         "application/pdf"))
        assert not r.success
        assert "Binary content" in r.error
        assert r.output == ""

    @pytest.mark.asyncio
    async def test_pdf_magic_without_content_type_refused(self):
        """The live SkyLance spec-sheet class: server said text/plain, body
        was a PDF."""
        r = await self._fetch(self._resp("%PDF-1.7 obj stream " * 40, "text/plain"))
        assert not r.success
        assert "Binary content" in r.error

    @pytest.mark.asyncio
    async def test_html_mentioning_pdf_not_blocked(self):
        r = await self._fetch(self._resp(
            "<html><body><p>Download the %PDF spec here for details on the "
            "portable document format.</p></body></html>", "text/html"))
        assert r.success
        assert "portable document format" in r.output


# ── F7: scaffold-header guard on tool-demand mining ─────────────────────────

class TestScaffoldQueryGuard:
    def test_scaffold_prompt_not_recorded(self, db):
        from app.core.tool_triggers import ToolCandidateStore
        store = ToolCandidateStore(db=db)
        before = db.fetchone("SELECT COUNT(*) c FROM auto_tool_candidates")["c"]
        rid = store.record(
            "=== AUTO-MONITOR DETECTION ===\nAnalyze recent topics and decide…",
            ["memory_search", "web_search"])
        assert rid == 0
        assert db.fetchone("SELECT COUNT(*) c FROM auto_tool_candidates")["c"] == before

    def test_leading_whitespace_scaffold_also_blocked(self, db):
        from app.core.tool_triggers import ToolCandidateStore
        assert ToolCandidateStore(db=db).record(
            "  === WILL MODULE ===\ninternal prompt", ["web_search"]) == 0

    def test_real_query_still_recorded(self, db):
        from app.core.tool_triggers import ToolCandidateStore
        rid = ToolCandidateStore(db=db).record(
            "compare 3090 vs 4090 prices === include used market ===",
            ["web_search", "calculator"])
        assert rid > 0

    def test_agent_loop_learn_has_same_guard(self):
        src = (APP / "core" / "agent_loop.py").read_text(encoding="utf-8")
        assert re.search(
            r'if \(result\.query or ""\)\.lstrip\(\)\.startswith\("==="\):\s*\n\s*return',
            src), "agent_loop._learn_from_run lost its scaffold-header guard"


# ── F8: anchored-due ranking floor in get_due ───────────────────────────────

class TestAnchoredDueFloor:
    FMT = "%Y-%m-%d %H:%M:%S"

    def _fix_local(self, monkeypatch, hour):
        """Pin MonitorStore's local clock and return the pinned datetime.

        All anchored-monitor seeds must be derived FROM this pin: a
        real-clock '30h ago' seed straddles one extra calendar date once the
        suite runs past UTC midnight (flaked live 2026-08-31 -> 09-01)."""
        from app.monitors.monitor_store import MonitorStore
        fixed = datetime.now(timezone.utc).replace(
            hour=hour, minute=0, second=0, microsecond=0)
        monkeypatch.setattr(MonitorStore, "_local_now", staticmethod(lambda: fixed))
        return fixed

    def _seed(self, db, fixed, *, anchored_days_behind=1):
        from app.monitors.monitor_store import MonitorStore
        store = MonitorStore(db)
        a_id = store.create("Morning Check-in", "query", {"anchor_hour": 7},
                            schedule_seconds=86400)
        r_id = store.create("Hourly Digest", "search", {"query": "news"},
                            schedule_seconds=3600)
        # Anchored: last ran N local days before the PINNED date (deterministic
        # date diff regardless of when the suite runs).
        a_last = (fixed - timedelta(days=anchored_days_behind)).replace(hour=14)
        db.execute("UPDATE monitors SET last_check_at = ? WHERE id = ?",
                   (a_last.strftime(self.FMT), a_id))
        # Rival: genuinely backlogged at ratio 2.5 against the REAL clock
        # (raw ratios use datetime.now, not the pinned local clock).
        r_last = datetime.now(timezone.utc) - timedelta(hours=2, minutes=30)
        db.execute("UPDATE monitors SET last_check_at = ? WHERE id = ?",
                   (r_last.strftime(self.FMT), r_id))
        return store, a_id, r_id

    def test_late_day_anchored_outranks_backlog(self, db, monkeypatch):
        """Aug 29/30 defect: due anchored daily at raw ratio ~0.55 sat behind
        31 monitors at ratio ≥2 and never ran. By evening the floor
        (1 + hours-past-anchor/6) must beat a ratio-2.5 digest."""
        fixed = self._fix_local(monkeypatch, 19)  # 12h past anchor → floor 3.0
        store, a_id, r_id = self._seed(db, fixed)
        due = store.get_due()
        ids = [m.id for m in due]
        assert ids.index(a_id) < ids.index(r_id)

    def test_early_day_anchored_does_not_jump_queue(self, db, monkeypatch):
        """Both directions: shortly after the anchor the floor is small — a
        genuinely backlogged monitor still goes first."""
        fixed = self._fix_local(monkeypatch, 9)  # 2h past anchor → floor ~1.33
        store, a_id, r_id = self._seed(db, fixed)
        due = store.get_due()
        ids = [m.id for m in due]
        assert a_id in ids, "anchored monitor must still be due"
        assert ids.index(r_id) < ids.index(a_id)

    def test_missed_full_day_jumps_queue_even_at_dawn(self, db, monkeypatch):
        """Live-observed 2026-08-31 16:58: on the post-outage day the /6
        floor (2.66) still lost to a 3-8x backlog — a THIRD consecutive
        missed day. An anchored daily ≥2 local days behind sorts with the
        never-run tier: max one missed day, ever."""
        fixed = self._fix_local(monkeypatch, 8)  # just past anchor, floor ~1.17
        store, a_id, r_id = self._seed(db, fixed, anchored_days_behind=2)
        due = store.get_due()
        ids = [m.id for m in due]
        assert ids.index(a_id) < ids.index(r_id), \
            "a day-missed anchored daily must outrank any backlogged digest"


# ── F10: NREM reflexion prune cleans chroma ─────────────────────────────────

class TestDreamPruneVectors:
    @pytest.mark.asyncio
    async def test_prune_removes_vectors_too(self, db, monkeypatch):
        from app.core.dream import (ConsolidationResult, DreamConsolidator,
                                    GatherSignals)
        from app.core.reflexion import ReflexionStore
        from app.database import AsyncSafeDB
        ids = []
        for i in range(3):
            cur = db.execute(
                "INSERT INTO reflexions (task_summary, outcome, reflection, "
                "quality_score) VALUES (?,?,?,?)",
                (f"junk task {i}", "failure", "low quality", 0.1))
            ids.append(cur.lastrowid)
        removed: list[int] = []
        monkeypatch.setattr(ReflexionStore, "_remove_from_vector",
                            lambda self, rids: removed.extend(rids))
        dc = DreamConsolidator(AsyncSafeDB(db))
        signals = GatherSignals(low_quality_reflexions=[{"id": i} for i in ids])
        result = ConsolidationResult()
        await dc._prune_reflexions(signals, result)
        assert result.reflexions_pruned == 3
        assert sorted(removed) == sorted(ids), \
            "SQL rows deleted but chroma vectors left as ghosts"
        assert db.fetchone("SELECT COUNT(*) c FROM reflexions")["c"] == 0

    @pytest.mark.asyncio
    async def test_empty_prune_is_noop(self, db, monkeypatch):
        from app.core.dream import (ConsolidationResult, DreamConsolidator,
                                    GatherSignals)
        from app.core.reflexion import ReflexionStore
        from app.database import AsyncSafeDB
        called: list[int] = []
        monkeypatch.setattr(ReflexionStore, "_remove_from_vector",
                            lambda self, rids: called.extend(rids))
        await DreamConsolidator(AsyncSafeDB(db))._prune_reflexions(
            GatherSignals(), ConsolidationResult())
        assert called == []


# ── F11: active_memory ephemeral gates ──────────────────────────────────────

class TestActiveMemoryEphemeralGate:
    def _tool(self, db, monkeypatch):
        from app.tools.active_memory import ActiveMemoryTool
        monkeypatch.setattr(ActiveMemoryTool, "_get_collection", lambda self: None)
        return ActiveMemoryTool(db=db)

    @pytest.mark.asyncio
    async def test_ephemeral_add_persists_nothing(self, db, monkeypatch):
        """The Prometheus-9 channel: an eval probe saying 'remember this'
        passes the intent gate by construction — the ephemeral gate must
        stop it while looking success-shaped to the probe."""
        from app.tools.active_memory import current_user_query
        tool = self._tool(db, monkeypatch)
        tok = current_user_query.set("remember that Prometheus-9 launches in October")
        try:
            EPHEMERAL_REQUEST.set(True)
            r = await tool.execute(action="add",
                                   content="Prometheus-9 launches in October")
            assert r.success
            assert db.fetchone("SELECT COUNT(*) c FROM active_memories")["c"] == 0
        finally:
            current_user_query.reset(tok)

    @pytest.mark.asyncio
    async def test_real_add_still_persists(self, db, monkeypatch):
        from app.tools.active_memory import current_user_query
        tool = self._tool(db, monkeypatch)
        tok = current_user_query.set("remember that my favorite editor is vim")
        try:
            r = await tool.execute(action="add",
                                   content="Owner's favorite editor is vim")
            assert r.success
            row = db.fetchone("SELECT content FROM active_memories")
            assert row and "vim" in row["content"]
        finally:
            current_user_query.reset(tok)

    @pytest.mark.asyncio
    async def test_ephemeral_update_and_delete_gated(self, db, monkeypatch):
        tool = self._tool(db, monkeypatch)
        db.execute(
            "INSERT INTO active_memories (content, category, last_accessed_at, "
            "created_at, updated_at) VALUES ('real fact','fact',datetime('now'),"
            "datetime('now'),datetime('now'))")
        mid = db.fetchone("SELECT id FROM active_memories")["id"]
        EPHEMERAL_REQUEST.set(True)
        r1 = await tool.execute(action="update", id=mid, content="poisoned")
        r2 = await tool.execute(action="delete", id=mid)
        assert r1.success and r2.success
        row = db.fetchone("SELECT content FROM active_memories WHERE id = ?", (mid,))
        assert row is not None and row["content"] == "real fact"


# ── F12/F13: source guards (behavioral paths need the full chat pipeline) ───

class TestSourceGuards:
    def test_brain_topic_tracking_gated_on_ephemeral(self):
        src = (APP / "core" / "brain.py").read_text(encoding="utf-8")
        m = re.search(r"^\s*if (.*):\s*\n(?:.*\n){0,3}?.*record_topic",
                      src, re.MULTILINE)
        assert m, "topic-tracking call site not found"
        assert "not ephemeral" in m.group(1), \
            "topic_frequency writer lost its ephemeral gate (Prometheus-9 row 35 class)"

    def test_extension_register_tasks_held(self):
        src = (APP / "core" / "extensions.py").read_text(encoding="utf-8")
        assert "_bg_register_tasks" in src
        assert "add_done_callback" in src

    def test_activity_write_tasks_held(self):
        src = (APP / "main.py").read_text(encoding="utf-8")
        assert "_activity_write_tasks" in src
        assert "add_done_callback" in src


# ── F15: QA meta-annotations excluded from dossier sources ──────────────────

class TestStorylineSourceMetaExclusion:
    def _seed(self, db):
        cur = db.execute(
            "INSERT INTO storylines (story_key, title, summary, monitors_csv, "
            "update_count, last_updated) VALUES ('opec-supply','OPEC supply',"
            "'OPEC weighing further cuts','Energy Watch',1,datetime('now'))")
        sid = cur.lastrowid
        events = [
            "OPEC agreed to cut production by 1M barrels per day",
            "⚠ 2 of 3 spot-checked claims could NOT be confirmed",
            "_Sourcing note: 2 of 5 sources are primary_",
            "read 5 sources across 3 domains",
        ]
        for ev in events:
            db.execute(
                "INSERT INTO storyline_events (storyline_id, summary, "
                "source_monitor, is_new) VALUES (?,?, 'Energy Watch', 1)",
                (sid, ev))
        return sid

    def test_meta_rows_filtered_real_events_kept(self, db):
        from app.core.dossiers import _storyline_sources
        sid = self._seed(db)
        out = _storyline_sources(db, sid, None)
        assert "OPEC agreed to cut production" in out
        assert "OPEC weighing further cuts" in out  # thread state line
        assert "⚠" not in out
        assert "Sourcing note" not in out
        assert "read 5 sources" not in out

    def test_since_branch_filters_too(self, db):
        from app.core.dossiers import _storyline_sources
        sid = self._seed(db)
        out = _storyline_sources(db, sid, "2000-01-01 00:00:00")
        assert "OPEC agreed to cut production" in out
        assert "⚠" not in out
        assert "Sourcing note" not in out

    def test_api_display_filter_shares_the_sql(self):
        src = (APP / "api" / "monitors.py").read_text(encoding="utf-8")
        assert "from app.core.storylines import EVENT_META_EXCL_SQL" in src or \
               "EVENT_META_EXCL_SQL" in src, \
            "display-side filter no longer shares the canonical exclusion SQL"


# ── SEC Form-4 enrichment retry + coverage observability (2026-09-01) ───────

class TestForm4EnrichRetry:
    """Owner-reported "bare links" (2026-09-01): the 00:29 post-outage SEC
    digest shipped 8/15 items with no trade annotation. Replay parsed ALL 15
    — transient sec.gov throttling during the catch-up burst, and the
    enricher's `except: pass` had zero retries and zero log evidence."""

    def _items(self, n=4):
        from types import SimpleNamespace
        return [SimpleNamespace(url=f"https://www.sec.gov/Archives/edgar/data/"
                                    f"{100+i}/{'0'*12}{100000+i}/x-index.htm",
                                meta={}) for i in range(n)]

    @pytest.mark.asyncio
    async def test_transient_failure_recovered_by_retry(self, monkeypatch):
        import app.monitors.domain_study_runner as dsr
        calls = {}

        async def flaky(url, client):
            calls[url] = calls.get(url, 0) + 1
            if calls[url] == 1:
                raise RuntimeError("HTTP 403 (sec.gov throttle)")
            return {"direction": "buy", "buy_shares": 10, "buy_value": 500.0,
                    "sell_shares": 0, "sell_value": 0, "codes": ["P"]}

        monkeypatch.setattr(dsr, "_fetch_form4_txn", flaky)
        monkeypatch.setattr(dsr.asyncio, "sleep", AsyncMock())
        items = self._items()
        out = await dsr._enrich_sec_form4(items)
        assert all(it.meta.get("form4") for it in out), \
            "one paced retry must recover the transient sec.gov throttle class"
        assert all(c == 2 for c in calls.values())

    @pytest.mark.asyncio
    async def test_permanent_failure_logs_low_coverage(self, monkeypatch, caplog):
        import logging

        import app.monitors.domain_study_runner as dsr

        async def dead(url, client):
            raise RuntimeError("HTTP 403")

        monkeypatch.setattr(dsr, "_fetch_form4_txn", dead)
        monkeypatch.setattr(dsr.asyncio, "sleep", AsyncMock())
        with caplog.at_level(logging.INFO):
            await dsr._enrich_sec_form4(self._items())
        assert any("form4 coverage LOW" in r.message for r in caplog.records), \
            "silent enrichment degradation must be loud"

    @pytest.mark.asyncio
    async def test_parsed_first_try_does_not_retry(self, monkeypatch):
        import app.monitors.domain_study_runner as dsr
        calls = {}

        async def good(url, client):
            calls[url] = calls.get(url, 0) + 1
            return {"direction": "sell", "buy_shares": 0, "buy_value": 0,
                    "sell_shares": 5, "sell_value": 100.0, "codes": ["S"]}

        monkeypatch.setattr(dsr, "_fetch_form4_txn", good)
        await dsr._enrich_sec_form4(self._items())
        assert all(c == 1 for c in calls.values()), "no wasteful double-fetch"
