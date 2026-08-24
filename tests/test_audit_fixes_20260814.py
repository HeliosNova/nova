"""Regression tests for the 2026-08-14 deep-audit fixes.

Each test pins a specific defect the audit found so it can't silently return.
The `db` fixture (tmp_path + init_schema) comes from conftest.py.
"""

from __future__ import annotations

import asyncio


# --- detect_change: prose briefings must not be suppressed by a header number ---
class TestDetectChangeProse:
    def test_prose_change_not_suppressed_by_source_count(self):
        """The live bug: a fully-rewritten World Awareness briefing was suppressed
        because its '_read N sources_' header count moved <5% (27→28)."""
        from app.monitors.monitor_store import detect_change
        old = ("## world — domain overview\n_read 27 sources_\n\n"
               "US Navy carrier repositioned to the eastern Mediterranean amid tensions. "
               + "Detailed analysis of naval movements and force posture. " * 40)
        new = ("## world — domain overview\n_read 28 sources_\n\n"
               "DRC Ebola outbreak escalates to historic threat levels, WHO warns. "
               + "A completely different briefing about a health emergency. " * 40)
        result = detect_change(old, new)
        assert result is not None
        assert result["type"] == "text"

    def test_numeric_readout_still_uses_numeric(self):
        from app.monitors.monitor_store import detect_change
        assert detect_change("42/42 sources live", "42/42 sources live") is None
        r = detect_change("CPU 40%", "CPU 80%")
        assert r is not None and r["type"] == "numeric"

    def test_short_numeric_within_threshold_none(self):
        from app.monitors.monitor_store import detect_change
        assert detect_change("$100", "$102", threshold_pct=5.0) is None


# --- _strip_placeholders: [FIGURE NOT IN SOURCES] must never ship ---
class TestStripPlaceholders:
    def test_bracket_placeholder_sentence_dropped(self):
        from app.monitors.deep_research import _strip_placeholders
        text = ("Bitcoin holdings rose to [FIGURE NOT IN SOURCES] this quarter. "
                "MicroStrategy added 5,000 BTC (theblock.co).")
        out = _strip_placeholders(text)
        assert "FIGURE NOT IN SOURCES" not in out
        assert "MicroStrategy added 5,000 BTC" in out

    def test_empty_label_parenthetical_dropped(self):
        from app.monitors.deep_research import _strip_placeholders
        text = ("Freedom House scored it (Political Rights:; Civil Liberties:). "
                "Next fact stands (reuters.com).")
        out = _strip_placeholders(text)
        assert "Political Rights:;" not in out
        assert "Next fact stands" in out

    def test_clean_text_untouched(self):
        from app.monitors.deep_research import _strip_placeholders
        text = "The Fed held rates (reuters.com). Markets rallied 2%."
        assert _strip_placeholders(text) == text


# --- sentence splitter must not decapitate at U.S./U.K./U.N./E.U. ---
class TestSentenceSplitter:
    def test_un_not_split(self):
        from app.monitors.deep_research import _SENT_SPLIT_RE
        parts = _SENT_SPLIT_RE.split("The U.N. Security Council met today. Officials spoke.")
        assert parts[0] == "The U.N. Security Council met today."
        assert len(parts) == 2

    def test_us_uk_eu_not_split(self):
        from app.monitors.deep_research import _SENT_SPLIT_RE
        for abbr in ("U.S.", "U.K.", "E.U."):
            parts = _SENT_SPLIT_RE.split(f"The {abbr} announced sanctions today. Markets fell.")
            assert len(parts) == 2, f"{abbr} wrongly split: {parts}"


# --- curiosity future-filter: catch embedded 'will', keep present-capability ---
class TestFutureQuestion:
    def test_leading_will_is_future(self):
        from app.core.dossiers import _is_future_question
        assert _is_future_question("Will allied governments impose matching curbs next quarter")

    def test_embedded_will_is_future(self):
        from app.core.dossiers import _is_future_question
        # the exact 2026-08-14 live leak
        assert _is_future_question(
            "What legislative or administrative reforms will Congress implement")

    def test_present_capability_is_researchable(self):
        from app.core.dossiers import _is_future_question
        assert not _is_future_question("Can Qwen3.8 run on a single 3090")
        assert not _is_future_question("What matching rules have allied fabs adopted so far")

    def test_could_with_future_marker_is_future(self):
        from app.core.dossiers import _is_future_question
        assert _is_future_question("Could OpenAI reach a $1T valuation next year")


# --- _domain_sources: over cap, the NEWEST digest must survive intact ---
class TestDomainSourcesNewest:
    def test_newest_digest_survives_cap(self, db):
        from app.core import dossiers as mod
        cur = db.execute(
            "INSERT INTO monitors (name, check_type, check_config, schedule_seconds, "
            "enabled, cooldown_minutes, notify_condition) "
            "VALUES ('Domain Study: T', 'query', '{}', 86400, 1, 0, 'on_change')")
        mid = cur.lastrowid
        big = "x" * 12000
        for i, marker in enumerate(("OLDEST", "MIDDLE", "NEWEST")):
            db.execute(
                "INSERT INTO monitor_results (monitor_id, status, value, created_at) "
                "VALUES (?, 'alert', ?, ?)",
                (mid, f"[{marker}] {big}", f"2026-08-14 0{i}:00:00"))
        out = mod._domain_sources(db, "Domain Study: T", None)
        assert "NEWEST" in out
        assert "OLDEST" not in out


# --- numeric_tensions: comma-grouped numbers parse whole ---
class TestNumericTensionComma:
    def test_comma_grouped_number_parses_whole(self, db):
        from app.core import dossiers as mod
        # two dossiers, same quantity, values that only diverge if commas parse right
        for dkey, title, val in (("finance", "Finance", "5,300 million losses reported"),
                                  ("economics", "Economics", "9,900 million losses reported")):
            db.execute(
                "INSERT INTO dossiers (kind, dkey, title, body) VALUES ('domain', ?, ?, ?)",
                (dkey, title,
                 f"## Key facts & figures\n* Estimated aggregate {val} nationwide (cnbc.com)\n"))
        tensions = mod._numeric_tensions(db)
        # 5300 vs 9900 million is an ~87% divergence — must flag, and the numbers
        # in the message must be the full comma-grouped values, not '300'/'900'.
        assert tensions
        assert "5300" in tensions[0] and "9900" in tensions[0]


# --- storylines auto-close ---
class TestCloseStale:
    def test_closes_only_stale_active(self, db):
        from app.core.storylines import close_stale
        db.execute("INSERT INTO storylines (story_key, title, status, summary, "
                   "monitors_csv, update_count, last_updated) VALUES "
                   "('a','A','active','','',1, datetime('now','-30 days'))")
        db.execute("INSERT INTO storylines (story_key, title, status, summary, "
                   "monitors_csv, update_count, last_updated) VALUES "
                   "('b','B','active','','',1, datetime('now','-2 days'))")
        db.execute("INSERT INTO storylines (story_key, title, status, summary, "
                   "monitors_csv, update_count, last_updated) VALUES "
                   "('c','C','closed','','',1, datetime('now','-90 days'))")
        n = close_stale(db, days=21)
        assert n == 1
        assert db.fetchone("SELECT status FROM storylines WHERE story_key='a'")["status"] == "closed"
        assert db.fetchone("SELECT status FROM storylines WHERE story_key='b'")["status"] == "active"


# --- curiosity churn: a recently-failed topic is not re-minted ---
class TestCuriosityChurn:
    def test_recently_failed_topic_not_readded(self, db):
        from app.core.curiosity import CuriosityQueue
        q = CuriosityQueue(db)
        topic = "how does the new export control regime affect allied semiconductor fabs"
        db.execute("INSERT INTO curiosity_queue (topic, source, urgency, status, created_at) "
                   "VALUES (?, 'dossier_tension', 0.7, 'failed', datetime('now'))", (topic,))
        assert q.add(topic, source="dossier_tension", urgency=0.7) == -1

    def test_old_failed_topic_may_return(self, db):
        from app.core.curiosity import CuriosityQueue
        q = CuriosityQueue(db)
        topic = "what are the current cost dynamics of grid-scale battery storage in europe"
        db.execute("INSERT INTO curiosity_queue (topic, source, urgency, status, created_at) "
                   "VALUES (?, 'dossier_tension', 0.7, 'failed', datetime('now','-30 days'))", (topic,))
        assert q.add(topic, source="dossier_tension", urgency=0.7) > 0


# --- kg dead-alias prune ---
class TestPruneDeadAliases:
    def test_dead_alias_pruned_live_kept(self, db):
        from app.core.kg import KnowledgeGraph
        kg = KnowledgeGraph(db)
        db.execute("INSERT INTO kg_facts (subject, predicate, object) VALUES ('OpenAI','is_a','company')")
        db.execute("INSERT INTO kg_entity_aliases (alias_lower, canonical) VALUES ('openai inc','OpenAI')")
        db.execute("INSERT INTO kg_entity_aliases (alias_lower, canonical) VALUES ('defunctco','DefunctCo')")
        n = asyncio.run(kg.prune_dead_aliases())
        assert n == 1
        remaining = {r["canonical"] for r in db.fetchall("SELECT canonical FROM kg_entity_aliases")}
        assert remaining == {"OpenAI"}
