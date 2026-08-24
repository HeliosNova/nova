"""Knowing tier (living dossiers, 2026-08-12) — store, revision trail, retrieval,
consolidation cycle. LLM mocked throughout; asyncio.run keeps tests plugin-agnostic."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest


GOOD_BODY = (
    "## Current understanding\n"
    "The semiconductor export-control regime tightened materially in August; NVIDIA's "
    "H30 shipments to China now require per-customer licenses (reuters.com). This "
    "reshapes the mid-term supply picture.\n"
    "## How we got here\n"
    "- (2026-07-01) Initial curbs announced (wsj.com)\n"
    "- (2026-08-10) License regime expanded (reuters.com)\n"
    "## Key facts & figures\n"
    "- H30 license backlog: 412 applications (reuters.com)\n"
    "## Open questions\n"
    "- Will allied governments impose matching curbs next quarter\n"
    "- What matching rules have allied fabs adopted so far\n"
    "CHANGED: license regime expanded to per-customer scope"
)


@pytest.fixture()
def db(tmp_path):
    from app.database import SafeDB
    d = SafeDB(str(tmp_path / "dossier_test.db"))
    d.init_schema()
    yield d
    d.close()


def _seed_domain(db, name="Domain Study: Semiconductors", value="digest " + "x" * 500):
    cur = db.execute(
        "INSERT INTO monitors (name, check_type, check_config, schedule_seconds, "
        "enabled, cooldown_minutes, notify_condition) VALUES (?, 'query', '{}', 86400, 1, 0, 'on_change')",
        (name,),
    )
    mid = cur.lastrowid
    db.execute(
        "INSERT INTO monitor_results (monitor_id, status, value) VALUES (?, 'ok', ?)",
        (mid, value),
    )
    return mid


class TestMigration:
    def test_tables_exist(self, db):
        names = {r["name"] for r in db.fetchall(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        assert "dossiers" in names
        assert "dossier_revisions" in names


class TestBound:
    def test_under_cap_passthrough(self):
        from app.core.dossiers import _bound
        assert _bound("short text.", 100) == "short text."

    def test_over_cap_ends_at_sentence(self):
        from app.core.dossiers import _bound
        text = ("Alpha sentence one. " * 30) + "Beta tail that will be cut mid"
        out = _bound(text, 300)
        assert len(out) <= 300
        assert out.endswith(".")   # never mid-thought

    def test_slug(self):
        from app.core.dossiers import _slug
        assert _slug("AI and ML") == "ai-and-ml"
        assert _slug("Domain Study: Finance!") == "domain-study-finance"


class TestUpdateDossier:
    def _cand(self):
        return {"kind": "domain", "dkey": "semiconductors", "title": "Semiconductors",
                "monitor_name": "Domain Study: Semiconductors", "since": None}

    def test_first_consolidation_creates(self, db):
        from app.core import dossiers as mod
        with patch.object(mod, "llm") as m:
            m.invoke_nothink = AsyncMock(return_value=GOOD_BODY)
            res = asyncio.run(mod._update_dossier(db, self._cand(), "s" * 400, None))
        assert res and res["title"] == "Semiconductors"
        row = mod.get_dossier(db, "domain", "semiconductors")
        assert row is not None and row["update_count"] == 1
        assert "per-customer licenses" in row["body"]
        assert "CHANGED:" not in row["body"]                 # marker stripped from body
        assert "license regime expanded" in row["changed_note"]
        assert db.fetchone("SELECT COUNT(*) n FROM dossier_revisions")["n"] == 0

    def test_revision_trail_on_update(self, db):
        from app.core import dossiers as mod
        with patch.object(mod, "llm") as m:
            m.invoke_nothink = AsyncMock(return_value=GOOD_BODY)
            asyncio.run(mod._update_dossier(db, self._cand(), "s" * 400, None))
            m.invoke_nothink = AsyncMock(
                return_value=GOOD_BODY.replace("412", "988") + " v2")
            asyncio.run(mod._update_dossier(db, self._cand(), "t" * 400, None))
        row = mod.get_dossier(db, "domain", "semiconductors")
        assert row["update_count"] == 2 and "988" in row["body"]
        revs = db.fetchall("SELECT * FROM dossier_revisions")
        assert len(revs) == 1 and "412" in revs[0]["body"]   # prior body archived

    def test_malformed_output_writes_nothing(self, db):
        from app.core import dossiers as mod
        with patch.object(mod, "llm") as m:
            m.invoke_nothink = AsyncMock(return_value="I could not do that.")
            res = asyncio.run(mod._update_dossier(db, self._cand(), "s" * 400, None))
        assert res is None
        assert mod.get_dossier(db, "domain", "semiconductors") is None

    def test_thin_sources_skip(self, db):
        from app.core import dossiers as mod
        with patch.object(mod, "llm") as m:
            m.invoke_nothink = AsyncMock(return_value=GOOD_BODY)
            res = asyncio.run(mod._update_dossier(db, self._cand(), "tiny", None))
        assert res is None
        m.invoke_nothink.assert_not_awaited()               # no LLM call on empty material


class TestRetrieval:
    def _install(self, db):
        from app.core import dossiers as mod
        with patch.object(mod, "llm") as m:
            m.invoke_nothink = AsyncMock(return_value=GOOD_BODY)
            asyncio.run(mod._update_dossier(
                db, {"kind": "domain", "dkey": "semiconductors", "title": "Semiconductors",
                     "monitor_name": "x", "since": None}, "s" * 400, None))

    def test_relevant_match_returns_excerpt(self, db):
        from app.core.dossiers import get_relevant_dossiers
        self._install(db)
        out = get_relevant_dossiers(db, "what is the state of semiconductor export controls?")
        assert len(out) == 1
        assert out[0]["title"] == "Semiconductors"
        assert "per-customer licenses" in out[0]["excerpt"]
        assert "## " not in out[0]["excerpt"]               # excerpt is prose, not headings

    def test_no_overlap_returns_empty(self, db):
        from app.core.dossiers import get_relevant_dossiers
        self._install(db)
        assert get_relevant_dossiers(db, "recipe for sourdough bread") == []

    def test_domain_label_mapping(self, db):
        from app.core.dossiers import get_domain_dossier
        self._install(db)
        assert get_domain_dossier(db, "Semiconductors") is not None
        assert get_domain_dossier(db, "Finance") is None


class TestCandidates:
    def test_new_digest_makes_candidate(self, db):
        from app.core.dossiers import _domains_needing_update
        _seed_domain(db)
        cands = _domains_needing_update(db)
        assert any(c["dkey"] == "semiconductors" and c["since"] is None for c in cands)

    def test_current_dossier_not_candidate(self, db):
        from app.core import dossiers as mod
        _seed_domain(db)
        with patch.object(mod, "llm") as m:
            m.invoke_nothink = AsyncMock(return_value=GOOD_BODY)
            asyncio.run(mod.consolidate_dossiers(db))
        # dossier now newer than the digest → no longer a candidate
        assert all(c["dkey"] != "semiconductors" for c in mod._domains_needing_update(db))

    def test_numeric_tension_detected_across_dossiers(self, db):
        # Thinking rung brick one: two CURRENT dossiers asserting materially
        # different values for the same quantity get flagged; agreement within
        # 5% and different-quantity numbers do not.
        from app.core import dossiers as mod

        def _seed(dkey, title, bullet):
            db.execute(
                "INSERT INTO dossiers (kind, dkey, title, body) VALUES ('domain', ?, ?, ?)",
                (dkey, title,
                 f"## Current understanding\nprose\n## Key facts & figures\n{bullet}\n"))

        _seed("finance", "Finance",
              "* Headline CPI inflation cooled to 3.4% annually, shelter-driven (cnbc.com)")
        _seed("economics", "Economics",
              "* Headline CPI inflation printed at 2.9% annually per the report (apnews.com)")
        _seed("energy", "Energy",
              "* Brent crude fell 4.7% after the strike pause (reuters.com)")
        tensions = mod._numeric_tensions(db)
        assert len(tensions) == 1
        assert "3.4%" in tensions[0] and "2.9%" in tensions[0]
        assert "Finance" in tensions[0] and "Economics" in tensions[0]

    def test_numeric_agreement_not_flagged(self, db):
        from app.core import dossiers as mod
        for dkey, title, val in (("finance", "Finance", "3.4"), ("economics", "Economics", "3.4")):
            db.execute(
                "INSERT INTO dossiers (kind, dkey, title, body) VALUES ('domain', ?, ?, ?)",
                (dkey, title,
                 f"## Key facts & figures\n* Headline CPI inflation cooled to {val}% annually (cnbc.com)\n"))
        assert mod._numeric_tensions(db) == []

    def test_weak_citation_flagging(self, db):
        # Consolidation-time authority check (audit 2026-08-13): a Key-facts
        # bullet supported ONLY by a dataset-confirmed junk host gets tagged;
        # unknown hosts score neutral and never trip it; other sections and
        # co-cited reputable hosts stay untouched.
        from app.core import dossiers as mod
        body = ("## Current understanding\n"
                "Prose citing junk (junkfarm.com) stays untouched here.\n"
                "## Key facts & figures\n"
                "* CPI hit 3.4% annually (junkfarm.com)\n"
                "* Oil fell 4.7% (junkfarm.com; reuters.com)\n"
                "* Niche detail from an unknown host (obscure-lab.edu)\n"
                "## Open questions\n- One?\n")
        fake_auth = {"junkfarm.com": 0.1, "reuters.com": 0.95}
        with patch("app.core.source_authority.authority",
                   side_effect=lambda h: fake_auth.get(h, 0.5)):
            out = mod._flag_weak_citations(body)
        lines = out.split("\n")
        assert any("CPI" in l and "⚠ low-authority" in l for l in lines)
        assert all("⚠" not in l for l in lines if "Oil fell" in l)        # co-cite saves it
        assert all("⚠" not in l for l in lines if "Niche detail" in l)    # unknown = neutral
        assert all("⚠" not in l for l in lines if "Prose citing" in l)    # only Key facts

    def test_entity_cold_start_reserved_slot(self, db):
        # Audit 2026-08-13: '~new' staleness sorts after every real timestamp
        # and ~37 domains regenerate staleness daily, so the top-8 cut was
        # always full of domains — no entity ever earned its FIRST dossier
        # (zero existed after 60+ consolidations). One slot per cycle is now
        # reserved for the top entity candidate when none makes the cut.
        from app.core import dossiers as mod
        domains = [{"kind": "domain", "dkey": f"d{i}", "title": f"D{i}",
                    "monitor_name": f"Domain Study: D{i}", "since": None,
                    "staleness": f"2026-08-0{i + 1}"} for i in range(8)]
        entity = {"kind": "entity", "dkey": "openai", "title": "OpenAI",
                  "subject": "OpenAI", "since": None, "staleness": "~new"}
        seen = []

        async def fake_update(db_, cand, sources, syn_model, **kw):
            seen.append((cand["kind"], cand["dkey"]))
            return None

        with patch.object(mod, "_domains_needing_update", return_value=domains), \
             patch.object(mod, "_storylines_needing_update", return_value=[]), \
             patch.object(mod, "_entities_needing_update", return_value=[entity]), \
             patch.object(mod, "_update_dossier", side_effect=fake_update):
            asyncio.run(mod.consolidate_dossiers(db))
        assert ("entity", "openai") in seen           # reserved slot honored
        assert len(seen) == mod._MAX_UPDATES_PER_CYCLE
        assert sum(1 for k, _ in seen if k == "domain") == mod._MAX_UPDATES_PER_CYCLE - 1

    def test_mature_storyline_is_candidate(self, db):
        from app.core.dossiers import _storylines_needing_update
        cur = db.execute(
            "INSERT INTO storylines (story_key, title, summary, update_count) "
            "VALUES ('iran-hormuz', 'Iran-Hormuz tensions', 'state', 5)")
        db.execute(
            "INSERT INTO storyline_events (storyline_id, summary, is_new) VALUES (?, 'ev', 1)",
            (cur.lastrowid,))
        assert any(c["dkey"] == "iran-hormuz" for c in _storylines_needing_update(db))

    def test_immature_storyline_ignored(self, db):
        from app.core.dossiers import _storylines_needing_update
        db.execute(
            "INSERT INTO storylines (story_key, title, summary, update_count) "
            "VALUES ('minor', 'Minor thread', 's', 1)")
        assert all(c["dkey"] != "minor" for c in _storylines_needing_update(db))


class TestConsolidateCycle:
    def test_end_to_end_digest_string(self, db):
        from app.core import dossiers as mod
        _seed_domain(db)
        with patch.object(mod, "llm") as m:
            m.invoke_nothink = AsyncMock(return_value=GOOD_BODY)
            out = asyncio.run(mod.consolidate_dossiers(db))
        assert "KNOWING" in out and "Semiconductors" in out
        assert mod.get_dossier(db, "domain", "semiconductors") is not None

    def test_nothing_to_do_marker(self, db):
        from app.core.dossiers import consolidate_dossiers
        out = asyncio.run(consolidate_dossiers(db))
        assert "nothing to consolidate" in out


GOOD_BODY_WITH_FORECAST = GOOD_BODY + (
    "\nFORECAST: Allied fabs will adopt matching export rules | 60 days | 0.6"
)


class TestKnowingRungs:
    def test_forecast_minted_and_stripped(self, db):
        from app.core import dossiers as mod
        with patch.object(mod, "llm") as m:
            m.invoke_nothink = AsyncMock(return_value=GOOD_BODY_WITH_FORECAST)
            asyncio.run(mod._update_dossier(
                db, {"kind": "domain", "dkey": "semiconductors", "title": "Semiconductors",
                     "monitor_name": "x", "since": None}, "s" * 400, None))
        row = mod.get_dossier(db, "domain", "semiconductors")
        assert "FORECAST:" not in row["body"]          # marker never stored
        f = db.fetchone("SELECT claim, source_monitor, storyline_key FROM forecasts ORDER BY rowid DESC LIMIT 1")
        assert f is not None and "matching export rules" in f["claim"]
        assert f["source_monitor"] == "Knowledge Consolidation"
        assert f["storyline_key"].startswith("dossier:")

    def test_open_questions_extraction(self):
        from app.core.dossiers import _extract_open_questions
        qs = _extract_open_questions(GOOD_BODY, limit=2)
        assert len(qs) == 2                            # extraction is shape-blind
        assert qs[0].lower().startswith("will allied governments")
        assert "matching rules have allied fabs" in qs[1].lower()
        assert _extract_open_questions("no sections here") == []

    def test_cycle_feeds_curiosity(self, db):
        from app.core import dossiers as mod
        _seed_domain(db)
        with patch.object(mod, "llm") as m:
            m.invoke_nothink = AsyncMock(return_value=GOOD_BODY)
            asyncio.run(mod.consolidate_dossiers(db))
        r = db.fetchone(
            "SELECT topic, source FROM curiosity_queue WHERE source = 'dossier_open_question' "
            "ORDER BY id DESC LIMIT 1")
        assert r is not None
        assert "Semiconductors" in r["topic"] and "matching rules" in r["topic"]
        # future-shaped questions are forecast material, never research targets
        assert not r["topic"].split(": ", 1)[-1].lower().startswith("will ")

    def test_state_of_world_needs_three_domains(self, db):
        from app.core import dossiers as mod
        db.execute("INSERT INTO dossiers (kind, dkey, title, body, update_count) "
                   "VALUES ('domain','a','A', ?, 1)", (GOOD_BODY,))
        db.execute("INSERT INTO dossiers (kind, dkey, title, body, update_count) "
                   "VALUES ('domain','b','B', ?, 1)", (GOOD_BODY,))
        with patch.object(mod, "llm") as m:
            m.invoke_nothink = AsyncMock(return_value=GOOD_BODY)
            res = asyncio.run(mod._update_state_of_world(db, None))
        assert res is None                              # <3 domains → no world-view
        m.invoke_nothink.assert_not_awaited()

    def test_state_of_world_capstone(self, db):
        from app.core import dossiers as mod
        for k in ("a", "b", "c"):
            db.execute("INSERT INTO dossiers (kind, dkey, title, body, update_count) "
                       "VALUES ('domain', ?, ?, ?, 1)", (k, k.upper(), GOOD_BODY))
        world = GOOD_BODY.replace("Alpha", "Macro")
        with patch.object(mod, "llm") as m:
            m.invoke_nothink = AsyncMock(return_value=world)
            res = asyncio.run(mod._update_state_of_world(db, None))
        assert res is not None
        row = mod.get_dossier(db, "meta", "state-of-the-world")
        assert row is not None and row["title"] == "State of the World"

    def test_world_retrievable_in_chat(self, db):
        from app.core import dossiers as mod
        for k in ("a", "b", "c"):
            db.execute("INSERT INTO dossiers (kind, dkey, title, body, update_count) "
                       "VALUES ('domain', ?, ?, ?, 1)", (k, k.upper(), GOOD_BODY))
        with patch.object(mod, "llm") as m:
            m.invoke_nothink = AsyncMock(return_value=GOOD_BODY)
            asyncio.run(mod._update_state_of_world(db, None))
        out = mod.get_relevant_dossiers(db, "what is the state of the world right now?")
        assert out and out[0]["title"] == "State of the World"


class TestEntityDossiers:
    def _seed_facts(self, db, subject, n):
        # Production always initializes KnowledgeGraph at startup, which owns and
        # upgrades kg_facts (adds superseded_at etc. beyond the base schema) —
        # mirror that here or the entity queries see a pre-bitemporal table.
        from app.core.kg import KnowledgeGraph
        KnowledgeGraph(db)
        for i in range(n):
            db.execute(
                "INSERT INTO kg_facts (subject, predicate, object, confidence, created_at) "
                "VALUES (?, 'related_to', ?, 0.8, datetime('now'))",
                (subject, f"thing-{i}"))

    def test_active_entity_becomes_candidate(self, db):
        from app.core.dossiers import _entities_needing_update
        self._seed_facts(db, "Anthropic", 9)
        cands = _entities_needing_update(db)
        assert any(c["dkey"] == "anthropic" and c["kind"] == "entity" for c in cands)
        # New entities must yield to the domain backlog in the staleness sort.
        assert all(c["staleness"] == "~new" for c in cands if c["dkey"] == "anthropic")

    def test_thin_or_generic_entities_filtered(self, db):
        from app.core.dossiers import _entities_needing_update
        self._seed_facts(db, "Anthropic", 3)          # below _ENTITY_MIN_FACTS
        self._seed_facts(db, "market", 12)            # generic stopword
        assert _entities_needing_update(db) == []

    def test_entity_sources_render_facts(self, db):
        from app.core.dossiers import _entity_sources
        self._seed_facts(db, "Anthropic", 9)
        src = _entity_sources(db, "Anthropic", None)
        assert "KG FACTS" in src and "Anthropic related_to thing-0" in src

    def test_cycle_builds_entity_dossier(self, db):
        from app.core import dossiers as mod
        self._seed_facts(db, "Anthropic", 9)
        with patch.object(mod, "llm") as m:
            m.invoke_nothink = AsyncMock(return_value=GOOD_BODY)
            out = asyncio.run(mod.consolidate_dossiers(db))
        assert "Anthropic" in out
        row = mod.get_dossier(db, "entity", "anthropic")
        assert row is not None and row["title"] == "Anthropic"

    def test_current_entity_dossier_not_recandidated(self, db):
        from app.core import dossiers as mod
        self._seed_facts(db, "Anthropic", 9)
        with patch.object(mod, "llm") as m:
            m.invoke_nothink = AsyncMock(return_value=GOOD_BODY)
            asyncio.run(mod.consolidate_dossiers(db))
        assert all(c["dkey"] != "anthropic" for c in mod._entities_needing_update(db))
