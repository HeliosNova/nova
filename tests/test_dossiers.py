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
    "- Whether allied fabs adopt matching rules\n"
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
