"""Procedure skills (migration 30) — NL skills, semantic-match only.

The regex+steps formalism produced 0 organic skills in Nova's lifetime (the
LLM can't reliably author brittle trigger regexes). Procedure skills store
natural-language guidance and match semantically; an empty trigger_pattern
must never reach re.search (it would match every query).
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from app.core.skills import SkillStore
from app.database import SafeDB


@pytest.fixture
def db(tmp_path):
    d = SafeDB(str(tmp_path / "test.db"))
    d.init_schema()
    yield d
    d.close()


@pytest.fixture
def store(db, monkeypatch):
    monkeypatch.setenv("ENABLE_SEMANTIC_SKILL_MATCHING", "false")
    from app.config import reset_config
    reset_config()
    s = SkillStore(db)
    yield s
    reset_config()


PROC = (
    "For queries like: compare two crypto prices and compute their ratio\n"
    "Effective tool sequence:\n"
    "1. web_search(query='BTC price')\n"
    "2. web_search(query='ETH price')\n"
    "Then synthesize the tool results into a direct answer with specific figures."
)


class TestCreateProcedureSkill:
    def test_create_and_roundtrip(self, store):
        sid = store.create_procedure_skill(
            "crypto_ratio_procedure", "compare two crypto prices", PROC)
        assert sid is not None
        row = store._db.fetchone("SELECT * FROM skills WHERE id = ?", (sid,))
        skill = store._row_to_skill(row)
        assert skill.kind == "procedure"
        assert skill.procedure_text == PROC
        assert skill.trigger_pattern == ""
        assert skill.steps == []

    def test_too_short_rejected(self, store):
        assert store.create_procedure_skill("x", "desc", "too short") is None

    def test_too_long_rejected(self, store):
        assert store.create_procedure_skill("x", "desc", "y" * 3000) is None

    def test_same_name_updates_in_place(self, store):
        sid1 = store.create_procedure_skill("proc_a", "first description", PROC)
        sid2 = store.create_procedure_skill(
            "proc_a", "second description", PROC + " Updated guidance.")
        assert sid2 == sid1
        row = store._db.fetchone("SELECT procedure_text FROM skills WHERE id = ?", (sid1,))
        assert "Updated guidance." in row["procedure_text"]


class TestEmptyPatternNeverRegexMatches:
    def test_regex_path_skips_procedure_skills(self, store):
        sid = store.create_procedure_skill(
            "proc_skip", "some procedural knowledge", PROC)
        assert sid is not None
        # With semantic matching disabled, an empty trigger must never match —
        # re.search("") would match EVERY query.
        with patch.object(store, "_semantic_match", return_value=None):
            assert store.get_matching_skill("completely unrelated query") is None


class TestProcedureFallbackText:
    @pytest.mark.asyncio
    async def test_fallback_creates_procedure_on_guard_reject(self, store, monkeypatch):
        """When strict extraction fails a guard on a multi-tool interaction,
        the procedure fallback memorializes it instead of discarding."""
        monkeypatch.setenv("ENABLE_AUTO_SKILL_CREATION", "true")
        from app.config import reset_config
        reset_config()
        from app.core import auto_skills

        async def fake_invoke(*a, **k):
            # Invalid regex → strict path rejects → fallback fires
            return ('{"name": "metal_prices", "trigger_pattern": "([unclosed", '
                    '"steps": [{"tool": "web_search", "args_template": {"query": "{query}"}}]}')

        monkeypatch.setattr(auto_skills.llm, "invoke_nothink", fake_invoke)
        trs = [
            {"tool": "web_search", "args": {"query": "gold price"}, "output": "gold $2400"},
            {"tool": "web_search", "args": {"query": "silver price"}, "output": "silver $29"},
        ]
        await auto_skills.maybe_extract_skill(
            "what is the gold to silver ratio right now", trs,
            "The ratio is about 82.", store, quality_score=0.9)
        row = store._db.fetchone(
            "SELECT kind, procedure_text FROM skills WHERE name = 'metal_prices'")
        assert row is not None
        assert row["kind"] == "procedure"
        assert "web_search" in row["procedure_text"]
        reset_config()
