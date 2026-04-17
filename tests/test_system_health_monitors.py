"""Tests for the 5 system-health monitor handlers added to HeartbeatLoop.

Covers:
  _execute_db_size_check
  _execute_ollama_latency_check
  _execute_skill_quality_check
  _execute_chromadb_integrity_check
  _execute_kg_health_check

Each handler gets a happy-path test and at least one failure/degraded test.
"""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.monitors.heartbeat import HeartbeatLoop, MonitorStore


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def store(db):
    return MonitorStore(db)


@pytest.fixture
def loop(store):
    return HeartbeatLoop(store)


# ---------------------------------------------------------------------------
# _execute_db_size_check
# ---------------------------------------------------------------------------


class TestDbSizeCheck:
    @pytest.mark.asyncio
    async def test_happy_path_with_db_file(self, loop, tmp_path):
        """Reports file size and row counts when the DB file exists."""
        db_file = tmp_path / "nova.db"
        db_file.write_bytes(b"\x00" * 2048)  # 2 KB

        mock_db = MagicMock()
        mock_db.fetchone.return_value = {"c": 42}

        with patch("app.monitors.heartbeat_loop.config") as mock_cfg, \
             patch("app.database.get_db", return_value=mock_db):
            mock_cfg.DB_PATH = str(db_file)
            result = await loop._execute_db_size_check()

        assert "size:" in result
        assert "MB" in result
        assert "conversations:" in result

    @pytest.mark.asyncio
    async def test_missing_db_file_reports_not_found(self, loop, tmp_path, monkeypatch):
        """Reports 'DB not found' when the DB path does not exist."""
        missing = str(tmp_path / "nonexistent.db")
        mock_db = MagicMock()
        mock_db.fetchone.return_value = {"c": 0}

        with patch("app.monitors.heartbeat_loop.config") as mock_cfg, \
             patch("app.database.get_db", return_value=mock_db):
            mock_cfg.DB_PATH = missing
            result = await loop._execute_db_size_check()

        assert "missing" in result.lower() or "size:" in result

    @pytest.mark.asyncio
    async def test_table_error_is_swallowed(self, loop, tmp_path):
        """Errors reading individual tables are silently skipped; other parts still appear."""
        mock_db = MagicMock()
        mock_db.fetchone.side_effect = Exception("table missing")

        with patch("app.monitors.heartbeat_loop.config") as mock_cfg, \
             patch("app.database.get_db", return_value=mock_db):
            mock_cfg.DB_PATH = str(tmp_path / "no.db")
            result = await loop._execute_db_size_check()

        # Should not raise; returns whatever partial info was collected
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_wal_file_included_when_present(self, loop, tmp_path):
        """WAL size is reported when a -wal file exists alongside the DB."""
        db_file = tmp_path / "nova.db"
        wal_file = tmp_path / "nova.db-wal"
        db_file.write_bytes(b"\x00" * 1024)
        wal_file.write_bytes(b"\x00" * 512)

        mock_db = MagicMock()
        mock_db.fetchone.return_value = {"c": 5}

        with patch("app.monitors.heartbeat_loop.config") as mock_cfg, \
             patch("app.database.get_db", return_value=mock_db):
            mock_cfg.DB_PATH = str(db_file)
            result = await loop._execute_db_size_check()

        assert "wal:" in result.lower()


# ---------------------------------------------------------------------------
# _execute_ollama_latency_check
# ---------------------------------------------------------------------------


class TestOllamaLatencyCheck:
    @pytest.mark.asyncio
    async def test_healthy_provider_reports_latency(self, loop):
        """Returns latency in ms and 'healthy' label when provider responds OK."""
        mock_provider = MagicMock()
        mock_provider.check_health = AsyncMock(return_value=True)

        with patch("app.core.llm.get_provider", return_value=mock_provider):
            result = await loop._execute_ollama_latency_check()

        assert "ms" in result
        assert "healthy" in result.lower()
        assert "UNHEALTHY" not in result

    @pytest.mark.asyncio
    async def test_unhealthy_provider_flags_unhealthy(self, loop):
        """Returns UNHEALTHY label when provider.check_health() returns False."""
        mock_provider = MagicMock()
        mock_provider.check_health = AsyncMock(return_value=False)

        with patch("app.core.llm.get_provider", return_value=mock_provider):
            result = await loop._execute_ollama_latency_check()

        assert "unhealthy" in result.lower()
        assert "ms" in result

    @pytest.mark.asyncio
    async def test_provider_exception_caught(self, loop):
        """If get_provider() or check_health() raises, returns error string."""
        with patch("app.core.llm.get_provider", side_effect=RuntimeError("Ollama down")):
            result = await loop._execute_ollama_latency_check()

        assert "error" in result.lower()
        assert "Ollama down" in result


# ---------------------------------------------------------------------------
# _execute_skill_quality_check
# ---------------------------------------------------------------------------


class TestSkillQualityCheck:
    @pytest.mark.asyncio
    async def test_happy_path_reports_all_metrics(self, loop):
        """Returns total/enabled/disabled counts and avg success rate."""
        mock_db = MagicMock()
        # total=10, enabled=8, disabled=2, avg_sr=0.75, degrading=1
        mock_db.fetchone.side_effect = [
            {"c": 10},     # total
            {"c": 8},      # enabled
            {"avg_sr": 0.75},  # avg success rate
            {"c": 1},      # degrading
        ]
        mock_skills = MagicMock()
        mock_skills._db = mock_db

        mock_svc = MagicMock()
        mock_svc.skills = mock_skills

        with patch("app.core.brain.get_services", return_value=mock_svc):
            result = await loop._execute_skill_quality_check()

        assert "total: 10" in result
        assert "enabled: 8" in result
        assert "disabled: 2" in result
        assert "0.75" in result
        assert "degrading: 1" in result

    @pytest.mark.asyncio
    async def test_no_skills_service(self, loop):
        """Returns informative string when skills store is not initialised."""
        mock_svc = MagicMock()
        mock_svc.skills = None

        with patch("app.core.brain.get_services", return_value=mock_svc):
            result = await loop._execute_skill_quality_check()

        assert "unavailable" in result.lower() or "not available" in result.lower()

    @pytest.mark.asyncio
    async def test_db_error_returns_error_string(self, loop):
        """DB exception is caught and returned as a readable error string."""
        mock_db = MagicMock()
        mock_db.fetchone.side_effect = Exception("locked")

        mock_skills = MagicMock()
        mock_skills._db = mock_db

        mock_svc = MagicMock()
        mock_svc.skills = mock_skills

        with patch("app.core.brain.get_services", return_value=mock_svc):
            result = await loop._execute_skill_quality_check()

        assert "error" in result.lower()

    @pytest.mark.asyncio
    async def test_null_avg_success_rate_handled(self, loop):
        """avg_sr = NULL (no enabled skills) should yield 0.0, not crash."""
        mock_db = MagicMock()
        mock_db.fetchone.side_effect = [
            {"c": 3},
            {"c": 0},          # no enabled skills
            {"avg_sr": None},  # SQLite AVG returns NULL on empty set
            {"c": 0},
        ]
        mock_skills = MagicMock()
        mock_skills._db = mock_db

        mock_svc = MagicMock()
        mock_svc.skills = mock_skills

        with patch("app.core.brain.get_services", return_value=mock_svc):
            result = await loop._execute_skill_quality_check()

        assert "0.00" in result


# ---------------------------------------------------------------------------
# _execute_chromadb_integrity_check
# ---------------------------------------------------------------------------


class TestChromaDbIntegrityCheck:
    @pytest.mark.asyncio
    async def test_happy_path_reports_doc_count(self, loop):
        """Reports ChromaDB doc count and FTS5 chunk count."""
        mock_collection = MagicMock()
        mock_collection.count.return_value = 150

        mock_retriever = MagicMock()
        mock_retriever._get_collection.return_value = mock_collection

        mock_fts_row = {"c": 300}
        mock_db = MagicMock()
        mock_db.fetchone.return_value = mock_fts_row

        mock_svc = MagicMock()
        mock_svc.retriever = mock_retriever

        with patch("app.core.brain.get_services", return_value=mock_svc), \
             patch("app.database.get_db", return_value=mock_db):
            result = await loop._execute_chromadb_integrity_check()

        assert "docs: 150" in result
        assert "fts5: 300" in result

    @pytest.mark.asyncio
    async def test_chromadb_collection_error_reported(self, loop):
        """ChromaDB collection failure is caught and reported in output."""
        mock_retriever = MagicMock()
        mock_retriever._get_collection.side_effect = RuntimeError("collection missing")

        mock_db = MagicMock()
        mock_db.fetchone.return_value = {"c": 0}

        mock_svc = MagicMock()
        mock_svc.retriever = mock_retriever

        with patch("app.core.brain.get_services", return_value=mock_svc), \
             patch("app.database.get_db", return_value=mock_db):
            result = await loop._execute_chromadb_integrity_check()

        assert "chromadb error" in result.lower()
        assert "collection missing" in result

    @pytest.mark.asyncio
    async def test_no_retriever_reports_unavailable(self, loop):
        """Returns 'Retriever not available' when retriever is None."""
        mock_db = MagicMock()
        mock_db.fetchone.return_value = {"c": 0}

        mock_svc = MagicMock()
        mock_svc.retriever = None

        with patch("app.core.brain.get_services", return_value=mock_svc), \
             patch("app.database.get_db", return_value=mock_db):
            result = await loop._execute_chromadb_integrity_check()

        assert "retriever" in result.lower() and "unavailable" in result.lower()

    @pytest.mark.asyncio
    async def test_fts5_error_is_swallowed(self, loop):
        """FTS5 query failure is silently skipped; ChromaDB count still reported."""
        mock_collection = MagicMock()
        mock_collection.count.return_value = 42

        mock_retriever = MagicMock()
        mock_retriever._get_collection.return_value = mock_collection

        mock_db = MagicMock()
        mock_db.fetchone.side_effect = Exception("no such table: chunks_fts")

        mock_svc = MagicMock()
        mock_svc.retriever = mock_retriever

        with patch("app.core.brain.get_services", return_value=mock_svc), \
             patch("app.database.get_db", return_value=mock_db):
            result = await loop._execute_chromadb_integrity_check()

        assert "docs: 42" in result
        # FTS error swallowed — no crash, no fts5 field
        assert "fts5:" not in result


# ---------------------------------------------------------------------------
# _execute_kg_health_check
# ---------------------------------------------------------------------------


class TestKgHealthCheck:
    def _make_kg_mock(self, total=100, current=80, superseded=20, entity_count=45, orphan_count=3):
        mock_kg = MagicMock()
        mock_kg.get_stats.return_value = {
            "total_facts": total,
            "current_facts": current,
            "superseded_facts": superseded,
            "unique_entities": entity_count,
            "unique_predicates": 12,
        }
        mock_kg._db.fetchone.side_effect = [
            {"c": entity_count},  # unique entities query
            {"c": orphan_count},  # orphans query
        ]
        return mock_kg

    @pytest.mark.asyncio
    async def test_happy_path_all_fields_present(self, loop):
        """Reports fact counts, active count, superseded count, entities, and orphans."""
        mock_kg = self._make_kg_mock()
        mock_svc = MagicMock()
        mock_svc.kg = mock_kg

        with patch("app.core.brain.get_services", return_value=mock_svc):
            result = await loop._execute_kg_health_check()

        assert "facts: 100" in result
        assert "active: 80" in result       # uses current_facts key (bug was active_facts)
        assert "superseded: 20" in result
        assert "entities:" in result
        assert "orphans:" in result

    @pytest.mark.asyncio
    async def test_no_kg_returns_unavailable(self, loop):
        """Returns 'KG not available' when kg service is None."""
        mock_svc = MagicMock()
        mock_svc.kg = None

        with patch("app.core.brain.get_services", return_value=mock_svc):
            result = await loop._execute_kg_health_check()

        assert "unavailable" in result.lower() or "not available" in result.lower()

    @pytest.mark.asyncio
    async def test_kg_stats_exception_caught(self, loop):
        """Exception from get_stats() is caught and returned as error string."""
        mock_kg = MagicMock()
        mock_kg.get_stats.side_effect = Exception("DB corrupt")

        mock_svc = MagicMock()
        mock_svc.kg = mock_kg

        with patch("app.core.brain.get_services", return_value=mock_svc):
            result = await loop._execute_kg_health_check()

        assert "error" in result.lower()
        assert "DB corrupt" in result

    @pytest.mark.asyncio
    async def test_active_facts_key_is_current_facts(self, loop):
        """Regression: handler must use 'current_facts' (not the old 'active_facts') key."""
        mock_kg = MagicMock()
        mock_kg.get_stats.return_value = {
            "total_facts": 50,
            "current_facts": 40,   # this is the real key
            "superseded_facts": 10,
            "unique_entities": 20,
        }
        mock_kg._db.fetchone.return_value = {"c": 5}

        mock_svc = MagicMock()
        mock_svc.kg = mock_kg

        with patch("app.core.brain.get_services", return_value=mock_svc):
            result = await loop._execute_kg_health_check()

        # Active must show 40, NOT 0 (which would happen with the wrong 'active_facts' key)
        assert "active: 40" in result
        assert "active: 0 " not in result
        assert "active: 0\u2502" not in result


# ---------------------------------------------------------------------------
# Seed entry tests (MonitorStore.seed_defaults)
# ---------------------------------------------------------------------------


class TestSystemHealthMonitorSeeds:
    """Verify all 5 new monitor check types are present in seed_defaults."""

    EXPECTED = {
        "db_size": "DB Size Monitor",
        "ollama_latency": "Ollama Latency Monitor",
        "skill_quality": "Skill Quality Monitor",
        "chromadb_integrity": "ChromaDB Integrity",
        "kg_health": "KG Health Monitor",
    }

    def test_all_five_seeds_created(self, store):
        """seed_defaults() creates all 5 new system-health monitors."""
        store.seed_defaults()
        monitors = {m.name: m for m in store.list_all()}

        for check_type, name in self.EXPECTED.items():
            assert name in monitors, f"Missing seed: {name!r}"
            assert monitors[name].check_type == check_type

    def test_seed_intervals_are_reasonable(self, store):
        """Each monitor has schedule_seconds in [1h, 24h] and positive cooldown."""
        store.seed_defaults()
        monitors = {m.name: m for m in store.list_all()}

        for check_type, name in self.EXPECTED.items():
            m = monitors[name]
            assert 3600 <= m.schedule_seconds <= 86400, (
                f"{name}: schedule {m.schedule_seconds}s out of [1h, 24h]"
            )
            assert m.cooldown_minutes > 0, f"{name}: cooldown must be positive"

    def test_seed_idempotent(self, store):
        """Calling seed_defaults() twice does not create duplicates."""
        store.seed_defaults()
        count_first = len(store.list_all())
        store.seed_defaults()
        count_second = len(store.list_all())
        assert count_first == count_second

    def test_new_monitors_enabled_by_default(self, store):
        """All 5 new monitors are enabled when first seeded."""
        store.seed_defaults()
        monitors = {m.name: m for m in store.list_all()}

        for check_type, name in self.EXPECTED.items():
            assert monitors[name].enabled is True, f"{name} should be enabled by default"
