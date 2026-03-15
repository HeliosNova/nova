"""Tests for user fact access tracking (Phase 1D)."""

from __future__ import annotations

import pytest

from app.core.memory import UserFactStore


class TestUserFactAccessTracking:
    @pytest.fixture
    def store(self, db):
        return UserFactStore(db)

    def test_refresh_access_updates_count(self, store):
        store.set("name", "Alex")
        store.refresh_access(["name"])
        # Access count should be 1
        row = store._db.fetchone(
            "SELECT access_count FROM user_facts WHERE key = ?", ("name",)
        )
        assert row["access_count"] == 1

    def test_refresh_access_increments(self, store):
        store.set("name", "Alex")
        store.refresh_access(["name"])
        store.refresh_access(["name"])
        row = store._db.fetchone(
            "SELECT access_count FROM user_facts WHERE key = ?", ("name",)
        )
        assert row["access_count"] == 2

    def test_refresh_access_updates_timestamp(self, store):
        store.set("name", "Alex")
        store.refresh_access(["name"])
        row = store._db.fetchone(
            "SELECT last_accessed_at FROM user_facts WHERE key = ?", ("name",)
        )
        assert row["last_accessed_at"] is not None

    def test_refresh_access_empty_list(self, store):
        # Should not raise
        store.refresh_access([])

    def test_get_stale_facts(self, db):
        store = UserFactStore(db)
        store.set("old_fact", "old value")
        store.set("new_fact", "new value")
        # Mark new_fact as recently accessed
        store.refresh_access(["new_fact"])
        # Force old_fact to have old access time
        db.execute(
            "UPDATE user_facts SET last_accessed_at = datetime('now', '-90 days') WHERE key = ?",
            ("old_fact",),
        )
        stale = store.get_stale_facts(days=60)
        stale_keys = [f.key for f in stale]
        assert "old_fact" in stale_keys

    def test_stale_facts_includes_never_accessed(self, db):
        store = UserFactStore(db)
        store.set("never_accessed", "some value")
        stale = store.get_stale_facts(days=60)
        stale_keys = [f.key for f in stale]
        assert "never_accessed" in stale_keys
