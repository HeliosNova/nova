"""Dossier API (knowing tier read surface, 2026-08-12) — list/detail/revisions.
Mounts only the dossiers router with auth overridden: the router is read-only
SQL, so it needs no services and no full app lifespan."""

from __future__ import annotations

import pytest


BODY = "## Current understanding\nUnderstanding text.\n## Open questions\n- q1"


@pytest.fixture()
def db(tmp_path):
    from app.database import SafeDB
    d = SafeDB(str(tmp_path / "dossier_api_test.db"))
    d.init_schema()
    yield d
    d.close()


@pytest.fixture()
def client(db, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    import app.api.dossiers as mod
    from app.auth import require_auth

    monkeypatch.setattr(mod, "get_db", lambda: db)
    api = FastAPI()
    api.include_router(mod.router, prefix="/api")
    api.dependency_overrides[require_auth] = lambda: None
    return TestClient(api)


def _seed(db):
    cur = db.execute(
        "INSERT INTO dossiers (kind, dkey, title, body, changed_note, update_count) "
        "VALUES ('domain', 'finance', 'Finance', ?, 'rates shifted', 3)", (BODY,))
    dom_id = cur.lastrowid
    cur = db.execute(
        "INSERT INTO dossiers (kind, dkey, title, body, changed_note, update_count) "
        "VALUES ('meta', 'state-of-the-world', 'State of the World', ?, 'macro moved', 5)", (BODY,))
    meta_id = cur.lastrowid
    cur = db.execute(
        "INSERT INTO dossier_revisions (dossier_id, body, valid_from) "
        "VALUES (?, 'old understanding', datetime('now','-2 day'))", (dom_id,))
    rev_id = cur.lastrowid
    return dom_id, meta_id, rev_id


class TestList:
    def test_meta_sorts_first(self, client, db):
        _seed(db)
        out = client.get("/api/dossiers").json()["dossiers"]
        assert len(out) == 2
        assert out[0]["kind"] == "meta"          # capstone pinned to the top
        assert "body" not in out[0]              # list stays light
        assert out[0]["body_chars"] > 0

    def test_kind_filter(self, client, db):
        _seed(db)
        out = client.get("/api/dossiers?kind=domain").json()["dossiers"]
        assert len(out) == 1 and out[0]["dkey"] == "finance"

    def test_bad_kind_422(self, client):
        assert client.get("/api/dossiers?kind=junk").status_code == 422


class TestDetail:
    def test_detail_includes_body_and_revision_count(self, client, db):
        dom_id, _, _ = _seed(db)
        out = client.get(f"/api/dossiers/{dom_id}").json()
        assert out["title"] == "Finance"
        assert "Understanding text" in out["body"]
        assert out["revision_count"] == 1

    def test_404(self, client):
        assert client.get("/api/dossiers/9999").status_code == 404


class TestRevisions:
    def test_revision_index_and_body(self, client, db):
        dom_id, _, rev_id = _seed(db)
        idx = client.get(f"/api/dossiers/{dom_id}/revisions").json()["revisions"]
        assert len(idx) == 1 and idx[0]["id"] == rev_id and "body" not in idx[0]
        rev = client.get(f"/api/dossiers/{dom_id}/revisions/{rev_id}").json()
        assert rev["body"] == "old understanding"

    def test_revision_scoped_to_dossier(self, client, db):
        dom_id, meta_id, rev_id = _seed(db)
        # A revision id fetched under the WRONG dossier must 404, not leak.
        assert client.get(f"/api/dossiers/{meta_id}/revisions/{rev_id}").status_code == 404
