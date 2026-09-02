"""Backup sidecar picks the newest VERIFIED snapshot by mtime (2026-09-02).

First live run: the lexicographic sort chose `nova-premove.db` (an Aug 28
leftover that sorts after every `nova-2026*.db`), it failed verification, and
the sync copied nothing — the DR legs would have silently stopped updating.
"""
from __future__ import annotations

import os
import sqlite3
import time

import pytest

from scripts import backup_sidecar as bs


def _db(path, ok=True):
    if ok:
        c = sqlite3.connect(path)
        c.execute("CREATE TABLE t (x)")
        c.execute("INSERT INTO t VALUES (1)")
        c.commit()
        c.close()
    else:
        path.write_bytes(b"not a database at all")


def _touch(path, age_s):
    t = time.time() - age_s
    os.utime(path, (t, t))


@pytest.fixture
def world(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    leg = tmp_path / "leg"
    leg.mkdir()
    old_ok = src / "nova-20260827.db"
    _db(old_ok)
    _touch(old_ok, 6 * 86400)
    newest_ok = src / "nova-20260902.db"
    _db(newest_ok)
    _touch(newest_ok, 60)
    bad_late_name = src / "nova-premove.db"
    _db(bad_late_name, ok=False)
    _touch(bad_late_name, 5 * 86400)
    return src, leg


def test_newest_verified_by_mtime_is_copied(world):
    src, leg = world
    report = bs.sync_once(src, [leg], keep=7)
    assert (leg / "nova-20260902.db").exists()
    assert not (leg / "nova-premove.db").exists()
    assert report["copied"] == [str(leg / "nova-20260902.db")]
    assert not report["failed"]


def test_corrupt_newest_falls_back_to_the_next_verified(world):
    src, leg = world
    corrupt_newest = src / "nova-20260903.db"
    _db(corrupt_newest, ok=False)
    _touch(corrupt_newest, 1)
    report = bs.sync_once(src, [leg], keep=7)
    assert report["failed"] == ["source snapshot nova-20260903.db failed verification"]
    assert (leg / "nova-20260902.db").exists()


def test_prune_keeps_the_newest_by_mtime(world):
    src, leg = world
    for i in range(4):
        p = leg / f"nova-2026080{i}.db"
        _db(p)
        _touch(p, (10 - i) * 86400)
    bs.sync_once(src, [leg], keep=2)
    kept = sorted(p.name for p in leg.glob("nova-*.db"))
    assert kept == ["nova-20260803.db", "nova-20260902.db"]
