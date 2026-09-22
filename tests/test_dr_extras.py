"""The disaster-recovery extras follow the snapshot into the volume (2026-09-22).

The maintenance monitor wrote the models manifest and the config overrides
straight onto the off-volume mount. When the backup sidecar took the legs
over on 2026-09-01, nova-app lost that mount and both legs' copies froze at
that day's versions for three weeks. write_dr_extras() writes them beside the
snapshots inside the volume; the sidecar carries them from there.
"""
from __future__ import annotations

import inspect
import json

import pytest

from app.core import backup


class _Resp:
    def __init__(self, data):
        self._d = data

    def json(self):
        return self._d


class _Client:
    def __init__(self, data=None, fail=False):
        self._d, self._fail = data, fail

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def get(self, url):
        if self._fail:
            raise RuntimeError("ollama down")
        assert url.endswith("/api/tags")
        return _Resp(self._d)


@pytest.mark.asyncio
async def test_extras_are_written_beside_the_snapshots(tmp_path, monkeypatch):
    monkeypatch.setattr(backup.httpx, "AsyncClient",
                        lambda **kw: _Client({"models": [{"name": "qwen3.8:27b"}, {"name": "bge-m3:latest"}]}))
    ov = tmp_path / "config_overrides.json"
    ov.write_text(json.dumps({"ENABLE_MINICHECK": True}), encoding="utf-8")
    dest = tmp_path / "backups"
    written = await backup.write_dr_extras(dest, "http://ollama:11434/", overrides=ov)
    assert written == ["models_manifest.txt", "config_overrides.json"]
    assert (dest / "models_manifest.txt").read_text(encoding="utf-8") == "bge-m3:latest\nqwen3.8:27b\n"
    assert json.loads((dest / "config_overrides.json").read_text(encoding="utf-8")) == {"ENABLE_MINICHECK": True}


@pytest.mark.asyncio
async def test_an_unreachable_ollama_still_saves_the_overrides(tmp_path, monkeypatch):
    monkeypatch.setattr(backup.httpx, "AsyncClient", lambda **kw: _Client(fail=True))
    ov = tmp_path / "config_overrides.json"
    ov.write_text("{}", encoding="utf-8")
    dest = tmp_path / "backups"
    written = await backup.write_dr_extras(dest, "http://ollama:11434", overrides=ov)
    assert written == ["config_overrides.json"]
    assert not (dest / "models_manifest.txt").exists()


@pytest.mark.asyncio
async def test_no_overrides_file_means_manifest_only(tmp_path, monkeypatch):
    monkeypatch.setattr(backup.httpx, "AsyncClient", lambda **kw: _Client({"models": []}))
    written = await backup.write_dr_extras(tmp_path / "b", "http://ollama:11434",
                                           overrides=tmp_path / "nope.json")
    assert written == ["models_manifest.txt"]


def test_maintenance_writes_the_extras_into_the_volume():
    from app.monitors import maintenance
    src = inspect.getsource(maintenance)
    assert "write_dr_extras(backup_dir" in src, "extras must land where the sidecar reads"
