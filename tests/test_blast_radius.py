"""Blast-radius boundaries (audit 2026-09-01, defensive).

Measured: /backups (F:) and /offsite (E:) were mounted read-write inside the
container that runs LLM-authored `sh -c`; /data/extensions was writable by
file_ops and shell_exec and is imported into the uvicorn process at boot;
headless Chromium (Playwright 1.49.1, Nov-2024 build) launched with the
process environment including the API and bot tokens; the socket proxy's
POST=1 master switch admitted container delete/prune/create.
"""
from __future__ import annotations

import os
import re
import sqlite3
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]

# docker-compose.yml, socket-proxy/ and scripts/ are repo files, not image
# files: these three checks lint the deployment, so they run on the host and
# in the repo-mounted ephemeral container and skip inside nova-app.
needs_checkout = pytest.mark.skipif(
    not (ROOT / "docker-compose.yml").exists(),
    reason="deployment-lint test: run against a checkout (host or repo-mounted container)")
needs_scripts = pytest.mark.skipif(
    not (ROOT / "scripts" / "backup_sidecar.py").exists(),
    reason="reads scripts/backup_sidecar.py, which the image does not ship")


def test_extensions_and_chroma_dirs_are_protected_from_file_ops():
    from app.tools.file_ops import FileOpsTool
    assert {"extensions", "chromadb"} <= {d.lower() for d in FileOpsTool._PROTECTED_DIRS}


def test_shell_blocklist_covers_extensions_and_dr_legs():
    from app.tools.shell_exec import _BLOCKED_PATTERNS
    def blocked(cmd):
        return any(re.search(p, cmd) for p in _BLOCKED_PATTERNS)
    assert blocked("echo 'import os' > /data/extensions/evil.py")
    assert blocked("rm -rf /offsite/*")
    assert blocked("rm /backups/nova-20260901.db")
    assert not blocked("grep -c error /data/logs/nova-app.log")
    assert not blocked("ls /data/backups")


def test_browser_env_carries_no_secrets(monkeypatch):
    from app.tools import browser as browser_mod
    monkeypatch.setenv("NOVA_API_KEY", "secret-canary")
    monkeypatch.setenv("TELEGRAM_TOKEN", "secret-canary-2")
    monkeypatch.setenv("PLAYWRIGHT_BROWSERS_PATH", "/opt/playwright")
    env = browser_mod._browser_env()
    joined = " ".join(f"{k}={v}" for k, v in env.items())
    assert "secret-canary" not in joined
    assert not any("TOKEN" in k or "KEY" in k for k in env)
    assert env.get("PLAYWRIGHT_BROWSERS_PATH") == "/opt/playwright"
    assert "PATH" in env


@needs_checkout
def test_playwright_is_current():
    req = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    m = re.search(r"^playwright==(\d+)\.(\d+)", req, re.M)
    assert m and (int(m.group(1)), int(m.group(2))) >= (1, 60), m.group(0) if m else "no pin"


def _compose():
    return yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))


@needs_checkout
def test_app_container_holds_no_dr_legs():
    svc = _compose()["services"]["nova"]
    mounts = " ".join(str(v) for v in svc.get("volumes", []))
    assert "/backups" not in mounts and "/offsite" not in mounts


@needs_checkout
def test_backup_sidecar_is_declared_read_only_on_data():
    svcs = _compose()["services"]
    assert "backup" in svcs, "backup sidecar missing"
    b = svcs["backup"]
    vols = [str(v) for v in b.get("volumes", [])]
    assert any(v.startswith("nova_data:/data") and v.endswith(":ro") for v in vols), vols
    assert any("/backups" in v for v in vols) and any("/offsite" in v for v in vols)
    assert b.get("network_mode") == "none"
    assert "backup_sidecar.py" in " ".join(str(x) for x in b.get("entrypoint", b.get("command", [])))


@needs_checkout
def test_socket_proxy_is_an_explicit_allowlist():
    svcs = _compose()["services"]
    p = svcs["socket-proxy"]
    assert p["image"].startswith("nginx"), p["image"]
    assert "POST" not in (p.get("environment") or {})
    assert any("socket-proxy:/etc/nginx/conf.d" in str(v) for v in p.get("volumes", []))
    conf = (ROOT / "socket-proxy" / "default.conf").read_text(encoding="utf-8")
    assert "restart$" in conf and "limit_except POST" in conf
    assert "return 403" in conf
    live = "\n".join(l for l in conf.splitlines() if not l.strip().startswith("#"))
    assert "containers/create" not in live and "prune" not in live


@needs_scripts
def test_backup_sidecar_copies_verifies_and_prunes(tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("backup_sidecar", ROOT / "scripts" / "backup_sidecar.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    src = tmp_path / "src"; src.mkdir()
    leg = tmp_path / "leg"; leg.mkdir()
    for i in range(9):
        con = sqlite3.connect(src / f"nova-2026090{i}.db")
        con.execute("CREATE TABLE t (x)"); con.execute("INSERT INTO t VALUES (1)"); con.commit(); con.close()
        os.utime(src / f"nova-2026090{i}.db", (1_800_000_000 + i, 1_800_000_000 + i))
    for i in range(9):  # pre-existing old copies in the leg
        (leg / f"nova-2026080{i}.db").write_bytes(b"old")
        os.utime(leg / f"nova-2026080{i}.db", (1_700_000_000 + i, 1_700_000_000 + i))
    rep = mod.sync_once(src, [leg, tmp_path / "missing"], keep=7)
    assert rep["copied"] and rep["copied"][0].endswith("nova-20260908.db")
    assert any("not mounted" in s for s in rep["skipped"])
    assert len(list(leg.glob("nova-*.db"))) == 7
    # a corrupt "snapshot" is never propagated
    (src / "nova-20260909.db").write_bytes(b"garbage")
    os.utime(src / "nova-20260909.db", (1_900_000_000, 1_900_000_000))
    rep2 = mod.sync_once(src, [leg], keep=7)
    assert rep2["failed"] and not (leg / "nova-20260909.db").exists()
