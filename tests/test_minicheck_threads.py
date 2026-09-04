"""The entailment sidecar must size its thread pool from its cgroup quota.

torch reads os.cpu_count(), which reports the HOST's CPUs and ignores the
container's quota. Measured 2026-09-04 on flan-t5-large, 8 pairs of 8,000-char
documents: 10.05 s/pair with torch's default 6 threads against a 4-CPU quota,
8.26 s/pair with the threads matched to the quota. Raising the quota to 8 CPUs
and 8 threads was WORSE (13.09 s/pair) — this model does not scale past a few
threads, so the fix is matching, not adding.

Entailment is 64% of a digest's wall clock (19.8 of 30.8 minutes measured over
25 digests) and runs while the GPU is idle, so this is the cheapest throughput
the system has.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

import app as _app_pkg

ROOT = Path(_app_pkg.__file__).resolve().parents[1]
SERVICE = ROOT / "minicheck_service" / "app.py"

pytestmark = pytest.mark.skipif(
    not SERVICE.exists(),
    reason="reads minicheck_service/app.py, which the nova image does not ship")


def _quota_fn(tmp_path, monkeypatch, cgroup2=None, cgroup1=None, affinity=12):
    """Load _cpu_quota with the cgroup files faked, without importing fastapi."""
    src = SERVICE.read_text(encoding="utf-8")
    start = src.index("def _cpu_quota()")
    end = src.index("_THREADS = int(")
    ns: dict = {}
    exec("import os\n" + src[start:end], ns)                      # noqa: S102 - test harness

    real_open = open

    def fake_open(path, *a, **k):
        p = str(path)
        if p == "/sys/fs/cgroup/cpu.max":
            if cgroup2 is None:
                raise FileNotFoundError(p)
            return real_open(_write(tmp_path, "cpu.max", cgroup2))
        if p.startswith("/sys/fs/cgroup/cpu/cpu.cfs_"):
            if cgroup1 is None:
                raise FileNotFoundError(p)
            name = "quota" if "quota" in p else "period"
            return real_open(_write(tmp_path, name, cgroup1[0 if name == "quota" else 1]))
        return real_open(path, *a, **k)

    monkeypatch.setitem(ns, "open", fake_open)
    ns["os"].sched_getaffinity = lambda _pid: set(range(affinity))
    return ns["_cpu_quota"]


def _write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(str(text))
    return p


def test_cgroup_v2_quota_wins_over_host_cpu_count(tmp_path, monkeypatch):
    fn = _quota_fn(tmp_path, monkeypatch, cgroup2="400000 100000", affinity=12)
    assert fn() == 4, "4 CPUs of quota on a 12-CPU host must read as 4"


def test_fractional_and_large_quotas_round_sensibly(tmp_path, monkeypatch):
    assert _quota_fn(tmp_path, monkeypatch, cgroup2="50000 100000")() == 1     # 0.5 CPU
    assert _quota_fn(tmp_path, monkeypatch, cgroup2="800000 100000")() == 8


def test_unlimited_cgroup_v2_falls_through_to_affinity(tmp_path, monkeypatch):
    fn = _quota_fn(tmp_path, monkeypatch, cgroup2="max 100000", affinity=7)
    assert fn() == 7


def test_cgroup_v1_is_read_when_v2_is_absent(tmp_path, monkeypatch):
    fn = _quota_fn(tmp_path, monkeypatch, cgroup2=None, cgroup1=(200000, 100000))
    assert fn() == 2


def test_no_cgroup_at_all_falls_back_to_affinity(tmp_path, monkeypatch):
    assert _quota_fn(tmp_path, monkeypatch, affinity=3)() == 3


def test_threads_are_pinned_before_torch_is_imported():
    """OMP reads its thread count at first use, so the env must be set at import."""
    src = SERVICE.read_text(encoding="utf-8")
    assert src.index('os.environ.setdefault("OMP_NUM_THREADS"') < src.index("import torch")
    assert "MINICHECK_THREADS" in src, "an operator override must exist"
    assert "torch.set_num_threads(_THREADS)" in src


def test_health_reports_the_thread_count():
    src = SERVICE.read_text(encoding="utf-8")
    assert '"threads": _THREADS' in src
