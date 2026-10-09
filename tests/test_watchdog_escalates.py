"""A staleness restart that did not revive the loop escalates to the model
containers (2026-10-08).

Six times in sixteen hours the watchdog logged "heartbeat stale 150m — loop
alive but monitors dead" and restarted nova-app, and six times the heartbeat
stayed dead, because the thing that was wrong was nova-ollama (running the 9B
on the CPU after a timed-out GPU discovery). Restarting the same container on
the same clock forever is not a watchdog; it is a metronome.

Rule: the staleness rule counts consecutive firings with no fresh heartbeat in
between. On the WATCHDOG_STALE_ESCALATE-th (default 2) it restarts the aux
targets (nova-ollama, nova-embed) first, then nova-app, and says so in the
alert. A fresh heartbeat reading resets the streak.

The test runs the real script under bash with a fake curl/jq/sqlite3 on PATH:
nova-app always "healthy", the heartbeat always 200 minutes stale, every
restart POST recorded to a file.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

import app as _app_pkg

ROOT = Path(_app_pkg.__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "watchdog.sh"
BASH = shutil.which("bash")

pytestmark = [
    pytest.mark.skipif(not SCRIPT.exists(), reason="scripts/ is not shipped in the image"),
    pytest.mark.skipif(BASH is None or sys.platform == "win32", reason="needs bash"),
]


def _fakes(tmp_path: Path) -> tuple[dict[str, str], Path]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    calls = tmp_path / "calls.log"
    (bin_dir / "curl").write_text(
        "#!/usr/bin/env bash\n"
        "for a in \"$@\"; do case \"$a\" in\n"
        f"  *restart*) echo \"$a\" >> '{calls}'; exit 0;;\n"
        "esac; done\n"
        "echo '{\"State\":{\"Health\":{\"Status\":\"healthy\"}}}'\n",
        encoding="utf-8",
    )
    (bin_dir / "jq").write_text("#!/usr/bin/env bash\necho healthy\n", encoding="utf-8")
    # stale query → 200 minutes; quiet-window query → 0
    (bin_dir / "sqlite3").write_text(
        "#!/usr/bin/env bash\n"
        "case \"$*\" in *quiet_until*) echo 0;; *) echo 200;; esac\n",
        encoding="utf-8",
    )
    for f in bin_dir.iterdir():
        f.chmod(0o755)
    env = {
        "PATH": f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}",
        "WATCHDOG_DOCKER_API": "http://fake",
        "WATCHDOG_INTERVAL": "0",
        "WATCHDOG_COOLDOWN": "0",
        "WATCHDOG_GRACE_SEC": "0",
        "WATCHDOG_HB_STALE_MIN": "150",
    }
    return env, calls


def _run_until(env: dict[str, str], calls: Path, want: int, wait: float = 15.0) -> tuple[list[str], str]:
    proc = subprocess.Popen([BASH, str(SCRIPT)], env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True)
    deadline = time.monotonic() + wait
    lines: list[str] = []
    while time.monotonic() < deadline:
        if calls.exists():
            lines = [ln for ln in calls.read_text(encoding="utf-8").splitlines() if ln]
            if len(lines) >= want:
                break
        if proc.poll() is not None:
            break
        time.sleep(0.1)
    proc.kill()
    out = proc.stdout.read() if proc.stdout else ""
    return lines, out


def test_second_consecutive_staleness_restart_restarts_the_model_containers(tmp_path):
    env, calls = _fakes(tmp_path)
    lines, out = _run_until(env, calls, want=4)
    assert len(lines) >= 4, (lines, out)
    assert "/containers/nova-app/restart" in lines[0], lines
    # escalation: aux targets go first, then nova-app again
    assert "/containers/nova-ollama/restart" in lines[1], lines
    assert "/containers/nova-embed/restart" in lines[2], lines
    assert "/containers/nova-app/restart" in lines[3], lines
    assert "did not revive" in out, out


def test_the_escalation_threshold_is_configurable(tmp_path):
    env, calls = _fakes(tmp_path)
    env["WATCHDOG_STALE_ESCALATE"] = "3"
    lines, out = _run_until(env, calls, want=5)
    assert [ln.split("/containers/")[1].split("/")[0] for ln in lines[:5]] == [
        "nova-app", "nova-app", "nova-ollama", "nova-embed", "nova-app",
    ], (lines, out)


def test_a_fresh_heartbeat_resets_the_streak(tmp_path):
    env, calls = _fakes(tmp_path)
    # stale on the 1st and 3rd read, fresh on the 2nd → never two in a row
    counter = tmp_path / "n"
    (tmp_path / "bin" / "sqlite3").write_text(
        "#!/usr/bin/env bash\n"
        "case \"$*\" in *quiet_until*) echo 0; exit;; esac\n"
        f"n=$(cat '{counter}' 2>/dev/null || echo 0); n=$((n+1)); echo $n > '{counter}'\n"
        "if [ $((n % 2)) -eq 0 ]; then echo 1; else echo 200; fi\n",
        encoding="utf-8",
    )
    lines, out = _run_until(env, calls, want=3, wait=6)
    assert all("/containers/nova-app/restart" in ln for ln in lines[:3]), (lines, out)
    assert "nova-ollama" not in "".join(lines), lines
