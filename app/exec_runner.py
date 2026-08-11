"""Network-isolated code-execution sidecar (audit 2026-07-08).

Runs as its own container with `network_mode: none` — model-written code
physically cannot exfiltrate data or reach in-network services (Ollama's
unauthenticated API being the important one), no matter what slips past the
AST screen and import-hook guard. 2026 consensus: static screens are a
pre-filter, not a boundary; the boundary must be structural.

Protocol (shared named volume, default /exec_queue):
    jobs/{id}.json     written by nova-app: {"code": str, "timeout": int}
    running/{id}.json  claimed atomically via rename (single runner)
    results/{id}.json  {"stdout": str, "stderr": str, "returncode": int,
                        "timed_out": bool}

The job payload deliberately does NOT carry executable runner code — this
sidecar builds its own guarded runner from the same module nova-app uses,
so a compromised app container can't smuggle a different runner in.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
import shutil
from pathlib import Path

QUEUE = Path("/exec_queue")
JOBS = QUEUE / "jobs"
RUNNING = QUEUE / "running"
RESULTS = QUEUE / "results"
POLL_S = 0.25
STALE_S = 3600


def _log(msg: str) -> None:
    print(f"[exec-runner] {msg}", flush=True)


def process_job(job_path: Path) -> None:
    from app.tools.code_exec import (
        _build_guarded_runner, _posix_rlimits, IS_WINDOWS, get_safe_env,
    )

    job_id = job_path.stem
    try:
        job = json.loads(job_path.read_text(encoding="utf-8"))
        code = str(job.get("code", ""))
        timeout = min(int(job.get("timeout", 30)), 300)
    except Exception as e:
        _write_result(job_id, "", f"invalid job payload: {e}", 1, False)
        return

    sandbox = tempfile.mkdtemp(prefix="nova_exec_")
    try:
        script = Path(sandbox) / "_script.py"
        script.write_text(code, encoding="utf-8")
        runner = Path(sandbox) / "_runner.py"
        runner.write_text(_build_guarded_runner(), encoding="utf-8")
        try:
            proc = subprocess.run(
                [sys.executable, "-I", str(runner), str(script)],
                capture_output=True, text=True, timeout=timeout,
                cwd=sandbox, env=get_safe_env(),
                preexec_fn=None if IS_WINDOWS else _posix_rlimits,
            )
            _write_result(job_id, proc.stdout, proc.stderr, proc.returncode, False)
        except subprocess.TimeoutExpired:
            _write_result(job_id, "", f"timed out after {timeout}s", 124, True)
    except Exception as e:
        _write_result(job_id, "", f"runner error: {e}", 1, False)
    finally:
        shutil.rmtree(sandbox, ignore_errors=True)


def _write_result(job_id: str, stdout: str, stderr: str, rc: int, timed_out: bool) -> None:
    tmp = RESULTS / f"{job_id}.tmp"
    final = RESULTS / f"{job_id}.json"
    tmp.write_text(json.dumps({
        "stdout": stdout[-200_000:], "stderr": stderr[-50_000:],
        "returncode": rc, "timed_out": timed_out,
    }), encoding="utf-8")
    tmp.replace(final)  # atomic — the app never reads a half-written result


def _gc() -> None:
    cutoff = time.time() - STALE_S
    for d in (JOBS, RUNNING, RESULTS):
        for f in d.glob("*"):
            try:
                if f.stat().st_mtime < cutoff:
                    f.unlink()
            except OSError:
                pass


def main() -> None:
    for d in (JOBS, RUNNING, RESULTS):
        d.mkdir(parents=True, exist_ok=True)
    _log("started — network-isolated code execution sidecar")
    last_gc = time.time()
    while True:
        claimed = None
        for job in sorted(JOBS.glob("*.json")):
            target = RUNNING / job.name
            try:
                job.replace(target)  # atomic claim
                claimed = target
                break
            except OSError:
                continue
        if claimed is not None:
            _log(f"job {claimed.stem}")
            process_job(claimed)
            claimed.unlink(missing_ok=True)
        else:
            time.sleep(POLL_S)
        if time.time() - last_gc > 300:
            _gc()
            last_gc = time.time()


if __name__ == "__main__":
    main()
