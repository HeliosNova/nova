"""The Ollama container refuses to run CPU-only on a machine that has a GPU
(2026-10-08).

Ollama 0.32.15 gives GPU discovery a 30-second deadline. One minute after a
reboot the CUDA runner libraries are still cold on the external disk, the
deadline passed for cuda_v12 and cuda_v13, and Ollama logged
``inference compute id=cpu library=cpu`` — then kept that decision for the
container's whole life. Every model call timed out, no monitor completed, and
the watchdog restarted nova-app every 150 minutes for 16 hours (six times)
without ever touching the container that was actually wrong. The healthcheck
(``ollama list``) passes on CPU, so nothing noticed.

scripts/ollama_entrypoint.sh wraps ``ollama serve``: it pre-reads the runner
libraries so discovery sees a warm page cache, then watches the serve log for
the discovery verdict. CPU-only while nvidia-smi sees a card means exit 1, so
Docker's restart policy tries again — with the libraries now cached.
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
SCRIPT = ROOT / "scripts" / "ollama_entrypoint.sh"

BASH = shutil.which("bash")
pytestmark = [
    pytest.mark.skipif(not SCRIPT.exists(), reason="scripts/ is not shipped in the image"),
    pytest.mark.skipif(BASH is None or sys.platform == "win32", reason="needs bash"),
]

CPU_LINE = ('time=2026-10-08T14:26:17.586Z level=INFO source=types.go:50 msg="inference compute" '
            'id=cpu library=cpu compute="" name=cpu description=cpu libdirs=ollama total="32.0 GiB"')
GPU_LINE = ('time=2026-10-09T06:38:43.421Z level=INFO source=types.go:32 msg="inference compute" '
            'id=0 filter_id=0 library=CUDA compute=8.6 name=CUDA0 description="NVIDIA GeForce RTX 3090" '
            'libdirs=ollama,cuda_v13 driver=13.4')


def _fakes(tmp_path: Path, *, serve_line: str, gpu_present: bool) -> dict[str, str]:
    """A PATH with a fake `ollama` (prints one discovery line, then idles) and a
    fake `nvidia-smi` (exit 0 with a card listed, or exit 9 like a box with no
    driver)."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    ollama = bin_dir / "ollama"
    ollama.write_text(
        "#!/usr/bin/env bash\n"
        "echo 'time=x level=INFO msg=\"Listening on [::]:11434 (version 0.32.15)\"' >&2\n"
        "echo 'time=x level=INFO source=runner.go:60 msg=\"discovering available GPUs...\"' >&2\n"
        "sleep 1\n"
        f"echo '{serve_line}' >&2\n"
        "exec sleep 60\n",
        encoding="utf-8",
    )
    ollama.chmod(0o755)
    smi = bin_dir / "nvidia-smi"
    if gpu_present:
        smi.write_text("#!/usr/bin/env bash\necho 'GPU 0: NVIDIA GeForce RTX 3090 (UUID: x)'\nexit 0\n")
    else:
        smi.write_text("#!/usr/bin/env bash\necho 'NVIDIA-SMI has failed' >&2\nexit 9\n")
    smi.chmod(0o755)
    libs = tmp_path / "libs" / "cuda_v13"
    libs.mkdir(parents=True)
    for i in range(3):
        (libs / f"lib{i}.so").write_bytes(b"\0" * 4096)
    env = dict(os.environ)
    env["PATH"] = f"{bin_dir}{os.pathsep}{env.get('PATH', '')}"
    env["OLLAMA_PREREAD_DIRS"] = str(libs)
    env["OLLAMA_GPU_CHECK_SEC"] = "20"
    env["OLLAMA_SERVE_LOG"] = str(tmp_path / "serve.log")
    return env


# terminated by the test: raw SIGTERM, or 128+15 once the script's TERM trap
# has forwarded the signal to serve and exited through `wait`
TERMINATED = {-15, 143}


def _run(env: dict[str, str], wait: float) -> tuple[subprocess.Popen, str]:
    proc = subprocess.Popen(
        [BASH, str(SCRIPT)], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    deadline = time.monotonic() + wait
    while time.monotonic() < deadline and proc.poll() is None:
        time.sleep(0.2)
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
    out = proc.stdout.read() if proc.stdout else ""
    return proc, out


def test_the_script_is_lf_only():
    src = SCRIPT.read_text(encoding="utf-8")
    assert "\r" not in src, "CRLF in a shell script kills the container entrypoint"


def test_cpu_only_with_a_gpu_present_exits_nonzero(tmp_path):
    env = _fakes(tmp_path, serve_line=CPU_LINE, gpu_present=True)
    proc, out = _run(env, wait=15)
    assert proc.returncode == 1, out
    assert "CPU" in out and "GPU" in out, out


def test_a_gpu_verdict_keeps_serving(tmp_path):
    env = _fakes(tmp_path, serve_line=GPU_LINE, gpu_present=True)
    proc, out = _run(env, wait=6)
    # still running when we gave up waiting → terminated by the test, not by the check
    assert proc.returncode in TERMINATED, out
    assert "CUDA" in out, out


def test_cpu_only_without_a_gpu_is_a_legitimate_box(tmp_path):
    env = _fakes(tmp_path, serve_line=CPU_LINE, gpu_present=False)
    proc, out = _run(env, wait=6)
    assert proc.returncode in TERMINATED, out


def test_the_runner_libraries_are_pre_read_before_serve(tmp_path):
    env = _fakes(tmp_path, serve_line=GPU_LINE, gpu_present=True)
    _proc, out = _run(env, wait=6)
    assert "pre-read 3 file" in out, out
    serve_log = Path(env["OLLAMA_SERVE_LOG"]).read_text(encoding="utf-8")
    assert out.index("pre-read 3 file") < out.index("discovering available GPUs"), out
    assert "inference compute" in serve_log
