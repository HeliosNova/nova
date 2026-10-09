#!/usr/bin/env bash
# Ollama container entrypoint: refuse to run CPU-only on a box that has a GPU.
#
# Why (2026-10-08): Ollama 0.32.15 gives GPU discovery a 30-second deadline.
# One minute after a reboot the CUDA runner libraries (/usr/lib/ollama/cuda_v*,
# ~2 GB) are still cold on the external disk; discovery timed out for cuda_v12
# AND cuda_v13, Ollama logged `inference compute id=cpu library=cpu`, and kept
# that for the container's whole life. Every model call timed out, no monitor
# completed, and the watchdog restarted nova-app every 150 minutes for 16 hours
# without touching this container. `ollama list` (the healthcheck) passes on
# CPU, so nothing noticed. There is no env knob for the discovery deadline.
#
# What this does:
#   1. pre-reads the runner libraries so discovery sees a warm page cache
#      (OLLAMA_PREREAD_DIRS, default /usr/lib/ollama);
#   2. starts `ollama serve` and tees its log to OLLAMA_SERVE_LOG;
#   3. waits up to OLLAMA_GPU_CHECK_SEC for the discovery verdict. CPU-only
#      while `nvidia-smi -L` lists a card → log it, stop serve, exit 1, and
#      Docker's restart policy tries again with the libraries now cached.
#      A GPU verdict, a box with no GPU, or no verdict in time → keep serving.
#
# Tested by tests/test_ollama_entrypoint_gpu_check.py with a fake ollama and
# nvidia-smi on PATH. LF line endings only (CRLF kills a container entrypoint).

set -u

# Only what discovery dlopens: the CUDA runners (cuda_v13 first — the dir the
# current driver selects — then cuda_v12) and the top-level ggml/llama libs.
# Measured 2026-10-09: the whole tree is 4.3 GB (2.2 GB of it an MLX backend
# that is never loaded) and took 105 s cold; these ~2 GB take about half that.
# Each dir is read one level deep — the runner dirs are flat.
PREREAD_DIRS=${OLLAMA_PREREAD_DIRS:-/usr/lib/ollama/cuda_v13 /usr/lib/ollama/cuda_v12 /usr/lib/ollama}
CHECK_SEC=${OLLAMA_GPU_CHECK_SEC:-240}
SERVE_LOG=${OLLAMA_SERVE_LOG:-/tmp/ollama-serve.log}
GPU_LIBS='library=(CUDA|ROCm|Vulkan|Metal|HIP)'

log() { echo "[ollama-entrypoint] $(date -u '+%Y-%m-%d %H:%M:%S') $*"; }

gpu_present() {
    command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1
}

preread() {
    local start count
    start=$(date +%s)
    count=0
    for d in $PREREAD_DIRS; do
        [ -d "$d" ] || continue
        # cat every regular file once; the bytes land in the page cache and
        # llama-server's dlopen of the CUDA runner no longer waits on the disk.
        while IFS= read -r f; do
            cat "$f" > /dev/null 2>&1
            count=$((count + 1))
        done < <(find "$d" -maxdepth 1 -type f 2>/dev/null)
    done
    log "pre-read ${count} file(s) from ${PREREAD_DIRS} in $(( $(date +%s) - start ))s"
}

preread

: > "$SERVE_LOG"
ollama serve > >(tee -a "$SERVE_LOG") 2>&1 &
OLLAMA_PID=$!
trap 'kill -TERM "$OLLAMA_PID" 2>/dev/null' TERM INT

verdict=""
deadline=$(( $(date +%s) + CHECK_SEC ))
while [ "$(date +%s)" -lt "$deadline" ]; do
    if grep -q 'inference compute' "$SERVE_LOG" 2>/dev/null; then
        # discovery logs one line per device; give the batch a moment to land
        sleep 2
        if grep 'inference compute' "$SERVE_LOG" | grep -Eq "$GPU_LIBS"; then
            verdict=gpu
        else
            verdict=cpu
        fi
        break
    fi
    if ! kill -0 "$OLLAMA_PID" 2>/dev/null; then
        break
    fi
    sleep 1
done

case "$verdict" in
    gpu)
        log "GPU discovery OK: $(grep 'inference compute' "$SERVE_LOG" | grep -Eo 'library=[A-Za-z]+' | sort -u | tr '\n' ' ')"
        ;;
    cpu)
        if gpu_present; then
            timed_out=$(grep -c 'GPU discovery watchdog timed out' "$SERVE_LOG" 2>/dev/null || echo 0)
            log "Ollama chose CPU (library=cpu) but nvidia-smi lists a GPU (discovery timeouts: ${timed_out}) — exiting so Docker restarts this container with the runner libraries cached"
            kill -TERM "$OLLAMA_PID" 2>/dev/null
            wait "$OLLAMA_PID" 2>/dev/null
            exit 1
        fi
        log "CPU-only verdict and no GPU visible to nvidia-smi — serving on CPU"
        ;;
    *)
        log "no discovery verdict within ${CHECK_SEC}s — leaving serve alone (check 'ollama ps' PROCESSOR)"
        ;;
esac

wait "$OLLAMA_PID"
exit $?
