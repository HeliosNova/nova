#!/usr/bin/env bash
# check_container_freshness.sh — diff host app/ vs container /app/app/.
# Exits 0 if container is fresh, 1 if any tracked file differs in size.
#
# Run before assuming code edits are live:
#   bash scripts/check_container_freshness.sh
#
# Or add a pre-deploy hook:
#   bash scripts/check_container_freshness.sh || docker compose build nova && docker compose up -d nova
set -u
# Stop Git Bash from rewriting absolute /app paths into Windows paths
export MSYS_NO_PATHCONV=1

CONTAINER="${1:-nova-app}"
HOST_DIR="${2:-app}"
CONTAINER_DIR="${3:-/app/app}"

if ! docker ps --filter "name=^${CONTAINER}$" --format '{{.Names}}' | grep -q "${CONTAINER}"; then
    echo "[freshness] container '${CONTAINER}' is not running — nothing to check"
    exit 0
fi

stale=0
checked=0
while IFS= read -r f; do
    rel="${f#${HOST_DIR}/}"
    host_size=$(wc -c < "$f")
    container_size=$(docker exec "$CONTAINER" stat -c %s "${CONTAINER_DIR}/${rel}" 2>/dev/null)
    checked=$((checked + 1))
    if [ -z "$container_size" ]; then
        echo "[freshness] MISSING in container: ${rel}"
        stale=$((stale + 1))
    elif [ "$host_size" != "$container_size" ]; then
        echo "[freshness] STALE: ${rel}  host=${host_size}  container=${container_size}"
        stale=$((stale + 1))
    fi
done < <(find "${HOST_DIR}" -type f -name '*.py' 2>/dev/null)

echo "[freshness] checked=${checked} stale=${stale}"
if [ "$stale" -gt 0 ]; then
    echo "[freshness] container is stale — run: docker compose build nova && docker compose up -d nova"
    exit 1
fi
exit 0
