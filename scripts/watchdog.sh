#!/bin/sh
# Nova watchdog sidecar — restarts nova-app on sustained-unhealthy or a stale
# heartbeat, and alerts via the Discord REST API. Runs OUT-OF-PROCESS so it
# survives anything that kills the app (approved by owner 2026-07-06 after the
# 54h silent write-lock freeze: Docker marked the container unhealthy and
# nothing acted on it).
#
# Two independent detectors:
#   1. container health: `unhealthy` for UNHEALTHY_LIMIT consecutive checks
#      (the /api/health probe runs on the event loop — a frozen loop fails it);
#   2. heartbeat staleness: health OK but no monitor checked in HB_STALE_MIN
#      minutes (catches a dead heartbeat task behind a live HTTP server; the
#      busiest gap in a healthy system is ~1h — Curiosity Research is hourly).
#
# Never prints or sends DISCORD_TOKEN. WATCHDOG_DRY_RUN=1 logs+alerts without
# restarting (used to validate detection end-to-end).

SOCK=/var/run/docker.sock
# Blast-radius reduction (audit 2026-07-08): when WATCHDOG_DOCKER_API is set
# (docker-socket-proxy, e.g. http://socket-proxy:2375) the raw socket is not
# mounted at all — the proxy allows ONLY container list/inspect + restart, so
# a compromise of this container can no longer become root on the host.
DOCKER_API=${WATCHDOG_DOCKER_API:-}
DB=/data/nova.db
TARGET=${WATCHDOG_TARGET:-nova-app}
INTERVAL=${WATCHDOG_INTERVAL:-60}
UNHEALTHY_LIMIT=${WATCHDOG_UNHEALTHY_LIMIT:-3}
HB_STALE_MIN=${WATCHDOG_HB_STALE_MIN:-90}
COOLDOWN=${WATCHDOG_COOLDOWN:-600}
DRY_RUN=${WATCHDOG_DRY_RUN:-0}

log() { echo "[watchdog] $(date -u '+%Y-%m-%d %H:%M:%S') $*"; }

alert() {
    if [ -z "$DISCORD_TOKEN" ] || [ -z "$DISCORD_CHANNEL_ID" ]; then
        log "alert (no discord creds): $1"
        return
    fi
    curl -sS -m 15 -X POST \
        "https://discord.com/api/v10/channels/${DISCORD_CHANNEL_ID}/messages" \
        -H "Authorization: Bot ${DISCORD_TOKEN}" \
        -H "Content-Type: application/json" \
        -d "{\"content\": \"🛡️ **Watchdog**: $1\"}" >/dev/null 2>&1 \
        || log "discord alert failed: $1"
}

docker_api() {
    # $1 = curl max-time, $2 = API path, rest = extra curl args
    _t="$1"; _p="$2"; shift 2
    if [ -n "$DOCKER_API" ]; then
        curl -s -m "$_t" "$@" "${DOCKER_API}${_p}"
    else
        curl -s -m "$_t" --unix-socket "$SOCK" "$@" "http://localhost${_p}"
    fi
}

health() {
    docker_api 10 "/containers/${TARGET}/json" \
        | jq -r '.State.Health.Status // .State.Status // "unknown"' 2>/dev/null \
        || echo unknown
}

hb_stale_minutes() {
    # Minutes since the heartbeat last checked ANY monitor (WAL read, concurrent-
    # safe). Non-numeric/empty output is treated as "can't tell" by the caller.
    sqlite3 -cmd '.timeout 3000' "$DB" \
        "SELECT CAST((julianday('now') - julianday(MAX(last_check_at)))*1440 AS INTEGER) \
         FROM monitors WHERE enabled=1 AND last_check_at IS NOT NULL;" 2>/dev/null
}

restart_target() {
    if [ "$DRY_RUN" = "1" ]; then
        log "DRY RUN — would restart ${TARGET}: $1"
        alert "DRY RUN — would restart nova-app: $1"
        sleep "$COOLDOWN"
        return
    fi
    log "restarting ${TARGET}: $1"
    alert "nova-app is frozen ($1) — restarting it. Monitors resume automatically after startup."
    docker_api 90 "/containers/${TARGET}/restart?t=30" -X POST >/dev/null 2>&1
    sleep "$COOLDOWN"
}

fails=0
log "started (target=${TARGET} interval=${INTERVAL}s unhealthy_limit=${UNHEALTHY_LIMIT} hb_stale=${HB_STALE_MIN}m dry_run=${DRY_RUN})"
while true; do
    st=$(health)
    if [ "$st" = "unhealthy" ]; then
        fails=$((fails + 1))
        log "health=unhealthy (${fails}/${UNHEALTHY_LIMIT})"
        if [ "$fails" -ge "$UNHEALTHY_LIMIT" ]; then
            restart_target "health check failing ${fails}x"
            fails=0
            continue
        fi
    else
        [ "$fails" -gt 0 ] && log "health recovered (${st})"
        fails=0
    fi
    if [ "$st" = "healthy" ]; then
        stale=$(hb_stale_minutes)
        case "$stale" in
            ''|*[!0-9]*) : ;;   # unreadable/negative — skip, health rule still guards
            *) if [ "$stale" -ge "$HB_STALE_MIN" ]; then
                   restart_target "heartbeat stale ${stale}m — loop alive but monitors dead"
                   continue
               fi ;;
        esac
    fi
    sleep "$INTERVAL"
done
