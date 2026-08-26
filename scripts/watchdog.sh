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
# Post-restart grace for the STALENESS rule only (2026-08-12). last_check_at
# updates only when a monitor COMPLETES; after a restart the heartbeat re-runs
# the stalest monitor first, and if that is a long digest (~30-40 min), the
# staleness clock cannot reset within COOLDOWN — so the watchdog killed the
# recovering app every 10 min forever (live death-spiral: stale 90→100→110m,
# first real firing of this watchdog). The staleness rule now waits GRACE_SEC
# after any restart (and after watchdog startup — it has no history then).
# The health detector is UNAFFECTED: a truly frozen loop still gets caught.
GRACE_SEC=${WATCHDOG_GRACE_SEC:-2700}
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
    LAST_RESTART=$(date +%s)
    sleep "$COOLDOWN"
}

# Secondary targets (2026-08-25): a hung nova-ollama or nova-embed used to
# stall monitors until the heartbeat went stale — then the watchdog restarted
# the WRONG container (nova-app) forever. These have compose healthchecks;
# sustained `unhealthy` gets the same limit+cooldown treatment as the primary.
# Space-separated, override with WATCHDOG_AUX_TARGETS.
AUX_TARGETS=${WATCHDOG_AUX_TARGETS:-nova-ollama nova-embed}

aux_health() {
    docker_api 10 "/containers/$1/json" \
        | jq -r '.State.Health.Status // .State.Status // "unknown"' 2>/dev/null \
        || echo unknown
}

restart_aux() {
    # $1 = container, $2 = reason
    if [ "$DRY_RUN" = "1" ]; then
        log "DRY RUN — would restart $1: $2"
        alert "DRY RUN — would restart $1: $2"
        return
    fi
    log "restarting $1: $2"
    alert "$1 is unhealthy ($2) — restarting it."
    docker_api 90 "/containers/$1/restart?t=30" -X POST >/dev/null 2>&1
}

check_aux_targets() {
    now=$(date +%s)
    for c in $AUX_TARGETS; do
        ast=$(aux_health "$c")
        if [ "$ast" = "unhealthy" ]; then
            n=$(eval "echo \${AUXFAIL_$(echo "$c" | tr -c 'A-Za-z0-9' '_'):-0}")
            n=$((n + 1))
            eval "AUXFAIL_$(echo "$c" | tr -c 'A-Za-z0-9' '_')=$n"
            log "aux $c health=unhealthy (${n}/${UNHEALTHY_LIMIT})"
            last=$(eval "echo \${AUXRESTART_$(echo "$c" | tr -c 'A-Za-z0-9' '_'):-0}")
            if [ "$n" -ge "$UNHEALTHY_LIMIT" ] && [ $((now - last)) -ge "$COOLDOWN" ]; then
                restart_aux "$c" "health check failing ${n}x"
                eval "AUXRESTART_$(echo "$c" | tr -c 'A-Za-z0-9' '_')=$now"
                eval "AUXFAIL_$(echo "$c" | tr -c 'A-Za-z0-9' '_')=0"
            fi
        else
            eval "AUXFAIL_$(echo "$c" | tr -c 'A-Za-z0-9' '_')=0"
        fi
    done
}

fails=0
unknowns=0
# Startup counts as a grace start: the watchdog has no restart history yet and
# the app may be mid-recovery from whatever preceded this watchdog boot.
LAST_RESTART=$(date +%s)
log "started (target=${TARGET} aux=[${AUX_TARGETS}] interval=${INTERVAL}s unhealthy_limit=${UNHEALTHY_LIMIT} hb_stale=${HB_STALE_MIN}m grace=${GRACE_SEC}s dry_run=${DRY_RUN})"
while true; do
    check_aux_targets
    st=$(health)
    # "unknown" means the watchdog itself is blind (socket-proxy dead, target
    # renamed/gone) — both detectors silently skip in that state, so a blind
    # watchdog looked identical to a healthy system (audit 2026-08-19). Log
    # every miss and alert once after 10 consecutive (~10 min at default 60s).
    if [ "$st" = "unknown" ]; then
        unknowns=$((unknowns + 1))
        log "health=unknown (${unknowns}x) — cannot see ${TARGET} via docker API"
        if [ "$unknowns" -eq 10 ]; then
            alert "watchdog is BLIND: docker API/${TARGET} unreachable for ${unknowns} checks — nova-app is UNPROTECTED (socket-proxy down or container renamed)"
        fi
    else
        [ "$unknowns" -ge 10 ] && alert "watchdog can see ${TARGET} again (state=${st})"
        unknowns=0
    fi
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
                   since_restart=$(( $(date +%s) - LAST_RESTART ))
                   if [ "$since_restart" -lt "$GRACE_SEC" ]; then
                       # In grace: the heartbeat is (re)working through a long
                       # head-of-queue task; killing it now would re-queue the
                       # same task with the same stale clock — the death spiral.
                       log "heartbeat stale ${stale}m but within post-restart grace (${since_restart}/${GRACE_SEC}s) — deferring"
                   else
                       restart_target "heartbeat stale ${stale}m — loop alive but monitors dead"
                       continue
                   fi
               fi ;;
        esac
    fi
    sleep "$INTERVAL"
done
