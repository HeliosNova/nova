#!/bin/sh
# Fix /data ownership for mounted volumes, then run as nova user.
# Only chown top-level dirs (not recursive) to avoid slow startup on large volumes.
chown nova:nova /data /data/screenshots /data/mcp 2>/dev/null || true

# Start virtual display for desktop automation (headless mode)
if [ "$ENABLE_DESKTOP_AUTOMATION" = "true" ]; then
    echo "[entrypoint] Starting Xvfb virtual display for desktop automation..."
    mkdir -p /tmp/.X11-unix
    chmod 1777 /tmp/.X11-unix
    export XAUTHORITY=/tmp/.Xauthority
    touch "$XAUTHORITY"
    chown nova:nova "$XAUTHORITY"
    Xvfb :99 -screen 0 1920x1080x24 -nolisten tcp -ac &
    sleep 1
    echo "[entrypoint] Virtual display ready (DISPLAY=:99, 1920x1080)"
fi

exec "$@"
