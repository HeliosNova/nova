#!/bin/sh
# Fix /data ownership for mounted volumes, then run as nova user.
# Only chown top-level dirs (not recursive) to avoid slow startup on large volumes.
chown nova:nova /data /data/screenshots /data/mcp 2>/dev/null || true

exec "$@"
