# --- SQLite 3.53.4 builder (audit 2026-08-24) ---
# Debian trixie ships libsqlite3 3.46.1, which carries the 16-year-latent
# WAL-reset corruption bug (fixed upstream in 3.51.3; found via Tailscale's
# 2025 outages). Trigger: two connections on the same WAL DB writing /
# checkpointing at the same instant -> pages silently never migrate from WAL
# to the main file -> permanent corruption. Nova's entire brain is ONE WAL
# SQLite file and docker-exec write scripts DO run alongside the app, so the
# risk window is real even though in-process writes serialize on SafeDB's
# write lock. Debian's +deb13u1 backports only the FTS5 CVE, NOT this fix
# (verified via the package changelog), so we compile a fixed SQLite and let
# the stdlib pick it up via /usr/local/lib (ld.so precedence + stable C ABI).
FROM python:3.12-slim AS sqlite-builder
# libc6-dev is EXPLICIT: it is only a Recommends of gcc, so
# --no-install-recommends leaves gcc unable to link a test program and
# sqlite's configure reports the misleading "No working C compiler found".
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libc6-dev make curl ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && echo 'int main(void){return 0;}' > /tmp/cc-probe.c \
    && gcc /tmp/cc-probe.c -o /tmp/cc-probe && /tmp/cc-probe
# Note: no custom CFLAGS — 3.53's autosetup configure rejects them and already
# enables math functions, DBSTAT and JSON by default (verified in a probe build;
# python picked up 3.53.4 with FTS5 working).
RUN curl -fsSL https://sqlite.org/2026/sqlite-autoconf-3530400.tar.gz -o /tmp/sqlite.tar.gz \
    && tar -xzf /tmp/sqlite.tar.gz -C /tmp \
    && cd /tmp/sqlite-autoconf-3530400 \
    && ./configure --prefix=/usr/local --enable-fts5 --enable-fts4 --enable-fts3 --enable-rtree \
    && make -j"$(nproc)" && make install

FROM python:3.12-slim

WORKDIR /app

# Fixed SQLite (see sqlite-builder stage): /usr/local/lib outranks /usr/lib in
# ld.so search order, so python's stdlib _sqlite3 binds 3.53.4 at import.
# COPY dereferences symlinks, so copy the real .so once and recreate the
# SONAME links (_sqlite3 links libsqlite3.so.0).
COPY --from=sqlite-builder /usr/local/lib/libsqlite3.so.3.53.4 /usr/local/lib/
RUN ln -sf libsqlite3.so.3.53.4 /usr/local/lib/libsqlite3.so.0 \
    && ln -sf libsqlite3.so.3.53.4 /usr/local/lib/libsqlite3.so \
    && ldconfig

# System deps + Playwright Chromium dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libnss3 libnspr4 libatk1.0-0 libatk-bridge2.0-0 libcups2 \
    libdrm2 libdbus-1-3 libxkbcommon0 libatspi2.0-0 \
    libxcomposite1 libxdamage1 libxfixes3 libxrandr2 libgbm1 \
    libpango-1.0-0 libcairo2 libasound2 libwayland-client0 \
    fonts-liberation fonts-noto-color-emoji \
    # Voice transcription (Whisper needs ffmpeg for audio decoding)
    ffmpeg \
    # Desktop automation deps (PyAutoGUI — optional, used with ENABLE_DESKTOP_AUTOMATION)
    xvfb scrot x11-utils python3-tk python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Playwright Chromium to a shared location accessible by all users
ENV PLAYWRIGHT_BROWSERS_PATH=/opt/playwright
RUN playwright install chromium

# Application code.
# ORDER MATTERS for build-cache reuse: COPY the rarely-changing trees FIRST and
# the frequently-edited app/ LAST. A COPY layer invalidates every layer after
# it, so with app/ first, every app edit re-ran the tests/ and evals/ COPYs too
# — and on this host the build context lives on a slow 9p mount where each COPY
# costs 60-85s. App-last keeps the big tests/evals layers cached across app edits.
COPY tests/ tests/
COPY evals/ evals/
COPY pytest.ini .
# scripts/ package marker only. The fine-tuning / GRPO / RLVR-trainer stack was
# archived 2026-06-12 (see archive/training/ + CLAUDE.md) — the in-context memory
# loop is the product, weight training was 0-successful-deploy experimental. The
# runtime image no longer ships any trainer.
COPY scripts/__init__.py scripts/__init__.py
# app/ last: the most frequently edited tree, so its cache miss never cascades
# into re-copying tests/ or evals/.
# CACHEBUST: Docker Desktop on Windows over the 9p F: mount does not reliably
# propagate file-content changes to BuildKit's COPY cache key, so edited app/
# files were silently served from a stale cached layer (the image sat unchanged
# for hours across "successful" builds). Passing --build-arg CACHEBUST=<epoch>
# each build forces this layer to re-copy. Cheap: only the small app/ layer.
ARG CACHEBUST=0
RUN echo "cachebust=${CACHEBUST}" > /tmp/.cachebust
COPY app/ app/

# Data directory + non-root user
RUN mkdir -p /data /data/screenshots /data/mcp && \
    useradd -m -u 1000 nova && \
    chown -R nova:nova /app /data /home/nova

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

USER nova
EXPOSE 8000

ENTRYPOINT ["/entrypoint.sh"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
