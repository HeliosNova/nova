FROM python:3.12-slim

WORKDIR /app

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
# Phase-0: include the bootstrap verification script so it can be run via
# `docker exec nova-app python -m scripts.verify_phase_0`. We copy a single
# file rather than the whole scripts/ directory because most other scripts
# are heavyweight training pipelines that don't belong in the runtime image.
COPY scripts/__init__.py scripts/__init__.py
COPY scripts/verify_phase_0.py scripts/verify_phase_0.py
# GRPO trainer — small, no torch import at module load. Lets us run
# `docker exec nova-app python -m scripts.grpo_train --dry-run` without
# staging the file via /data each time. The actual training step still
# requires the full finetune venv outside the container.
COPY scripts/grpo_train.py scripts/grpo_train.py
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
