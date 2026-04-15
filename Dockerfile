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

# Application code
COPY app/ app/
COPY tests/ tests/
COPY pytest.ini .

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
