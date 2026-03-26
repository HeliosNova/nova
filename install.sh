#!/usr/bin/env bash
# Nova — One-line installer
# curl -fsSL https://raw.githubusercontent.com/HeliosNova/nova/main/install.sh | bash
set -euo pipefail

echo ""
echo "  ╔═══════════════════════════════════════════╗"
echo "  ║  Nova — The personal AI that learns       ║"
echo "  ║  https://github.com/HeliosNova/nova       ║"
echo "  ╚═══════════════════════════════════════════╝"
echo ""

# Check Docker
if ! command -v docker &>/dev/null; then
    echo "ERROR: Docker is not installed."
    echo "  Install: https://docs.docker.com/get-docker/"
    exit 1
fi

if ! docker compose version &>/dev/null; then
    echo "ERROR: Docker Compose (v2) is not installed."
    echo "  Install: https://docs.docker.com/compose/install/"
    exit 1
fi

# Check GPU
HAS_GPU=false
if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
    VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
    if [ -n "$VRAM" ] && [ "$VRAM" -ge 20000 ] 2>/dev/null; then
        HAS_GPU=true
        echo "  GPU detected: ${VRAM}MB VRAM"
    else
        echo "  GPU detected but <20GB VRAM (${VRAM:-unknown}MB)"
    fi
else
    echo "  No NVIDIA GPU detected"
fi

# Clone
if [ -d "nova" ]; then
    echo "  Directory 'nova' already exists — pulling latest..."
    cd nova && git pull
else
    echo "  Cloning..."
    git clone https://github.com/HeliosNova/nova.git
    cd nova
fi

# Create .env if missing
if [ ! -f .env ]; then
    cp .env.example .env
    echo "  Created .env from .env.example"
fi

# Choose compose file
if [ "$HAS_GPU" = true ]; then
    COMPOSE_FILE="docker-compose.yml"
    echo ""
    echo "  Starting with LOCAL inference (Ollama + GPU)..."
    docker compose up -d --build

    echo ""
    echo "  Pulling models (this may take a few minutes)..."
    docker exec nova-ollama ollama pull qwen3.5:27b
    docker exec nova-ollama ollama pull nomic-embed-text-v2-moe
else
    echo ""
    echo "  No GPU — using cloud LLM mode."
    echo ""
    echo "  You need to set your LLM provider in .env:"
    echo "    LLM_PROVIDER=openai    (or anthropic, google)"
    echo "    OPENAI_API_KEY=sk-..."
    echo ""
    read -p "  Have you configured .env? [y/N] " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "  Edit nova/.env, then run:"
        echo "    cd nova && docker compose -f docker-compose.cloud.yml up -d --build"
        exit 0
    fi
    COMPOSE_FILE="docker-compose.cloud.yml"
    docker compose -f "$COMPOSE_FILE" up -d --build
fi

echo ""
echo "  Nova is starting up..."
echo ""
echo "  Web UI:  http://localhost:5173"
echo "  API:     http://localhost:8000/api/health"
echo ""
echo "  Logs:    docker compose logs -f nova"
echo "  Stop:    docker compose down"
echo ""
