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

# Detect hardware tier: full GPU (20GB+), small GPU, or CPU-only
TIER="cpu"
VRAM=""
if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
    VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
    if [ -n "$VRAM" ] && [ "$VRAM" -ge 20000 ] 2>/dev/null; then
        TIER="gpu_full"
        echo "  GPU detected: ${VRAM}MB VRAM — full local tier (qwen3.5:27b)"
    elif [ -n "$VRAM" ] && [ "$VRAM" -ge 7000 ] 2>/dev/null; then
        TIER="gpu_small"
        echo "  GPU detected: ${VRAM}MB VRAM — small-model tier (qwen3.5:9b)"
    else
        echo "  GPU detected but very low VRAM (${VRAM:-unknown}MB) — CPU tier"
    fi
else
    echo "  No NVIDIA GPU detected — CPU tier (slow but functional)"
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

# Start the stack for the detected tier. Nova is Ollama-only by design
# (local inference is the point) — smaller hardware just means a smaller model.
set_model() {
    # Set LLM_MODEL in .env unless the user already chose one
    if grep -qE '^LLM_MODEL=.+' .env 2>/dev/null; then
        echo "  Keeping existing LLM_MODEL from .env"
    elif grep -qE '^#? ?LLM_MODEL=' .env 2>/dev/null; then
        sed -i.bak "s|^#\{0,1\} \{0,1\}LLM_MODEL=.*|LLM_MODEL=$1|" .env && rm -f .env.bak
        echo "  Set LLM_MODEL=$1 in .env"
    else
        echo "LLM_MODEL=$1" >> .env
        echo "  Set LLM_MODEL=$1 in .env"
    fi
}

case "$TIER" in
    gpu_full)
        echo ""
        echo "  Starting with local GPU inference (qwen3.5:27b)..."
        docker compose up -d --build
        echo ""
        echo "  Pulling models (this may take a few minutes)..."
        docker exec nova-ollama ollama pull qwen3.5:27b
        docker exec nova-ollama ollama pull bge-m3
        ;;
    gpu_small)
        set_model "qwen3.5:9b"
        echo ""
        echo "  Starting with local GPU inference (qwen3.5:9b)..."
        docker compose up -d --build
        echo ""
        echo "  Pulling models (this may take a few minutes)..."
        docker exec nova-ollama ollama pull qwen3.5:9b
        docker exec nova-ollama ollama pull bge-m3
        ;;
    cpu)
        set_model "qwen3.5:4b"
        echo ""
        echo "  Starting in CPU mode (qwen3.5:4b) — responses will be slow"
        echo "  but everything works. Add a GPU later for full speed."
        docker compose -f docker-compose.yml -f docker-compose.cpu.yml up -d --build
        echo ""
        echo "  Pulling models (this may take a few minutes)..."
        docker exec nova-ollama ollama pull qwen3.5:4b
        docker exec nova-ollama ollama pull bge-m3
        ;;
esac

echo ""
echo "  Nova is starting up..."
echo ""
echo "  Web UI:  http://localhost:5173"
echo "  API:     http://localhost:8000/api/health"
echo ""
echo "  Logs:    docker compose logs -f nova"
echo "  Stop:    docker compose down"
echo ""
