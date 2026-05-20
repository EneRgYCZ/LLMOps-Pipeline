#!/usr/bin/env bash
# Start the single-node LLMOps stack and pull the inference model.
# Usage: ./scripts/start.sh [MODEL]
#   MODEL defaults to the value in .env or ministral-3:8b-instruct-2512-q4_K_M
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPOSE_FILE="$REPO_ROOT/single-node/docker-compose.yml"

# Read model name from .env if present, else use default.
MODEL="${1:-$(grep -E '^OLLAMA_MODEL=' "$REPO_ROOT/.env" 2>/dev/null | cut -d= -f2 | tr -d '"' || echo 'ministral-3:8b-instruct-2512-q4_K_M')}"
EMBED_MODEL="${EVAL_EMBEDDING_MODEL:-$(grep -E '^EVAL_EMBEDDING_MODEL=' "$REPO_ROOT/.env" 2>/dev/null | cut -d= -f2 | tr -d '"' || echo 'nomic-embed-text')}"

# --- NVIDIA Container Toolkit check ---
toolkit_installed() {
  command -v nvidia-ctk &>/dev/null && return 0
  command -v dpkg &>/dev/null && dpkg -s nvidia-container-toolkit &>/dev/null && return 0
  command -v rpm  &>/dev/null && rpm -q  nvidia-container-toolkit &>/dev/null && return 0
  return 1
}

if toolkit_installed; then
  echo "NVIDIA Container Toolkit is already installed."
else
  echo "Installing NVIDIA Container Toolkit..."
  if command -v apt-get &>/dev/null; then
    sudo apt-get update -q
    sudo apt-get install -y nvidia-container-toolkit
    sudo systemctl restart docker
  elif command -v dnf &>/dev/null; then
    sudo dnf install -y nvidia-container-toolkit
    sudo systemctl restart docker
  else
    echo "Unsupported package manager. Install the toolkit manually:"
    echo "  https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
    exit 1
  fi
  echo "NVIDIA Container Toolkit installed."
fi

# --- Start the stack ---
echo "Starting stack from $COMPOSE_FILE ..."
docker compose -f "$COMPOSE_FILE" up -d --wait

# --- Pull the model into Ollama ---
echo "Pulling model '$MODEL' into Ollama (skipped if already cached)..."
docker compose -f "$COMPOSE_FILE" exec ollama ollama pull "$MODEL"

echo "Pulling embedding model '$EMBED_MODEL' into Ollama (skipped if already cached)..."
docker compose -f "$COMPOSE_FILE" exec ollama ollama pull "$EMBED_MODEL"

echo ""
echo "Stack is ready."
echo "  Ollama:     http://localhost:11434"
echo "  Prometheus: http://localhost:9090"
echo "  Grafana:    http://localhost:3000  (admin / admin)"
echo ""
echo "Run the chat client:"
echo "  cd $REPO_ROOT && python src/chat.py"
