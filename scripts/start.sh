#!/usr/bin/env bash
# Start the single-node LLMOps stack and pull the inference model.
# Usage: ./scripts/start.sh [MODEL]
#   MODEL defaults to the value in .env or ministral-3:8b-instruct-2512-q4_K_M
#
# Reads TARGET from .env (GPU or CPU, case-insensitive, default CPU).
# GPU mode loads docker-compose.gpu.yml as an overlay; CPU mode uses the base file only.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPOSE_FILE="$REPO_ROOT/single-node/docker-compose.yml"
COMPOSE_GPU_FILE="$REPO_ROOT/single-node/docker-compose.gpu.yml"

# Read model name from .env if present, else use default.
MODEL="${1:-$(grep -E '^OLLAMA_MODEL=' "$REPO_ROOT/.env" 2>/dev/null | cut -d= -f2 | tr -d '"' || echo 'ministral-3:8b-instruct-2512-q4_K_M')}"
EMBED_MODEL="${EVAL_EMBEDDING_MODEL:-$(grep -E '^EVAL_EMBEDDING_MODEL=' "$REPO_ROOT/.env" 2>/dev/null | cut -d= -f2 | tr -d '"' || echo 'nomic-embed-text')}"

# Read TARGET from .env, normalize to uppercase.
TARGET="${TARGET:-$(grep -E '^TARGET=' "$REPO_ROOT/.env" 2>/dev/null | cut -d= -f2 | tr -d '"' || echo 'CPU')}"
TARGET="${TARGET^^}"

echo "==> TARGET=${TARGET}  MODEL=${MODEL}"

# --- NVIDIA Container Toolkit check (GPU only) --------------------------------
if [[ "$TARGET" == "GPU" ]]; then
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
fi

# --- Compose file selection ---------------------------------------------------
COMPOSE_ARGS=(-f "$COMPOSE_FILE")
if [[ "$TARGET" == "GPU" ]]; then
  COMPOSE_ARGS+=(-f "$COMPOSE_GPU_FILE")
  echo "==> GPU mode: loading overlay $COMPOSE_GPU_FILE"
fi

# --- Start the stack ----------------------------------------------------------
echo "Starting stack from $COMPOSE_FILE ..."
docker compose "${COMPOSE_ARGS[@]}" up -d --wait

# --- Pull the model into Ollama -----------------------------------------------
echo "Pulling model '$MODEL' into Ollama (skipped if already cached)..."
docker compose "${COMPOSE_ARGS[@]}" exec ollama ollama pull "$MODEL"

echo "Pulling embedding model '$EMBED_MODEL' into Ollama (skipped if already cached)..."
docker compose "${COMPOSE_ARGS[@]}" exec ollama ollama pull "$EMBED_MODEL"

echo ""
echo "Stack is ready."
echo "  Ollama:     http://localhost:11434"
echo "  Prometheus: http://localhost:9090"
echo "  Grafana:    http://localhost:3000  (admin / admin)"
echo ""
echo "Run the chat client:"
echo "  cd $REPO_ROOT && python src/chat.py"
