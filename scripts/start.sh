#!/bin/bash

set -e
cd "$(dirname "$0")"

# Load .env into shell
if [ -f ../.env ]; then
  export $(grep -v '^#' ../.env | xargs)
else
  echo "No .env file found, defaulting to CPU mode"
  USE_GPU=false
fi

# Run docker compose based on USE_GPU
if [ "$USE_GPU" = "true" ]; then
  echo "Starting with GPU support..."
  docker compose -f ../docker-compose.yml -f ../docker-compose.gpu.override.yml up -d
else
  echo "Starting in CPU-only mode..."
  docker compose -f ../docker-compose.yml up -d
fi
