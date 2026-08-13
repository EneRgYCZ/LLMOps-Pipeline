#!/usr/bin/env bash
set -uo pipefail

cd /home/catalin/LLMOps-Pipeline

TAGS=(
  "llama3.1:8b-instruct-q2_K"
  "llama3.1:8b-instruct-q3_K_S"
  "llama3.1:8b-instruct-q3_K_M"
  "llama3.1:8b-instruct-q3_K_L"
  "llama3.1:8b-instruct-q4_0"
  "llama3.1:8b-instruct-q4_1"
  "llama3.1:8b-instruct-q4_K_S"
  "llama3.1:8b-instruct-q5_0"
  "llama3.1:8b-instruct-q5_1"
  "llama3.1:8b-instruct-q5_K_S"
  "llama3.1:8b-instruct-q5_K_M"
  "llama3.1:8b-instruct-q6_K"
)

FAILED=()

for tag in "${TAGS[@]}"; do
  echo "==================================================================="
  echo "=== $tag : $(date -u +%FT%TZ)"
  echo "==================================================================="
  df -h / | tail -1

  if ! docker exec ollama ollama pull "$tag"; then
    echo "!!! PULL FAILED for $tag, skipping"
    FAILED+=("$tag (pull)")
    continue
  fi

  if ! ./venv/bin/python analysis/rq4/rq4_experiment.py --model-tag "$tag"; then
    echo "!!! EXPERIMENT FAILED for $tag"
    FAILED+=("$tag (experiment)")
  fi

  docker exec ollama ollama rm "$tag" || echo "!!! could not remove $tag (non-fatal)"
  echo "=== done with $tag, disk now:"
  df -h / | tail -1
done

echo "==================================================================="
echo "Sweep complete: $(date -u +%FT%TZ)"
if [ "${#FAILED[@]}" -gt 0 ]; then
  echo "Failures:"
  printf '  - %s\n' "${FAILED[@]}"
else
  echo "All tags succeeded."
fi
