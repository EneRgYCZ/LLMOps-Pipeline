#!/bin/bash

set -e

if [ "$USE_GPU" != "true" ]; then
  echo "GPU mode is disabled. Skipping NVIDIA setup."
  exit 0
fi

echo "Installing NVIDIA Container Toolkit..."

# Detect package manager
if command -v apt-get &>/dev/null; then
  sudo apt-get update
  sudo apt-get install -y nvidia-container-toolkit
  sudo systemctl restart docker
elif command -v dnf &>/dev/null; then
  sudo dnf install -y nvidia-container-toolkit
  sudo systemctl restart docker
else
  echo "Unsupported OS. Please install the toolkit manually:"
  echo "https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
  exit 1
fi

echo "✅ NVIDIA Container Toolkit installed."
