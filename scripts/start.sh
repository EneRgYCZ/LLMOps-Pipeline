#!/bin/bash

set -e
cd "$(dirname "$0")"

toolkit_installed() {
  command -v nvidia-ctk &>/dev/null && return 0

  if command -v dpkg &>/dev/null; then
    dpkg -s nvidia-container-toolkit &>/dev/null && return 0
  fi

  if command -v rpm &>/dev/null; then
    rpm -q nvidia-container-toolkit &>/dev/null && return 0
  fi

  return 1
}

if toolkit_installed; then
  echo "✅ NVIDIA Container Toolkit is already installed."
else
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
fi

docker compose -f ../docker-compose.yml up -d