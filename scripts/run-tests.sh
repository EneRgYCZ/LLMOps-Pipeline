#!/usr/bin/env bash
# Run the test suite locally without any external services.
# Usage: ./scripts/run-tests.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Use existing venv/ if present; otherwise create .venv-test/
if [[ -d "$REPO_ROOT/venv" ]]; then
    VENV_DIR="$REPO_ROOT/venv"
else
    VENV_DIR="$REPO_ROOT/.venv-test"
fi
PYTHON_MIN_MINOR=10

# ---------------------------------------------------------------------------
# Python version check
# ---------------------------------------------------------------------------

PYTHON_BIN="$(command -v python3 || command -v python)"
PYTHON_VERSION="$("$PYTHON_BIN" -c 'import sys; print(sys.version_info.minor)')"
PYTHON_MAJOR="$("$PYTHON_BIN" -c 'import sys; print(sys.version_info.major)')"

if [[ "$PYTHON_MAJOR" -lt 3 || ("$PYTHON_MAJOR" -eq 3 && "$PYTHON_VERSION" -lt "$PYTHON_MIN_MINOR") ]]; then
    echo "ERROR: Python >= 3.${PYTHON_MIN_MINOR} required (found $("$PYTHON_BIN" --version))" >&2
    exit 1
fi

echo "Using $("$PYTHON_BIN" --version)"

# ---------------------------------------------------------------------------
# Virtual environment
# ---------------------------------------------------------------------------

if [[ ! -d "$VENV_DIR" ]]; then
    echo "Creating virtual environment at $VENV_DIR ..."
    "$PYTHON_BIN" -m venv "$VENV_DIR"
else
    echo "Using existing virtual environment at $VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo "Installing dependencies ..."
pip install --quiet --upgrade pip
pip install --quiet -r "$REPO_ROOT/requirements.txt"
pip install --quiet -r "$REPO_ROOT/requirements-test.txt"

# ---------------------------------------------------------------------------
# Run tests
# ---------------------------------------------------------------------------

echo ""
echo "Running test suite ..."
echo "========================================"

export RAG_ENABLED=false
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318

pytest \
    "$REPO_ROOT/tests" \
    --cov=src \
    --cov-report=term-missing \
    --tb=short \
    --timeout=10 \
    -v

EXIT_CODE=$?

echo "========================================"
if [[ $EXIT_CODE -eq 0 ]]; then
    echo "All tests passed."
else
    echo "Tests FAILED (exit code $EXIT_CODE)." >&2
fi

exit $EXIT_CODE
