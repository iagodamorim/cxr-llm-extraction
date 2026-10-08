#!/usr/bin/env bash
# setup.sh — one-command environment bootstrap.
# Usage: bash setup.sh
set -euo pipefail

cd "$(dirname "$0")"

# Detect Python
if command -v python3 &>/dev/null; then
    PYTHON=python3
elif command -v python &>/dev/null; then
    PYTHON=python
else
    echo "ERROR: python3 not found. Install Python 3.10+ first."
    exit 1
fi

echo "Using $($PYTHON --version)"

# Create venv if missing
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment in .venv/ ..."
    $PYTHON -m venv .venv
fi

# Activate + install
# shellcheck disable=SC1091
source .venv/bin/activate
echo "Upgrading pip..."
pip install --upgrade pip > /dev/null

echo "Installing requirements..."
pip install -r requirements.txt

# Copy .env.example to .env if missing
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo ""
    echo "==============================================================="
    echo "Created .env from template."
    echo "Edit it with your real API keys before running api_inference.py:"
    echo "  $(pwd)/.env"
    echo "==============================================================="
fi

echo ""
echo "Setup complete. To activate the environment in future sessions:"
echo "  source .venv/bin/activate"
echo ""
echo "Next step: run 'bash run_pipeline.sh steps1to4'  (skips API calls)"
echo "Or just step by step from the README."
