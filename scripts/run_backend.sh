#!/usr/bin/env bash
# Sets up a virtualenv (if missing) and runs the FastAPI backend with reload.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$SCRIPT_DIR/../backend"

cd "$BACKEND_DIR"

if [ ! -d ".venv" ]; then
  echo "Creating virtual environment..."
  python3 -m venv .venv
fi

source .venv/bin/activate
pip install --upgrade pip -q
pip install -r requirements.txt -q

if [ ! -f ".env" ]; then
  cp .env.example .env
  echo "Created backend/.env from .env.example"
fi

echo "Starting backend at http://localhost:8000 (docs at /docs)..."
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
