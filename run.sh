#!/usr/bin/env bash
# Starts the API and the web app together.
#
#   ./run.sh
#
# API  -> http://localhost:8000   (docs at /docs)
# Web  -> http://localhost:3000
set -e
cd "$(dirname "$0")"

if [ ! -d .venv ]; then
  echo "==> creating .venv"
  python3 -m venv .venv
  ./.venv/bin/python -m pip install --upgrade pip --quiet
  ./.venv/bin/python -m pip install -r api/requirements.txt
fi

if [ ! -d web/node_modules ]; then
  echo "==> installing web dependencies"
  (cd web && npm install)
fi

echo "==> starting API on :8000"
(cd api && ../.venv/bin/python -m uvicorn main:app --port 8000) &
API_PID=$!
trap 'kill $API_PID 2>/dev/null' EXIT

echo "==> starting web on :3000"
cd web && npm run dev
