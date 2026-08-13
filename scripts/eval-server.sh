#!/usr/bin/env bash
# scripts/eval-server.sh
# Serves the repo root on a local port so the eval harness can resolve
# ../extension/heuristics/heuristic-filter.js without the http.server's
# directory traversal complaint.
#
# Usage:  bash scripts/eval-server.sh [PORT]   (default 8090)
# Open:   http://127.0.0.1:<PORT>/eval/

set -euo pipefail
PORT="${1:-8090}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

cd "$REPO_ROOT"
echo "Serving $REPO_ROOT on http://127.0.0.1:${PORT}/"
echo "Open: http://127.0.0.1:${PORT}/eval/"
exec python3 -m http.server "$PORT" --bind 127.0.0.1
