#!/bin/bash
# Launcher for the dubbing web UI.
# Resolves the directory containing this script so it works from /workspace/scripts
# (after 04_setup.sh) or from the repo checkout.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PORT="${DUBBING_WEB_PORT:-7860}"
HOST="${DUBBING_WEB_HOST:-0.0.0.0}"

echo "Starting dubbing web UI on ${HOST}:${PORT}"
echo "Workspace: ${DUBBING_WORKSPACE:-/workspace}"
if [ -z "${DUBBING_UI_TOKEN:-}" ]; then
    echo "WARNING: DUBBING_UI_TOKEN not set — the UI is UNAUTHENTICATED." >&2
    echo "         Set it before exposing this port: DUBBING_UI_TOKEN=\$(openssl rand -hex 24)" >&2
fi
exec python -m uvicorn web.app:app \
    --host "$HOST" --port "$PORT" \
    --log-level info \
    --no-access-log
