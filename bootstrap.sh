#!/usr/bin/env bash
# RunPod bootstrap for the French Dubbing Pipeline.
#
# Container start command:
#   bash -c "curl -fsSL https://raw.githubusercontent.com/<you>/french-dubbing/main/bootstrap.sh | bash"
#
# Required env vars (set in the RunPod template):
#   HF_TOKEN     — HuggingFace token; must have accepted the pyannote license.
# Optional env vars:
#   REPO_URL     — defaults to https://github.com/<you>/french-dubbing.git
#   GIT_REF      — branch or tag to deploy (default: main)
#   GITHUB_TOKEN — for private repos; injected into the clone URL.
#
# Persistence: mount a network volume at /workspace. All caches (HF, Ollama,
# the repo itself, the setup marker) live under /workspace so a container
# restart is a ~30 s no-op instead of a 10 min re-install.

set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/REPLACE_ME/french-dubbing.git}"
GIT_REF="${GIT_REF:-main}"
REPO_DIR="/workspace/french-dubbing"
MARKER="/workspace/.setup_done"
LOG="/workspace/logs/bootstrap.log"

mkdir -p /workspace/logs
exec > >(tee -a "$LOG") 2>&1
echo "[$(date -Is)] bootstrap start  ref=$GIT_REF  repo=$REPO_URL"

# Inject GITHUB_TOKEN for private repos.
CLONE_URL="$REPO_URL"
if [[ -n "${GITHUB_TOKEN:-}" && "$REPO_URL" == https://github.com/* ]]; then
    CLONE_URL="${REPO_URL/https:\/\//https:\/\/${GITHUB_TOKEN}@}"
fi

# 1. Source code: clone if missing, else fast-forward to the requested ref.
if [[ ! -d "$REPO_DIR/.git" ]]; then
    echo "Cloning $REPO_URL @ $GIT_REF → $REPO_DIR"
    git clone --depth 1 --branch "$GIT_REF" "$CLONE_URL" "$REPO_DIR"
else
    echo "Refreshing $REPO_DIR to $GIT_REF"
    git -C "$REPO_DIR" remote set-url origin "$CLONE_URL"
    git -C "$REPO_DIR" fetch --depth 1 origin "$GIT_REF"
    git -C "$REPO_DIR" reset --hard "origin/$GIT_REF"
    git -C "$REPO_DIR" clean -fd
fi

# 2. Setup: idempotent. Re-runs only if requirements.txt changed since last boot.
REQ_HASH=$(sha256sum "$REPO_DIR/requirements.txt" | awk '{print $1}')
if [[ -f "$MARKER" && "$(cat "$MARKER")" == "$REQ_HASH" ]]; then
    echo "Setup already current (requirements.txt unchanged) — skipping."
else
    echo "Running 04_setup.sh (this takes ~10 min on first boot) …"
    bash "$REPO_DIR/04_setup.sh"
    echo "$REQ_HASH" > "$MARKER"
fi

# 3. Source the env file 04_setup.sh wrote (HF_HUB_ENABLE_HF_TRANSFER,
#    COQUI_TOS_AGREED, OLLAMA_MODELS). Subsequent boots need these too.
# shellcheck disable=SC1091
[[ -f /workspace/.env ]] && source /workspace/.env

# 4. Ensure Ollama is running on cached restarts (setup-skip path).
if ! pgrep -x ollama > /dev/null 2>&1 && command -v ollama &>/dev/null; then
    echo "Starting Ollama (cached models at $OLLAMA_MODELS) …"
    OLLAMA_MODELS="$OLLAMA_MODELS" nohup ollama serve > /workspace/logs/ollama.log 2>&1 &
    for i in $(seq 1 15); do
        sleep 2
        curl -s http://localhost:11434/api/tags > /dev/null && break
    done
fi

# 5. Launch the web UI in the foreground. exec → web becomes PID 1, so a crash
#    triggers a RunPod container restart.
echo "[$(date -Is)] launching web UI on :${DUBBING_WEB_PORT:-7860}"
exec bash /workspace/scripts/05_web.sh
