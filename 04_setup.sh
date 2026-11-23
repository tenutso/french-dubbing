#!/bin/bash
# French Dubbing Pipeline - Setup v1.0
# Target: Ubuntu 24.04 / Python 3.12 with an NVIDIA GPU (driver >= 580,
#         CUDA 13.x). Installs PyTorch 2.15 itself from requirements.txt, so any
#         RunPod PyTorch/CUDA image works (an older preinstalled torch is upgraded).
#
# Design: each step handles its own errors.
# One failing step does NOT abort the whole setup.
# Critical prerequisites (GPU, Python, PyTorch) DO abort on failure.
# A final summary lists every error and warning.

set -uo pipefail
set PYTHONUTF8=1
# ── Colour helpers ────────────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
ERRORS=(); WARNINGS=()

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

LOGFILE=/tmp/dubbing_setup.log
mkdir -p "$(dirname "$LOGFILE")"
: > "$LOGFILE"

ts()          { date +%H:%M:%S; }
log_step()    { echo -e "\n${YELLOW}[$(ts)] >>> $1${NC}" | tee -a "$LOGFILE"; }
log_success() { echo -e "${GREEN}[$(ts)] ✓ $1${NC}"      | tee -a "$LOGFILE"; }
log_warn()    { echo -e "${YELLOW}[$(ts)] ⚠ $1${NC}"     | tee -a "$LOGFILE"; WARNINGS+=("$1"); }
log_error()   { echo -e "${RED}[$(ts)] ✗ $1${NC}"        | tee -a "$LOGFILE"; ERRORS+=("$1"); }
die()         { echo -e "${RED}[$(ts)] FATAL: $1${NC}"    | tee -a "$LOGFILE"; exit 1; }

echo "==========================================" | tee -a "$LOGFILE"
echo "  French Dubbing Pipeline — Setup v1.0"    | tee -a "$LOGFILE"
echo "  $(date)"                                  | tee -a "$LOGFILE"
echo "==========================================" | tee -a "$LOGFILE"

# ── CRITICAL PREREQUISITES (abort on failure) ─────────────────────────────

log_step "Checking critical prerequisites …"

command -v nvidia-smi &>/dev/null \
    || die "nvidia-smi not found. This pipeline requires an NVIDIA GPU."
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | tee -a "$LOGFILE"

PYTHON=$(command -v python3 2>/dev/null || command -v python 2>/dev/null)
[[ -z "$PYTHON" ]] && die "Python not found in PATH."
PYTHON_VER=$($PYTHON --version 2>&1 | awk '{print $2}')
log_success "Python $PYTHON_VER at $PYTHON"

# PyTorch itself is installed from requirements.txt (Step 3). Its CUDA 13.x
# wheels need an NVIDIA driver >= 580.
DRIVER_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
DRIVER_MAJOR=${DRIVER_VER%%.*}
if [[ "$DRIVER_MAJOR" =~ ^[0-9]+$ ]] && (( DRIVER_MAJOR < 580 )); then
    die "NVIDIA driver $DRIVER_VER is too old for the CUDA 13 PyTorch wheels (need >= 580). Pick a pod/host with CUDA 13.x."
fi
log_success "NVIDIA driver $DRIVER_VER"

log_success "Prerequisites OK"

# ── Load configuration ─────────────────────────────────────────────────────
# Read paths and model names from config.yaml so setup and the pipeline stay in
# sync. Falls back to the documented defaults if a key (or the file) is missing.

log_step "Loading configuration from config.yaml …"

CONFIG_FILE="$SCRIPT_DIR/config.yaml"

# PyYAML is installed properly in Step 3; ensure it's importable now so the
# config reader below works before the main dependency install.
$PYTHON -c "import yaml" 2>/dev/null \
    || $PYTHON -m pip install --quiet --no-cache-dir pyyaml 2>/dev/null || true

cfg() {
    # cfg <dotted.key> <default> — prints the config value or the default.
    local key="$1" default="$2" val
    val=$($PYTHON - "$CONFIG_FILE" "$key" <<'PYEOF' 2>/dev/null
import sys
path, key = sys.argv[1], sys.argv[2]
try:
    import yaml
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    for part in key.split("."):
        data = data[part]
    if data is None or data == "":
        sys.exit(1)
    print(data)
except Exception:
    sys.exit(1)
PYEOF
)
    if [[ $? -ne 0 || -z "$val" ]]; then
        echo "$default"
    else
        echo "$val"
    fi
}

if [[ -f "$CONFIG_FILE" ]]; then
    log_success "Using config: $CONFIG_FILE"
else
    log_warn "config.yaml not found at $CONFIG_FILE — using built-in defaults"
fi

# Paths
INPUT_FOLDER=$(cfg  pipeline.input_folder  /workspace/videos/input)
OUTPUT_FOLDER=$(cfg pipeline.output_folder /workspace/outputs)
MODELS_FOLDER=$(cfg pipeline.models_folder /workspace/models)
LOGS_FOLDER=$(cfg   pipeline.logs_folder   /workspace/logs)
TEMP_FOLDER=$(cfg   pipeline.temp_folder   /workspace/temp)
WHISPER_CACHE="$MODELS_FOLDER/whisper"

# Models
WHISPER_MODEL=$(cfg     whisper.model      large-v3)
OLLAMA_MODEL=$(cfg      translation.model  mistral-small:22b)
F5TTS_MODEL=$(cfg       tts.f5tts_model    F5TTS_v1_Multilingual)
DIARIZATION_MODEL=$(cfg diarization.model  pyannote/speaker-diarization-community-1)

log_success "Paths: input=$INPUT_FOLDER output=$OUTPUT_FOLDER models=$MODELS_FOLDER"
log_success "Models: whisper=$WHISPER_MODEL ollama=$OLLAMA_MODEL f5tts=$F5TTS_MODEL"

# ── Step 1: System packages ────────────────────────────────────────────────

log_step "Installing system packages …"

if apt-get update -qq 2>&1 | tail -2 | tee -a "$LOGFILE" && \
   DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
       ffmpeg git wget curl sox \
       libsndfile1 libsndfile1-dev \
       libportaudio2 portaudio19-dev \
       build-essential python3-dev 2>&1 | tail -5 | tee -a "$LOGFILE"; then
    log_success "System packages installed"
else
    log_error "Some system packages failed — ffmpeg is required for this pipeline"
fi

# ── Step 2: Workspace structure ────────────────────────────────────────────

log_step "Creating workspace at /workspace …"

if mkdir -p "$INPUT_FOLDER" "$OUTPUT_FOLDER" "$WHISPER_CACHE" \
            "$LOGS_FOLDER" "$TEMP_FOLDER" /workspace/scripts && \
   chmod 755 "$INPUT_FOLDER" "$OUTPUT_FOLDER" "$WHISPER_CACHE" \
             "$LOGS_FOLDER" "$TEMP_FOLDER" /workspace/scripts; then
    log_success "Workspace ready"
else
    log_error "Workspace creation failed"
fi

# Relocate log to workspace
LOGFILE="$LOGS_FOLDER/setup.log"
cat /tmp/dubbing_setup.log >> "$LOGFILE" 2>/dev/null || true

# ── Step 3: Python dependencies (PyTorch stack + pipeline) ────────────────
# requirements.txt is the single source of truth — it pins torch / torchaudio /
# torchcodec together (CUDA 13.x wheels from PyPI), so a base image with an
# older torch is simply upgraded in place.

log_step "Upgrading pip / setuptools / wheel …"
$PYTHON -m pip install --upgrade --no-cache-dir pip setuptools wheel packaging \
       2>&1 | tail -3 | tee -a "$LOGFILE" \
    || log_warn "pip upgrade had issues — continuing anyway"

log_step "Installing Python dependencies from requirements.txt (torch + pipeline) …"
if $PYTHON -m pip install --no-cache-dir -r "$SCRIPT_DIR/requirements.txt" \
       2>&1 | tail -5 | tee -a "$LOGFILE"; then
    log_success "Python dependencies installed"
else
    die "pip install -r requirements.txt failed — see $LOGFILE"
fi

$PYTHON - <<'PYCHECK' 2>&1 | tee -a "$LOGFILE" || die "PyTorch stack failed its GPU check — see $LOGFILE"
import torch, torchaudio, torchcodec
from torchcodec.decoders import AudioDecoder  # fails if torchcodec doesn't match torch
assert torch.cuda.is_available(), "torch.cuda.is_available() returned False (driver too old for CUDA 13?)"
x = torch.randn(8, 8, device="cuda"); _ = x @ x
print(f"PyTorch {torch.__version__} (CUDA {torch.version.cuda}), torchaudio {torchaudio.__version__}, "
      f"torchcodec {torchcodec.__version__}, GPU {torch.cuda.get_device_name(0)}")
PYCHECK
log_success "PyTorch stack OK"

# hf_transfer is opt-in via env var; export now and persist in /workspace/.env
# so it's active for all subsequent pre-downloads and pipeline runs.
export HF_HUB_ENABLE_HF_TRANSFER=1
grep -q "HF_HUB_ENABLE_HF_TRANSFER" /workspace/.env 2>/dev/null \
    || echo "HF_HUB_ENABLE_HF_TRANSFER=1" >> /workspace/.env

# ── Step 4: Ollama ───────────────────────────────────────────────────────

log_step "Setting up Ollama …"

# Persist model weights on /workspace (the only volume that survives container
# restarts). Without this, qwen3:14b (~9 GB) re-downloads every boot.
export OLLAMA_MODELS="${OLLAMA_MODELS:-/workspace/.ollama/models}"
mkdir -p "$OLLAMA_MODELS"
grep -q "OLLAMA_MODELS" /workspace/.env 2>/dev/null \
    || echo "export OLLAMA_MODELS=$OLLAMA_MODELS" >> /workspace/.env

if command -v ollama &>/dev/null; then
    log_success "Ollama already installed: $(ollama --version 2>&1 | head -1)"
else
    if curl -fsSL https://ollama.ai/install.sh | sh 2>&1 | tail -5 | tee -a "$LOGFILE"; then
        log_success "Ollama installed"
    else
        log_error "Ollama installation failed — run manually: curl -fsSL https://ollama.ai/install.sh | sh"
    fi
fi

# ── Step 5: Start Ollama service ─────────────────────────────────────────

log_step "Starting Ollama service …"

if pgrep -x ollama > /dev/null 2>&1; then
    log_success "Ollama already running"
elif command -v ollama &>/dev/null; then
    OLLAMA_MODELS="$OLLAMA_MODELS" nohup ollama serve > "$LOGS_FOLDER/ollama.log" 2>&1 &
    # Wait up to 30 s for readiness
    READY=0
    for i in $(seq 1 15); do
        sleep 2
        if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
            log_success "Ollama started ($(ollama --version 2>&1 | head -1))"
            READY=1
            break
        fi
    done
    [[ $READY -eq 0 ]] && log_error "Ollama did not start in 30 s — check $LOGS_FOLDER/ollama.log"
else
    log_warn "Ollama not installed — skipping service start"
fi

# ── Step 6: Pull the translation model ─────────────────────────────────

pull_with_retry() {
    local model="$1"
    local max_tries=3
    local attempt
    for attempt in $(seq 1 $max_tries); do
        log_step "Pulling $model (attempt $attempt/$max_tries) …"
        if ollama pull "$model" 2>&1 | tee -a "$LOGFILE"; then
            log_success "$model ready"
            return 0
        fi
        [[ $attempt -lt $max_tries ]] && sleep 10
    done
    log_error "Could not pull $model — run manually: ollama pull $model"
    return 1
}

if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    pull_with_retry "$OLLAMA_MODEL"
else
    log_warn "Ollama not reachable — skipping model download. Run: ollama pull $OLLAMA_MODEL"
fi

# ── Step 7: HuggingFace token (pyannote diarization is a gated model) ──────

log_step "Configuring HuggingFace access for pyannote diarization …"

# Load from project-root .env (where user keeps it) and /workspace/.env if present.
for envfile in "$SCRIPT_DIR/.env" /workspace/.env; do
    [[ -f "$envfile" ]] && set -a && source "$envfile" 2>/dev/null && set +a || true
done

# Accept any of the canonical HF env-var names; normalize to HF_TOKEN.
HF_TOKEN="${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-${HUGGINGFACE_HUB_TOKEN:-}}}"

if [[ -z "$HF_TOKEN" ]]; then
    echo ""
    echo -e "${YELLOW}pyannote diarization uses a gated HuggingFace model.${NC}"
    echo "  1. Accept license: https://huggingface.co/$DIARIZATION_MODEL"
    echo "  2. Get your token: https://huggingface.co/settings/tokens (read permission)"
    echo ""
    read -rp "  Enter HuggingFace token (or press Enter to skip): " HF_TOKEN
    echo ""
fi

if [[ -n "$HF_TOKEN" ]]; then
    # Strip non-ASCII/non-token chars (e.g. Unicode ellipsis from copy-paste truncation)
    HF_TOKEN=$(echo "$HF_TOKEN" | LC_ALL=C tr -cd 'A-Za-z0-9_-')
    export HF_TOKEN="$HF_TOKEN"
    # Persist so future setup runs and pipeline runs can find it
    grep -q "HF_TOKEN" /workspace/.env 2>/dev/null \
        || echo "HF_TOKEN=\"$HF_TOKEN\"" >> /workspace/.env

    if $PYTHON -c "
from huggingface_hub import HfApi
try:
    info = HfApi().whoami(token='$HF_TOKEN')
    print(f\"Authenticated as: {info.get('name', 'unknown')}\")
except Exception as e:
    print(f'Auth failed: {e}')
    raise
" 2>&1 | tee -a "$LOGFILE"; then
        log_success "HuggingFace authenticated — token saved to /workspace/.env"
    else
        log_warn "HuggingFace token invalid — diarization will fail"
    fi
else
    log_warn "No HuggingFace token provided — diarization will be disabled at runtime"
fi

# ── Step 8: Pre-download Whisper large-v3 ────────────────────────────────

log_step "Pre-downloading Whisper $WHISPER_MODEL (~3 GB) …"

if WHISPER_MODEL="$WHISPER_MODEL" WHISPER_CACHE="$WHISPER_CACHE" \
   $PYTHON - <<'PYEOF' 2>&1 | tee -a "$LOGFILE"; then
import os
from faster_whisper import WhisperModel
model = os.environ["WHISPER_MODEL"]
cache = os.environ["WHISPER_CACHE"]
print(f"Downloading Whisper {model} to {cache} …")
m = WhisperModel(model, device="cpu", compute_type="int8", download_root=cache)
del m
print(f"✓ Whisper {model} cached")
PYEOF
    log_success "Whisper $WHISPER_MODEL cached"
else
    log_warn "Whisper download failed — it will auto-download on first use"
fi

# ── Step 9: Pre-download F5-TTS weights (~1.5 GB) ───────────────────────────

log_step "Pre-downloading F5-TTS model (~1.5 GB) …"

if F5TTS_MODEL="$F5TTS_MODEL" $PYTHON - <<'PYEOF' 2>&1 | tee -a "$LOGFILE"; then
import os
try:
    from f5_tts.api import F5TTS
    model = os.environ.get("F5TTS_MODEL", "F5TTS_v1_Small")
    print(f"Downloading F5-TTS model: {model} …")
    if "/" in model:
        # HuggingFace repo ID (e.g. RASPIAUDIO/F5-French-MixedSpeakers-reduced).
        # Download checkpoint + vocab and load with F5TTS_Base architecture.
        from huggingface_hub import hf_hub_download
        print(f"  Fetching weights from HuggingFace: {model} …")
        try:
            ckpt_file = hf_hub_download(repo_id=model, filename="model_last_reduced.pt")
        except Exception:
            ckpt_file = hf_hub_download(repo_id=model, filename="model_last.pt")
        vocab_file = hf_hub_download(repo_id=model, filename="vocab.txt")
        t = F5TTS(model="F5TTS_Base", ckpt_file=ckpt_file, vocab_file=vocab_file)
    else:
        t = F5TTS(model=model)
    del t
    print("✓ F5-TTS cached")
except Exception as e:
    print(f"F5-TTS download error: {e}")
    raise
PYEOF
    log_success "F5-TTS cached"
else
    log_error "F5-TTS failed to import or download — TTS will not work (see $LOGFILE)"
fi

# ── Step 10: Install scripts & config ─────────────────────────────────────

log_step "Installing pipeline scripts …"

for script in 02_pipeline.py 03_batch_runner.py verify_setup.py 05_web.sh metrics.py voice_preview.py; do
    if [[ -f "$SCRIPT_DIR/$script" ]]; then
        cp "$SCRIPT_DIR/$script" /workspace/scripts/
        chmod +x "/workspace/scripts/$script"
        log_success "$script → /workspace/scripts/"
    else
        log_warn "$script not found in $SCRIPT_DIR — copy manually to /workspace/scripts/"
    fi
done

# Install config only if one doesn't already exist
if [[ -f "$SCRIPT_DIR/config.yaml" ]]; then
    if [[ ! -f /workspace/config.yaml ]]; then
        cp "$SCRIPT_DIR/config.yaml" /workspace/config.yaml
        log_success "config.yaml → /workspace/config.yaml"
    else
        log_success "config.yaml already at /workspace/config.yaml (not overwritten)"
    fi
else
    log_warn "config.yaml not found in $SCRIPT_DIR"
fi

# Install Canadian French glossary (always overwrite — it's a vocabulary file, not user config)
if [[ -f "$SCRIPT_DIR/canadian_glossary.yaml" ]]; then
    cp "$SCRIPT_DIR/canadian_glossary.yaml" /workspace/canadian_glossary.yaml
    log_success "canadian_glossary.yaml → /workspace/canadian_glossary.yaml"
else
    log_warn "canadian_glossary.yaml not found in $SCRIPT_DIR — Canadian French locale will skip glossary"
fi

# ── Step 11: Web UI package + runtime directories ────────────────────────────
log_step "Installing web UI (FastAPI) …"

if [[ -d "$SCRIPT_DIR/web" ]]; then
    rm -rf /workspace/scripts/web
    cp -r "$SCRIPT_DIR/web" /workspace/scripts/web
    log_success "web/ → /workspace/scripts/web/"
else
    log_warn "web/ not found in $SCRIPT_DIR — web UI will not be available"
fi

mkdir -p /workspace/web/uploads /workspace/web/outputs \
    && log_success "Runtime dirs at /workspace/web/{uploads,outputs}" \
    || log_warn "Could not create /workspace/web/{uploads,outputs}"

# ── Final Summary ──────────────────────────────────────────────────────────

echo ""
echo "==========================================" | tee -a "$LOGFILE"
if [[ ${#ERRORS[@]} -eq 0 ]]; then
    echo -e "${GREEN}Setup complete — no errors!${NC}" | tee -a "$LOGFILE"
else
    echo -e "${YELLOW}Setup complete with ${#ERRORS[@]} error(s):${NC}" | tee -a "$LOGFILE"
    for err in "${ERRORS[@]}"; do
        echo -e "  ${RED}✗ $err${NC}" | tee -a "$LOGFILE"
    done
fi
if [[ ${#WARNINGS[@]} -gt 0 ]]; then
    echo -e "${YELLOW}Warnings (${#WARNINGS[@]}):${NC}" | tee -a "$LOGFILE"
    for warn in "${WARNINGS[@]}"; do
        echo -e "  ${YELLOW}⚠ $warn${NC}" | tee -a "$LOGFILE"
    done
fi
echo "==========================================" | tee -a "$LOGFILE"
echo ""
echo "Next steps:"
echo "  1. Verify:  python /workspace/scripts/verify_setup.py"
echo "  2. Copy:    cp *.mp4 $INPUT_FOLDER/"
echo "  3. Single:  python /workspace/scripts/02_pipeline.py \\"
echo "                --video $INPUT_FOLDER/webinar.mp4"
echo "  4. Batch:   python /workspace/scripts/03_batch_runner.py"
echo "  5. Web UI:  bash /workspace/scripts/05_web.sh"
echo "              then open the RunPod-proxied URL on port 7860"
echo ""
echo "Setup log: $LOGFILE"
echo ""

[[ ${#ERRORS[@]} -gt 0 ]] && exit 1 || exit 0
