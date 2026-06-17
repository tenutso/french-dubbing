FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV COQUI_TOS_AGREED=1

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3-pip \
    python3-setuptools \
    build-essential \
    pkg-config \
    ffmpeg \
    git \
    wget \
    curl \
    sox \
    libsndfile1 \
    libsndfile1-dev \
    portaudio19-dev \
    && rm -rf /var/lib/apt/lists/*

# Make python3.11 the default interpreter
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
 && update-alternatives --install /usr/bin/python  python  /usr/bin/python3.11 1

WORKDIR /workspace
RUN mkdir -p /workspace/{videos/input,models/whisper,outputs,scripts,logs,temp}

# Upgrade build tools first
RUN pip install --upgrade --no-cache-dir pip setuptools wheel packaging

# Phase 1: Core scientific stack
# Main env is unpinned (modern numpy is fine); the numpy<2 constraint applies
# only inside the TTS engine venv built in Phase 5.
RUN pip install --no-cache-dir \
    "numpy>=1.26.4" \
    "scipy>=1.13.1"

# Phase 2: PyTorch 2.8.0 + torchaudio (CUDA 12.8.1 = cu128)
RUN pip install --no-cache-dir \
    torch==2.8.0 \
    torchaudio==2.8.0 \
    --index-url https://download.pytorch.org/whl/cu128

# Phase 3: Audio libraries (depend on numpy)
RUN pip install --no-cache-dir \
    "librosa>=0.10.2" \
    "soundfile>=0.12.1" \
    "pydub>=0.25.1"

# Phase 4: Transcription — faster-whisper (CTranslate2)
RUN pip install --no-cache-dir "faster-whisper>=1.0.0"

# Phase 5: TTS engine — built into its OWN venv, not the main env, so its
# transformers/numpy/torch pins stay isolated. ENGINE selects the adapter +
# requirements-<engine>.txt (default xtts). Build with:
#   docker build --build-arg TTS_ENGINE=chatterbox .
ARG TTS_ENGINE=xtts
ENV TTS_VENV_ROOT=/workspace/venvs
COPY requirements-${TTS_ENGINE}.txt /workspace/
COPY tts/ /workspace/scripts/tts/
RUN python3 -m venv "/workspace/venvs/${TTS_ENGINE}" \
 && /workspace/venvs/${TTS_ENGINE}/bin/pip install --no-cache-dir --upgrade pip wheel \
 && /workspace/venvs/${TTS_ENGINE}/bin/pip install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cu128 \
        -r "/workspace/requirements-${TTS_ENGINE}.txt"

# Phase 6: Utilities
RUN pip install --no-cache-dir \
    "pysrt>=1.1.2" \
    "requests>=2.32.0" \
    "tqdm>=4.66.0" \
    "click>=8.1.7" \
    "pyyaml>=6.0.1"

# Phase 7: RunPod Serverless (optional, for serverless deployment)
RUN pip install --no-cache-dir \
    "runpod>=0.10.0" \
    "boto3>=1.26.0"

# Install Ollama
RUN curl -fsSL https://ollama.ai/install.sh | sh

# Copy application files
COPY 02_pipeline.py       /workspace/scripts/
COPY 03_batch_runner.py   /workspace/scripts/
COPY verify_setup.py      /workspace/scripts/
COPY runpod_handler.py    /workspace/scripts/
COPY config.yaml          /workspace/

RUN chmod +x /workspace/scripts/*.py

# Verify critical imports at build time — main env (no TTS) + the engine adapter
# inside its venv.
RUN python3 -c "\
import torch; \
assert torch.__version__.startswith('2.8'), f'Wrong PyTorch: {torch.__version__}'; \
print(f'PyTorch {torch.__version__}'); \
from faster_whisper import WhisperModel; print('✓ faster-whisper'); \
print('Main env imports OK') \
"
RUN PYTHONPATH=/workspace/scripts/tts \
    "/workspace/venvs/${TTS_ENGINE}/bin/python" -c \
    "import engines.${TTS_ENGINE} as e; assert hasattr(e, 'Adapter'); print('✓ TTS engine adapter:', '${TTS_ENGINE}')"

VOLUME ["/workspace/videos", "/workspace/outputs", "/workspace/models"]

CMD ["bash", "-c", "\
    echo 'Starting Ollama ...'; \
    nohup ollama serve > /workspace/logs/ollama.log 2>&1 & \
    sleep 5 && ollama pull qwen2.5:14b; \
    echo ''; \
    echo '✓ Pipeline ready!'; \
    echo 'Usage:'; \
    echo '  python /workspace/scripts/02_pipeline.py --video /workspace/videos/input/webinar.mp4'; \
    echo '  python /workspace/scripts/03_batch_runner.py'; \
    exec bash \
"]
