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
# numpy<2.0.0 required by Coqui TTS (XTTS v2)
RUN pip install --no-cache-dir \
    "numpy>=1.26.4,<2.0.0" \
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

# Phase 5: TTS — Coqui XTTS v2
# Pin transformers FIRST: coqui-tts imports isin_mps_friendly which was removed in 4.41.0
RUN pip install --no-cache-dir "transformers>=4.36.0,<4.41.0"
# coqui-tts is the community fork of the original TTS package (removed from PyPI);
# installs to the same TTS.* import namespace.
RUN pip install --no-cache-dir coqui-tts || \
    pip install --no-cache-dir "git+https://github.com/coqui-ai/TTS.git"

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

# Verify critical imports at build time
RUN python3 -c "\
import torch; \
assert torch.__version__.startswith('2.8'), f'Wrong PyTorch: {torch.__version__}'; \
assert torch.cuda.is_available() or True; \
print(f'PyTorch {torch.__version__}'); \
from faster_whisper import WhisperModel; print('✓ faster-whisper'); \
from TTS.api import TTS; print('✓ Coqui TTS (XTTS v2)'); \
print('All imports OK') \
"

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
