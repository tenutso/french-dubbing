"""TTS engine adapter contract.

Every engine is a self-contained module in ``tts/engines/<name>.py`` exposing a
class named ``Adapter`` that implements the duck-typed interface below. The
worker (``tts/tts_worker.py``) loads it by name — it has no per-engine code — so
adding a new TTS tool never touches the core pipeline or the worker.

------------------------------------------------------------------------------
How to add a new engine (e.g. "myttts")
------------------------------------------------------------------------------
1. Create ``tts/engines/myttts.py`` with an ``Adapter`` class (see contract).
2. Create ``requirements-myttts.txt`` at the repo root pinning its deps.
3. Select it in config.yaml:  ``tts.engine: myttts`` and put any engine-specific
   knobs under ``tts.engine_params: { ... }`` (handed to ``Adapter.__init__``).
4. Build its venv:  ``python -m venv /workspace/venvs/myttts`` then
   ``pip install -r requirements-myttts.txt`` (04_setup.sh does this from config).

No edits to 02_pipeline.py, load_config, or tts_worker.py are required.

------------------------------------------------------------------------------
Contract
------------------------------------------------------------------------------
class Adapter:
    sample_rate: int
        # Native output rate of synthesized waveforms, in Hz. Read after __init__.

    def __init__(self, params: dict, device: str) -> None:
        # params  — the opaque dict from config tts.engine_params (engine-specific).
        # device  — "cuda" or "cpu". Load the model here.

    def synth(self, text: str, ref_wav: str, lang: str) -> "numpy.ndarray":
        # Synthesize ``text`` in language ``lang`` (ISO code, e.g. "fr") cloning the
        # voice in ``ref_wav``. Return a 1-D float32 numpy array at self.sample_rate.
        # Handle any internal chunking/concatenation for long text here.

    @staticmethod
    def prefetch(params: dict) -> None:        # optional
        # Download model weights ahead of time (called by 04_setup.sh inside the
        # engine venv). Omit if the engine downloads lazily on first use.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class TTSAdapter(Protocol):
    """Reference Protocol for type-checkers; adapters need not subclass it."""

    sample_rate: int

    def __init__(self, params: dict, device: str) -> None: ...

    def synth(self, text: str, ref_wav: str, lang: str) -> np.ndarray: ...
