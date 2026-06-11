"""XTTS-v2 engine adapter (Coqui / Idiap fork, ``coqui-tts``).

Multilingual zero-shot voice cloning, 24 kHz native output, native French.
Self-contained: the per-language chunk limits and sentence-aware splitting that
used to live in 02_pipeline.py now live here, so the core pipeline carries no
XTTS-specific logic.

engine_params (all optional, defaults match the historical config.yaml values):
    model                 str   tts_models/multilingual/multi-dataset/xtts_v2
    temperature           float 0.65
    length_penalty        float 1.0
    repetition_penalty    float 2.0
    top_k                 int   50
    top_p                 float 0.85
"""

from __future__ import annotations

import os
import re
from typing import List

import numpy as np

# XTTS-v2 hard per-language character cap. Values are well under the model's
# internal token cap — XTTS loops well before its hard cap, so we split
# aggressively on sentence boundaries. 180 chars/chunk for Romance languages
# eliminates almost all internal looping while keeping prosody continuous.
_XTTS_CHUNK_LIMITS = {
    "fr": 180, "en": 200, "es": 180, "de": 200, "it": 170, "pt": 180,
    "pl": 200, "nl": 200, "ru": 160, "cs": 160, "ar": 150, "tr": 180,
    "hu": 160, "zh": 80, "ja": 80, "ko": 80, "hi": 200,
}

_DEFAULT_MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"


def _split_for_xtts(text: str, limit: int) -> List[str]:
    """Break text into ≤limit-char chunks on natural boundaries.

    Tries sentences first (. ! ? …), then clauses (; : ,), then a hard split.
    """
    text = text.strip()
    if len(text) <= limit:
        return [text]
    pieces = re.split(r"(?<=[.!?…])\s+", text)
    out: List[str] = []
    for piece in pieces:
        if len(piece) <= limit:
            out.append(piece)
            continue
        sub = re.split(r"(?<=[,;:])\s+", piece)
        buf = ""
        for s in sub:
            if len(s) > limit:  # one clause is itself too long → hard wrap
                if buf:
                    out.append(buf.strip()); buf = ""
                for i in range(0, len(s), limit):
                    out.append(s[i:i + limit].strip())
                continue
            if len(buf) + 1 + len(s) <= limit:
                buf = f"{buf} {s}".strip()
            else:
                if buf:
                    out.append(buf.strip())
                buf = s
        if buf:
            out.append(buf.strip())
    return [c for c in out if c]


class Adapter:
    sample_rate = 24000  # XTTS-v2 native rate

    def __init__(self, params: dict, device: str) -> None:
        from TTS.api import TTS  # heavy import, only in the engine venv

        self.model_name = params.get("model", _DEFAULT_MODEL)
        self.temperature = float(params.get("temperature", 0.65))
        self.length_penalty = float(params.get("length_penalty", 1.0))
        self.repetition_penalty = float(params.get("repetition_penalty", 2.0))
        self.top_k = int(params.get("top_k", 50))
        self.top_p = float(params.get("top_p", 0.85))
        # Auto-accept the CPML license non-interactively.
        os.environ.setdefault("COQUI_TOS_AGREED", "1")
        self._tts = TTS(self.model_name).to(device)
        # 80 ms silence between concatenated chunks of one segment.
        self._gap = np.zeros(int(self.sample_rate * 0.08), dtype=np.float32)

    def _synth_one(self, text: str, ref_wav: str, lang: str) -> np.ndarray:
        return np.asarray(
            self._tts.tts(
                text=text,
                speaker_wav=ref_wav,
                language=lang,
                temperature=self.temperature,
                length_penalty=self.length_penalty,
                repetition_penalty=self.repetition_penalty,
                top_k=self.top_k,
                top_p=self.top_p,
                split_sentences=True,
            ),
            dtype=np.float32,
        )

    def synth(self, text: str, ref_wav: str, lang: str) -> np.ndarray:
        limit = _XTTS_CHUNK_LIMITS.get(lang, 240)
        chunks = _split_for_xtts(text, limit)
        parts = [self._synth_one(c, ref_wav, lang) for c in chunks]
        if len(parts) == 1:
            return parts[0]
        joined: List[np.ndarray] = []
        for i, part in enumerate(parts):
            if i > 0:
                joined.append(self._gap)
            joined.append(part)
        return np.concatenate(joined)

    @staticmethod
    def prefetch(params: dict) -> None:
        from TTS.api import TTS

        os.environ.setdefault("COQUI_TOS_AGREED", "1")
        model = params.get("model", _DEFAULT_MODEL)
        TTS(model)  # triggers the HuggingFace download into the cache
