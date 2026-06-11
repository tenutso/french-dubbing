"""Chatterbox Multilingual engine adapter (Resemble AI, ``chatterbox-tts``).

Zero-shot voice cloning across 23+ languages including French (language_id="fr").
Ships its own pinned deps (transformers 4.46.x, torch 2.6.x, numpy 1.26.x), which
is exactly why it must run in its own venv — they conflict with XTTS's pins.

engine_params (all optional):
    exaggeration   float  0.5    emotion/intensity control
    cfg_weight     float  0.5    classifier-free guidance weight
    temperature    float  0.8    sampling temperature
    t3_model       str    "v3"   backbone variant passed to from_pretrained
    max_chars      int    300    sentence-split threshold for long segments
"""

from __future__ import annotations

import re
from typing import List

import numpy as np


def _import_multilingual():
    # Class lives in chatterbox.mtl_tts in current releases; fall back defensively.
    try:
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS
        return ChatterboxMultilingualTTS
    except ImportError:
        from chatterbox.tts import ChatterboxMultilingualTTS  # type: ignore
        return ChatterboxMultilingualTTS


def _split_sentences(text: str, limit: int) -> List[str]:
    text = text.strip()
    if len(text) <= limit:
        return [text]
    pieces = re.split(r"(?<=[.!?…])\s+", text)
    out: List[str] = []
    buf = ""
    for p in pieces:
        if len(buf) + 1 + len(p) <= limit:
            buf = f"{buf} {p}".strip()
        else:
            if buf:
                out.append(buf)
            buf = p
    if buf:
        out.append(buf)
    return [c for c in out if c]


class Adapter:
    sample_rate = 24000  # overwritten from model.sr in __init__

    def __init__(self, params: dict, device: str) -> None:
        import torch  # noqa: F401  (ensure the venv's torch is importable)

        cls = _import_multilingual()
        t3 = params.get("t3_model", "v3")
        try:
            self._model = cls.from_pretrained(device=device, t3_model=t3)
        except TypeError:
            # Older signature without t3_model.
            self._model = cls.from_pretrained(device=device)
        self.sample_rate = int(getattr(self._model, "sr", 24000))
        self.exaggeration = float(params.get("exaggeration", 0.5))
        self.cfg_weight = float(params.get("cfg_weight", 0.5))
        self.temperature = float(params.get("temperature", 0.8))
        self.max_chars = int(params.get("max_chars", 300))

    def _to_numpy(self, wav) -> np.ndarray:
        try:
            import torch
            if isinstance(wav, torch.Tensor):
                wav = wav.squeeze().detach().cpu().numpy()
        except ImportError:
            pass
        return np.asarray(wav, dtype=np.float32).reshape(-1)

    def _synth_one(self, text: str, ref_wav: str, lang: str) -> np.ndarray:
        wav = self._model.generate(
            text,
            language_id=lang,
            audio_prompt_path=ref_wav,
            exaggeration=self.exaggeration,
            cfg_weight=self.cfg_weight,
            temperature=self.temperature,
        )
        return self._to_numpy(wav)

    def synth(self, text: str, ref_wav: str, lang: str) -> np.ndarray:
        chunks = _split_sentences(text, self.max_chars)
        parts = [self._synth_one(c, ref_wav, lang) for c in chunks]
        if len(parts) == 1:
            return parts[0]
        gap = np.zeros(int(self.sample_rate * 0.08), dtype=np.float32)
        joined: List[np.ndarray] = []
        for i, part in enumerate(parts):
            if i > 0:
                joined.append(gap)
            joined.append(part)
        return np.concatenate(joined)

    @staticmethod
    def prefetch(params: dict) -> None:
        cls = _import_multilingual()
        t3 = params.get("t3_model", "v3")
        try:
            cls.from_pretrained(device="cpu", t3_model=t3)
        except TypeError:
            cls.from_pretrained(device="cpu")
