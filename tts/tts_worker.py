#!/usr/bin/env python
"""Out-of-process TTS worker.

Runs inside an engine venv (/workspace/venvs/<engine>). Reads a manifest JSON,
loads the matching adapter from tts/engines/, synthesizes one WAV per segment,
and writes a results JSON the main pipeline reads back.

Usage:  python tts_worker.py /path/to/manifest.json

Manifest:
  {
    "engine": "xtts",
    "lang": "fr",
    "speaker_wav": "/path/speaker.wav" | null,
    "speaker_profiles": {"SPEAKER_00": "/path.wav", ...},
    "engine_params": { ... },          # opaque, engine-specific
    "out_dir": "/path/tts_out",
    "segments": [{"id": 1, "text": "...", "start": 1.2, "end": 3.4,
                  "speaker": "SPEAKER_00"}, ...]
  }

Results (written to <out_dir>/results.json):
  {"sample_rate": 24000,
   "segments": [{"id": 1, "wav": "/path/seg_1.wav", "start": 1.2, "end": 3.4}, ...]}
"""

import importlib
import json
import os
import sys

# Make the sibling ``engines`` subpackage importable regardless of cwd.
_WORKER_DIR = os.path.dirname(os.path.abspath(__file__))
if _WORKER_DIR not in sys.path:
    sys.path.insert(0, _WORKER_DIR)


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def main() -> int:
    if len(sys.argv) != 2:
        _log("usage: tts_worker.py <manifest.json>")
        return 2
    with open(sys.argv[1]) as f:
        manifest = json.load(f)

    engine = manifest["engine"]
    lang = (manifest.get("lang") or "fr").lower().split("-")[0]
    speaker_wav = manifest.get("speaker_wav")
    profiles = manifest.get("speaker_profiles") or {}
    params = manifest.get("engine_params") or {}
    out_dir = manifest["out_dir"]
    segments = manifest.get("segments") or []
    os.makedirs(out_dir, exist_ok=True)

    import numpy as np  # noqa: F401  (engines return numpy arrays)
    import soundfile as sf

    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        device = "cpu"

    try:
        mod = importlib.import_module(f"engines.{engine}")
    except Exception as e:
        _log(f"FATAL: cannot load engine adapter '{engine}': {e}")
        return 1

    _log(f"Loading TTS engine '{engine}' on {device} …")
    adapter = mod.Adapter(params, device)
    sr = int(adapter.sample_rate)
    _log(f"✓ engine ready (output {sr} Hz)")

    def pick_wav(seg):
        if profiles:
            p = profiles.get(seg.get("speaker", "SPEAKER_00"))
            if p and os.path.exists(p):
                return p
        return speaker_wav

    results = []
    total = len(segments)
    for i, seg in enumerate(segments, 1):
        text = (seg.get("text") or "").strip()
        sid = seg["id"]
        if not text:
            continue
        ref = pick_wav(seg)
        if not ref or not os.path.exists(ref):
            _log(f"  [{i}/{total}] seg {sid}: no speaker reference, skipping")
            continue
        try:
            wav = adapter.synth(text, ref, lang)
            wav_path = os.path.join(out_dir, f"seg_{sid}.wav")
            # Float32 subtype keeps the handoff lossless — the segment arrays were
            # kept in float32 in-process before this refactor; PCM_16 (soundfile's
            # WAV default) would quantize them.
            sf.write(wav_path, wav, sr, subtype="FLOAT")
            results.append({
                "id": sid, "wav": wav_path,
                "start": seg["start"], "end": seg["end"],
            })
            _log(f"  [{i}/{total}] seg {sid} ✓")
        except Exception as e:
            _log(f"  [{i}/{total}] seg {sid} FAILED: {e}")

    results_path = os.path.join(out_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump({"sample_rate": sr, "segments": results}, f)
    _log(f"✓ synthesized {len(results)}/{total} segments → {results_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
