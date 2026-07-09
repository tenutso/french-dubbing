#!/usr/bin/env python3
"""One-shot voice-clone preview for the review UI.

Builds a reference clip from a job spec (a time range of the vocals track, or
a library clip), synthesizes one short sentence with F5-TTS, and writes a WAV.
Runs as a subprocess of the web app so model VRAM is freed on exit and a crash
can never take the UI down.

Usage:
    voice_preview.py --config /workspace/config.yaml --job job.json --out out.wav

job.json:
    {
      "source":   "range" | "library",
      "vocals":   "/path/vocals.wav",      # range source
      "start":    123.0,                    # range source (seconds)
      "duration": 12.0,                     # range source (seconds)
      "path":     "/workspace/voices/x.wav" # library source
      "denoise":  false,
      "text":     "optional custom sentence"
    }
"""
import argparse
import importlib.util
import json
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path

DEFAULT_TEXT = (
    "Bonjour à tous et merci de votre présence aujourd'hui "
    "pour cette discussion très importante."
)
MAX_TEXT_CHARS = 200


def _load_pipeline_module():
    """importlib-load 02_pipeline.py (its numeric prefix blocks normal import)."""
    here = Path(__file__).resolve().parent
    path = here / "02_pipeline.py"
    spec = importlib.util.spec_from_file_location("dubbing_pipeline", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--job", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s",
                        stream=sys.stderr)
    log = logging.getLogger("voice_preview")

    with open(args.job, encoding="utf-8") as f:
        job = json.load(f)

    pl = _load_pipeline_module()
    config = pl.load_config(args.config)

    # ── Build the reference clip (mirrors _ref_from_override) ────────────────
    sr_ref = max(int(config.tts_speaker_profile_sr), 24000)
    workdir = tempfile.mkdtemp(prefix="voice_preview_")
    ref_raw = os.path.join(workdir, "ref_raw.wav")

    source = job.get("source")
    if source == "range":
        vocals = job.get("vocals") or ""
        if not os.path.exists(vocals):
            print(f"vocals track not found: {vocals}", file=sys.stderr)
            return 2
        start = float(job.get("start", 0.0))
        dur = float(job.get("duration", config.tts_speaker_duration))
        subprocess.run(
            ["ffmpeg", "-y", "-i", vocals, "-ss", str(start), "-t", str(dur),
             "-ar", str(sr_ref), "-ac", "1", ref_raw],
            check=True, capture_output=True, timeout=60,
        )
    elif source == "library":
        libpath = job.get("path") or ""
        if not os.path.exists(libpath):
            print(f"library clip not found: {libpath}", file=sys.stderr)
            return 2
        subprocess.run(
            ["ffmpeg", "-y", "-i", libpath, "-ar", str(sr_ref), "-ac", "1", ref_raw],
            check=True, capture_output=True, timeout=60,
        )
    else:
        print(f"unknown source: {source!r}", file=sys.stderr)
        return 2

    # Pause-condense so the preview clones the same pace Phase 2 will.
    import librosa
    import soundfile as sf
    y, _ = librosa.load(ref_raw, sr=sr_ref, mono=True)
    condensed, removed = pl._condense_silences(y, sr_ref)
    if len(condensed) >= sr_ref * 2:
        y = condensed
        if removed > 0.25:
            log.info(f"condensed {removed:.1f}s of pauses from the reference")
    # Same hard cap Phase 2 applies (F5 duration formula breaks past ~22 s).
    max_samples = int(15.0 * sr_ref)
    if len(y) > max_samples:
        y = y[:max_samples]
    ref_wav = os.path.join(workdir, "ref.wav")
    sf.write(ref_wav, y, sr_ref)

    if job.get("denoise"):
        dn = os.path.join(workdir, "ref_dn.wav")
        ref_wav = pl.denoise_audio(ref_wav, dn, log,
                                   prop_decrease=config.tts_reference_denoise_strength)

    # ── Test text: spoken form + pronunciation lexicon, like the real run ────
    text = (job.get("text") or "").strip()[:MAX_TEXT_CHARS] or DEFAULT_TEXT
    pron = pl.load_pronunciations(config.glossary_path, log)
    text = pl._tts_spoken_form(text, pron, spell_acronyms=config.tts_spell_acronyms)
    if text and text[-1] not in ".!?…":
        text += "."
    log.info(f"synthesizing: {text!r}")

    # ── Synthesize ────────────────────────────────────────────────────────────
    ref_text = ""
    if not pl._torchcodec_ok():
        # Same fallback as the pipeline: pre-transcribe the reference so F5
        # never touches torchcodec.
        try:
            from faster_whisper import WhisperModel
            wm = WhisperModel("small", device="cpu", compute_type="int8",
                              download_root=config.models_folder)
            segs, _ = wm.transcribe(ref_wav, language=None, beam_size=1,
                                    condition_on_previous_text=False)
            ref_text = " ".join(s.text.strip() for s in segs).strip()
            del wm
        except Exception as e:
            log.warning(f"ref-text fallback failed ({e}) — trying ref_text=''")

    f5, sr = pl.load_f5tts_model(config, log)
    wav, _, _ = f5.infer(
        ref_file=ref_wav,
        ref_text=ref_text,
        gen_text=text,
        nfe_step=config.f5tts_nfe_step,
        cfg_strength=config.f5tts_cfg_strength,
        speed=config.f5tts_speed,
        remove_silence=False,
        file_wave=None,
        seed=None,
    )
    import numpy as np
    wav = np.asarray(wav, dtype=np.float32)
    if len(wav) < sr * 0.2:
        print("synthesis produced near-empty audio — try a longer/cleaner range",
              file=sys.stderr)
        return 3
    peak = float(np.max(np.abs(wav)))
    if peak > 0:
        wav *= 0.95 / peak
    sf.write(args.out, wav, sr)
    log.info(f"wrote {args.out} ({len(wav) / sr:.1f}s)")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except subprocess.CalledProcessError as e:
        print(f"ffmpeg failed: {(e.stderr or b'').decode('utf-8', 'ignore')[-300:]}",
              file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"voice preview failed: {e}", file=sys.stderr)
        sys.exit(1)
