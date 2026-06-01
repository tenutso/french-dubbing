#!/usr/bin/env python3
"""
French Dubbing Pipeline v4.0 — refined & simplified

English video → French audio track + SRT subtitles.

Stack (single, fixed path):
  Source separation : Demucs htdemucs           (vocals + background split)
  Transcription     : faster-whisper large-v3   (word timestamps, VAD)
  Segment merging   : sentence-scale chunks     (8–12s, no sub-second fragments)
  Diarization (opt) : pyannote-audio            (per-speaker voice profiles)
  Translation       : Qwen3:14b via Ollama      (single natural pass)
  Speaker denoising : DeepFilterNet             (clean voice reference)
  TTS               : VoxCPM2 2B at 48 kHz      (voice cloning)
  Assembly          : FFmpeg atempo + crossfade (fit French to original timing)
  Subtitles         : direct from merged spans  (no WhisperX force-align)
  Output            : AAC 192 kbps 48 kHz stereo + UTF-8 SRT
"""

import gc
import json
import logging
import os
import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional, Tuple

# Accelerated HF downloads — must be set before huggingface_hub is imported.
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    for candidate in (
        Path(__file__).resolve().parent / ".env",
        Path("/workspace/.env"),
    ):
        if candidate.exists():
            load_dotenv(candidate, override=False)

_load_dotenv()

# Normalize the HF token env var names HuggingFace libs accept.
_hf_tok = (
    os.environ.get("HF_TOKEN")
    or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    or ""
)
if _hf_tok:
    os.environ["HF_TOKEN"] = _hf_tok
    os.environ["HUGGING_FACE_HUB_TOKEN"] = _hf_tok

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")
warnings.filterwarnings("ignore", category=UserWarning, module="pyannote")

import click
import librosa
import numpy as np
import pysrt
import requests
import soundfile as sf
import torch
import torchaudio
import yaml

# Enable TF32 for better performance on Ampere+ GPUs (RTX 30/40)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from pysrt import SubRipItem, SubRipTime
from tqdm import tqdm

from faster_whisper import WhisperModel


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class PipelineConfig:
    input_folder: str
    output_folder: str
    models_folder: str
    logs_folder: str
    temp_folder: str

    # Transcription
    whisper_model: str = "large-v3"
    whisper_device: str = "cuda"
    whisper_compute_type: str = "float16"

    # Source separation
    use_demucs: bool = True
    demucs_model: str = "htdemucs"
    preserve_background: bool = True

    # Diarization
    use_diarization: bool = False
    diarization_model: str = "pyannote/speaker-diarization-community-1"
    diarization_min_speakers: int = 2
    diarization_max_speakers: int = 10
    diarization_profile_duration: float = 25.0

    # Translation — Qwen via Ollama (single backend)
    translation_model: str = "qwen3:14b"
    translation_temperature: float = 0.3
    translation_batch_size: int = 20
    translation_review: bool = False
    target_lang: str = "fr"

    # Speaker reference
    use_deepfilter: bool = True
    tts_speaker_duration: float = 25.0
    tts_speaker_skip: float = 20.0

    # TTS — VoxCPM2 (single engine)
    tts_model: str = "openbmb/VoxCPM2"
    tts_max_stretch: float = 1.25
    tts_min_stretch: float = 0.8
    tts_cfg_value: float = 2.5
    tts_inference_timesteps: int = 24
    tts_use_prompt_text: bool = False

    # Audio
    output_volume_boost_pct: float = 0.0

    # HF token (only used by diarization gated model)
    huggingface_token: str = ""

    # Locale + glossary (Canadian French enforcement)
    locale: str = "fr"
    glossary_path: str = ""

    # Segment merging — sentence-scale chunks
    segment_merge_gap: float = 1.5
    segment_merge_max_duration: float = 12.0
    segment_merge_min_duration: float = 2.0

    # SRT
    subtitle_offset_ms: int = 0

    # Sample rates
    synthesis_sample_rate: int = 48000
    output_sample_rate: int = 48000

    timeout_seconds: int = 7200

    # Debug — keep all temp files and dump intermediate segment JSON
    keep_temp: bool = False


@dataclass
class GlossaryEntry:
    en: str
    fr_ca: str
    fr_std: str = ""
    mode: str = "suggest"
    category: str = ""
    note: str = ""


@dataclass
class Glossary:
    entries: List[GlossaryEntry]
    formatting_rules: List[str]
    inclusive_language: List[str]

    @property
    def has_content(self) -> bool:
        return bool(self.entries or self.formatting_rules or self.inclusive_language)


def load_config(path: str) -> PipelineConfig:
    with open(path) as f:
        c = yaml.safe_load(f)
    p    = c.get("pipeline", {})
    w    = c.get("whisper", {})
    t    = c.get("translation", {})
    tts  = c.get("tts", {})
    proc = c.get("processing", {})
    aud  = c.get("audio", {})
    sep  = c.get("source_separation", {})
    sub  = c.get("subtitles", {})
    dia  = c.get("diarization", {})
    return PipelineConfig(
        input_folder=p.get("input_folder", "/workspace/videos/input"),
        output_folder=p.get("output_folder", "/workspace/outputs"),
        models_folder=p.get("models_folder", "/workspace/models"),
        logs_folder=p.get("logs_folder", "/workspace/logs"),
        temp_folder=p.get("temp_folder", "/workspace/temp"),
        whisper_model=w.get("model", "large-v3"),
        whisper_device=w.get("device", "cuda"),
        whisper_compute_type=w.get("compute_type", "float16"),
        use_demucs=sep.get("enabled", True),
        demucs_model=sep.get("model", "htdemucs"),
        preserve_background=sep.get("preserve_background", True),
        use_diarization=dia.get("enabled", False),
        diarization_model=dia.get("model", "pyannote/speaker-diarization-community-1"),
        diarization_min_speakers=dia.get("min_speakers", 1),
        diarization_max_speakers=dia.get("max_speakers", 10),
        diarization_profile_duration=dia.get("profile_duration", 25.0),
        translation_model=t.get("model", "qwen3:14b"),
        translation_temperature=t.get("temperature", 0.3),
        translation_batch_size=t.get("batch_size", 20),
        translation_review=t.get("review_pass", False),
        target_lang=t.get("target_lang", "fr"),
        use_deepfilter=tts.get("use_deepfilter", True),
        tts_speaker_duration=tts.get("speaker_profile_duration", 25.0),
        tts_speaker_skip=tts.get("speaker_profile_skip", 20.0),
        tts_model=tts.get("model", "openbmb/VoxCPM2"),
        tts_max_stretch=tts.get("max_stretch", 1.25),
        tts_min_stretch=tts.get("min_stretch", 0.8),
        tts_cfg_value=tts.get("cfg_value", 2.5),
        tts_inference_timesteps=tts.get("inference_timesteps", 24),
        tts_use_prompt_text=tts.get("use_prompt_text", False),
        output_volume_boost_pct=float(aud.get("volume_boost_pct", 0.0)),
        huggingface_token=(
            t.get("huggingface_token", "")
            or os.environ.get("HF_TOKEN", "")
        ),
        locale=t.get("locale", "fr"),
        glossary_path=t.get("glossary_path", ""),
        segment_merge_gap=tts.get("segment_merge_gap", 1.5),
        segment_merge_max_duration=tts.get("segment_merge_max_duration", 12.0),
        segment_merge_min_duration=tts.get("segment_merge_min_duration", 2.0),
        subtitle_offset_ms=sub.get("sync_offset_ms", 0),
        synthesis_sample_rate=aud.get("synthesis_sample_rate", 48000),
        output_sample_rate=aud.get("output_sample_rate", 48000),
        timeout_seconds=proc.get("timeout_seconds", 7200),
        keep_temp=bool(proc.get("keep_temp", False)),
    )


def setup_logging(log_dir: str, name: str) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S")
    fh = logging.FileHandler(os.path.join(log_dir, f"{name}.log"))
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def free_vram(log: Optional[logging.Logger] = None) -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if log:
            used_gb = torch.cuda.memory_allocated() / 1e9
            log.debug(f"VRAM after cleanup: {used_gb:.1f} GB")


# ============================================================================
# Glossary — Canadian French vocabulary enforcement
# ============================================================================

def load_glossary(path: str, log: logging.Logger) -> "Glossary":
    empty = Glossary(entries=[], formatting_rules=[], inclusive_language=[])
    if not path:
        return empty
    if not os.path.exists(path):
        log.warning(f"Glossary file not found: {path} — continuing without glossary")
        return empty
    try:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        entries = [
            GlossaryEntry(
                en=str(t.get("en", "")),
                fr_ca=str(t.get("fr_ca", "")),
                fr_std=str(t.get("fr_std", "")),
                mode=str(t.get("mode", "suggest")),
                category=str(t.get("category", "")),
                note=str(t.get("note", "")),
            )
            for t in (data.get("terms") or [])
            if t.get("fr_ca")
        ]
        formatting_rules   = [str(r) for r in (data.get("formatting_rules")  or [])]
        inclusive_language = [str(r) for r in (data.get("inclusive_language") or [])]
        glossary = Glossary(
            entries=entries,
            formatting_rules=formatting_rules,
            inclusive_language=inclusive_language,
        )
        log.info(
            f"✓ Glossary loaded: {len(entries)} terms, "
            f"{len(formatting_rules)} formatting rules, "
            f"{len(inclusive_language)} inclusive language rules ({path})"
        )
        return glossary
    except Exception as e:
        log.warning(f"Glossary load failed ({e}) — continuing without glossary")
        return empty


def _build_glossary_section(glossary: "Glossary", locale: str) -> str:
    if locale != "fr-ca" or not glossary.has_content:
        return ""

    blocks: List[str] = []

    if glossary.entries:
        lines = ["MANDATORY VOCABULARY — use Québécois/Canadian French forms:"]
        for e in glossary.entries:
            line = f"  {e.en} → {e.fr_ca}"
            if e.fr_std and e.fr_std.lower() != e.fr_ca.lower():
                line += f"  (NOT: {e.fr_std})"
            if e.note:
                line += f"  [{e.note}]"
            lines.append(line)
        blocks.append("\n".join(lines))

    if glossary.formatting_rules:
        lines = ["FORMATTING RULES (Canadian French / CAPS standards):"]
        for rule in glossary.formatting_rules:
            lines.append(f"  • {rule}")
        blocks.append("\n".join(lines))

    if glossary.inclusive_language:
        lines = ["INCLUSIVE LANGUAGE (CAPS standard — apply to every translation):"]
        for rule in glossary.inclusive_language:
            lines.append(f"  • {rule}")
        blocks.append("\n".join(lines))

    return "\n\n".join(blocks) + "\n"


def _match_case(original: str, replacement: str) -> str:
    if original.isupper():
        return replacement.upper()
    if original and original[0].isupper():
        return replacement[0].upper() + replacement[1:] if replacement else replacement
    return replacement


def apply_glossary(
    segments: List[dict],
    entries: List[GlossaryEntry],
    log: logging.Logger,
    text_keys: Tuple[str, ...] = ("text_fr", "text_fr_natural"),
) -> List[dict]:
    """Apply deterministic 'always'-mode glossary substitutions post-translation."""
    always = [e for e in entries if e.mode == "always"]
    if not always:
        return segments

    out = [dict(s) for s in segments]
    total_subs = 0

    for seg in out:
        for key in text_keys:
            text = seg.get(key)
            if not text:
                continue
            for e in always:
                if e.fr_std:
                    new = re.sub(
                        r"\b" + re.escape(e.fr_std) + r"\b",
                        lambda m, rep=e.fr_ca: _match_case(m.group(), rep),
                        text,
                        flags=re.IGNORECASE,
                    )
                    if new != text:
                        total_subs += 1
                        text = new
                if e.en:
                    new = re.sub(
                        r"\b" + re.escape(e.en) + r"\b",
                        lambda m, rep=e.fr_ca: _match_case(m.group(), rep),
                        text,
                        flags=re.IGNORECASE,
                    )
                    if new != text:
                        total_subs += 1
                        text = new
            seg[key] = text

    log.info(f"✓ Glossary: {total_subs} substitution(s) across {len(out)} segments")
    return out


# ============================================================================
# Step 1: Source Separation — Demucs
# ============================================================================

def separate_vocals(
    video_path: str,
    temp_dir: str,
    model_name: str,
    log: logging.Logger,
) -> Tuple[Optional[str], Optional[str]]:
    """Split speaker vocals from background. Cleaner vocals → better Whisper."""
    log.info(f"[Demucs] Separating vocals — model: {model_name} …")
    try:
        subprocess.run(
            [
                sys.executable, "-m", "demucs",
                "--two-stems", "vocals",
                "-n", model_name,
                "--out", temp_dir,
                "--device", "cuda" if torch.cuda.is_available() else "cpu",
                video_path,
            ],
            check=True, capture_output=True, text=True, timeout=1800,
        )
        vocals_files    = sorted(Path(temp_dir).rglob("vocals.wav"))
        no_vocals_files = sorted(Path(temp_dir).rglob("no_vocals.wav"))

        if not vocals_files:
            log.error("Demucs output vocals.wav not found")
            return None, None

        vocals_path    = str(vocals_files[-1])
        no_vocals_path = str(no_vocals_files[-1]) if no_vocals_files else None
        log.info(f"✓ Vocals separated: {os.path.getsize(vocals_path) / 1e6:.1f} MB")
        return vocals_path, no_vocals_path

    except subprocess.TimeoutExpired:
        log.error("Demucs timed out (30 min) — falling back to raw audio")
    except Exception as e:
        log.error(f"Demucs failed ({e}) — falling back to raw audio")
    return None, None


# ============================================================================
# Step 2: Audio Extraction + Duration
# ============================================================================

def extract_audio(
    video_path: str,
    wav_path: str,
    sample_rate: int,
    log: logging.Logger,
) -> bool:
    log.info(f"Extracting audio → {Path(wav_path).name}")
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", video_path,
                "-vn", "-ac", "1",
                "-ar", str(sample_rate),
                "-acodec", "pcm_s16le",
                wav_path,
            ],
            check=True, capture_output=True, timeout=600,
        )
        log.info(f"✓ Audio extracted: {os.path.getsize(wav_path) / 1e6:.1f} MB")
        return True
    except Exception as e:
        log.error(f"Audio extraction failed: {e}")
        return False


def get_duration(video_path: str, log: logging.Logger) -> float:
    try:
        r = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                video_path,
            ],
            capture_output=True, text=True, timeout=30,
        )
        return float(r.stdout.strip())
    except Exception as e:
        log.warning(f"Could not read duration ({e}), defaulting to 3600 s")
        return 3600.0


# ============================================================================
# Step 3: Transcription — faster-whisper
# ============================================================================

def transcribe_audio(
    wav_path: str,
    model_name: str,
    device: str,
    compute_type: str,
    models_dir: str,
    log: logging.Logger,
) -> Optional[List[dict]]:
    log.info(f"Loading faster-whisper {model_name} [{compute_type}] …")
    try:
        model = WhisperModel(
            model_name,
            device=device,
            compute_type=compute_type,
            download_root=os.path.join(models_dir, "whisper"),
        )
        log.info("Transcribing with VAD filter + word timestamps …")
        segments_gen, info = model.transcribe(
            wav_path,
            language="en",
            beam_size=5,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 300},
            word_timestamps=True,
            initial_prompt="Hello. Welcome to this video. Today we will discuss several important topics.",
        )
        segments = [
            {"id": i, "start": s.start, "end": s.end, "text": s.text.strip()}
            for i, s in enumerate(segments_gen)
            if s.text.strip()
        ]
        log.info(f"✓ {len(segments)} segments ({info.language} detected, {info.duration:.0f} s)")
        del model
        free_vram(log)
        return segments
    except Exception as e:
        log.error(f"Transcription failed: {e}")
        return None


# ============================================================================
# Step 3a: Dedupe Whisper repetition across adjacent segments
# ============================================================================

def _tokenize_for_dedup(text: str) -> List[str]:
    """Word-level tokens, lower-cased and stripped of trailing punctuation."""
    return [w.strip(".,!?;:\"'()[]").lower() for w in text.split() if w.strip(".,!?;:\"'()[]")]


def dedupe_whisper_segments(segments: List[dict], log: logging.Logger) -> List[dict]:
    """Strip word-level overlap between adjacent Whisper segments.

    Whisper's VAD can re-emit the last few words of segment N as the start of
    segment N+1, especially around hesitations or end-of-audio silence. The
    duplicate phrase then survives merge_segments, gets translated twice, and
    appears as a "repeating subtitle" pair in the SRT.

    For each adjacent pair, finds the longest suffix of N's words that matches
    a prefix of N+1's words (≥ 3 words) and trims that prefix from N+1. If the
    overlap eats N+1 entirely, drops N+1 and extends N's end.
    """
    if len(segments) < 2:
        return segments

    out: List[dict] = [dict(segments[0])]
    dropped = 0
    trimmed = 0

    for seg in segments[1:]:
        prev = out[-1]
        prev_words = _tokenize_for_dedup(prev["text"])
        next_raw   = seg["text"].split()
        next_words = _tokenize_for_dedup(seg["text"])

        # Find the longest suffix-of-prev == prefix-of-next match (≥ 3 words).
        max_k = min(len(prev_words), len(next_words), 15)
        best_k = 0
        for k in range(max_k, 2, -1):
            if prev_words[-k:] == next_words[:k]:
                best_k = k
                break

        if best_k == 0:
            out.append(dict(seg))
            continue

        # Trim the duplicated prefix from seg.text. Re-walk next_raw skipping
        # tokens that would lower-case-strip to the duplicated words.
        kept_raw: List[str] = []
        skipped = 0
        for raw in next_raw:
            if skipped < best_k and raw.strip(".,!?;:\"'()[]").lower():
                skipped += 1
                continue
            kept_raw.append(raw)
        trimmed_text = " ".join(kept_raw).strip()

        if not trimmed_text:
            # Pure repetition — drop seg entirely, extend prev's end.
            prev["end"] = max(prev["end"], seg["end"])
            dropped += 1
        else:
            new_seg = dict(seg)
            new_seg["text"] = trimmed_text
            out.append(new_seg)
            trimmed += 1

    for i, s in enumerate(out):
        s["id"] = i

    if trimmed or dropped:
        log.info(
            f"✓ Dedup: trimmed {trimmed} overlapping prefix(es), "
            f"dropped {dropped} fully-repeated segment(s) "
            f"({len(segments)} → {len(out)})"
        )
    else:
        log.debug("✓ Dedup: no adjacent-segment overlap detected")
    return out


# ============================================================================
# Step 3b: Segment merging — into sentence-scale chunks
# ============================================================================

def merge_segments(
    segments: List[dict],
    max_gap: float,
    max_duration: float,
    min_duration: float,
    log: logging.Logger,
) -> List[dict]:
    """Merge Whisper fragments into sentence-scale chunks.

    Strategy:
      - Keep merging across pauses ≤ max_gap until either max_duration is hit
        or we cross min_duration AND hit a sentence-ending punctuation.
      - Never emit a chunk shorter than min_duration: force-merge tiny chunks
        with their previous neighbour as a final sweep.

    This eliminates the sub-second fragments that broke SRT timing in v3 and
    gives the TTS enough context per call for natural prosody.
    """
    if not segments:
        return segments

    merged: List[dict] = []
    current = dict(segments[0])

    for seg in segments[1:]:
        gap          = seg["start"] - current["end"]
        combined_dur = seg["end"]   - current["start"]
        current_dur  = current["end"] - current["start"]
        ends_sent    = bool(re.search(r"[.!?]\s*$", current["text"].rstrip()))

        # Merge if gap small enough, combined stays under cap, AND
        # either we haven't reached min_duration or the last clause didn't end.
        should_merge = (
            gap <= max_gap
            and combined_dur <= max_duration
            and (current_dur < min_duration or not ends_sent)
        )
        if should_merge:
            current["end"]  = seg["end"]
            current["text"] = current["text"].rstrip() + " " + seg["text"].lstrip()
        else:
            merged.append(current)
            current = dict(seg)

    merged.append(current)

    # Final sweep: absorb any still-too-short chunks into the previous chunk.
    if len(merged) > 1:
        cleaned: List[dict] = [merged[0]]
        for chunk in merged[1:]:
            dur = chunk["end"] - chunk["start"]
            prev_dur = cleaned[-1]["end"] - cleaned[-1]["start"]
            if dur < min_duration and (prev_dur + dur) <= max_duration * 1.25:
                cleaned[-1]["end"]  = chunk["end"]
                cleaned[-1]["text"] = cleaned[-1]["text"].rstrip() + " " + chunk["text"].lstrip()
            else:
                cleaned.append(chunk)
        merged = cleaned

    for i, s in enumerate(merged):
        s["id"] = i

    log.info(
        f"✓ Merged {len(segments)} Whisper fragments → {len(merged)} chunks "
        f"(target {min_duration:.0f}s–{max_duration:.0f}s)"
    )
    return merged


# ============================================================================
# Step 3c: Speaker Diarization — pyannote.audio (optional)
# ============================================================================

def diarize_audio(
    wav_path: str,
    model_name: str,
    hf_token: str,
    min_speakers: int,
    max_speakers: int,
    log: logging.Logger,
) -> Optional[List[Tuple[float, float, str]]]:
    """Run pyannote.audio diarization on a mono WAV. Returns (start, end, label) tuples."""
    try:
        from pyannote.audio import Pipeline as PyannotePipeline
    except ImportError:
        log.error("pyannote.audio not installed.  Fix: pip install pyannote.audio")
        return None

    import warnings
    tok_tail = hf_token[-4:] if hf_token else "none"
    try:
        log.info(f"Loading diarization model: {model_name} (HF token …{tok_tail})")
        try:
            audio_info = sf.info(wav_path)
            log.info(
                f"  Input audio: {audio_info.duration:.1f}s, "
                f"{audio_info.samplerate} Hz, {audio_info.channels}ch"
            )
        except Exception as ie:
            log.debug(f"  sf.info failed: {ie}")

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module="pyannote")
            pipeline = PyannotePipeline.from_pretrained(model_name, token=hf_token)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            pipeline.to(device)

            diarize_kwargs: dict = {}
            if min_speakers and min_speakers >= 1:
                diarize_kwargs["min_speakers"] = int(min_speakers)
            if max_speakers and max_speakers >= max(min_speakers or 1, 1):
                diarize_kwargs["max_speakers"] = int(max_speakers)

            log.info(f"  Pyannote params: {diarize_kwargs}")

            # Workaround for pyannote.audio name 'AudioDecoder' is not defined bug:
            # We inject a mock into the pyannote.audio.core.io module if it's missing.
            try:
                import pyannote.audio.core.io as py_io
                if not hasattr(py_io, "AudioDecoder"):
                    class MockAudioDecoder:
                        def __init__(self, *args, **kwargs): pass
                    py_io.AudioDecoder = MockAudioDecoder
            except ImportError:
                pass

            waveform, sample_rate = torchaudio.load(wav_path)
            result = pipeline({"waveform": waveform, "sample_rate": sample_rate}, **diarize_kwargs)

        # pyannote ≥ 3.3 wraps the output in a namedtuple whose annotation field
        # has been renamed across versions — probe known shapes.
        annotation = result
        if not hasattr(annotation, "itertracks"):
            candidate = None
            for attr in ("speaker_diarization", "diarization", "annotation"):
                inner = getattr(result, attr, None)
                if inner is not None and hasattr(inner, "itertracks"):
                    candidate = inner
                    break
            if candidate is None and hasattr(result, "_fields") and result._fields:
                first = getattr(result, result._fields[0])
                if hasattr(first, "itertracks"):
                    candidate = first
            if candidate is None:
                fields = getattr(result, "_fields", None) or dir(result)
                raise RuntimeError(
                    f"unrecognised pyannote output: type={type(result).__name__}, "
                    f"fields={list(fields)[:10]}"
                )
            annotation = candidate

        turns = [
            (turn.start, turn.end, speaker)
            for turn, _, speaker in annotation.itertracks(yield_label=True)
        ]
        speaker_ids = sorted({t[2] for t in turns})
        log.info(f"✓ Diarization complete — {len(speaker_ids)} speaker(s): {speaker_ids}")

        per_speaker: dict = {}
        for (t_start, t_end, spk) in turns:
            per_speaker.setdefault(spk, []).append(t_end - t_start)
        for spk, durs in sorted(per_speaker.items()):
            log.info(
                f"  {spk}: {len(durs)} turns, {sum(durs):.1f}s total, "
                f"longest {max(durs):.1f}s"
            )

        del pipeline
        free_vram(log)
        return turns

    except Exception as e:
        log.error(
            f"Diarization failed: {e}\n"
            f"  HF token tail: …{tok_tail}\n"
            f"  If 401/403: accept the license at https://huggingface.co/{model_name}"
        )
        return None


def assign_speakers(
    segments: List[dict],
    turns: List[Tuple[float, float, str]],
) -> List[dict]:
    """Tag each segment with the speaker that occupies the most of its window."""
    out = [dict(s) for s in segments]
    for seg in out:
        seg_start, seg_end = seg["start"], seg["end"]
        best_spk, best_overlap = "SPEAKER_00", 0.0
        for (t_start, t_end, speaker) in turns:
            if t_end <= seg_start or t_start >= seg_end:
                continue
            overlap = min(t_end, seg_end) - max(t_start, seg_start)
            if overlap > best_overlap:
                best_overlap, best_spk = overlap, speaker
        seg["speaker"] = best_spk
    return out


def build_speaker_profiles(
    vocals_wav: str,
    segments: List[dict],
    temp_dir: str,
    profile_duration: float,
    log: logging.Logger,
    use_deepfilter: bool = True,
    diarization_turns: Optional[List[Tuple[float, float, str]]] = None,
) -> dict:
    """Build a per-speaker voice-clone reference clip from the cleanest turns."""
    from collections import defaultdict

    TARGET_SR = 16000
    MIN_PROFILE_S = 3.0

    log.info(f"  Loading vocals at {TARGET_SR} Hz for profile extraction …")
    try:
        full_audio, _ = librosa.load(vocals_wav, sr=TARGET_SR, mono=True)
    except Exception as e:
        log.error(f"Cannot load vocals for speaker profiles: {e}")
        return {}

    total_s = len(full_audio) / TARGET_SR

    by_speaker: dict = defaultdict(list)
    if diarization_turns:
        for t_start, t_end, spk in diarization_turns:
            by_speaker[spk].append((t_start, t_end))
    else:
        for seg in segments:
            spk = seg.get("speaker", "SPEAKER_00")
            by_speaker[spk].append((float(seg["start"]), float(seg["end"])))

    profiles: dict = {}

    for speaker, windows in by_speaker.items():
        windows = sorted(windows, key=lambda w: w[1] - w[0], reverse=True)
        chunks: List[np.ndarray] = []
        collected = 0.0

        for (w_start, w_end) in windows:
            if collected >= profile_duration:
                break
            s_start = max(0.0, w_start)
            s_end   = min(total_s, w_end)
            dur     = s_end - s_start
            if dur < 0.3:
                continue
            want      = min(dur, profile_duration - collected)
            idx_start = int(s_start * TARGET_SR)
            idx_end   = int((s_start + want) * TARGET_SR)
            chunks.append(full_audio[idx_start:idx_end])
            collected += want

        if not chunks or collected < MIN_PROFILE_S:
            log.warning(
                f"  {speaker}: only {collected:.1f}s available — "
                f"skipping profile (need ≥{MIN_PROFILE_S}s)"
            )
            profiles[speaker] = None
            continue

        combined = np.concatenate(chunks)
        raw_path = os.path.join(temp_dir, f"profile_{speaker}_raw.wav")
        sf.write(raw_path, combined, TARGET_SR)

        if use_deepfilter:
            denoised_path = os.path.join(temp_dir, f"profile_{speaker}_denoised.wav")
            profiles[speaker] = denoise_audio(raw_path, denoised_path, log)
        else:
            profiles[speaker] = raw_path

        log.info(f"  {speaker}: {collected:.1f}s profile built → {profiles[speaker]}")

    return profiles


# ============================================================================
# Step 4: Translation — Qwen3 via Ollama (single pass)
# ============================================================================

_LANG_NAMES = {
    "fr": "French", "es": "Spanish", "de": "German", "it": "Italian",
    "pt": "Portuguese", "nl": "Dutch", "pl": "Polish", "ru": "Russian",
    "ja": "Japanese", "ko": "Korean", "zh": "Chinese", "ar": "Arabic",
    "tr": "Turkish", "hi": "Hindi", "vi": "Vietnamese",
}

_TRANSLATE_PROMPT = """\
You are a professional {language} dubbing translator.
Translate each numbered English segment into natural, conversational {language}
suitable for a dubbed voice-over.

Each segment includes its duration in seconds [N.Ns].
Your translation should be natural and match the speaker's pacing.
Aim for approximately 13-15 characters per second of duration.

RULES:
- Preserve key technical terms and proper nouns.
- Adapt idioms naturally; do not translate literally.
- Use a spoken register (contractions, common phrasing) — not literary French.
- Do NOT excessively shorten the translation; let it flow naturally.
- Output ONLY the numbered translations, one per line, same numbering as input.
- Do NOT add character counts, parentheticals, notes, brackets, or explanations.
{glossary_section}
English segments:
{segments}

{language} translations:"""

_REVIEW_PROMPT = """\
You are a native {language} editor reviewing dubbed video subtitles.{locale_note}
Correct unnatural phrasing, Anglicisms, grammar errors, and register slips.

Each segment includes its duration in seconds [N.Ns].
Ensure the corrected text remains concise enough to fit the timing.

- Output only the corrected numbered list, same numbering as input.
- Do NOT add character counts, parentheticals, notes, brackets, or explanations.
{glossary_section}
{language} subtitles to review:
{segments}

Corrected {language} subtitles:"""


def check_ollama(model: str, log: logging.Logger) -> bool:
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=5)
    except requests.exceptions.ConnectionError:
        log.error(
            "Ollama is NOT running.\n"
            "  Start it: nohup ollama serve > /workspace/logs/ollama.log 2>&1 &\n"
            "  Wait 5 s, then re-run."
        )
        return False

    if r.status_code != 200:
        log.error(f"Ollama returned HTTP {r.status_code}")
        return False

    available  = [m["name"] for m in r.json().get("models", [])]
    model_base = model.split(":")[0]
    if not any(model_base in m for m in available):
        log.error(
            f"Model '{model}' not found.\n"
            f"  Available: {available or ['(none)']}\n"
            f"  Fix: ollama pull {model}"
        )
        return False

    log.info(f"✓ Ollama ready — '{model}' available")
    return True


def _verify_translation_quality(segments: List[dict], log: logging.Logger) -> None:
    unchanged = sum(1 for s in segments if s.get("text_fr") == s["text"])
    pct = 100 * unchanged / max(len(segments), 1)
    if pct > 50:
        log.error(
            f"TRANSLATION FAILURE: {unchanged}/{len(segments)} segments ({pct:.0f}%) "
            f"still in English. Check Ollama and the model is loaded."
        )
    elif unchanged > 0:
        log.warning(f"{unchanged} segment(s) could not be translated — kept in English")


def _ollama_call(prompt: str, model: str, temperature: float, log: logging.Logger) -> Optional[str]:
    # keep_alive=30m pins the model in VRAM between batches.
    try:
        r = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "keep_alive": "30m",
                "options": {"temperature": temperature, "num_predict": 4096},
            },
            timeout=(15, 600),
        )
        if r.status_code == 200:
            return r.json().get("response", "").strip()
        log.warning(f"Ollama HTTP {r.status_code}")
        return None
    except requests.exceptions.ConnectionError:
        log.error("Cannot reach Ollama at localhost:11434.")
        return None
    except requests.exceptions.ReadTimeout:
        log.error(
            "Ollama call timed out after 600s. Check `nvidia-smi` and `ollama ps`."
        )
        return None
    except Exception as e:
        log.error(f"Ollama call failed: {e}")
        return None


def _parse_numbered(text: str, count: int) -> List[str]:
    """Parse numbered LLM output. Strip stray annotations the LLM may emit.

    Cleans:
      - Qwen3 chain-of-thought blocks
      - leading "(MAX N chars)" budget hints from legacy prompts
      - trailing "(20)" character-count self-reports — this is the bug that
        leaked into the v3 SRT
      - trailing "[note]" translator notes
      - surrounding whitespace and stray quotes
    """
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    result: dict = {}
    for line in text.splitlines():
        m = re.match(r"^[\(\[]?(\d+)[\.\)\]]\s+(.*)", line.strip())
        if not m:
            continue
        idx, content = int(m.group(1)), m.group(2).strip()
        content = re.sub(r"^\(MAX\s+\d+\s+chars?\)\s*", "", content, flags=re.IGNORECASE)
        # Clean leading duration hints [12.3s] the LLM may copy from the prompt
        content = re.sub(r"^\[\d+(\.\d+)?s\]\s*", "", content)
        content = re.sub(r"\s*\(\s*\d+\s*\)\s*$", "", content)
        content = re.sub(r"\s*\[[^\]]*\]\s*$", "", content)
        content = content.strip(" \t\"'")
        if 1 <= idx <= count and content:
            result[idx] = content
    return [result.get(i + 1, "") for i in range(count)]


def translate_segments_qwen(
    segments: List[dict],
    model: str,
    temperature: float,
    batch_size: int,
    log: logging.Logger,
    target_lang: str = "fr",
    glossary_section: str = "",
) -> List[dict]:
    """Translate segments via Qwen3 over Ollama. One natural pass.

    No character-budget micromanagement — the audio assembler handles
    overflow with atempo (≤1.25×) and gap-borrowing. Trying to LLM-fit
    each segment to a tight character count was the root cause of the
    summarized / nonsensical output in v3.

    On batch failure, retries the batch once. If individual segments still
    come back empty, retries them one at a time so a single bad line doesn't
    poison the rest of the batch.
    """
    language = _LANG_NAMES.get(target_lang, target_lang.upper())
    out = [dict(s) for s in segments]
    think_prefix = "/no_think\n" if "qwen3" in model.lower() else ""

    log.info(f"Translating {len(segments)} segments with {model} (Ollama) …")

    def _translate(items: List[dict]) -> List[str]:
        if not items:
            return []
        numbered = "\n".join(
            f"{i + 1}. [{s['end'] - s['start']:.1f}s] {s['text']}" 
            for i, s in enumerate(items)
        )
        prompt = think_prefix + _TRANSLATE_PROMPT.format(
            language=language, segments=numbered, glossary_section=glossary_section
        )
        response = _ollama_call(prompt, model, temperature, log)
        if not response:
            return [""] * len(items)
        return _parse_numbered(response, len(items))

    batches = [segments[i : i + batch_size] for i in range(0, len(segments), batch_size)]
    
    def _process_batch(batch: List[dict]) -> List[str]:
        translations = _translate(batch)
        # Retry the whole batch once if mostly empty.
        missing = sum(1 for t in translations if not t)
        if missing > len(batch) // 2:
            translations = _translate(batch)
        # Per-segment fallback for stragglers.
        for i, t in enumerate(translations):
            if not t:
                single = _translate([batch[i]])
                if single and single[0]:
                    translations[i] = single[0]
        return translations

    # Parallel processing of batches
    all_translations = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(tqdm(
            executor.map(_process_batch, batches),
            total=len(batches),
            desc=f"Translating ({target_lang})",
            leave=False
        ))
        for batch_translations in results:
            all_translations.extend(batch_translations)

    for i, text in enumerate(all_translations):
        text = text or out[i]["text"]
        out[i]["text_fr"] = text
        out[i]["text_fr_natural"] = text

    log.info(f"✓ Translation complete ({target_lang})")
    return out


def review_translations(
    segments: List[dict],
    model: str,
    temperature: float,
    log: logging.Logger,
    batch_size: int = 20,
    target_lang: str = "fr",
    locale: str = "fr",
    glossary_section: str = "",
) -> List[dict]:
    """Optional second pass — Qwen3 self-review for register / Anglicisms."""
    log.info(f"Review pass with {model} — {len(segments)} segments …")
    language = _LANG_NAMES.get(target_lang, target_lang.upper())
    locale_note = (
        "\nUse Québécois/Canadian French register throughout "
        "(e.g. courriel, fin de semaine, dîner for lunch, souper for supper)."
        if locale == "fr-ca" else ""
    )
    think_prefix = "/no_think\n" if "qwen3" in model.lower() else ""
    out = [dict(s) for s in segments]

    for start in tqdm(range(0, len(segments), batch_size), desc=f"Reviewing ({target_lang})"):
        batch    = segments[start : start + batch_size]
        numbered = "\n".join(
            f"{i + 1}. [{s['end'] - s['start']:.1f}s] {s.get('text_fr', '')}" 
            for i, s in enumerate(batch)
        )
        prompt = think_prefix + _REVIEW_PROMPT.format(
            language=language,
            segments=numbered,
            locale_note=locale_note,
            glossary_section=glossary_section,
        )
        response = _ollama_call(prompt, model, temperature, log)
        if not response:
            continue
        corrected = _parse_numbered(response, len(batch))
        for i in range(len(batch)):
            if corrected[i]:
                out[start + i]["text_fr"] = corrected[i]
                out[start + i]["text_fr_natural"] = corrected[i]

    log.info("✓ Review complete")
    return out


# ============================================================================
# Step 5: Speaker Reference — extraction + denoising
# ============================================================================

def extract_speaker_sample(
    wav_path: str,
    duration: float,
    output_path: str,
    log: logging.Logger,
    skip_seconds: float = 20.0,
) -> bool:
    """Extract a 25s speaker reference clip skipping past intro music / titles."""
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", wav_path,
                "-ss", str(skip_seconds),
                "-t",  str(duration),
                "-ar", "16000",
                "-ac", "1",
                output_path,
            ],
            check=True, capture_output=True, timeout=60,
        )
        log.info(f"✓ Speaker sample: {duration:.0f} s at {skip_seconds:.0f} s offset")
        return True
    except Exception as e:
        log.error(f"Speaker sample extraction failed: {e}")
        return False


def denoise_audio(
    audio_path: str,
    output_path: str,
    log: logging.Logger,
) -> str:
    """Denoise speaker reference. Tries DeepFilterNet → noisereduce → FFmpeg anlmdn."""
    try:
        from df.enhance import enhance, init_df
        try:
            from df.enhance import load_audio, save_audio
        except ImportError:
            from df.io import load_audio, save_audio

        log.info("Denoising with DeepFilterNet …")
        model, df_state, _ = init_df()
        audio, _  = load_audio(audio_path, sr=df_state.sr())
        enhanced  = enhance(model, df_state, audio)
        save_audio(output_path, enhanced, df_state.sr())
        log.info("✓ Speaker reference denoised (DeepFilterNet)")
        return output_path
    except ImportError:
        log.debug("DeepFilterNet not available — trying noisereduce")
    except Exception as e:
        log.warning(f"DeepFilterNet failed ({e}) — trying noisereduce")

    try:
        import noisereduce as nr
        import soundfile as _sf

        log.info("Denoising with noisereduce …")
        data, rate = _sf.read(audio_path)
        reduced    = nr.reduce_noise(y=data, sr=rate, prop_decrease=0.75)
        _sf.write(output_path, reduced, rate)
        log.info("✓ Speaker reference denoised (noisereduce)")
        return output_path
    except ImportError:
        log.debug("noisereduce not available — trying FFmpeg anlmdn")
    except Exception as e:
        log.warning(f"noisereduce failed ({e}) — trying FFmpeg anlmdn")

    try:
        log.info("Denoising with FFmpeg anlmdn …")
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", audio_path,
                "-af", "anlmdn=s=7:p=0.002:r=0.002:m=15",
                output_path,
            ],
            check=True, capture_output=True, timeout=60,
        )
        log.info("✓ Speaker reference denoised (FFmpeg anlmdn)")
        return output_path
    except Exception as e:
        log.warning(f"FFmpeg anlmdn failed ({e}) — using raw speaker reference")
        return audio_path


def _get_reference_transcript(
    segments: List[dict],
    skip_seconds: float,
    duration: float,
) -> str:
    """Collect the English transcript window matching the speaker reference clip."""
    ref_end = skip_seconds + duration + 5.0
    texts = [
        s["text"]
        for s in segments
        if s["start"] >= skip_seconds - 2.0 and s["end"] <= ref_end
    ]
    return " ".join(texts).strip()


# ============================================================================
# Step 6: TTS Synthesis — VoxCPM2 (only engine)
# ============================================================================

def _seg_text(seg: dict) -> str:
    return (seg.get("text_fr") or seg.get("text") or "").strip()


def synthesize_all_segments(
    segments: List[dict],
    speaker_wav: Optional[str],
    reference_transcript: str,
    config: "PipelineConfig",
    log: logging.Logger,
    speaker_profiles: Optional[dict] = None,
) -> Tuple[List[Tuple[np.ndarray, float, float]], int]:
    """Synthesize every French segment with VoxCPM2."""
    try:
        from voxcpm import VoxCPM
    except ImportError:
        log.error("voxcpm not installed. Install: pip install voxcpm")
        return [], 48000

    def _pick_wav(seg: dict) -> Optional[str]:
        if speaker_profiles:
            profile = speaker_profiles.get(seg.get("speaker", "SPEAKER_00"))
            if profile and os.path.exists(profile):
                return profile
        return speaker_wav

    log.info(f"Loading VoxCPM2: {config.tts_model} …")
    model = VoxCPM.from_pretrained(config.tts_model)
    sr = model.tts_model.sample_rate
    log.info(f"✓ VoxCPM2 ready (output: {sr} Hz)")

    synthesized: List[Tuple[np.ndarray, float, float]] = []
    with tqdm(total=len(segments), desc="Synthesizing (VoxCPM2)") as pbar:
        for seg in segments:
            text = _seg_text(seg)
            if not text:
                pbar.update(1)
                continue
            try:
                ref_wav = _pick_wav(seg)
                kwargs: dict = {
                    "text": text,
                    "cfg_value": config.tts_cfg_value,
                    "inference_timesteps": config.tts_inference_timesteps,
                    # normalize=False protects French diacritics/digits from
                    # VoxCPM2's English-centric text normalizer.
                    "normalize": False,
                    "retry_badcase": True,
                    "retry_badcase_max_times": 3,
                }
                if ref_wav and os.path.exists(ref_wav):
                    kwargs["reference_wav_path"] = ref_wav
                    if config.tts_use_prompt_text and reference_transcript:
                        kwargs["prompt_text"] = reference_transcript
                wav = model.generate(**kwargs)
                synthesized.append((np.array(wav, dtype=np.float32), seg["start"], seg["end"]))
            except Exception as e:
                log.warning(f"Segment {seg['id']} VoxCPM2 failed: {e}")
            pbar.update(1)

    log.info(f"✓ Synthesized {len(synthesized)} segments at {sr} Hz")
    del model
    free_vram(log)
    return synthesized, sr


# ============================================================================
# Step 7: Audio Assembly + Encoding + Background Re-mix
# ============================================================================

_CROSSFADE_MS = 120.0
_FADE_OUT_MS  = 100.0
_FADE_IN_MS   = 30.0


def _trim_silence(audio: np.ndarray, top_db: int = 30) -> np.ndarray:
    """Remove leading and trailing silence from synthesized audio."""
    try:
        # librosa.effects.trim returns (trimmed_audio, index)
        trimmed, _ = librosa.effects.trim(audio, top_db=top_db)
        return trimmed
    except Exception:
        return audio


def _apply_fade_in(audio: np.ndarray, fade_samples: int) -> np.ndarray:
    n = min(fade_samples, len(audio))
    if n <= 0:
        return audio
    out = audio.copy()
    # Cosine fade-in ramp
    ramp = 0.5 * (1.0 - np.cos(np.linspace(0, np.pi, n, dtype=np.float32)))
    out[:n] *= ramp
    return out


def _atempo_stretch(
    audio: np.ndarray,
    target_samples: int,
    src_rate: int,
    max_ratio: float,
    min_ratio: float,
    temp_dir: str,
    log: logging.Logger,
) -> np.ndarray:
    ratio = len(audio) / max(target_samples, 1)
    if 0.98 <= ratio <= 1.02:
        return audio

    ratio   = max(min_ratio, min(max_ratio, ratio))
    tmp_in  = os.path.join(temp_dir, "_at_in.wav")
    tmp_out = os.path.join(temp_dir, "_at_out.wav")
    sf.write(tmp_in, audio, src_rate)

    af = f"atempo={ratio:.4f}" if ratio <= 2.0 else f"atempo=2.0,atempo={ratio / 2:.4f}"
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", tmp_in, "-af", af, tmp_out],
            check=True, capture_output=True, timeout=60,
        )
        stretched, _ = librosa.load(tmp_out, sr=src_rate, mono=True)
        return stretched
    except Exception as e:
        log.debug(f"atempo failed ({e}), truncating instead")
        return audio[:target_samples]
    finally:
        for f in (tmp_in, tmp_out):
            try:
                os.unlink(f)
            except OSError:
                pass


def _apply_fade_out(audio: np.ndarray, fade_samples: int) -> np.ndarray:
    n = min(fade_samples, len(audio))
    if n <= 0:
        return audio
    out = audio.copy()
    ramp = 0.5 * (1.0 + np.cos(np.linspace(0, np.pi, n, dtype=np.float32)))
    out[-n:] *= ramp
    return out


def _equal_power_crossfade(buf: np.ndarray, start: int, new_audio: np.ndarray, xfade_samples: int) -> int:
    """Mix new_audio into buf at `start` with equal-power xfade on overlap."""
    end = min(start + len(new_audio), len(buf))
    n   = end - start
    if n <= 0:
        return 0

    xfade = max(0, min(xfade_samples, n))
    existing = buf[start : start + xfade]
    # Relaxed overlap check: if any signal exists in the buffer, crossfade it.
    has_overlap = xfade > 0 and float(np.max(np.abs(existing))) > 1e-6

    if has_overlap:
        t       = np.linspace(0, 1, xfade, dtype=np.float32)
        fade_out = np.cos(0.5 * np.pi * t)
        fade_in  = np.sin(0.5 * np.pi * t)
        buf[start : start + xfade] = existing * fade_out + new_audio[:xfade] * fade_in
        if n > xfade:
            buf[start + xfade : end] = new_audio[xfade : n]
    else:
        buf[start : end] = new_audio[:n]
    return n


def assemble_and_encode(
    synthesized: List[Tuple[np.ndarray, float, float]],
    total_duration: float,
    wav_path: str,
    aac_path: str,
    src_rate: int,
    out_rate: int,
    max_stretch: float,
    min_stretch: float,
    temp_dir: str,
    log: logging.Logger,
    volume_boost_pct: float = 0.0,
) -> bool:
    """Place each synthesized segment into the timeline.

    Per-segment policy:
      1. Fit in the original window + 1 borrowed gap from the next segment.
      2. If audio still overflows by ≤ max_stretch: atempo stretch.
      3. Otherwise truncate with an 80 ms cosine fade-out.
    Adjacent segments are joined with a 50 ms equal-power crossfade on overlap.
    """
    log.info(
        f"Assembling {len(synthesized)} segments at {src_rate} Hz "
        f"(stretch ≤ {max_stretch:.2f}, crossfade {_CROSSFADE_MS:.0f}ms) …"
    )

    total_samples = int((total_duration + 2) * src_rate)
    assembled     = np.zeros(total_samples, dtype=np.float32)
    ordered       = sorted(synthesized, key=lambda x: x[1])

    xfade_samples = int(_CROSSFADE_MS / 1000.0 * src_rate)
    fade_in_samples = int(_FADE_IN_MS / 1000.0 * src_rate)
    fade_out_samples = int(_FADE_OUT_MS / 1000.0 * src_rate)

    stretched_count = 0
    truncated_count = 0

    for i, (audio, start, end) in enumerate(ordered):
        # 1. Clean the synthesized audio
        audio = _trim_silence(audio)
        
        # 2. Apply micro-fades to prevent sharp cuts
        audio = _apply_fade_in(audio, fade_in_samples)
        audio = _apply_fade_out(audio, fade_in_samples) # Use fade_in_samples for symmetric micro-fade

        start_s  = int(start * src_rate)
        window_s = int((end - start) * src_rate)

        if i + 1 < len(ordered):
            next_start_s = int(ordered[i + 1][1] * src_rate)
            available    = max(window_s, next_start_s - start_s - xfade_samples)
        else:
            available = max(window_s, total_samples - start_s)

        ratio = len(audio) / max(available, 1)
        if ratio > max_stretch:
            # Speed up as much as allowed, then truncate
            audio = _atempo_stretch(audio, int(len(audio) / max_stretch), src_rate, max_stretch, min_stretch, temp_dir, log)
            audio = _apply_fade_out(audio[:available], fade_out_samples)
            truncated_count += 1
            log.debug(f"  segment {i}: truncated ({ratio:.2f}x over budget, available={available / src_rate:.2f}s)")
        elif ratio > 1.02 or ratio < 0.98:
            # Stretch to fit (either speed up or slow down)
            audio = _atempo_stretch(audio, available, src_rate, max_stretch, min_stretch, temp_dir, log)
            stretched_count += 1

        _equal_power_crossfade(assembled, start_s, audio, xfade_samples)

    if stretched_count or truncated_count:
        log.info(
            f"  Overflow/gap handling: {stretched_count} stretched ({min_stretch:.2f}x\u2013{max_stretch:.2f}x), "
            f"{truncated_count} truncated with fade-out"
        )

    peak = np.max(np.abs(assembled))
    if peak > 0:
        assembled *= 0.95 / peak
    if volume_boost_pct:
        gain = 1.0 + volume_boost_pct / 100.0
        assembled = np.clip(assembled * gain, -1.0, 1.0)
        log.info(f"  Applied {volume_boost_pct:+.0f}% volume boost (gain {gain:.2f}×)")

    sf.write(wav_path, assembled, src_rate)
    log.info(f"✓ WAV assembled: {os.path.getsize(wav_path) / 1e6:.1f} MB")

    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", wav_path,
                "-ar", str(out_rate), "-ac", "2",
                "-c:a", "aac", "-b:a", "192k",
                aac_path,
            ],
            check=True, capture_output=True, timeout=300,
        )
        log.info(
            f"✓ AAC encoded: {os.path.getsize(aac_path) / 1e6:.1f} MB "
            f"@ 192 kbps {out_rate} Hz stereo"
        )
        return True
    except Exception as e:
        log.error(f"AAC encoding failed: {e}")
        return False


def remix_with_background(
    french_wav: str,
    no_vocals_wav: str,
    output_aac: str,
    log: logging.Logger,
    bg_gain_db: float = -3.0,
    volume_boost_pct: float = 0.0,
) -> bool:
    """Mix French vocals with original background using sidechain ducking.

    Vocal Chain: highpass (80Hz) + compand (normalize)
    Background Chain: sidechaincompress (ducked by vocals)
    """
    log.info("Re-mixing French vocals with sidechain auto-ducking …")
    voice_gain = 1.0 + (volume_boost_pct or 0.0) / 100.0

    # sidechaincompress:
    #   threshold: level above which compression starts (0.1)
    #   ratio: how much to reduce bg (20:1)
    #   attack/release: timing of ducking (15ms / 400ms)
    filt = (
        f"[0:a]highpass=f=80,compand,volume={voice_gain:.3f},asplit=2[v_f][v_s];"
        f"[1:a]volume={bg_gain_db}dB[bg_pre];"
        "[bg_pre][v_s]sidechaincompress=threshold=0.08:ratio=12:attack=15:release=400[bg_ducked];"
        "[v_f][bg_ducked]amix=inputs=2:duration=first:dropout_transition=0[out]"
    )
    try:
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", french_wav,
                "-i", no_vocals_wav,
                "-filter_complex", filt,
                "-map", "[out]",
                "-ar", "48000", "-ac", "2",
                "-c:a", "aac", "-b:a", "192k",
                output_aac,
            ],
            check=True, capture_output=True, timeout=600,
        )
        log.info(f"✓ Background re-mixed (auto-ducked): {os.path.getsize(output_aac) / 1e6:.1f} MB")
        return True
    except Exception as e:
        log.error(f"Background re-mix failed: {e}")
        return False


# ============================================================================
# Step 8: SRT generation — direct from merged segment timings
# ============================================================================

def retime_segments_to_audio(
    segments: List[dict],
    synthesized: List[Tuple[np.ndarray, float, float]],
    src_rate: int,
    total_duration: float,
    log: logging.Logger,
) -> List[dict]:
    """Match each segment's SRT timing to where its dubbed audio actually plays.

    The audio assembler places each TTS chunk at seg["start"] and lets it
    extend up to the next segment's start (minus a 50 ms crossfade). When the
    TTS is shorter than the original Whisper window, the dubbed audio finishes
    early and the rest of the window is silent. Without this retime, the SRT
    keeps the subtitle on screen through that silence — which the viewer
    perceives as the subtitles "falling behind" the audio.

    Also drops segments whose TTS produced no audio so the SRT doesn't list
    entries with nothing playing underneath them.
    """
    XFADE_S = _CROSSFADE_MS / 1000.0
    syn_by_start: dict = {}
    for audio, start, _end in synthesized:
        syn_by_start[round(start * 1000)] = audio

    by_start = sorted(segments, key=lambda s: s["start"])

    kept: List[dict] = []
    dropped = 0
    tightened = 0

    for i, seg in enumerate(by_start):
        audio = syn_by_start.get(round(seg["start"] * 1000))
        if audio is None or len(audio) == 0:
            dropped += 1
            continue

        if i + 1 < len(by_start):
            available = by_start[i + 1]["start"] - seg["start"] - XFADE_S
        else:
            available = (total_duration + 2.0) - seg["start"]

        tts_dur  = len(audio) / src_rate
        played   = min(tts_dur, max(available, 0.5))
        new_end  = seg["start"] + max(played, 0.5)
        orig_end = seg["end"]

        new_seg = dict(seg)
        new_seg["end"] = new_end
        if abs(new_end - orig_end) > 0.3:
            tightened += 1
        kept.append(new_seg)

    if dropped or tightened:
        log.info(
            f"  SRT retime: tightened {tightened} entries to actual audio length, "
            f"dropped {dropped} entries with no synthesized audio"
        )
    return kept


def _wrap_subtitle(text: str, max_chars: int = 42) -> str:
    if len(text) <= max_chars:
        return text
    words  = text.split()
    lines: List[str] = []
    line:  List[str] = []
    for word in words:
        if line and sum(len(w) for w in line) + len(line) - 1 + 1 + len(word) > max_chars:
            lines.append(" ".join(line))
            line = [word]
        else:
            line.append(word)
    if line:
        lines.append(" ".join(line))
    return "\n".join(lines)


def create_srt(
    segments: List[dict],
    output_path: str,
    log: logging.Logger,
    offset_ms: int = 0,
) -> bool:
    """Write SRT using merged segment timings directly.

    Pick text from: seg["text_fr"] → seg["text"] (English fallback).
    Enforces a 1 s minimum entry duration so subtitles don't flash by.
    Includes a CPS (Characters Per Second) check.
    """
    try:
        offset_s = offset_ms / 1000.0
        subs     = pysrt.SubRipFile()
        high_cps_count = 0

        global_idx = 1
        for seg in segments:
            full_text = _wrap_subtitle(seg.get("text_fr") or seg["text"])
            lines     = full_text.split("\n")
            
            # Split into 2-line blocks
            blocks = ["\n".join(lines[i:i+2]) for i in range(0, len(lines), 2)]
            n_blocks = len(blocks)
            
            seg_start = max(0.0, seg["start"] + offset_s)
            seg_end   = max(seg_start + 1.0, seg["end"] + offset_s)
            seg_dur   = seg_end - seg_start
            
            block_dur = seg_dur / n_blocks
            
            for b_idx, text in enumerate(blocks):
                start = seg_start + (b_idx * block_dur)
                end   = start + block_dur
                
                # CPS Check
                dur = end - start
                if dur > 0:
                    cps = len(text) / dur
                    if cps > 20:
                        high_cps_count += 1
                        log.debug(f"  High CPS ({cps:.1f}) at index {global_idx}: '{text[:30]}...'")

                subs.append(SubRipItem(
                    index=global_idx,
                    start=SubRipTime(seconds=start),
                    end=SubRipTime(seconds=end),
                    text=text,
                ))
                global_idx += 1
        subs.save(output_path, encoding="utf-8")
        log.info(f"✓ SRT: {len(subs)} entries" + (f" (offset {offset_ms:+d} ms)" if offset_ms else ""))
        if high_cps_count > 0:
            log.warning(f"  {high_cps_count} subtitle(s) exceed 20 CPS (Characters Per Second).")
        return True
    except Exception as e:
        log.error(f"SRT creation failed: {e}")
        return False


# ============================================================================
# Main Pipeline
# ============================================================================

def _dump_segments(segments: List[dict], temp_dir: str, label: str, log: logging.Logger) -> None:
    """Write segments to a numbered JSON file for post-run inspection."""
    path = os.path.join(temp_dir, f"segments_{label}.json")
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(segments, f, ensure_ascii=False, indent=2)
        log.debug(f"[keep_temp] {len(segments)} segments → {Path(path).name}")
    except Exception as e:
        log.warning(f"Could not write debug dump {path}: {e}")


def process_video(
    video_path: str,
    output_dir: str,
    config: PipelineConfig,
    log: logging.Logger,
    force: bool = False,
) -> bool:
    name = Path(video_path).stem
    os.makedirs(output_dir, exist_ok=True)
    temp_dir = os.path.join(config.temp_folder, name)
    os.makedirs(temp_dir, exist_ok=True)

    final_aac = os.path.join(output_dir, f"{name}_french.m4a")
    final_srt = os.path.join(output_dir, f"{name}_french.srt")

    if not force and os.path.exists(final_aac) and os.path.exists(final_srt):
        log.info(f"SKIP {name} — outputs exist (use --force to reprocess)")
        return True

    log.info(f"\n{'=' * 60}\nPipeline v4.0: {name}\n{'=' * 60}")

    # Ollama required for translation.
    if not check_ollama(config.translation_model, log):
        return False

    # Load glossary only when running Canadian French.
    glossary = (
        load_glossary(config.glossary_path, log)
        if config.locale == "fr-ca"
        else Glossary([], [], [])
    )
    glossary_section = _build_glossary_section(glossary, config.locale)
    if glossary.has_content:
        always_n  = sum(1 for e in glossary.entries if e.mode == "always")
        suggest_n = sum(1 for e in glossary.entries if e.mode == "suggest")
        log.info(
            f"  Locale: {config.locale} — {always_n} always-substitute, "
            f"{suggest_n} suggest-only terms, "
            f"{len(glossary.formatting_rules)} formatting rules, "
            f"{len(glossary.inclusive_language)} inclusive language rules"
        )

    total_duration = get_duration(video_path, log)

    # ── 1. Source separation ────────────────────────────────────────────────
    vocals_wav:    Optional[str] = None
    no_vocals_wav: Optional[str] = None

    if config.use_demucs:
        log.info("\n[1/6] SOURCE SEPARATION (Demucs)")
        vocals_wav, no_vocals_wav = separate_vocals(
            video_path, temp_dir, config.demucs_model, log
        )

    if not vocals_wav:
        log.info("\n[1/6] EXTRACTING AUDIO")
        raw_wav = os.path.join(temp_dir, f"{name}.wav")
        if not extract_audio(video_path, raw_wav, config.synthesis_sample_rate, log):
            return False
        vocals_wav = raw_wav

    # ── 2. Transcribe + merge into sentence chunks ──────────────────────────
    log.info("\n[2/6] TRANSCRIBING (faster-whisper)")
    segments = transcribe_audio(
        vocals_wav,
        config.whisper_model,
        config.whisper_device,
        config.whisper_compute_type,
        config.models_folder,
        log,
    )
    if not segments:
        return False
    free_vram(log)

    if config.keep_temp:
        _dump_segments(segments, temp_dir, "01_whisper_raw", log)

    segments = dedupe_whisper_segments(segments, log)

    if config.keep_temp:
        _dump_segments(segments, temp_dir, "02_deduped", log)

    segments = merge_segments(
        segments,
        max_gap=config.segment_merge_gap,
        max_duration=config.segment_merge_max_duration,
        min_duration=config.segment_merge_min_duration,
        log=log,
    )

    if config.keep_temp:
        _dump_segments(segments, temp_dir, "03_merged", log)

    # Optional diarization.
    diarization_turns: Optional[List[Tuple[float, float, str]]] = None
    if config.use_diarization:
        log.info("\n[2b/6] SPEAKER DIARIZATION (pyannote.audio)")
        diarization_turns = diarize_audio(
            vocals_wav,
            config.diarization_model,
            config.huggingface_token,
            config.diarization_min_speakers,
            config.diarization_max_speakers,
            log,
        )
        if diarization_turns:
            segments = assign_speakers(segments, diarization_turns)
            speaker_counts: dict = {}
            for seg in segments:
                spk = seg.get("speaker", "?")
                speaker_counts[spk] = speaker_counts.get(spk, 0) + 1
            for spk, n in sorted(speaker_counts.items()):
                log.info(f"  {spk}: {n} segment(s)")
            if config.keep_temp:
                _dump_segments(segments, temp_dir, "04_diarized", log)
        else:
            log.warning("  Diarization failed — all segments assigned to SPEAKER_00")
            for seg in segments:
                seg["speaker"] = "SPEAKER_00"

    # ── 3. Translate ────────────────────────────────────────────────────────
    log.info(f"\n[3/6] TRANSLATING ({config.translation_model} via Ollama)")
    segments = translate_segments_qwen(
        segments,
        config.translation_model,
        config.translation_temperature,
        config.translation_batch_size,
        log,
        target_lang=config.target_lang,
        glossary_section=glossary_section,
    )
    _verify_translation_quality(segments, log)

    if config.keep_temp:
        _dump_segments(segments, temp_dir, "05_translated", log)

    if config.translation_review:
        log.info(f"\n[3b/6] REVIEWING TRANSLATIONS ({config.translation_model})")
        segments = review_translations(
            segments,
            config.translation_model,
            config.translation_temperature,
            log,
            batch_size=config.translation_batch_size,
            target_lang=config.target_lang,
            locale=config.locale,
            glossary_section=glossary_section,
        )
        if config.keep_temp:
            _dump_segments(segments, temp_dir, "06_reviewed", log)

    if glossary.entries:
        log.info("\n[3c/6] APPLYING GLOSSARY (deterministic substitution)")
        segments = apply_glossary(segments, glossary.entries, log)
        if config.keep_temp:
            _dump_segments(segments, temp_dir, "07_glossary", log)

    # ── 4. Speaker reference(s) ─────────────────────────────────────────────
    log.info("\n[4/6] PREPARING SPEAKER REFERENCE(S)")
    speaker_wav: Optional[str] = None
    speaker_profiles: Optional[dict] = None

    raw_speaker_wav = os.path.join(temp_dir, "speaker_raw.wav")
    if extract_speaker_sample(
        vocals_wav,
        config.tts_speaker_duration,
        raw_speaker_wav,
        log,
        skip_seconds=config.tts_speaker_skip,
    ):
        if config.use_deepfilter:
            denoised_wav = os.path.join(temp_dir, "speaker_denoised.wav")
            speaker_wav  = denoise_audio(raw_speaker_wav, denoised_wav, log)
        else:
            speaker_wav = raw_speaker_wav
    else:
        log.warning("Voice cloning disabled — using default VoxCPM2 voice")

    if config.use_diarization and any("speaker" in s for s in segments):
        speaker_profiles = build_speaker_profiles(
            vocals_wav,
            segments,
            temp_dir,
            config.diarization_profile_duration,
            log,
            use_deepfilter=config.use_deepfilter,
            diarization_turns=diarization_turns,
        )
        valid = sum(1 for v in speaker_profiles.values() if v)
        log.info(f"  Built {valid}/{len(speaker_profiles)} speaker profile(s)")

    reference_transcript = _get_reference_transcript(
        segments,
        skip_seconds=config.tts_speaker_skip,
        duration=config.tts_speaker_duration,
    )
    if reference_transcript:
        log.info(
            f"  Reference transcript ({len(reference_transcript)} chars): "
            f"{reference_transcript[:80]}…"
        )

    # ── 5. TTS synthesis ────────────────────────────────────────────────────
    log.info("\n[5/6] SYNTHESIZING FRENCH AUDIO (VoxCPM2)")
    synthesized, actual_sr = synthesize_all_segments(
        segments,
        speaker_wav,
        reference_transcript,
        config,
        log,
        speaker_profiles=speaker_profiles,
    )
    if not synthesized:
        return False
    free_vram(log)

    # ── 6. Assemble + encode + SRT ──────────────────────────────────────────
    log.info("\n[6/6] ASSEMBLING & ENCODING")
    interim_wav = os.path.join(temp_dir, f"{name}_french.wav")

    if not assemble_and_encode(
        synthesized,
        total_duration,
        interim_wav,
        final_aac,
        src_rate=actual_sr,
        out_rate=config.output_sample_rate,
        max_stretch=config.tts_max_stretch,
        min_stretch=config.tts_min_stretch,
        temp_dir=temp_dir,
        log=log,
        volume_boost_pct=config.output_volume_boost_pct,
    ):
        return False

    if config.preserve_background and no_vocals_wav and os.path.exists(no_vocals_wav):
        remixed_aac = os.path.join(output_dir, f"{name}_french_full.m4a")
        if remix_with_background(
            interim_wav, no_vocals_wav, remixed_aac, log,
            volume_boost_pct=config.output_volume_boost_pct,
        ):
            log.info(f"  Full mix (vocals + background): {Path(remixed_aac).name}")

    srt_segments = retime_segments_to_audio(
        segments, synthesized, actual_sr, total_duration, log
    )

    if config.keep_temp:
        _dump_segments(srt_segments, temp_dir, "08_retimed", log)

    create_srt(srt_segments, final_srt, log, offset_ms=config.subtitle_offset_ms)

    if config.keep_temp:
        log.info(f"  [keep_temp] Temp files preserved at: {temp_dir}")
    else:
        shutil.rmtree(temp_dir, ignore_errors=True)

    log.info(f"\n{'=' * 60}")
    log.info(f"DONE: {name}")
    log.info(f"  Audio : {final_aac}")
    log.info(f"  Subs  : {final_srt}")
    log.info(f"{'=' * 60}\n")
    return True


# ============================================================================
# CLI
# ============================================================================

@click.command()
@click.option("--video",      type=click.Path(exists=True), required=True, help="Input video file")
@click.option("--output-dir", default="/workspace/outputs",  help="Output directory")
@click.option("--config", "config_path", default="/workspace/config.yaml",
              type=click.Path(exists=True), help="Path to config.yaml")
@click.option("--force", is_flag=True, default=False, help="Overwrite existing outputs")
@click.option(
    "--locale",
    type=click.Choice(["fr", "fr-ca"], case_sensitive=False),
    default=None,
    help="Output locale. 'fr-ca' loads the Canadian French glossary.",
)
@click.option(
    "--volume-boost",
    type=float,
    default=None,
    help="Boost output loudness by this percent (e.g. 20 → +20%). 0 = off.",
)
@click.option(
    "--keep-temp",
    is_flag=True,
    default=False,
    help="Keep all temp files and dump intermediate segment JSON for debugging.",
)
def main(
    video: str,
    output_dir: str,
    config_path: str,
    force: bool,
    locale: Optional[str],
    volume_boost: Optional[float],
    keep_temp: bool,
) -> None:
    """Dub a video to French (audio track + SRT subtitles).

    Canadian French:
      --locale fr-ca
    """
    try:
        config = load_config(config_path)
    except Exception as e:
        print(f"Config error: {e}", file=sys.stderr)
        sys.exit(1)
    if locale:
        config.locale = locale.lower()
    if volume_boost is not None:
        config.output_volume_boost_pct = float(volume_boost)
    if keep_temp:
        config.keep_temp = True
    log     = setup_logging(config.logs_folder, Path(video).stem)
    success = process_video(video, output_dir, config, log, force=force)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
