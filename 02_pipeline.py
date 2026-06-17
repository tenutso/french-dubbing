#!/usr/bin/env python3
"""
French Dubbing Pipeline v1.0

English video → French audio track + SRT subtitles.

Stack (single, fixed path):
  Source separation : Demucs htdemucs           (vocals + background split)
  Transcription     : faster-whisper large-v3   (word timestamps, VAD)
  Segment merging   : sentence-scale chunks     (8–12s, no sub-second fragments)
  Diarization (opt) : pyannote-audio            (per-speaker voice profiles)
  Translation       : Qwen3:14b via Ollama      (natural pass + English-echo guard)
  Speaker denoising : noisereduce              (clean voice reference)
  TTS               : Coqui XTTS-v2 at 24 kHz   (multilingual voice cloning)
  Assembly          : Rubber Band stretch       (no_drop: never truncate; reading-pace slow-down)
  Subtitles         : hybrid BBC/Netflix shaper (≤2 lines, ≤42 cpl, ≤17 CPS)
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
from typing import Dict, List, Optional, Tuple

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
    # Anti-hallucination tuning (faster-whisper / CT2)
    whisper_condition_on_previous_text: bool = False
    whisper_compression_ratio_threshold: float = 2.2
    whisper_no_speech_threshold: float = 0.6
    whisper_log_prob_threshold: float = -1.0

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
    # Targeted second pass that compresses only segments still over budget.
    # Cheap (only over-budget segments are re-prompted) and high-leverage —
    # eliminates the final ~1–2 stretched segments per dub on info-dense talks.
    translation_compression_pass: bool = True
    target_lang: str = "fr"

    # Speaker reference
    use_deepfilter: bool = True
    tts_speaker_duration: float = 25.0
    tts_speaker_skip: float = 20.0

    # TTS — Coqui XTTS-v2 (Idiap fork)
    # In "anchored" mode tts_max_stretch is the per-group *speed-up* cap used to
    # keep the dub inside the source timeline (1.30 ≈ inaudible on speech).
    tts_max_stretch: float = 1.3
    tts_min_stretch: float = 0.8
    # Timing policy when a group's dub can't fit its time window:
    #   "anchored" → hold the source timeline: speed dense groups up (≤ max_stretch)
    #                and re-anchor drift at each pause, so audio length ≈ source.
    #                Slows a group only when it already fits and has spare room.
    #   "no_drop"  → never speed up; extend the timeline (push later groups back).
    #                No words lost, but output runs progressively longer than source.
    #   "lock"     → keep exact source timing; speed up to max_stretch then
    #                hard-truncate the overflowing tail (legacy behaviour).
    timing_policy: str = "anchored"
    # Reading-speed coupling: groups whose text is denser than this many chars/sec
    # are *slowed* toward this pace so the dub — and the subtitles timed to it —
    # read comfortably. In "anchored" mode the slow-down is capped by the window
    # edge (never pushes past the slot); bounded by tts_max_slowdown.
    tts_reading_cps: float = 16.0
    tts_max_slowdown: float = 1.25   # a group may be stretched at most this much longer
    # Grouped tempo smoothing: segments separated by ≤ this many seconds of
    # original silence share a single uniform stretch ratio, so speed-ups/
    # slow-downs are spread across the group instead of hitting one segment.
    tts_group_gap: float = 0.4
    # Stretch engine: "rubberband" (natural, formant-preserving) or "atempo".
    tts_stretcher: str = "rubberband"
    xtts_model: str = "tts_models/multilingual/multi-dataset/xtts_v2"
    xtts_temperature: float = 0.65
    xtts_length_penalty: float = 1.0
    xtts_repetition_penalty: float = 2.0
    xtts_top_k: int = 50
    xtts_top_p: float = 0.85

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

    # CPS guard — split translated segments whose French CPS exceeds this
    # at a sentence boundary before TTS. 0 disables the split pass.
    cps_split_threshold: float = 21.0
    # Character budget per second handed to Qwen as a per-segment limit.
    # Raise to give the LLM more headroom; lower to force tighter phrasing
    # (tighter = less French expansion = better sync against the fixed timeline).
    translation_budget_cps: int = 15
    # Max iterative compression rounds for segments still over budget. Each round
    # re-prompts only the remaining offenders against their latest text.
    translation_compression_rounds: int = 3

    # SRT — hybrid BBC/Netflix subtitle shaping
    subtitle_offset_ms: int = 0
    # "netflix" (≤42 cpl, ≤17 cps, 0.833-7s), "bbc" (≤37 cpl), or "kapwing"
    # (legacy karaoke single-line fragments, no reading-speed cap).
    subtitle_standard: str = "netflix"
    subtitle_max_cpl: int = 42
    subtitle_max_lines: int = 2
    subtitle_max_cps: float = 17.0
    subtitle_min_dur: float = 0.833
    subtitle_max_dur: float = 7.0
    subtitle_min_gap: float = 0.083
    # Fix #1: max seconds a subtitle may lag its audio to gain reading time in
    # dense speech (re-syncs at pauses). 0 keeps cues locked to the audio.
    subtitle_max_lag: float = 3.0
    # Fix #2: lightly LLM-shorten only the cues still over the reading-speed cap
    # after #1, so dense subtitles become readable with minimal divergence.
    subtitle_condense: bool = True

    # Sample rates
    synthesis_sample_rate: int = 48000
    output_sample_rate: int = 48000

    # Final muxed video — original video stream + dubbed audio (+ subtitles),
    # so sync is verifiable in a single file. Held to the source video length.
    mux_video: bool = True
    burn_subs: bool = False   # True = burn subtitles into the picture (re-encode);
                              # False = soft-embed the SRT track (mov_text, copy video).

    timeout_seconds: int = 7200

    # Debug — keep all temp files and dump intermediate segment JSON
    keep_temp: bool = False

    @property
    def target_locale(self) -> str:
        return self.locale


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
    acronyms: Dict[str, str] = None

    def __post_init__(self):
        if self.acronyms is None:
            self.acronyms = {}

    @property
    def has_content(self) -> bool:
        return bool(self.entries or self.formatting_rules or self.inclusive_language or self.acronyms)


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
    out  = c.get("output", {})
    return PipelineConfig(
        input_folder=p.get("input_folder", "/workspace/videos/input"),
        output_folder=p.get("output_folder", "/workspace/outputs"),
        models_folder=p.get("models_folder", "/workspace/models"),
        logs_folder=p.get("logs_folder", "/workspace/logs"),
        temp_folder=p.get("temp_folder", "/workspace/temp"),
        whisper_model=w.get("model", "large-v3"),
        whisper_device=w.get("device", "cuda"),
        whisper_compute_type=w.get("compute_type", "float16"),
        whisper_condition_on_previous_text=bool(w.get("condition_on_previous_text", False)),
        whisper_compression_ratio_threshold=float(w.get("compression_ratio_threshold", 2.2)),
        whisper_no_speech_threshold=float(w.get("no_speech_threshold", 0.6)),
        whisper_log_prob_threshold=float(w.get("log_prob_threshold", -1.0)),
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
        translation_compression_pass=bool(t.get("compression_pass", True)),
        target_lang=t.get("target_lang", "fr"),
        use_deepfilter=tts.get("use_deepfilter", True),
        tts_speaker_duration=tts.get("speaker_profile_duration", 25.0),
        tts_speaker_skip=tts.get("speaker_profile_skip", 20.0),
        xtts_model=tts.get("xtts_model", "tts_models/multilingual/multi-dataset/xtts_v2"),
        xtts_temperature=tts.get("xtts_temperature", 0.65),
        xtts_length_penalty=tts.get("xtts_length_penalty", 1.0),
        xtts_repetition_penalty=tts.get("xtts_repetition_penalty", 2.0),
        xtts_top_k=tts.get("xtts_top_k", 50),
        xtts_top_p=tts.get("xtts_top_p", 0.85),
        tts_max_stretch=tts.get("max_stretch", 1.3),
        tts_min_stretch=tts.get("min_stretch", 0.8),
        tts_group_gap=tts.get("group_gap", 0.4),
        tts_stretcher=tts.get("stretcher", "rubberband"),
        timing_policy=tts.get("timing_policy", "anchored"),
        tts_reading_cps=float(tts.get("reading_cps", 16.0)),
        tts_max_slowdown=float(tts.get("max_slowdown", 1.25)),
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
        cps_split_threshold=float(tts.get("cps_split_threshold", 21.0)),
        translation_budget_cps=int(t.get("budget_cps", 15)),
        translation_compression_rounds=int(t.get("compression_rounds", 3)),
        subtitle_offset_ms=sub.get("sync_offset_ms", 0),
        subtitle_standard=sub.get("standard", "netflix"),
        subtitle_max_cpl=int(sub.get("max_chars_per_line", 42)),
        subtitle_max_lines=int(sub.get("max_lines", 2)),
        subtitle_max_cps=float(sub.get("max_cps", 17.0)),
        subtitle_min_dur=float(sub.get("min_duration", 0.833)),
        subtitle_max_dur=float(sub.get("max_duration", 7.0)),
        subtitle_min_gap=float(sub.get("min_gap", 0.083)),
        subtitle_max_lag=float(sub.get("max_lag", 3.0)),
        subtitle_condense=bool(sub.get("condense", True)),
        synthesis_sample_rate=aud.get("synthesis_sample_rate", 48000),
        output_sample_rate=aud.get("output_sample_rate", 48000),
        mux_video=bool(out.get("mux_video", True)),
        burn_subs=bool(out.get("burn_subs", False)),
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
    empty = Glossary(entries=[], formatting_rules=[], inclusive_language=[], acronyms={})
    if not path:
        return empty
    if not os.path.exists(path):
        log.warning(f"Glossary file not found: {path} — continuing without glossary")
        return empty
    try:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        # Try new flat format first: glossary: { "speaker": "conférencier·ère", ... }
        glossary_dict = data.get("glossary") or {}
        entries = [
            GlossaryEntry(en=en, fr_ca=fr_ca, fr_std="", mode="suggest", category="", note="")
            for en, fr_ca in glossary_dict.items()
            if fr_ca and str(en) != str(fr_ca)
        ]

        # Fall back to old terms-list format if no entries found
        if not entries:
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

        # 'always' section: {find_form: fr_ca_form} — deterministically rewritten
        # in the output by apply_glossary (find_form may be standard French OR an
        # English term that leaked through). These enforce must-win FR-CA forms.
        always_dict = data.get("always") or {}
        for find_form, fr_ca in always_dict.items():
            if fr_ca and str(find_form) != str(fr_ca):
                entries.append(GlossaryEntry(
                    en="", fr_ca=str(fr_ca), fr_std=str(find_form),
                    mode="always", category="", note="",
                ))

        formatting_rules   = [str(r) for r in (data.get("formatting_rules")  or [])]
        inclusive_language = [str(r) for r in (data.get("inclusive_language") or [])]
        acronyms_dict = {str(k): str(v) for k, v in (data.get("acronyms") or {}).items()}

        glossary = Glossary(
            entries=entries,
            formatting_rules=formatting_rules,
            inclusive_language=inclusive_language,
            acronyms=acronyms_dict,
        )
        log.info(
            f"✓ Glossary loaded: {len(entries)} terms, "
            f"{len(formatting_rules)} formatting rules, "
            f"{len(inclusive_language)} inclusive language rules, "
            f"{len(acronyms_dict)} acronyms ({path})"
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

    if glossary.acronyms:
        lines = ["ACRONYMS — keep these in English exactly as-is (do not translate):"]
        for acronym, full_form in sorted(glossary.acronyms.items()):
            lines.append(f"  {acronym} ({full_form})")
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
    condition_on_previous_text: bool = False,
    compression_ratio_threshold: float = 2.2,
    no_speech_threshold: float = 0.6,
    log_prob_threshold: float = -1.0,
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
        # Anti-hallucination posture:
        #   condition_on_previous_text=False  → don't feed prior (possibly
        #     looped) output back as conditioning; the single biggest switch
        #     against runaway repetition.
        #   compression_ratio_threshold=2.2   → reject segments that are too
        #     compressible (a classic loop signature: "X. X. X.").
        #   no_speech_threshold=0.6           → drop silent windows VAD missed.
        #   log_prob_threshold=-1.0           → keep default, but explicit.
        segments_gen, info = model.transcribe(
            wav_path,
            language="en",
            beam_size=5,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 300},
            word_timestamps=True,
            condition_on_previous_text=condition_on_previous_text,
            compression_ratio_threshold=compression_ratio_threshold,
            no_speech_threshold=no_speech_threshold,
            log_prob_threshold=log_prob_threshold,
            initial_prompt="Hello. Welcome to this video. Today we will discuss several important topics.",
        )
        segments = [
            {
                "id": i,
                "start": s.start,
                "end": s.end,
                "text": s.text.strip(),
                "words": [
                    {"word": w.word, "start": w.start, "end": w.end}
                    for w in (getattr(s, "words", None) or [])
                ],
            }
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
            seg_words = seg.get("words") or []
            if seg_words and best_k <= len(seg_words):
                new_seg["words"] = seg_words[best_k:]
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


def collapse_intrasegment_loops(
    segments: List[dict], log: logging.Logger, min_ngram: int = 5, max_ngram: int = 12,
) -> List[dict]:
    """Collapse hallucinated phrase loops *within* a single segment.

    Pattern: a contiguous run of ≥ min_ngram words appears twice in a row,
    optionally separated by a short connector ("and", "then", "but", ",",
    ".", "I think"). E.g.:
        "I'm going to be married again. And he was like, okay, I'm going
         to be married again."
        "I think that's a really good question. And I think that's a
         really good"

    Legitimate repetition (the same noun phrase used twice with different
    surrounding clauses, e.g. "a Hall of Fame speaker") is NOT touched
    because we require the *entire run* of ≥5 words to match, not just a
    short phrase. The longer the ngram, the safer the collapse.
    """
    out: List[dict] = []
    collapses = 0
    for seg in segments:
        text = seg["text"]
        tokens = re.findall(r"\S+", text)
        if len(tokens) < 2 * min_ngram:
            out.append(seg)
            continue

        def _norm(words: List[str]) -> List[str]:
            return [w.strip(".,!?;:\"'()[]…").lower() for w in words]

        norm = _norm(tokens)
        n = len(tokens)

        # Two collapsible patterns:
        #   (A) Tail truncation: a ≥ min_ngram ngram appears once mid-segment
        #       and again at the very end (last ngram), with the second copy
        #       not followed by ≥ 2 new content words. This is the canonical
        #       Whisper "cut-off" hallucination.
        #   (B) Adjacent duplicate: a ≥ min_ngram ngram appears twice with at
        #       most max_gap tokens between copies. Humans rarely repeat 5+
        #       words verbatim within ~10 s of speech; Whisper does it often.
        max_gap = 8
        i = 0
        kept_idx: List[int] = []
        found = False
        while i < n:
            collapsed = False
            kmax = min(max_ngram, (n - i) // 2)
            for k in range(kmax, min_ngram - 1, -1):
                # Look for a matching copy starting at j ∈ [i+k, i+k+max_gap]
                for gap in range(0, max_gap + 1):
                    j = i + k + gap
                    if j + k > n:
                        break
                    if norm[i:i + k] != norm[j:j + k]:
                        continue
                    # Heuristic guard: don't collapse when the duplicate is
                    # surrounded by clearly distinct content on BOTH sides
                    # (i.e. legitimate recap). "Distinct" = ≥ 4 content words
                    # following the second copy.
                    trailing = n - (j + k)
                    if gap > 3 and trailing >= 4:
                        # Both copies embedded — only collapse for tighter loops
                        continue
                    # Collapse: keep first occurrence + connector tokens,
                    # drop only the duplicate copy. The connector is often
                    # real content (e.g. "question." in "...good question.
                    # And I think that's a really good") that we'd lose if
                    # we dropped it.
                    kept_idx.extend(range(i, i + k))
                    kept_idx.extend(range(i + k, j))   # keep gap/connector
                    i = j + k                          # skip duplicate copy
                    collapsed = True
                    found = True
                    collapses += 1
                    break
                if collapsed:
                    break
            if not collapsed:
                kept_idx.append(i)
                i += 1

        if found:
            new_text = " ".join(tokens[idx] for idx in kept_idx)
            new_seg = dict(seg)
            new_seg["text"] = new_text
            seg_words = seg.get("words") or []
            if seg_words and len(seg_words) == len(tokens):
                new_seg["words"] = [seg_words[idx] for idx in kept_idx]
            out.append(new_seg)
        else:
            out.append(seg)

    if collapses:
        log.info(f"✓ Intra-segment dedup: collapsed {collapses} phrase loop(s)")
    else:
        log.debug("✓ Intra-segment dedup: no hallucinated loops detected")
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
            if seg.get("words"):
                current["words"] = (current.get("words") or []) + seg["words"]
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
                if chunk.get("words"):
                    cleaned[-1]["words"] = (cleaned[-1].get("words") or []) + chunk["words"]
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
You are a professional {language} dubbing translator.{locale_note}
Translate each numbered English segment into natural, conversational {language}
suitable for a dubbed voice-over.

Each segment is tagged [N.Ns, ≤M chars] — the duration and the maximum
character budget for the translation. Stay within the budget so the audio
fits the timing; the budget already accounts for typical {language} expansion.

RULES:
- Preserve key technical terms and proper nouns.
- Adapt idioms naturally; do not translate literally.
- Use a spoken register (contractions, common phrasing) — not literary {language}.
- For info-dense segments, compress to the budget by tightening phrasing —
  keep the meaning, drop filler ("you know", "I mean" / "vous savez", "je veux dire"),
  prefer shorter synonyms. Do NOT summarise or omit content.
- For sparse segments, do NOT pad — match the source length naturally.
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


# Maximum sustainable characters-per-second of French speech. Beyond this,
# even max-stretch can't fit the text — the assembler ends up truncating
# or speeding past intelligibility. Splitting at a sentence boundary lets
# the two halves stretch independently within the same total time window.
_MAX_CPS_BEFORE_SPLIT = 21.0


def split_overflowing_segments(
    segments: List[dict], log: logging.Logger, max_cps: float = _MAX_CPS_BEFORE_SPLIT,
) -> List[dict]:
    """Split translated segments whose French text exceeds max_cps.

    For each over-budget segment, find the sentence boundary closest to
    the character-count midpoint. If one exists between 30% and 70% of
    the text, split the segment into two halves and prorate the time
    window by character count. The TTS then synthesises shorter, more
    stable utterances and the assembler can stretch each half locally.
    """
    SENT_END = re.compile(r"[.!?…»\"'\)]\s+")
    out: List[dict] = []
    splits = 0
    for seg in segments:
        fr = seg.get("text_fr_natural") or seg.get("text_fr") or ""
        dur = seg.get("end", 0) - seg.get("start", 0)
        if dur <= 0 or len(fr) < 80:
            out.append(seg)
            continue
        cps = len(fr) / dur
        if cps <= max_cps:
            out.append(seg)
            continue

        # Find sentence boundaries in the middle band [30%, 70%]
        lo, hi = int(len(fr) * 0.3), int(len(fr) * 0.7)
        candidates = [m.end() for m in SENT_END.finditer(fr) if lo <= m.end() <= hi]
        if not candidates:
            # Fall back to nearest comma in the middle band
            candidates = [m.end() for m in re.finditer(r",\s+", fr) if lo <= m.end() <= hi]
        if not candidates:
            out.append(seg)
            continue

        mid = len(fr) // 2
        cut = min(candidates, key=lambda x: abs(x - mid))
        a_fr = fr[:cut].rstrip()
        b_fr = fr[cut:].lstrip()
        if not a_fr or not b_fr:
            out.append(seg)
            continue

        # Prorate time window by character share
        share = len(a_fr) / (len(a_fr) + len(b_fr))
        cut_t = seg["start"] + dur * share

        # Mirror split on the English source so SRT/translation diagnostics stay coherent
        en = seg.get("text", "") or ""
        en_share = max(1, int(len(en) * share))
        a_en = en[:en_share]
        b_en = en[en_share:]

        base = {k: v for k, v in seg.items() if k not in ("start", "end", "text", "text_fr", "text_fr_natural", "id", "words")}
        a = dict(base, id=seg.get("id"), start=seg["start"], end=cut_t, text=a_en)
        b = dict(base, id=seg.get("id"), start=cut_t, end=seg["end"], text=b_en)
        for key in ("text_fr", "text_fr_natural"):
            if key in seg:
                a[key] = a_fr
                b[key] = b_fr
        seg_words = seg.get("words") or []
        if seg_words:
            cum = 0
            cut_w = len(seg_words)
            for wi, w in enumerate(seg_words):
                cum += len((w.get("word") or "").strip()) + 1
                if cum >= en_share:
                    cut_w = wi + 1
                    break
            a["words"] = seg_words[:cut_w]
            b["words"] = seg_words[cut_w:]
        out.append(a)
        out.append(b)
        splits += 1

    if splits:
        # Re-id sequentially
        for i, s in enumerate(out):
            s["id"] = i
        log.info(f"✓ CPS split: divided {splits} over-budget segment(s) (>{max_cps:.0f} CPS)")
    else:
        log.debug(f"✓ CPS split: no segments exceed {max_cps:.0f} CPS")
    return out


_COMPRESS_PROMPT = """\
You are a professional {language} dubbing editor.

Rewrite each numbered {language} segment so it sounds natural when spoken aloud and fits within its character budget.

Priorities (in order):

1. Preserve the speaker's intent, key message, facts, numbers, names, and technical terms.
2. Use natural spoken {language}, not literal translation.
3. Match the speaker's tone, formality, and emotional intensity.
4. Keep the segment concise enough for dubbing timing.
5. Remove fillers, hedges, repetitions, and unnecessary words when needed.

You may:

* Rephrase sentences.
* Use shorter synonyms.
* Restructure wording.
* Slightly condense non-essential details.

Do not:

* Change facts, numbers, names, or technical terminology.
* Add information not present in the source.
* Exceed the character limit.

Each line is tagged [≤N chars]. Your rewrite must not exceed N characters.

Output ONLY the numbered rewrites, one per line, preserving the original numbering. No explanations or notes.

{language} segments:
{segments}
"""

def compress_overflowing_translations(
    segments: List[dict],
    model: str,
    temperature: float,
    log: logging.Logger,
    budget_cps: int = 17,
    target_lang: str = "fr",
    margin: float = 1.05,
    rounds: int = 3,
) -> List[dict]:
    """Iterative Qwen pass that only touches segments still over budget.

    After the main translation, some segments will still exceed the per-segment
    character budget (info-dense English that resists compression in one shot).
    Each round batches the *remaining* offenders and re-prompts Qwen with a
    tighter brief, compressing against the latest text, so a segment that needs
    several passes to fit gets them. The loop stops early once every segment
    fits or a whole round makes no further progress. Compressions that still
    overshoot are kept iff strictly shorter (better than the original).

    margin: a small relaxation factor on the budget. Default 1.05 means
    "rewrite if FR is >5% over budget" — leaves a buffer for natural variance.
    rounds: maximum number of compression passes (length-aware iteration).
    """
    language = _LANG_NAMES.get(target_lang, target_lang.upper())
    think_prefix = "/no_think\n" if "qwen3" in model.lower() else ""

    def _budget(seg: dict) -> int:
        dur = max(seg["end"] - seg["start"], 0.5)
        return max(40, int(dur * budget_cps))

    out = [dict(s) for s in segments]
    BATCH = 15

    for rnd in range(1, max(1, rounds) + 1):
        # Recompute offenders from the *current* text so converged segments drop
        # out and stubborn ones get another, tighter attempt.
        offenders: List[Tuple[int, int]] = []  # (index, budget)
        for idx, s in enumerate(out):
            fr = s.get("text_fr") or ""
            if not fr:
                continue
            b = _budget(s)
            if len(fr) > int(b * margin):
                offenders.append((idx, b))
        if not offenders:
            if rnd == 1:
                log.debug("✓ Compression pass: no segments over budget")
            else:
                log.info(f"✓ Compression converged after {rnd - 1} round(s)")
            break

        log.info(
            f"Compression round {rnd}/{rounds}: {len(offenders)} segment(s) over "
            f"budget — re-prompting (budget={budget_cps} CPS, margin {int((margin-1)*100)}%)"
        )

        round_applied = 0
        for batch_start in range(0, len(offenders), BATCH):
            chunk = offenders[batch_start:batch_start + BATCH]
            numbered = "\n".join(
                f"{i + 1}. [≤{b} chars] {out[idx].get('text_fr','')}"
                for i, (idx, b) in enumerate(chunk)
            )
            prompt = think_prefix + _COMPRESS_PROMPT.format(language=language, segments=numbered)
            response = _ollama_call(prompt, model, temperature, log)
            if not response:
                continue
            rewrites = _parse_numbered(response, len(chunk))
            applied = 0
            still_over = 0
            for i, (idx, b) in enumerate(chunk):
                new = (rewrites[i] or "").strip()
                old = out[idx].get("text_fr", "")
                if not new:
                    continue
                # Accept only if strictly shorter than the current text — avoids
                # the LLM "rewriting" into longer prose, which has happened on
                # noisy inputs. If it's still over budget, keep it anyway iff
                # it's at least 10% shorter than the current text.
                if len(new) < len(old) and len(new) <= int(b * margin):
                    out[idx]["text_fr"] = new
                    if "text_fr_natural" in out[idx]:
                        out[idx]["text_fr_natural"] = new
                    applied += 1
                elif len(new) < int(len(old) * 0.9):
                    out[idx]["text_fr"] = new
                    if "text_fr_natural" in out[idx]:
                        out[idx]["text_fr_natural"] = new
                    applied += 1
                    still_over += 1
                # else: leave current — the rewrite didn't help
            round_applied += applied
            log.info(
                f"  round {rnd}: compressed {applied}/{len(chunk)} segment(s) "
                f"({still_over} still above budget but shorter)"
            )

        # A whole round that changed nothing won't improve on the next — stop.
        if round_applied == 0:
            log.info(f"  round {rnd}: no further compression possible — stopping")
            break

    return out


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


# English-only function words used to spot segments the LLM left untranslated
# (stochastic Qwen failures occasionally echo the source instead of translating;
# the silent fallback would otherwise dub English with the French voice).
# English-only tokens (deliberately excluding French homographs like "a", "on",
# "son", "par", "si", "ou", "des"…) so the ratio is a clean English signal.
_EN_ECHO_WORDS = {
    "the", "that", "you", "your", "with", "and", "have", "has", "had", "this",
    "they", "them", "what", "when", "would", "could", "should", "about", "there",
    "their", "from", "which", "been", "were", "was", "will", "just", "like",
    "don't", "i'm", "we're", "it's", "that's", "you're", "going", "really",
    "because", "something", "people", "into", "to", "of", "is", "are", "it",
    "we", "he", "she", "want", "get", "got", "do", "did", "does", "can", "not",
    "but", "all", "out", "who", "how", "then", "here", "more", "much", "two",
    "know", "think", "said", "where", "why", "well", "his", "her", "our",
    "these", "those", "very", "over", "after", "before", "while", "yeah",
}


def _looks_untranslated(text: str) -> bool:
    toks = re.findall(r"[a-zA-Z']+", text.lower())
    if len(toks) < 4:
        return False
    hits = sum(1 for t in toks if t in _EN_ECHO_WORDS)
    return hits >= 3 and hits / len(toks) >= 0.18


def _retranslate_leftover_english(
    segments: List[dict], model: str, log: logging.Logger,
    target_lang: str = "fr", locale: str = "fr", glossary_section: str = "",
) -> List[dict]:
    """Catch and re-translate any segment the batch pass left in English.

    A line is suspect if it equals its English source or reads as English by the
    function-word heuristic. Each suspect is re-translated individually with a
    strict single-line prompt; unrecoverable ones are warned about loudly."""
    language = _LANG_NAMES.get(target_lang, target_lang.upper())
    locale_note = (" Use Québécois/Canadian French." if locale == "fr-ca" else "")
    think = "/no_think\n" if "qwen3" in model.lower() else ""
    # A segment needs rescue when its OUTPUT still reads as English, or when the
    # LLM returned the source verbatim AND that source was English to begin with
    # (don't touch already-French source — some clips are partly in French).
    def _suspect(s: dict) -> bool:
        out = s.get("text_fr", "")
        if _looks_untranslated(out):
            return True
        return (out.strip() == s["text"].strip() and _looks_untranslated(s["text"]))

    suspects = [s for s in segments if _suspect(s)]
    if not suspects:
        return segments
    log.info(f"  Re-translating {len(suspects)} segment(s) that came back in English …")
    fixed = 0
    for s in suspects:
        prompt = (
            f"{think}Translate this single line into natural, spoken {language}."
            f"{locale_note}\nOutput ONLY the {language} translation — no quotes, "
            f"no notes.\n\nEnglish: {s['text']}\n{language}:"
        )
        out = _ollama_call(prompt, model, 0.2, log)
        if not out:
            continue
        out = re.sub(r"<think>.*?</think>", "", out, flags=re.DOTALL).strip()
        cand = next((ln.strip().strip('"') for ln in out.splitlines() if ln.strip()), "")
        if cand and cand != s["text"].strip() and not _looks_untranslated(cand):
            s["text_fr"] = cand
            s["text_fr_natural"] = cand
            fixed += 1
    still = sum(1 for s in suspects if _looks_untranslated(s.get("text_fr", "")))
    log.info(f"  ✓ Recovered {fixed}/{len(suspects)} segment(s)")
    if still:
        log.warning(f"  {still} segment(s) still not in {language} after retry — "
                    f"will be dubbed as-is")
    return segments


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


# Budget/length tags the translator is told NOT to emit but sometimes echoes,
# e.g. "[≤92 chars]", "[2.3s, ≤39 chars]", "(MAX 50 chars)", "[≤ 80 caractères]".
_BUDGET_BRACKET_RE = re.compile(
    r"^\s*[\(\[][^\]\)]*(?:≤|<=|chars?|caract[èe]res?)[^\]\)]*[\)\]]\s*",
    re.IGNORECASE,
)
_BUDGET_BARE_RE = re.compile(
    r"^\s*(?:max\s+)?[≤<]=?\s*\d+\s*(?:chars?|caract[èe]res?)\b[\s:.,\-–—]*",
    re.IGNORECASE,
)


def _strip_budget_tag(s: str) -> str:
    """Remove a leaked leading character-budget tag from a translated line."""
    prev = None
    while prev != s:
        prev = s
        s = _BUDGET_BRACKET_RE.sub("", s)
        s = _BUDGET_BARE_RE.sub("", s)
    return s.strip()


def _parse_numbered(text: str, count: int) -> List[str]:
    """Parse numbered LLM output. Multi-line aware: content lines that don't
    begin with a "<n>." marker belong to the *previous* numbered item.

    This matters because Qwen sometimes wraps long translations across two
    lines (especially after punctuation), and the legacy line-by-line parser
    would silently drop the continuation — causing the next "<n+1>." marker
    to inherit content from item n and shift all subsequent items.

    Cleans:
      - Qwen3 chain-of-thought blocks
      - leading "(MAX N chars)" / "[12.3s]" budget hints
      - trailing "(20)" character-count self-reports
      - trailing "[note]" translator notes
      - surrounding whitespace and stray quotes
    """
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    marker_re = re.compile(r"^[\(\[]?(\d+)[\.\)\]]\s+(.*)")

    result: dict = {}
    cur_idx: int = -1
    cur_buf: List[str] = []

    def _flush(idx: int, buf: List[str]) -> None:
        if not (1 <= idx <= count):
            return
        content = " ".join(s.strip() for s in buf if s.strip())
        content = _strip_budget_tag(content)                  # leading [≤N chars] / [2.3s, ≤N chars]
        content = re.sub(r"^\[\d+(\.\d+)?s\]\s*", "", content)  # bare leading [12.3s]
        content = _strip_budget_tag(content)                  # again if a duration tag preceded it
        content = re.sub(r"\s*\(\s*\d+\s*\)\s*$", "", content)
        content = re.sub(r"\s*\[[^\]]*\]\s*$", "", content)
        content = content.strip(" \t\"'")
        if content:
            result[idx] = content

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        m = marker_re.match(stripped)
        if m:
            # Heuristic guard: don't treat "2024." or similar bare years as
            # markers — require the number to be plausible as a 1-based index.
            idx_candidate = int(m.group(1))
            if idx_candidate <= count + 5:
                # New item — flush previous
                if cur_idx >= 1:
                    _flush(cur_idx, cur_buf)
                cur_idx = idx_candidate
                cur_buf = [m.group(2)]
                continue
        # Continuation line (or pre-first-marker preamble we ignore)
        if cur_idx >= 1:
            cur_buf.append(stripped)

    if cur_idx >= 1:
        _flush(cur_idx, cur_buf)

    return [result.get(i + 1, "") for i in range(count)]


def translate_segments_qwen(
    segments: List[dict],
    model: str,
    temperature: float,
    batch_size: int,
    log: logging.Logger,
    target_lang: str = "fr",
    locale: str = "fr",
    glossary_section: str = "",
    budget_cps: int = 17,
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
    locale_note = (
        "\nUse Québécois/Canadian French register throughout "
        "(e.g. courriel, fin de semaine, dîner for lunch, souper for supper)."
        if locale == "fr-ca" else ""
    )
    out = [dict(s) for s in segments]
    think_prefix = "/no_think\n" if "qwen3" in model.lower() else ""

    log.info(f"Translating {len(segments)} segments with {model} (Ollama) …")

    # Char budget per second of audio. 17 CPS is the upper bound of naturally
    # speakable French; the assembler will still stretch up to max_stretch
    # for the segments that overshoot, but most segments now arrive at a
    # speakable rate without any speed-up.
    def _budget(seg: dict) -> int:
        dur = max(seg["end"] - seg["start"], 0.5)
        # Floor at 40 chars so 1-2s segments don't get clipped to nothing.
        return max(40, int(dur * budget_cps))

    def _translate(items: List[dict]) -> List[str]:
        if not items:
            return []
        numbered = "\n".join(
            f"{i + 1}. [{s['end'] - s['start']:.1f}s, ≤{_budget(s)} chars] {s['text']}"
            for i, s in enumerate(items)
        )
        prompt = think_prefix + _TRANSLATE_PROMPT.format(
            language=language, locale_note=locale_note, segments=numbered, glossary_section=glossary_section
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
            f"{i + 1}. [{s['end'] - s['start']:.1f}s, ≤{max(40, int((s['end']-s['start'])*17))} chars] {s.get('text_fr', '')}"
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
    """Denoise speaker reference. Tries noisereduce → FFmpeg anlmdn."""
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


# ============================================================================
# Step 6: TTS Synthesis — Coqui XTTS-v2 (Idiap fork)
# ============================================================================

def _seg_text(seg: dict) -> str:
    return (seg.get("text_fr") or seg.get("text") or "").strip()


# XTTS-v2 hard per-language character cap. The values below are well under
# the model's internal token cap (FR: 273, EN: 250, DE/PL: 248, …) — XTTS
# becomes loop-prone well *before* its hard cap, so we split aggressively
# on sentence boundaries. Empirically, 180 chars/chunk for Romance languages
# eliminates almost all internal looping while keeping prosody continuous.
_XTTS_CHUNK_LIMITS = {
    "fr": 180, "en": 200, "es": 180, "de": 200, "it": 170, "pt": 180,
    "pl": 200, "nl": 200, "ru": 160, "cs": 160, "ar": 150, "tr": 180,
    "hu": 160, "zh": 80, "ja": 80, "ko": 80, "hi": 200,
}


def _split_for_xtts(text: str, limit: int) -> List[str]:
    """Break text into ≤limit-char chunks on natural boundaries.

    Tries sentences first (. ! ? …), then clauses (; : ,), then a hard split.
    Returns chunks that are each ≤ limit chars (with one-char slack for joins).
    """
    text = text.strip()
    if len(text) <= limit:
        return [text]
    import re as _re

    pieces = _re.split(r"(?<=[.!?…])\s+", text)
    out: List[str] = []
    for piece in pieces:
        if len(piece) <= limit:
            out.append(piece)
            continue
        # Sub-split on clause punctuation, then accumulate greedily.
        sub = _re.split(r"(?<=[,;:])\s+", piece)
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


def synthesize_all_segments(
    segments: List[dict],
    speaker_wav: Optional[str],
    config: "PipelineConfig",
    log: logging.Logger,
    speaker_profiles: Optional[dict] = None,
) -> Tuple[List[Tuple[np.ndarray, float, float]], int]:
    """Synthesize every segment with Coqui XTTS-v2."""
    try:
        from TTS.api import TTS
    except ImportError:
        log.error("coqui-tts not installed. Install: pip install 'transformers<5' coqui-tts")
        return [], 24000

    import torch as _torch

    def _pick_wav(seg: dict) -> Optional[str]:
        if speaker_profiles:
            profile = speaker_profiles.get(seg.get("speaker", "SPEAKER_00"))
            if profile and os.path.exists(profile):
                return profile
        return speaker_wav

    lang_code = (config.target_lang or "fr").lower().split("-")[0]
    log.info(f"Loading XTTS-v2: {config.xtts_model} (lang={lang_code}) …")
    # Auto-accept the CPML license non-interactively.
    os.environ.setdefault("COQUI_TOS_AGREED", "1")
    device = "cuda" if _torch.cuda.is_available() else "cpu"
    tts = TTS(config.xtts_model).to(device)
    sr = 24000  # XTTS-v2 native rate
    log.info(f"✓ XTTS-v2 ready (output: {sr} Hz, device: {device})")

    chunk_limit = _XTTS_CHUNK_LIMITS.get(lang_code, 240)
    # 80 ms silence between concatenated chunks of the same segment.
    chunk_gap = np.zeros(int(sr * 0.08), dtype=np.float32)

    def _synth_one(text: str, ref_wav: str) -> np.ndarray:
        return np.asarray(
            tts.tts(
                text=text,
                speaker_wav=ref_wav,
                language=lang_code,
                temperature=config.xtts_temperature,
                length_penalty=config.xtts_length_penalty,
                repetition_penalty=config.xtts_repetition_penalty,
                top_k=config.xtts_top_k,
                top_p=config.xtts_top_p,
                split_sentences=True,
            ),
            dtype=np.float32,
        )

    synthesized: List[Tuple[np.ndarray, float, float]] = []
    with tqdm(total=len(segments), desc="Synthesizing (XTTS-v2)") as pbar:
        for seg in segments:
            text = _seg_text(seg)
            if not text:
                pbar.update(1)
                continue
            try:
                ref_wav = _pick_wav(seg)
                if not ref_wav or not os.path.exists(ref_wav):
                    log.warning(f"Segment {seg['id']}: no speaker reference, skipping")
                    pbar.update(1)
                    continue
                chunks = _split_for_xtts(text, chunk_limit)
                if len(chunks) > 1:
                    log.info(
                        f"Segment {seg['id']}: {len(text)} chars > {chunk_limit} "
                        f"limit → split into {len(chunks)} chunks"
                    )
                parts = [_synth_one(c, ref_wav) for c in chunks]
                wav = parts[0] if len(parts) == 1 else np.concatenate(
                    [p for i, part in enumerate(parts) for p in
                     ((part,) if i == 0 else (chunk_gap, part))]
                )
                synthesized.append((wav, seg["start"], seg["end"]))
            except Exception as e:
                log.warning(f"Segment {seg['id']} XTTS-v2 failed: {e}")
            pbar.update(1)

    log.info(f"✓ Synthesized {len(synthesized)} segments at {sr} Hz")
    del tts
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


def _stretch_audio(
    audio: np.ndarray,
    speed_factor: float,
    src_rate: int,
    temp_dir: str,
    log: logging.Logger,
    stretcher: str = "rubberband",
) -> np.ndarray:
    """Time-stretch `audio` by `speed_factor` (>1 = faster, <1 = slower).

    Uses ffmpeg's rubberband filter by default — formant-preserving, sounds
    more natural than atempo at moderate ratios. Falls back to atempo if the
    rubberband filter isn't available.
    """
    if len(audio) == 0 or 0.98 <= speed_factor <= 1.02:
        return audio

    tmp_in  = os.path.join(temp_dir, "_st_in.wav")
    tmp_out = os.path.join(temp_dir, "_st_out.wav")
    sf.write(tmp_in, audio, src_rate)

    if stretcher == "rubberband":
        af = f"rubberband=tempo={speed_factor:.4f}"
    else:
        af = (
            f"atempo={speed_factor:.4f}"
            if 0.5 <= speed_factor <= 2.0
            else f"atempo=2.0,atempo={speed_factor / 2:.4f}"
        )

    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", tmp_in, "-af", af, tmp_out],
            check=True, capture_output=True, timeout=120,
        )
        stretched, _ = librosa.load(tmp_out, sr=src_rate, mono=True)
        return stretched
    except Exception as e:
        if stretcher == "rubberband":
            log.debug(f"rubberband failed ({e}), falling back to atempo")
            for f in (tmp_in, tmp_out):
                try: os.unlink(f)
                except OSError: pass
            return _stretch_audio(
                audio, speed_factor, src_rate, temp_dir, log, stretcher="atempo"
            )
        log.debug(f"atempo failed ({e}), returning unstretched")
        return audio
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
    group_gap: float = 0.4,
    stretcher: str = "rubberband",
    placements_out: Optional[Dict[int, Tuple[float, float]]] = None,
    timing_policy: str = "no_drop",
    seg_text_chars: Optional[Dict[int, int]] = None,
    read_cps: float = 16.0,
    max_slowdown: float = 1.25,
) -> bool:
    """Place synthesized segments into the timeline with grouped tempo smoothing.

    Consecutive segments separated by <= `group_gap` seconds of original
    silence share a single uniform stretch ratio. The ratio is only applied
    when the group's total audio can't fit in its time window — groups
    that already fit play at natural speed with natural gaps preserved.
    This avoids the noticeable per-segment speed-ups/slow-downs that came
    from stretching each chunk to fill its own window.

    If `placements_out` is provided, it's populated with
    {round(original_start * 1000): (new_start, played_duration)} so the SRT
    retime step can follow the actual audio placement.
    """
    log.info(
        f"Assembling {len(synthesized)} segments at {src_rate} Hz "
        f"(stretch <= {max_stretch:.2f}, group gap {group_gap:.2f}s, "
        f"engine {stretcher}, crossfade {_CROSSFADE_MS:.0f}ms) ..."
    )

    ordered       = [(a, s, e) for (a, s, e) in sorted(synthesized, key=lambda x: x[1]) if len(a) > 0]

    if not ordered:
        log.warning("No synthesized audio to assemble.")
        return False

    xfade_samples    = int(_CROSSFADE_MS / 1000.0 * src_rate)
    fade_in_samples  = int(_FADE_IN_MS / 1000.0 * src_rate)
    fade_out_samples = int(_FADE_OUT_MS / 1000.0 * src_rate)
    xfade_s          = _CROSSFADE_MS / 1000.0

    trimmed = [_trim_silence(a) for a, _, _ in ordered]

    # In no_drop mode the timeline can extend past the source, so size the
    # buffer for the worst case (everything placed back-to-back after the
    # source end). In lock mode the legacy bound is enough.
    policy   = (timing_policy or "anchored").lower()
    anchored = policy == "anchored"
    # Both "anchored" and "no_drop" use cursor-following placement (never
    # truncate); they differ only in how per-group speed is computed below.
    # "anchored" speeds dense groups up (capped at max_stretch) to hold the
    # source timeline; "no_drop" only ever slows, so its timeline can grow.
    no_drop  = policy in ("anchored", "no_drop")
    headroom = sum(len(a) for a in trimmed) if no_drop else 0
    total_samples = int((total_duration + 2) * src_rate) + headroom
    assembled     = np.zeros(total_samples, dtype=np.float32)

    # Build groups: consecutive segments whose original inter-gap <= group_gap.
    groups: List[List[int]] = []
    current = [0]
    for i in range(1, len(ordered)):
        prev_end   = ordered[i - 1][2]
        next_start = ordered[i][1]
        if next_start - prev_end <= group_gap:
            current.append(i)
        else:
            groups.append(current)
            current = [i]
    groups.append(current)

    stretched_groups = 0
    truncated_groups = 0
    natural_groups   = 0
    extended_groups  = 0
    slowed_groups    = 0
    write_cursor_s   = 0   # furthest sample written (no_drop: prevents overlap)

    for g_i, group in enumerate(groups):
        group_start    = ordered[group[0]][1]
        group_orig_end = ordered[group[-1]][2]

        # Window upper bound: extend up to the next group's start (minus
        # crossfade). This multi-gap borrowing lets a dense run lean on the
        # silence before the next run.
        if g_i + 1 < len(groups):
            next_group_start = ordered[groups[g_i + 1][0]][1]
            window_end = max(group_orig_end, next_group_start - xfade_s)
        else:
            window_end = max(group_orig_end, total_duration + 2.0)

        W = max(window_end - group_start, 1e-3)
        A = sum(len(trimmed[idx]) for idx in group) / src_rate

        group_chars = (
            sum(seg_text_chars.get(round(ordered[idx][1] * 1000), 0) for idx in group)
            if seg_text_chars else 0
        )
        if anchored:
            # Hold the source timeline: fit each group into its window W with
            # *bidirectional* stretch. Dense groups are sped up (capped at
            # max_stretch) instead of extending the timeline — this is what
            # stops drift accumulating across a long talk. A group is only
            # *slowed* when it already fits AND there is spare room, so the
            # subtitle reads at ≤ read_cps; the slow-down is bounded by the
            # window edge (W) and max_slowdown so it can never push past its slot.
            if A <= W * 1.02:
                t_read = group_chars / read_cps if (group_chars and read_cps > 0) else 0.0
                t_target = min(max(A, t_read), W, A * max_slowdown) if A > 0 else A
                speed = A / t_target if t_target > 0 else 1.0
                if speed < 0.995:
                    slowed_groups += 1
                else:
                    natural_groups += 1
            else:
                ideal_speed = A / W
                speed = min(ideal_speed, max_stretch)
                stretched_groups += 1
                if ideal_speed > max_stretch:
                    # Can't fully fit even at the cap; the small residual
                    # overrun is absorbed by re-anchoring at the next silence.
                    extended_groups += 1
        elif no_drop:
            # Never speed up (extend the timeline instead). Additionally, when
            # the group's TEXT is denser than read_cps, slow the audio toward
            # the reading pace (bounded by max_slowdown) so the dub — and the
            # subtitles timed to it — read comfortably. speed ≤ 1.0 always.
            t_read = group_chars / read_cps if (group_chars and read_cps > 0) else 0.0
            t_target = min(max(A, t_read), A * max_slowdown) if A > 0 else A
            speed = A / t_target if t_target > 0 else 1.0
            if speed < 0.995:
                slowed_groups += 1
            else:
                natural_groups += 1
        elif A <= W * 1.02:
            speed = 1.0
            natural_groups += 1
        else:
            ideal_speed = A / W
            speed = min(ideal_speed, max_stretch)
            stretched_groups += 1
            if ideal_speed > max_stretch:
                truncated_groups += 1
                log.debug(
                    f"  group {g_i}: ideal speed {ideal_speed:.2f}x exceeds "
                    f"max {max_stretch:.2f}x, will truncate tail"
                )

        audios = []
        for idx in group:
            a = trimmed[idx]
            if speed != 1.0:
                a = _stretch_audio(a, speed, src_rate, temp_dir, log, stretcher)
            a = _apply_fade_in(a, fade_in_samples)
            a = _apply_fade_out(a, fade_in_samples)
            audios.append(a)

        if no_drop:
            # Follow the write cursor so an earlier overrun pushes this group
            # later (timeline extends) instead of overwriting it. When nothing
            # upstream ran long and the group fits, segments keep their natural
            # onsets; otherwise they play back-to-back from the cursor.
            group_floor_s = max(int(group_start * src_rate), write_cursor_s)
            delayed = group_floor_s > int(group_start * src_rate)
            if speed == 1.0 and not delayed:
                for j, idx in enumerate(group):
                    orig_start = ordered[idx][1]
                    start_s = max(int(orig_start * src_rate), write_cursor_s)
                    _equal_power_crossfade(assembled, start_s, audios[j], xfade_samples)
                    if placements_out is not None:
                        placements_out[round(orig_start * 1000)] = (
                            start_s / src_rate, len(audios[j]) / src_rate
                        )
                    write_cursor_s = max(write_cursor_s, start_s + len(audios[j]))
            else:
                cursor_s = group_floor_s
                for j, idx in enumerate(group):
                    a = audios[j]
                    orig_start = ordered[idx][1]
                    # Isochrony: never start a line before its original onset.
                    # If the dub is running ahead (its French compressed shorter
                    # than the source run), wait for the real onset — a natural
                    # pause — instead of racing ahead of the picture. If it is
                    # running behind, play back-to-back from the cursor to catch
                    # up. This keeps every line anchored to the speaker on screen.
                    place_s = max(cursor_s, int(orig_start * src_rate))
                    _equal_power_crossfade(assembled, place_s, a, xfade_samples)
                    if placements_out is not None:
                        placements_out[round(orig_start * 1000)] = (
                            place_s / src_rate, len(a) / src_rate
                        )
                    cursor_s = place_s + len(a)
                write_cursor_s = max(write_cursor_s, cursor_s)
        elif speed == 1.0:
            for j, idx in enumerate(group):
                orig_start = ordered[idx][1]
                start_s = int(orig_start * src_rate)
                _equal_power_crossfade(assembled, start_s, audios[j], xfade_samples)
                if placements_out is not None:
                    placements_out[round(orig_start * 1000)] = (
                        orig_start, len(audios[j]) / src_rate
                    )
        else:
            cursor_s  = int(group_start * src_rate)
            max_end_s = int(window_end * src_rate)
            for j, idx in enumerate(group):
                a = audios[j]
                is_last = (j == len(group) - 1)
                if is_last and cursor_s + len(a) > max_end_s:
                    keep = max(0, max_end_s - cursor_s)
                    a = _apply_fade_out(a[:keep], fade_out_samples)
                placed_start_s = cursor_s
                _equal_power_crossfade(assembled, placed_start_s, a, xfade_samples)
                if placements_out is not None:
                    orig_start = ordered[idx][1]
                    placements_out[round(orig_start * 1000)] = (
                        placed_start_s / src_rate, len(a) / src_rate
                    )
                cursor_s += len(a)

    if anchored:
        total_out = (write_cursor_s / src_rate) if write_cursor_s else 0.0
        drift = total_out - total_duration
        log.info(
            f"  Grouped smoothing [anchored]: {natural_groups} natural, "
            f"{slowed_groups} slowed, {stretched_groups} sped up "
            f"(≤ {max_stretch:.2f}x), {extended_groups} over-cap — output "
            f"{total_out:.1f}s vs source {total_duration:.1f}s (drift {drift:+.1f}s)"
        )
    elif no_drop:
        total_out = (write_cursor_s / src_rate) if write_cursor_s else 0.0
        log.info(
            f"  Grouped smoothing [no_drop]: {natural_groups} natural, "
            f"{slowed_groups} slowed for reading (≥ {1/max_slowdown:.2f}x), "
            f"0 truncated — output {total_out:.1f}s"
        )
    elif stretched_groups or truncated_groups:
        log.info(
            f"  Grouped smoothing: {natural_groups} natural, "
            f"{stretched_groups} stretched (<= {max_stretch:.2f}x), "
            f"{truncated_groups} tail-truncated"
        )
    else:
        log.info(f"  Grouped smoothing: all {natural_groups} groups fit naturally")

    # The no_drop buffer is over-allocated; trim trailing silence so the WAV
    # ends shortly after the last placed sample.
    nz = np.nonzero(np.abs(assembled) > 1e-5)[0]
    if len(nz):
        assembled = assembled[: min(len(assembled), int(nz[-1] + 0.2 * src_rate))]

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


def mux_final_video(
    video_path: str,
    audio_path: str,
    srt_path: str,
    output_path: str,
    log: logging.Logger,
    burn_subs: bool = False,
) -> bool:
    """Mux the original video stream with the dubbed audio (+ subtitles).

    The dubbed audio is now held to the source length (anchored timing), so
    ``-shortest`` only trims the sub-second epsilon. With ``burn_subs`` the SRT
    is rendered into the picture (video is re-encoded); otherwise it is
    soft-embedded as a selectable ``mov_text`` track and the video is copied.
    """
    if burn_subs:
        # Escape the SRT path for the subtitles filter (\, :, ' are special).
        esc = srt_path.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")
        cmd = [
            "ffmpeg", "-y", "-i", video_path, "-i", audio_path,
            "-vf", f"subtitles='{esc}'",
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "libx264", "-preset", "medium", "-crf", "18",
            "-c:a", "aac", "-b:a", "192k", "-shortest", output_path,
        ]
    else:
        # No -shortest here: a soft subtitle track whose last cue ends before the
        # video (trailing no-speech footage) would otherwise truncate the video.
        # The source video length stays authoritative; the dubbed audio is already
        # held to ~that length by the anchored timing policy.
        cmd = [
            "ffmpeg", "-y", "-i", video_path, "-i", audio_path, "-i", srt_path,
            "-map", "0:v:0", "-map", "1:a:0", "-map", "2:0",
            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
            "-c:s", "mov_text", "-metadata:s:s:0", "language=fra",
            output_path,
        ]
    mode = "burned-in" if burn_subs else "soft"
    log.info(f"Muxing final video ({mode} subtitles) …")
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=1800)
        log.info(
            f"✓ Final video: {Path(output_path).name} "
            f"({os.path.getsize(output_path) / 1e6:.1f} MB)"
        )
        return True
    except subprocess.CalledProcessError as e:
        err = (e.stderr or b"").decode("utf-8", "ignore")[-500:]
        log.error(f"Final video mux failed: {err}")
        return False
    except Exception as e:
        log.error(f"Final video mux failed: {e}")
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
    placements: Optional[Dict[int, Tuple[float, float]]] = None,
) -> List[dict]:
    """Match each segment's SRT timing to where its dubbed audio actually plays.

    When `placements` is provided (the assembler's actual placement of each
    segment after grouped tempo smoothing), the SRT follows those positions
    exactly — keeping subtitles synced with the audio even when segments
    were shifted within a stretched group.

    Otherwise falls back to the previous behavior: assume each segment
    starts at its original timestamp and just tighten its end to the actual
    audio length.
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
        key = round(seg["start"] * 1000)
        audio = syn_by_start.get(key)
        if audio is None or len(audio) == 0:
            dropped += 1
            continue

        if placements and key in placements:
            new_start, played = placements[key]
            new_end = new_start + max(played, 0.5)
        else:
            if i + 1 < len(by_start):
                available = by_start[i + 1]["start"] - seg["start"] - XFADE_S
            else:
                available = (total_duration + 2.0) - seg["start"]
            tts_dur = len(audio) / src_rate
            played  = min(tts_dur, max(available, 0.5))
            new_start = seg["start"]
            new_end   = new_start + max(played, 0.5)

        orig_end = seg["end"]
        new_seg = dict(seg)
        new_seg["start"] = new_start
        new_seg["end"]   = new_end
        if abs(new_end - orig_end) > 0.3 or abs(new_start - seg["start"]) > 0.05:
            tightened += 1
        kept.append(new_seg)

    if dropped or tightened:
        log.info(
            f"  SRT retime: tightened {tightened} entries to actual audio length, "
            f"dropped {dropped} entries with no synthesized audio"
        )
    return kept


_PHRASE_HARD_RE = re.compile(r"[.!?…]+(?=\s|$)")
_PHRASE_SOFT_RE = re.compile(r"[,;:](?=\s)")

# French function words that should not be stranded at the end of line 1 of a
# two-line cue (BBC/Netflix: break at the highest syntactic node, never split a
# determiner/preposition from what it governs).
_FR_ORPHANS = {
    "le", "la", "les", "un", "une", "des", "du", "de", "au", "aux", "et", "ou",
    "mais", "donc", "car", "ni", "que", "qui", "à", "dans", "sur", "sous", "par",
    "pour", "avec", "sans", "vers", "chez", "en", "ce", "cette", "ces", "mon",
    "ma", "mes", "ton", "ta", "tes", "son", "sa", "ses", "notre", "votre", "nos",
    "vos", "leur", "leurs", "je", "tu", "il", "elle", "on", "nous", "vous", "ils",
    "elles", "ne", "se", "me", "te", "y", "l", "d", "qu", "n", "s", "c", "j",
}


def _split_into_chunks(text: str, char_cap: int) -> List[str]:
    """Split French text into cue-sized chunks (≤ char_cap chars each).

    Breaks at sentence boundaries first, then clause punctuation, then on word
    boundaries as a last resort. Consecutive pieces are greedily packed up to
    char_cap, and a chunk is closed early at a sentence end so cues align with
    sentences when they fit."""
    text = text.strip()
    if not text:
        return []

    # 1. hard sentence pieces
    pieces: List[str] = []
    last = 0
    for m in _PHRASE_HARD_RE.finditer(text):
        pieces.append(text[last:m.end()].strip()); last = m.end()
    if last < len(text):
        pieces.append(text[last:].strip())
    pieces = [p for p in pieces if p]

    # 2. split any over-long piece at clause punctuation, then by words
    units: List[Tuple[str, bool]] = []   # (text, ends_sentence)
    for p in pieces:
        ends_sent = bool(_PHRASE_HARD_RE.search(p[-2:])) or p[-1:] in ".!?…"
        if len(p) <= char_cap:
            units.append((p, ends_sent)); continue
        sub: List[str] = []
        l2 = 0
        for m in _PHRASE_SOFT_RE.finditer(p):
            sub.append(p[l2:m.end()].strip()); l2 = m.end()
        if l2 < len(p):
            sub.append(p[l2:].strip())
        for k, s in enumerate(sub):
            s = s.strip()
            if not s:
                continue
            last_in_p = (k == len(sub) - 1)
            if len(s) <= char_cap:
                units.append((s, ends_sent and last_in_p))
            else:  # word-level fallback
                cur = ""
                for w in s.split():
                    if cur and len(cur) + 1 + len(w) > char_cap:
                        units.append((cur, False)); cur = w
                    else:
                        cur = f"{cur} {w}".strip()
                if cur:
                    units.append((cur, ends_sent and last_in_p))

    # 3. greedily pack units into chunks; close a chunk at a sentence end
    chunks: List[str] = []
    buf = ""
    for u_text, ends_sent in units:
        if not buf:
            buf = u_text
        elif len(buf) + 1 + len(u_text) <= char_cap:
            buf = f"{buf} {u_text}"
        else:
            chunks.append(buf); buf = u_text
        if ends_sent:
            chunks.append(buf); buf = ""
    if buf:
        chunks.append(buf)
    return [c for c in chunks if c]


def _wrap_two_lines(text: str, max_cpl: int, max_lines: int = 2) -> str:
    """Wrap a cue into ≤ max_lines balanced lines at a logical break.

    Prefers a break near the middle, after punctuation if available, and avoids
    leaving a French function word stranded at the end of a line."""
    text = text.strip()
    if len(text) <= max_cpl or max_lines <= 1:
        return text
    words = text.split()
    # candidate break points (after word i, 0-based) with the length of line 1
    best = None
    cum = 0
    target = len(text) / 2
    for i in range(len(words) - 1):
        cum += len(words[i]) + (1 if i else 0)
        line1_len = cum
        line2_len = len(text) - cum - 1
        if line1_len > max_cpl or line2_len > max_cpl * (max_lines - 1):
            continue
        w = words[i].strip(".,;:!?…").lower()
        penalty = abs(line1_len - target)
        if w in _FR_ORPHANS:
            penalty += max_cpl        # discourage breaking after a function word
        if words[i][-1] in ",;:.!?…":
            penalty -= max_cpl * 0.5   # reward breaking after punctuation
        if best is None or penalty < best[0]:
            best = (penalty, i)
    if best is None:
        # No split keeps both lines ≤ max_cpl. Break at the last word boundary
        # that fits line 1 (accept an over-long line 2 — never split a word).
        cum, idx = 0, None
        for k in range(len(words) - 1):
            cum += len(words[k]) + (1 if k else 0)
            if cum <= max_cpl:
                idx = k
            else:
                break
        if idx is None:  # pathological: first word alone exceeds max_cpl
            return text
        return " ".join(words[: idx + 1]) + "\n" + " ".join(words[idx + 1:])
    i = best[1]
    return " ".join(words[: i + 1]) + "\n" + " ".join(words[i + 1:])


def _enforce_subtitle_timing(cues: List[dict], min_dur: float, max_dur: float,
                             max_cps: float, min_gap: float,
                             max_lag: float = 3.0) -> List[dict]:
    """Allocate readable on-screen time to each cue (fix #1).

    The dub holds the *source* timeline by speeding up dense speech, which
    shortens the audio span of those segments and would otherwise starve their
    subtitles (high CPS, flashing by). This forward cursor pass instead gives
    every cue its reading-time *need* (chars / max_cps, floored at min_dur,
    capped at max_dur) and at least enough time to cover its own audio. A cue
    appears when its audio starts; in dense back-to-back speech it may run past
    the next line's audio onset, so later cues *lag* the audio by a bounded
    amount (`max_lag`) and re-sync wherever a pause lets the cursor catch up —
    the subtitle analogue of the audio anchoring. Cues never overlap and never
    appear before their audio. Cues still too dense after this are handed to the
    condensation pass."""
    if not cues:
        return cues

    def _need(c):
        nchars = len(c["text"].replace("\n", " "))
        reading = nchars / max_cps if max_cps > 0 else min_dur
        return min(max(min_dur, reading), max_dur)

    cursor = 0.0
    for c in cues:
        audio_start = c["start"]
        audio_end = c["end"]
        start = max(audio_start, cursor)
        # Cover the spoken audio AND the reading-time need, capped at max_dur.
        end = min(max(start + _need(c), audio_end), start + max_dur)
        lag = start - audio_start
        if lag > max_lag:
            # Lagged too far behind the audio — trim this cue's tail so the
            # cursor catches up (it gets denser; condensation will shorten it).
            end = max(start + min_dur, end - (lag - max_lag))
        c["start"], c["end"] = start, end
        cursor = end + min_gap
    return cues


def _split_french_phrases(text: str, max_chars: int = 38) -> List[str]:
    text = text.strip()
    if not text:
        return []
    sentences: List[str] = []
    last = 0
    for m in _PHRASE_HARD_RE.finditer(text):
        end = m.end()
        sentences.append(text[last:end].strip())
        last = end
    if last < len(text):
        sentences.append(text[last:].strip())
    sentences = [s for s in sentences if s]

    cues: List[str] = []
    for sent in sentences:
        if len(sent) <= max_chars:
            cues.append(sent)
            continue
        parts: List[str] = []
        last = 0
        for m in _PHRASE_SOFT_RE.finditer(sent):
            end = m.end()
            parts.append(sent[last:end].strip())
            last = end
        if last < len(sent):
            parts.append(sent[last:].strip())
        parts = [p for p in parts if p]
        buf = ""
        for p in parts:
            if not buf:
                buf = p
            elif len(buf) + 1 + len(p) <= max_chars:
                buf = buf + " " + p
            else:
                cues.append(buf)
                buf = p
        if buf:
            cues.append(buf)

    final: List[str] = []
    for cue in cues:
        if len(cue) <= max_chars:
            final.append(cue)
            continue
        line: List[str] = []
        cur = 0
        for w in cue.split():
            add = (1 if line else 0) + len(w)
            if cur + add > max_chars and line:
                final.append(" ".join(line))
                line = [w]
                cur = len(w)
            else:
                line.append(w)
                cur += add
        if line:
            final.append(" ".join(line))
    return final


def _assign_phrase_times(
    fr_cues: List[str],
    seg_start: float,
    seg_end: float,
    words_en: Optional[List[dict]],
) -> List[Tuple[float, float]]:
    if not fr_cues:
        return []
    n = len(fr_cues)
    total_chars = sum(len(c) for c in fr_cues) or 1
    seg_dur = max(seg_end - seg_start, 0.1)

    if not words_en:
        cuts = [seg_start]
        acc = 0
        for c in fr_cues[:-1]:
            acc += len(c)
            cuts.append(seg_start + (acc / total_chars) * seg_dur)
        cuts.append(seg_end)
    else:
        first_s = words_en[0].get("start")
        last_e = words_en[-1].get("end")
        orig_start = float(first_s) if first_s is not None else seg_start
        orig_end_raw = float(last_e) if last_e is not None else seg_end
        orig_end = max(orig_end_raw, orig_start + 0.1)
        orig_dur = orig_end - orig_start

        def _scale(t: float) -> float:
            frac = (t - orig_start) / orig_dur
            frac = max(0.0, min(1.0, frac))
            return seg_start + frac * seg_dur

        en_lens = [len((w.get("word") or "").strip()) for w in words_en]
        en_total = sum(en_lens) or 1

        cuts = [seg_start]
        cum_fr = 0
        for c in fr_cues[:-1]:
            cum_fr += len(c)
            target = cum_fr / total_chars
            cum_en = 0
            anchor = None
            for w, wl in zip(words_en, en_lens):
                cum_en += wl
                if cum_en / en_total >= target:
                    we = w.get("end")
                    anchor = float(we) if we is not None else orig_end
                    break
            if anchor is None:
                anchor = orig_end
            cuts.append(_scale(anchor))
        cuts.append(seg_end)

    for i in range(1, len(cuts)):
        if cuts[i] < cuts[i - 1]:
            cuts[i] = cuts[i - 1]

    return [(cuts[i], cuts[i + 1]) for i in range(n)]


_SUB_CONDENSE_PROMPT = """\
You are editing {language} video subtitles for on-screen readability.
Each numbered line is a subtitle that is too long to read in the time it is shown.
Shorten each line to AT MOST its [≤N chars] budget while keeping the full meaning,
in natural spoken {language}. Drop fillers and redundancy, prefer shorter
synonyms; keep proper nouns, numbers and key technical terms. Do NOT merge or
reorder lines, add notes/brackets, or translate to another language.
Output ONLY the numbered shortened lines, one per line, same numbering.

Subtitles to shorten:
{lines}

Shortened {language} subtitles:"""


def condense_overlong_cues(
    cues: List[dict],
    model: str,
    temperature: float,
    max_cps: float,
    min_dur: float,
    log: logging.Logger,
    target_lang: str = "fr",
    margin: float = 1.08,
) -> List[dict]:
    """Fix #2 — lightly shorten only the cues still over the reading-speed cap.

    Runs after the timing pass, so it touches the *fewest* cues possible: those
    whose French is still denser than `max_cps` even with the bounded lag. Each
    offender is rewritten to fit `floor(duration × max_cps)` characters, keeping
    meaning. Rewrites are accepted only when strictly shorter, so a cue can never
    get longer. Timing is left untouched (the cue already covers its audio)."""
    language = _LANG_NAMES.get(target_lang, target_lang.upper())
    think_prefix = "/no_think\n" if "qwen3" in model.lower() else ""

    def _cps(c) -> float:
        d = c["end"] - c["start"]
        return (len(c["text"].replace("\n", " ")) / d) if d > 0 else 0.0

    offenders = [(i, c) for i, c in enumerate(cues) if _cps(c) > max_cps * margin]
    if not offenders:
        log.debug("✓ Subtitle condensation: all cues within reading speed")
        return cues

    log.info(
        f"Subtitle condensation: {len(offenders)} cue(s) over {max_cps:.0f} CPS "
        f"— shortening to fit reading speed"
    )
    BATCH = 15
    applied = 0
    for bs in range(0, len(offenders), BATCH):
        chunk = offenders[bs:bs + BATCH]
        numbered = "\n".join(
            f"{j + 1}. [≤{max(8, int((c['end'] - c['start']) * max_cps))} chars] "
            f"{c['text'].replace(chr(10), ' ')}"
            for j, (_i, c) in enumerate(chunk)
        )
        prompt = think_prefix + _SUB_CONDENSE_PROMPT.format(language=language, lines=numbered)
        resp = _ollama_call(prompt, model, temperature, log)
        if not resp:
            continue
        rewrites = _parse_numbered(resp, len(chunk))
        for j, (i, c) in enumerate(chunk):
            new = (rewrites[j] or "").strip()
            old = c["text"].replace("\n", " ")
            if new and len(new) < len(old):
                cues[i]["text"] = new
                applied += 1
    log.info(f"  condensed {applied}/{len(offenders)} over-speed cue(s)")
    return cues


def create_srt(
    segments: List[dict],
    output_path: str,
    log: logging.Logger,
    offset_ms: int = 0,
    *,
    standard: str = "netflix",
    max_cpl: int = 42,
    max_lines: int = 2,
    max_cps: float = 17.0,
    min_dur: float = 0.833,
    max_dur: float = 7.0,
    min_gap: float = 0.083,
    max_lag: float = 3.0,
    condense_model: Optional[str] = None,
    condense_temperature: float = 0.3,
    target_lang: str = "fr",
) -> bool:
    """Write broadcast-grade subtitles (hybrid BBC/Netflix) from the dub segments.

    Each segment's French text is split into cue-sized chunks (≤ max_cpl×max_lines
    chars) at sentence→clause→word boundaries, timed to the audio via English
    word anchors, then (fix #1) given readable on-screen time with a bounded
    subtitle lag (`max_lag`), and finally — when `condense_model` is set (fix #2)
    — any cue still above the reading-speed cap is lightly shortened by the LLM.

    standard="kapwing" falls back to the legacy karaoke single-line behaviour.
    """
    if standard == "kapwing":
        return _create_srt_legacy(segments, output_path, log, offset_ms=offset_ms)
    try:
        offset_s = offset_ms / 1000.0
        char_cap = max_cpl * max_lines
        raw: List[dict] = []
        for seg in segments:
            text = (seg.get("text_fr") or seg.get("text") or "").strip()
            if not text:
                continue
            chunks = _split_into_chunks(text, char_cap)
            if not chunks:
                continue
            seg_start = max(0.0, seg["start"] + offset_s)
            seg_end = max(seg_start + 0.1, seg["end"] + offset_s)
            times = _assign_phrase_times(chunks, seg_start, seg_end, seg.get("words") or [])
            for ctext, (s, e) in zip(chunks, times):
                raw.append({"text": ctext, "start": s, "end": e})

        raw.sort(key=lambda c: c["start"])

        # Merge a too-short cue into a contiguous neighbour when the combined
        # text still fits two lines — removes sub-minimum-duration fragments and
        # the zero-gaps between them.
        merged: List[dict] = []
        for c in raw:
            if merged:
                prev = merged[-1]
                gap = c["start"] - prev["end"]
                combined = len(prev["text"]) + 1 + len(c["text"])
                too_short = (prev["end"] - prev["start"] < min_dur
                             or c["end"] - c["start"] < min_dur)
                if combined <= char_cap and gap < 0.4 and too_short:
                    prev["text"] = f"{prev['text']} {c['text']}"
                    prev["end"] = c["end"]
                    continue
            merged.append(dict(c))
        raw = merged

        # Fix #1 — allocate readable on-screen time (bounded subtitle lag).
        _enforce_subtitle_timing(raw, min_dur, max_dur, max_cps, min_gap, max_lag)

        # Fix #2 — only the cues that are *still* too dense after #1 are lightly
        # shortened by the LLM, so the subtitle stays as close to the spoken dub
        # as readability allows.
        if condense_model:
            raw = condense_overlong_cues(
                raw, condense_model, condense_temperature, max_cps, min_dur, log,
                target_lang=target_lang,
            )

        subs = pysrt.SubRipFile()
        over_cps = 0
        for idx, c in enumerate(raw, 1):
            wrapped = _wrap_two_lines(c["text"], max_cpl, max_lines)
            dur = c["end"] - c["start"]
            if dur > 0 and len(c["text"].replace("\n", " ")) / dur > max_cps + 0.5:
                over_cps += 1
            subs.append(SubRipItem(
                index=idx,
                start=SubRipTime(seconds=c["start"]),
                end=SubRipTime(seconds=c["end"]),
                text=wrapped,
            ))
        subs.save(output_path, encoding="utf-8")
        log.info(
            f"✓ SRT: {len(subs)} entries [{standard}, ≤{max_cpl}cpl/{max_lines}ln, "
            f"≤{max_cps:.0f}cps]" + (f" (offset {offset_ms:+d} ms)" if offset_ms else "")
        )
        if over_cps:
            log.warning(f"  {over_cps} cue(s) still exceed {max_cps:.0f} CPS "
                        f"(speaker speech too dense to slow within timing).")
        return True
    except Exception as e:
        log.error(f"SRT creation failed: {e}")
        return False


def _create_srt_legacy(
    segments: List[dict],
    output_path: str,
    log: logging.Logger,
    offset_ms: int = 0,
    max_chars: int = 38,
    min_dur: float = 0.5,
) -> bool:
    """Write SRT as phrase-scale single-line cues with word-anchored timing.

    Each merged segment is re-split into short French phrase cues at
    punctuation boundaries. Cue boundaries within a segment are anchored
    to English word timestamps (preserved from Whisper) by character share.
    """
    try:
        offset_s = offset_ms / 1000.0
        subs = pysrt.SubRipFile()
        high_cps_count = 0
        global_idx = 1

        for seg in segments:
            text = (seg.get("text_fr") or seg.get("text") or "").strip()
            if not text:
                continue
            cues = _split_french_phrases(text, max_chars=max_chars)
            if not cues:
                continue

            seg_start = max(0.0, seg["start"] + offset_s)
            seg_end = max(seg_start + 1.0, seg["end"] + offset_s)
            words_en = seg.get("words") or []
            times = _assign_phrase_times(cues, seg_start, seg_end, words_en)

            # Cap merge length so we don't reassemble two-line monsters.
            merge_char_cap = int(max_chars * 1.3)
            packed_cues: List[str] = []
            packed_times: List[Tuple[float, float]] = []
            i = 0
            while i < len(cues):
                cue_text = cues[i]
                start, end = times[i]
                while (
                    end - start < min_dur
                    and i + 1 < len(cues)
                    and len(cue_text) + 1 + len(cues[i + 1]) <= merge_char_cap
                ):
                    i += 1
                    cue_text = cue_text + " " + cues[i]
                    end = times[i][1]
                packed_cues.append(cue_text)
                packed_times.append((start, end))
                i += 1
            if len(packed_cues) >= 2:
                last_s, last_e = packed_times[-1]
                if (
                    last_e - last_s < min_dur
                    and len(packed_cues[-2]) + 1 + len(packed_cues[-1]) <= merge_char_cap
                ):
                    prev_s, _ = packed_times[-2]
                    packed_cues[-2] = packed_cues[-2] + " " + packed_cues[-1]
                    packed_times[-2] = (prev_s, last_e)
                    packed_cues.pop(); packed_times.pop()

            for cue_text, (start, end) in zip(packed_cues, packed_times):
                dur = end - start
                if dur > 0:
                    cps = len(cue_text) / dur
                    if cps > 20:
                        high_cps_count += 1
                        log.debug(
                            f"  High CPS ({cps:.1f}) at index {global_idx}: "
                            f"'{cue_text[:30]}...'"
                        )
                subs.append(SubRipItem(
                    index=global_idx,
                    start=SubRipTime(seconds=start),
                    end=SubRipTime(seconds=end),
                    text=cue_text,
                ))
                global_idx += 1

        subs.save(output_path, encoding="utf-8")
        log.info(
            f"✓ SRT: {len(subs)} entries"
            + (f" (offset {offset_ms:+d} ms)" if offset_ms else "")
        )
        if high_cps_count > 0:
            log.warning(
                f"  {high_cps_count} subtitle(s) exceed 20 CPS (Characters Per Second)."
            )
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

    log.info(f"\n{'=' * 60}\nPipeline v1.0: {name}\n{'=' * 60}")

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
        condition_on_previous_text=config.whisper_condition_on_previous_text,
        compression_ratio_threshold=config.whisper_compression_ratio_threshold,
        no_speech_threshold=config.whisper_no_speech_threshold,
        log_prob_threshold=config.whisper_log_prob_threshold,
    )
    if not segments:
        return False
    free_vram(log)

    if config.keep_temp:
        _dump_segments(segments, temp_dir, "01_whisper_raw", log)

    segments = dedupe_whisper_segments(segments, log)

    if config.keep_temp:
        _dump_segments(segments, temp_dir, "02_deduped", log)

    segments = collapse_intrasegment_loops(segments, log)

    if config.keep_temp:
        _dump_segments(segments, temp_dir, "02b_loop_collapsed", log)

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
        locale=config.target_locale,
        glossary_section=glossary_section,
        budget_cps=config.translation_budget_cps,
    )
    _verify_translation_quality(segments, log)
    segments = _retranslate_leftover_english(
        segments, config.translation_model, log,
        target_lang=config.target_lang, locale=config.locale,
        glossary_section=glossary_section,
    )

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

    # Compression fallback: targeted second pass on segments still over budget.
    if config.translation_compression_pass:
        log.info("\n[3d/6] COMPRESSING OVER-BUDGET SEGMENTS")
        segments = compress_overflowing_translations(
            segments,
            config.translation_model,
            config.translation_temperature,
            log,
            budget_cps=config.translation_budget_cps,
            target_lang=config.target_lang,
            rounds=config.translation_compression_rounds,
        )
        if config.keep_temp:
            _dump_segments(segments, temp_dir, "06b_compressed", log)

    if glossary.entries:
        log.info("\n[3c/6] APPLYING GLOSSARY (deterministic substitution)")
        segments = apply_glossary(segments, glossary.entries, log)
        if config.keep_temp:
            _dump_segments(segments, temp_dir, "07_glossary", log)

    # CPS guard: split segments whose final French text would force the
    # assembler past max_stretch. Splits at sentence boundary; halves
    # are stretched independently and end up closer to natural rate.
    if config.cps_split_threshold > 0:
        segments = split_overflowing_segments(
            segments, log, max_cps=config.cps_split_threshold,
        )
        if config.keep_temp:
            _dump_segments(segments, temp_dir, "07b_cps_split", log)

    # Defensive sweep: strip any leaked "[≤N chars]" budget tag the LLM may have
    # echoed, so it never reaches the TTS or the SRT. (Root cause is handled in
    # _parse_numbered; this guarantees it for every path.)
    _tag_hits = 0
    for s in segments:
        for k in ("text_fr", "text_fr_natural"):
            v = s.get(k)
            if v:
                cleaned = _strip_budget_tag(v)
                if cleaned != v:
                    s[k] = cleaned
                    _tag_hits += 1
    if _tag_hits:
        log.warning(f"  Stripped {_tag_hits} leaked character-budget tag(s) from translations")

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
        log.warning("Voice cloning disabled — no speaker reference available")

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

    # ── 5. TTS synthesis ────────────────────────────────────────────────────
    log.info("\n[5/6] SYNTHESIZING FRENCH AUDIO (XTTS-v2)")
    synthesized, actual_sr = synthesize_all_segments(
        segments,
        speaker_wav,
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

    placements: Dict[int, Tuple[float, float]] = {}
    seg_text_chars = {
        round(s["start"] * 1000): len((s.get("text_fr") or "").replace("\n", " "))
        for s in segments
    }
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
        group_gap=config.tts_group_gap,
        stretcher=config.tts_stretcher,
        placements_out=placements,
        timing_policy=config.timing_policy,
        seg_text_chars=seg_text_chars,
        read_cps=config.tts_reading_cps,
        max_slowdown=config.tts_max_slowdown,
    ):
        return False

    mux_audio = final_aac   # voice-only by default; full mix preferred when made
    if config.preserve_background and no_vocals_wav and os.path.exists(no_vocals_wav):
        remixed_aac = os.path.join(output_dir, f"{name}_french_full.m4a")
        if remix_with_background(
            interim_wav, no_vocals_wav, remixed_aac, log,
            volume_boost_pct=config.output_volume_boost_pct,
        ):
            log.info(f"  Full mix (vocals + background): {Path(remixed_aac).name}")
            mux_audio = remixed_aac

    srt_segments = retime_segments_to_audio(
        segments, synthesized, actual_sr, total_duration, log,
        placements=placements,
    )

    if config.keep_temp:
        _dump_segments(srt_segments, temp_dir, "08_retimed", log)

    create_srt(
        srt_segments, final_srt, log,
        offset_ms=config.subtitle_offset_ms,
        standard=config.subtitle_standard,
        max_cpl=config.subtitle_max_cpl,
        max_lines=config.subtitle_max_lines,
        max_cps=config.subtitle_max_cps,
        min_dur=config.subtitle_min_dur,
        max_dur=config.subtitle_max_dur,
        min_gap=config.subtitle_min_gap,
        max_lag=config.subtitle_max_lag,
        condense_model=(config.translation_model if config.subtitle_condense else None),
        condense_temperature=config.translation_temperature,
        target_lang=config.target_lang,
    )

    final_mp4: Optional[str] = None
    if config.mux_video:
        candidate = os.path.join(output_dir, f"{name}_french.mp4")
        if mux_final_video(
            video_path, mux_audio, final_srt, candidate, log,
            burn_subs=config.burn_subs,
        ):
            final_mp4 = candidate

    if config.keep_temp:
        log.info(f"  [keep_temp] Temp files preserved at: {temp_dir}")
    else:
        shutil.rmtree(temp_dir, ignore_errors=True)

    log.info(f"\n{'=' * 60}")
    log.info(f"DONE: {name}")
    log.info(f"  Audio : {final_aac}")
    log.info(f"  Subs  : {final_srt}")
    if final_mp4:
        log.info(f"  Video : {final_mp4}")
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
