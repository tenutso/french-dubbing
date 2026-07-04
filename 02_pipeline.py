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
  TTS               : F5-TTS at 24 kHz            (multilingual flow-matching voice cloning)
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
import types
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

from metrics import MetricsCollector


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
    # Source language ("en", "fr", …). Empty = auto-detect per file — use for
    # bilingual sources so French passages aren't forced through English decoding.
    whisper_language: str = "en"
    # Optional domain-vocabulary prompt (proper nouns, acronyms). Empty = none.
    # Full sentences here get echoed into the transcript over silence/music —
    # keep it to a comma-separated term list if used at all.
    whisper_initial_prompt: str = ""
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
    diarization_profile_duration: float = 12.0

    # Translation — Qwen via Ollama (single backend)
    translation_model: str = "mistral-small:22b"
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
    tts_speaker_duration: float = 12.0
    tts_speaker_skip: float = 20.0
    # Reference clips are fed to F5-TTS for voice cloning. F5-TTS works at 24 kHz,
    # so references must be ≥24 kHz or the cloned voice inherits the band-limit and
    # sounds muffled/"underwater". reference_denoise_strength is noisereduce's
    # prop_decrease (0=off, 1=max); 0.5 is gentle enough to avoid watery artifacts.
    tts_speaker_profile_sr: int = 24000
    tts_reference_denoise_strength: float = 0.5
    voices_dir: str = "/workspace/voices"

    # TTS — F5-TTS (flow-matching zero-shot voice cloning)
    # In "anchored" mode tts_max_stretch is the per-group *speed-up* cap used to
    # keep the dub inside the source timeline (1.30 ≈ inaudible on speech).
    tts_max_stretch: float = 1.3
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
    f5tts_model: str = "F5TTS_v1_Multilingual"
    f5tts_nfe_step: int = 32
    f5tts_cfg_strength: float = 2.0
    f5tts_speed: float = 1.0

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

    # Metrics — emit per-phase CSV instrumentation to {output}/{name}_metrics/
    # for tuning translation length/timing fit. See metrics.py.
    metrics_enabled: bool = False

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
    met  = c.get("metrics", {})
    return PipelineConfig(
        input_folder=p.get("input_folder", "/workspace/videos/input"),
        output_folder=p.get("output_folder", "/workspace/outputs"),
        models_folder=p.get("models_folder", "/workspace/models"),
        logs_folder=p.get("logs_folder", "/workspace/logs"),
        temp_folder=p.get("temp_folder", "/workspace/temp"),
        whisper_model=w.get("model", "large-v3"),
        whisper_device=w.get("device", "cuda"),
        whisper_compute_type=w.get("compute_type", "float16"),
        whisper_language=str(w.get("language", "en") or ""),
        whisper_initial_prompt=str(w.get("initial_prompt", "") or ""),
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
        translation_model=t.get("model", "mistral-small:22b"),
        translation_temperature=t.get("temperature", 0.3),
        translation_batch_size=t.get("batch_size", 20),
        translation_review=t.get("review_pass", False),
        translation_compression_pass=bool(t.get("compression_pass", True)),
        target_lang=t.get("target_lang", "fr"),
        use_deepfilter=tts.get("use_deepfilter", True),
        tts_speaker_duration=tts.get("speaker_profile_duration", 25.0),
        tts_speaker_skip=tts.get("speaker_profile_skip", 20.0),
        tts_speaker_profile_sr=int(tts.get("speaker_profile_sr", 24000)),
        tts_reference_denoise_strength=float(tts.get("reference_denoise_strength", 0.5)),
        voices_dir=tts.get("voices_dir", "/workspace/voices"),
        f5tts_model=tts.get("f5tts_model", "F5TTS_v1_Multilingual"),
        f5tts_nfe_step=int(tts.get("f5tts_nfe_step", 32)),
        f5tts_cfg_strength=float(tts.get("f5tts_cfg_strength", 2.0)),
        f5tts_speed=float(tts.get("f5tts_speed", 1.0)),
        tts_max_stretch=tts.get("max_stretch", 1.3),
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
        metrics_enabled=bool(met.get("enabled", False)),
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
    """Source duration in seconds, or 0.0 on failure.

    Callers must treat 0.0 as fatal: the anchored timing policy fits the whole
    dub to this value, so a guessed duration silently corrupts the output."""
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
        log.error(f"Could not read source duration ({e})")
        return 0.0


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
    language: str = "en",
    initial_prompt: str = "",
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
        # language="" → auto-detect (bilingual sources); initial_prompt is a
        # hallucination seed on silence/music, so it defaults to empty and
        # should only ever carry domain vocabulary, never full sentences.
        segments_gen, info = model.transcribe(
            wav_path,
            language=language or None,
            beam_size=5,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 300},
            word_timestamps=True,
            condition_on_previous_text=condition_on_previous_text,
            compression_ratio_threshold=compression_ratio_threshold,
            no_speech_threshold=no_speech_threshold,
            log_prob_threshold=log_prob_threshold,
            initial_prompt=initial_prompt or None,
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
      - Never merge across a speaker change (segments carry a "speaker" key
        when diarization ran) — a chunk spanning two speakers would be dubbed
        entirely in one cloned voice.
      - Never emit a chunk shorter than min_duration: force-merge tiny chunks
        with their previous neighbour as a final sweep (same-speaker only).

    This eliminates the sub-second fragments that broke SRT timing in v3 and
    gives the TTS enough context per call for natural prosody.
    """
    if not segments:
        return segments

    def _same_speaker(a: dict, b: dict) -> bool:
        return a.get("speaker") == b.get("speaker")

    merged: List[dict] = []
    current = dict(segments[0])

    for seg in segments[1:]:
        gap          = seg["start"] - current["end"]
        combined_dur = seg["end"]   - current["start"]
        current_dur  = current["end"] - current["start"]
        ends_sent    = bool(re.search(r"[.!?]\s*$", current["text"].rstrip()))

        # Merge if gap small enough, combined stays under cap, same speaker, AND
        # either we haven't reached min_duration or the last clause didn't end.
        should_merge = (
            gap <= max_gap
            and combined_dur <= max_duration
            and _same_speaker(current, seg)
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
            if (dur < min_duration and (prev_dur + dur) <= max_duration * 1.25
                    and _same_speaker(cleaned[-1], chunk)):
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
            # The HF-token kwarg was renamed between pyannote versions: 4.x uses
            # `token=`, 3.x uses `use_auth_token=`. Pick whichever the installed
            # version accepts so the call works on either.
            import inspect
            _params = inspect.signature(PyannotePipeline.from_pretrained).parameters
            _tok_kw = "token" if "token" in _params else "use_auth_token"
            pipeline = PyannotePipeline.from_pretrained(model_name, **{_tok_kw: hf_token})
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            pipeline.to(device)

            diarize_kwargs: dict = {}
            if min_speakers and min_speakers >= 1:
                diarize_kwargs["min_speakers"] = int(min_speakers)
            if max_speakers and max_speakers >= max(min_speakers or 1, 1):
                diarize_kwargs["max_speakers"] = int(max_speakers)

            log.info(f"  Pyannote params: {diarize_kwargs}")

            # pyannote 4.x turned __call__ into a batch/streaming generator that
            # yields nothing for a single file — use the .apply() method instead,
            # which returns the diarization directly. .apply() reads file["uri"],
            # so pass the AudioFile *dict* form ({"audio": path, "uri": name}); a
            # bare path string raises `TypeError: string indices must be integers`.
            # On pyannote 3.x (no .apply) fall back to calling the pipeline directly.
            uri = os.path.splitext(os.path.basename(wav_path))[0]
            audio_file = {"audio": wav_path, "uri": uri}
            if hasattr(pipeline, "apply"):
                result = pipeline.apply(audio_file, **diarize_kwargs)
            else:
                result = pipeline(wav_path, **diarize_kwargs)

        # Defensive generator drain — kept in case a pyannote version wraps the
        # result in a generator (the 3.x __call__ path can).
        if isinstance(result, types.GeneratorType):
            try:
                result = next(result)
            except StopIteration:
                raise RuntimeError("pyannote pipeline returned an empty generator")
            if isinstance(result, tuple) and len(result) >= 2 and hasattr(result[-1], "itertracks"):
                result = result[-1]

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
    target_sr: int = 24000,
    denoise_strength: float = 0.5,
) -> dict:
    """Build a per-speaker voice-clone reference clip from the cleanest turns."""
    from collections import defaultdict

    # F5-TTS synthesizes at 24 kHz — extract at ≥24 kHz so the cloned voice keeps
    # its full bandwidth instead of sounding band-limited/muffled.
    TARGET_SR = max(int(target_sr), 24000)
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
            profiles[speaker] = denoise_audio(
                raw_path, denoised_path, log, prop_decrease=denoise_strength
            )
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

Translate each numbered English segment into natural spoken {language} suitable for AI voice dubbing.

Each segment is tagged:

[N.Ns, ~W words, ≤M chars]

where:

* N.Ns = target audio duration
* W = approximate word count at a comfortable spoken pace (self-check: count your words)
* M = maximum character budget

Your goal is to create dialogue that:

* sounds natural when spoken aloud
* preserves the speaker's intent, facts, names, numbers, and technical terms
* fits comfortably within the timing and character budget
* matches the speaker's tone, emotion, and level of formality

Translation Guidelines:

* Use natural spoken {language}, not literal translation.
* Prefer concise conversational phrasing.
* Adapt idioms and expressions naturally.
* Preserve proper nouns, product names, acronyms, numbers, and technical terminology.
* Remove filler words, repetitions, and unnecessary hedging when needed for timing.
* If timing is tight, condense wording while preserving meaning and intent.
* Do not add information.
* Do not summarize major content.
* Prefer shorter natural phrasing when multiple valid translations exist.
* Optimize for speech synthesis and dubbing, not written subtitles.

Character Limits:

* Do not exceed the stated character budget.
* If a segment is difficult to fit, prioritize preserving key meaning over literal wording.

Output Rules:

* Output ONLY the numbered translations.
* One translation per line.
* Preserve the original numbering.
* Do not output notes, explanations, brackets, character counts, or commentary.

{glossary_section}

English segments:
{segments}

{language} translations:"""

_REVIEW_PROMPT = """\
You are a native {language} dubbing editor.{locale_note}

Below are numbered {language} dubbing segments that have already been translated.
Lightly revise each one. Each line is tagged [N.Ns, ≤M chars] where N.Ns is the
target audio duration and M is the maximum character budget.

Revise each segment for:

* natural spoken rhythm
* voice dubbing suitability
* grammar and fluency
* {language} usage and register
* timing efficiency

When multiple valid phrasings exist:

* prefer shorter spoken forms
* prefer conversational language
* reduce Anglicisms
* preserve intent rather than literal wording

Do not make a segment longer unless necessary for clarity.
Do not change facts, numbers, names, or technical terms.
Keep each revision within its stated character budget.

Output Rules:

* Output ONLY the numbered revised segments — nothing else.
* One revised segment per line, preserving the original numbering.
* If a segment is already good, output it unchanged (still numbered).
* Do NOT output notes, explanations, headers, markdown, bullet points,
  character counts, the review criteria, or any commentary.

{glossary_section}

{language} segments to review:
{segments}

Revised {language} segments:"""


# Signatures of LLM meta-commentary / prompt-instruction leakage. When an LLM
# echoes its own brief (review criteria, markdown headers, rewrite arrows)
# instead of returning a clean translation, the text matches one of these. Used
# to drop such items so they can never reach the TTS or the subtitles. Kept
# high-precision (strong signals only) so ordinary French is never discarded.
_LEAK_SIGNATURE_RE = re.compile(
    r"\*\*"                                   # markdown bold
    r"|(?:^|\s)#{2,}\s"                       # markdown headers
    r"|→"                                     # rewrite arrow ("X → Y")
    r"|timing efficiency"
    r"|voice[\s-]*dubbing"
    r"|natural spoken rhythm"
    r"|grammar (?:and|/) fluency"
    r"|canadian french usage"
    r"|(?:reduce|éviter les) anglicism"
    r"|prefer shorter"
    r"|trim redundan"
    r"|privilégier la concision"          # French instruction phrasing seen
    r"|mots redondants"                   # in compression-pass rewrites of
    r"|termes québécois"                  # leaked review criteria
    r"|éviter les traductions",
    re.IGNORECASE,
)


def _looks_like_instruction_leak(text: str) -> bool:
    """True if `text` looks like echoed prompt instructions / meta-commentary
    rather than a real translation. See _LEAK_SIGNATURE_RE."""
    if not text:
        return False
    return bool(_LEAK_SIGNATURE_RE.search(text))


# Empirical French syllable density: ~0.32 syllables per character for typical
# spoken French (content + function words mixed). Used to derive syllable-per-
# second budgets from the character-per-second config values so that the
# compression trigger and CPS-split threshold are linguistically grounded
# rather than relying on character counts, which over-penalise long French
# words and under-penalise dense short ones.
_FR_SYL_PER_CHAR: float = 0.32

_pyphen_cache: dict = {}


def _count_syllables(text: str, lang: str = "fr") -> int:
    """Count syllables in `text` using pyphen's hyphenation dictionary.

    Hyphenation points ≈ syllable boundaries for timing purposes.
    Falls back to a vowel-group count if pyphen is unavailable.
    """
    words = re.findall(r"[a-zA-ZÀ-ÿ̀-ͯ']+", text)
    if not words:
        return 0
    try:
        import pyphen
        if lang not in _pyphen_cache:
            _pyphen_cache[lang] = pyphen.Pyphen(lang=lang)
        dic = _pyphen_cache[lang]
        return sum(len(dic.positions(w)) + 1 for w in words)
    except Exception:
        # Vowel-group fallback — counts runs of French vowels per word
        vowel_re = re.compile(r"[aeiouyàâéèêëîïôùûüœæ]+", re.IGNORECASE)
        return max(1, sum(len(vowel_re.findall(w)) for w in words))


def _word_budget(seg: dict, wps: float = 3.0) -> int:
    """Approximate word count a segment should contain at `wps` words/second.

    3.0 WPS is a comfortable dubbing pace for French. LLMs track word counts
    more reliably during generation than character counts, so including this
    in the prompt tag gives the model a self-checkable constraint.
    """
    dur = max(seg["end"] - seg["start"], 0.5)
    return max(3, round(dur * wps))


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

    The over-budget check uses syllables/second (derived from max_cps via
    _FR_SYL_PER_CHAR) rather than characters/second — syllables correlate
    more directly with F5-TTS output duration and avoid false triggers on
    segments that use many long words but few syllables.
    """
    max_sps = max_cps * _FR_SYL_PER_CHAR  # e.g. 21 CPS → ~6.7 SPS
    SENT_END = re.compile(r"[.!?…»\"'\)]\s+")
    out: List[dict] = []
    splits = 0
    for seg in segments:
        fr = seg.get("text_fr_natural") or seg.get("text_fr") or ""
        dur = seg.get("end", 0) - seg.get("start", 0)
        if dur <= 0 or len(fr) < 80:
            out.append(seg)
            continue
        sps = _count_syllables(fr) / dur
        if sps <= max_sps:
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

        # Mirror split on the English source so SRT/translation diagnostics stay coherent.
        # Snap to the nearest word boundary so the EN text never splits mid-word.
        en = seg.get("text", "") or ""
        en_share = max(1, int(len(en) * share))
        if en_share < len(en):
            bp = en.rfind(" ", 0, en_share)
            fp = en.find(" ", en_share)
            if bp > 0 or fp > 0:
                if bp <= 0:
                    en_share = fp + 1
                elif fp < 0:
                    en_share = bp + 1
                else:
                    en_share = (bp + 1) if (en_share - bp) <= (fp - en_share) else (fp + 1)
        a_en = en[:en_share].strip()
        b_en = en[en_share:].strip()

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

Preserve the speaker's intent, key message, facts, numbers, names, and technical terms.
Use natural spoken {language}, not literal translation.
Match the speaker's tone, formality, and emotional intensity.
Keep the segment concise enough for dubbing timing.
Remove fillers, hedges, repetitions, and unnecessary words when needed.

You may:

Rephrase sentences.
Use shorter synonyms.
Restructure wording.
Slightly condense non-essential details.

Do not:

Change facts, numbers, names, or technical terminology.
Add information not present in the source.
Exceed the character limit.

IMPORTANT — trim minimally:
Shorten ONLY as much as needed to fit within the limit. Most segments are just
slightly over budget and need only a light trim. Preserve as much of the
original meaning, detail, and natural phrasing as possible — aim to use most of
the available characters rather than producing the shortest possible text. Do
not gut a sentence or drop clauses when a small cut would have fit.

Each line is tagged [≤N chars]. Your rewrite must not exceed N characters, but
should stay reasonably close to N — do not compress far below it.

Output ONLY the numbered rewrites, one per line, preserving the original numbering. No explanations or notes.

{glossary_section}

{language} segments:
{segments}"""

def compress_overflowing_translations(
    segments: List[dict],
    model: str,
    temperature: float,
    log: logging.Logger,
    budget_cps: int = 17,
    target_lang: str = "fr",
    margin: float = 1.05,
    rounds: int = 3,
    glossary_section: str = "",
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

    Fidelity guard: Qwen tends to over-shorten, gutting a segment that was only
    slightly over budget (e.g. 185→61 chars against a 161 budget — dropping real
    content, not just filler). A rewrite that undershoots the budget drastically
    (< UNDERSHOOT_FLOOR × budget) is rejected UNLESS the original was heavily over
    budget (> HEAVY_OVER × budget) and therefore genuinely needs deep cutting.
    Rejected segments keep their faithful (mildly-over) translation; the small
    overflow is absorbed downstream by anchored timing + subtitle condensing.
    """
    language = _LANG_NAMES.get(target_lang, target_lang.upper())
    think_prefix = "/no_think\n" if "qwen3" in model.lower() else ""

    def _budget(seg: dict) -> int:
        """Character budget sent to the LLM in the compression prompt."""
        dur = max(seg["end"] - seg["start"], 0.5)
        return max(40, int(dur * budget_cps))

    def _syl_budget(seg: dict) -> int:
        """Syllable budget used to decide whether to trigger compression.

        Syllables correlate more directly with F5-TTS speech duration than
        characters, so this trigger fires only when the audio would genuinely
        be too long — not just because the LLM chose polysyllabic synonyms.
        Derived from budget_cps × _FR_SYL_PER_CHAR so the two budgets stay
        in sync when budget_cps is tuned.
        """
        dur = max(seg["end"] - seg["start"], 0.5)
        return max(8, int(dur * budget_cps * _FR_SYL_PER_CHAR))

    out = [dict(s) for s in segments]
    BATCH = 15
    # Fidelity guard thresholds (see docstring).
    UNDERSHOOT_FLOOR = 0.70   # reject rewrites below 70% of budget…
    HEAVY_OVER = 1.60         # …unless the original was >160% of budget
    rejected_overtrim = 0

    for rnd in range(1, max(1, rounds) + 1):
        # Recompute offenders from the *current* text so converged segments drop
        # out and stubborn ones get another, tighter attempt.
        offenders: List[Tuple[int, int]] = []  # (index, char_budget)
        for idx, s in enumerate(out):
            fr = s.get("text_fr") or ""
            if not fr:
                continue
            b = _budget(s)
            syl_b = _syl_budget(s)
            # Trigger on syllables (linguistically grounded) but send char
            # budget to the LLM so it has a concrete counting target.
            if _count_syllables(fr) > int(syl_b * margin):
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
            prompt = think_prefix + _COMPRESS_PROMPT.format(
                language=language, segments=numbered, glossary_section=glossary_section,
            )
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
                    # Fidelity guard: reject a drastic undershoot (content drop)
                    # on a segment that was only mildly over budget. Keep the
                    # faithful original; the small overflow is absorbed by
                    # anchored timing + subtitle condensing downstream.
                    if len(new) < int(b * UNDERSHOOT_FLOOR) and len(old) <= int(b * HEAVY_OVER):
                        rejected_overtrim += 1
                        continue
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

    if rejected_overtrim:
        log.info(
            f"✓ Compression fidelity guard: kept {rejected_overtrim} faithful "
            f"translation(s) over an over-trimmed rewrite"
        )
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


# Digit sequences worth checking for preservation (≥3 digits = year, phone,
# ID, price, etc. — short numbers like "5" or "10" are often spelled out in FR).
_DIGIT_SEQ_RE = re.compile(r'\b\d{3,}(?:[,.\-]\d+)*\b')


def _scan_and_fix_hallucinations(
    segments: List[dict],
    model: str,
    log: logging.Logger,
    target_lang: str = "fr",
    locale: str = "fr",
    glossary_section: str = "",
) -> List[dict]:
    """Scan translations for hallucination patterns and re-translate flagged ones.

    Checks (per segment):
      1. Number dropout  — digit sequences ≥3 digits in EN are missing from FR.
      2. FR phrase loop  — a ≥5-word run appears twice in the FR output (Qwen loop).
      3. Length explosion — FR text is >2.5× the EN length (possible fabrication).

    Flagged segments are re-translated individually with a strict single-line prompt
    that explicitly reminds the model to preserve numbers. The original is kept if
    the re-translation fails or still looks untranslated.
    """
    language = _LANG_NAMES.get(target_lang, target_lang.upper())
    locale_note = " Use Québécois/Canadian French." if locale == "fr-ca" else ""
    think = "/no_think\n" if "qwen3" in model.lower() else ""

    def _has_number_dropout(en: str, fr: str) -> bool:
        en_nums = set(_DIGIT_SEQ_RE.findall(en))
        fr_nums = set(_DIGIT_SEQ_RE.findall(fr.replace(" ", "").replace("\xa0", "")))
        return bool(en_nums - fr_nums)

    def _has_fr_phrase_loop(fr: str, min_k: int = 5) -> bool:
        toks = [t.strip(".,!?;:\"'()[]…").lower() for t in fr.split() if t.strip(".,!?;:\"'()[]…")]
        n = len(toks)
        for k in range(min(12, n // 2), min_k - 1, -1):
            for i in range(n - 2 * k + 1):
                if toks[i:i + k] == toks[i + k:i + 2 * k]:
                    return True
        return False

    flagged: List[Tuple[dict, List[str]]] = []
    for s in segments:
        en = s.get("text", "")
        fr = s.get("text_fr", "")
        if not en or not fr:
            continue
        reasons: List[str] = []
        if _has_number_dropout(en, fr):
            missing = set(_DIGIT_SEQ_RE.findall(en)) - set(_DIGIT_SEQ_RE.findall(fr.replace(" ", "").replace("\xa0", "")))
            reasons.append(f"number dropout: {sorted(missing)}")
        if _has_fr_phrase_loop(fr):
            reasons.append("repeated phrase loop in FR output")
        if len(en) > 20 and len(fr) > len(en) * 2.5:
            reasons.append(f"length explosion ({len(fr)/len(en):.1f}× EN)")
        if reasons:
            flagged.append((s, reasons))

    if not flagged:
        log.debug("✓ Hallucination scan: no issues detected in translations")
        return segments

    log.info(f"Hallucination scan: {len(flagged)} suspicious segment(s) — re-translating")
    fixed = 0
    for seg, reasons in flagged:
        log.info(f"  Segment {seg.get('id')}: {'; '.join(reasons)}")
        prompt = (
            f"{think}Translate this single line into natural, spoken {language}."
            f"{locale_note} Preserve ALL numbers exactly as written.\n"
            f"Output ONLY the {language} translation — no quotes, no notes.\n\n"
            f"English: {seg['text']}\n{language}:"
        )
        out = _ollama_call(prompt, model, 0.2, log)
        if not out:
            continue
        out = re.sub(r"<think>.*?</think>", "", out, flags=re.DOTALL).strip()
        cand = next((ln.strip().strip('"') for ln in out.splitlines() if ln.strip()), "")
        if cand and not _looks_untranslated(cand) and not _looks_like_instruction_leak(cand):
            seg["text_fr"] = cand
            seg["text_fr_natural"] = cand
            fixed += 1
        else:
            log.warning(f"  Segment {seg.get('id')}: re-translation failed, keeping original")

    log.info(f"  ✓ Fixed {fixed}/{len(flagged)} flagged segment(s)")
    return segments


# Explicit context window for every Ollama call. Ollama's default (2048-4096
# depending on version) silently truncates the FRONT of longer prompts — i.e.
# the instruction block and glossary — which looks exactly like "the model
# ignored the brief". 8192 covers the translate prompt + glossary + a
# 20-segment batch + the 4096-token output budget with room to spare.
_OLLAMA_NUM_CTX = 8192


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
                "options": {
                    "temperature": temperature,
                    "num_predict": 4096,
                    "num_ctx": _OLLAMA_NUM_CTX,
                },
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


def _ollama_unload(model: str, log: logging.Logger) -> None:
    """Tell Ollama to immediately unload *model* from VRAM (keep_alive=0).

    Called after all translation calls are done so the LLM doesn't compete
    with F5-TTS for GPU memory during synthesis.  Non-fatal if Ollama is
    unreachable (it may already have unloaded the model on its own).
    """
    try:
        requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model, "prompt": "", "keep_alive": 0},
            timeout=(5, 30),
        )
        log.info(f"  Ollama: unloaded {model} from VRAM")
    except Exception as e:
        log.debug(f"  Ollama unload skipped: {e}")


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
        # Defensive guard: drop items that are echoed prompt instructions /
        # meta-commentary rather than a real translation. The caller's
        # `if result[i]:` fallback then keeps the prior good text. Protects
        # every LLM pass that routes through _parse_numbered.
        if content and _looks_like_instruction_leak(content):
            return
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


def translate_segments(
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
            f"{i + 1}. [{s['end'] - s['start']:.1f}s, ~{_word_budget(s)} words, ≤{_budget(s)} chars] {s['text']}"
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

    # Sequential batches: Ollama serializes requests to a single loaded model
    # anyway (OLLAMA_NUM_PARALLEL=1 default), so a thread pool adds queueing
    # and read-timeout risk without throughput.
    all_translations = []
    for batch in tqdm(batches, desc=f"Translating ({target_lang})", leave=False):
        all_translations.extend(_process_batch(batch))

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
            f"{i + 1}. [{s['end'] - s['start']:.1f}s, ~{_word_budget(s)} words, ≤{max(40, int((s['end']-s['start'])*17))} chars] {s.get('text_fr', '')}"
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

def _best_voiced_offset(
    wav_path: str,
    duration: float,
    skip_seconds: float,
    log: logging.Logger,
) -> float:
    """Scan candidate windows ≥ skip_seconds and return the start (s) of the one
    with the highest voiced RMS energy — avoids landing on silence/music/faded
    regions. Falls back to skip_seconds if scanning fails."""
    try:
        y, sr = librosa.load(wav_path, sr=16000, mono=True)  # 16k is plenty for energy
        total = len(y) / sr
        if total <= skip_seconds + duration:
            return skip_seconds
        # Step across the file in half-window hops; score each candidate by RMS.
        hop = max(duration / 2.0, 2.0)
        best_off, best_rms = skip_seconds, -1.0
        off = skip_seconds
        while off + duration <= total:
            seg = y[int(off * sr):int((off + duration) * sr)]
            rms = float(np.sqrt(np.mean(seg ** 2))) if seg.size else 0.0
            if rms > best_rms:
                best_rms, best_off = rms, off
            off += hop
        log.info(f"  Reference auto-pick: best voiced window at {best_off:.0f} s (RMS {best_rms:.3f})")
        return best_off
    except Exception as e:
        log.debug(f"  voiced-offset scan failed ({e}) — using fixed skip {skip_seconds:.0f} s")
        return skip_seconds


def extract_speaker_sample(
    wav_path: str,
    duration: float,
    output_path: str,
    log: logging.Logger,
    skip_seconds: float = 20.0,
    target_sr: int = 24000,
) -> bool:
    """Extract a speaker reference clip, auto-picking the cleanest voiced window
    past skip_seconds. Extracts at ≥24 kHz so the cloned voice keeps full bandwidth."""
    sr = max(int(target_sr), 24000)
    offset = _best_voiced_offset(wav_path, duration, skip_seconds, log)
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", wav_path,
                "-ss", str(offset),
                "-t",  str(duration),
                "-ar", str(sr),
                "-ac", "1",
                output_path,
            ],
            check=True, capture_output=True, timeout=60,
        )
        log.info(f"✓ Speaker sample: {duration:.0f} s at {offset:.0f} s offset ({sr} Hz)")
        return True
    except Exception as e:
        log.error(f"Speaker sample extraction failed: {e}")
        return False


def denoise_audio(
    audio_path: str,
    output_path: str,
    log: logging.Logger,
    prop_decrease: float = 0.5,
) -> str:
    """Denoise speaker reference. Tries noisereduce → FFmpeg anlmdn.

    prop_decrease controls how much noise is removed (0=off, 1=max). Kept gentle
    by default — aggressive spectral gating produces the watery/"underwater"
    artifact that bleeds into the cloned voice."""
    if prop_decrease <= 0:
        log.info("  Reference denoise disabled (strength 0) — using raw reference")
        return audio_path
    try:
        import noisereduce as nr
        import soundfile as _sf

        log.info(f"Denoising with noisereduce (strength {prop_decrease:.2f}) …")
        data, rate = _sf.read(audio_path)
        reduced    = nr.reduce_noise(y=data, sr=rate, prop_decrease=float(prop_decrease))
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


def _ref_from_override(
    ov: dict,
    vocals_wav: str,
    temp_dir: str,
    key: str,
    config: "PipelineConfig",
    log: logging.Logger,
) -> Optional[str]:
    """Build a reference clip from a user override (a vocals time-range or a
    library clip), resampled to the F5-TTS rate. Returns None to signal the
    caller should fall back to the automatic reference."""
    sr = max(int(config.tts_speaker_profile_sr), 24000)
    out = os.path.join(temp_dir, f"ref_override_{key}.wav")
    src = ov.get("source")
    try:
        if src == "range":
            start = float(ov.get("start", 0.0))
            dur   = float(ov.get("duration", config.tts_speaker_duration))
            subprocess.run(
                ["ffmpeg", "-y", "-i", vocals_wav, "-ss", str(start), "-t", str(dur),
                 "-ar", str(sr), "-ac", "1", out],
                check=True, capture_output=True, timeout=60,
            )
        elif src == "library":
            libpath = ov.get("path") or ""
            if not os.path.exists(libpath):
                log.warning(f"  {key}: library reference not found ({libpath}) — using auto")
                return None
            subprocess.run(
                ["ffmpeg", "-y", "-i", libpath, "-ar", str(sr), "-ac", "1", out],
                check=True, capture_output=True, timeout=60,
            )
        else:
            return None
    except Exception as e:
        log.warning(f"  {key}: override reference build failed ({e}) — using auto")
        return None

    # Manual picks are assumed clean, so denoise is opt-in per override.
    if ov.get("denoise"):
        dn = os.path.join(temp_dir, f"ref_override_{key}_dn.wav")
        out = denoise_audio(out, dn, log, prop_decrease=config.tts_reference_denoise_strength)
    log.info(f"  {key}: using '{src}' reference override → {out}")
    return out


def resolve_speaker_references(
    vocals_wav: str,
    segments: List[dict],
    temp_dir: str,
    output_dir: str,
    name: str,
    config: "PipelineConfig",
    log: logging.Logger,
    diarization_turns: Optional[List[Tuple[float, float, str]]] = None,
) -> Tuple[Optional[str], Optional[dict]]:
    """Prepare the F5-TTS voice-clone reference(s) for synthesis.

    Honors per-speaker overrides written by the web review UI to
    ``{output_dir}/{name}_voice_refs.json`` (keys: speaker label or "default";
    each {"source": "range"|"library", ...}). Falls back to automatic extraction
    (now at the F5-TTS sample rate) for any speaker without an override.
    Returns (speaker_wav, speaker_profiles)."""
    overrides: dict = {}
    refs_path = os.path.join(output_dir, f"{name}_voice_refs.json")
    if os.path.exists(refs_path):
        try:
            with open(refs_path, encoding="utf-8") as f:
                overrides = json.load(f) or {}
            log.info(f"  Voice-reference overrides loaded: {sorted(overrides)}")
        except Exception as e:
            log.warning(f"  voice_refs.json unreadable ({e}) — ignoring")

    speaker_wav: Optional[str] = None
    speaker_profiles: Optional[dict] = None

    # Multi-speaker: auto-build all profiles, then apply per-speaker overrides.
    if config.use_diarization and any("speaker" in s for s in segments):
        speaker_profiles = build_speaker_profiles(
            vocals_wav, segments, temp_dir, config.diarization_profile_duration, log,
            use_deepfilter=config.use_deepfilter,
            diarization_turns=diarization_turns,
            target_sr=config.tts_speaker_profile_sr,
            denoise_strength=config.tts_reference_denoise_strength,
        )
        for spk in list(speaker_profiles.keys()):
            ov = overrides.get(spk)
            if ov:
                built = _ref_from_override(ov, vocals_wav, temp_dir, spk, config, log)
                if built:
                    speaker_profiles[spk] = built
        valid = sum(1 for v in speaker_profiles.values() if v)
        log.info(f"  Built {valid}/{len(speaker_profiles)} speaker profile(s)")

    # Single/default reference (fallback for any segment without a profile).
    default_ov = overrides.get("default") or overrides.get("SPEAKER_00")
    if default_ov:
        speaker_wav = _ref_from_override(default_ov, vocals_wav, temp_dir, "default", config, log)
    if speaker_wav is None:
        raw_speaker_wav = os.path.join(temp_dir, "speaker_raw.wav")
        if extract_speaker_sample(
            vocals_wav, config.tts_speaker_duration, raw_speaker_wav, log,
            skip_seconds=config.tts_speaker_skip, target_sr=config.tts_speaker_profile_sr,
        ):
            if config.use_deepfilter:
                denoised_wav = os.path.join(temp_dir, "speaker_denoised.wav")
                speaker_wav = denoise_audio(
                    raw_speaker_wav, denoised_wav, log,
                    prop_decrease=config.tts_reference_denoise_strength,
                )
            else:
                speaker_wav = raw_speaker_wav
        else:
            log.warning("Voice cloning disabled — no speaker reference available")

    return speaker_wav, speaker_profiles


# ============================================================================
# Step 6: TTS Synthesis — F5-TTS (flow-matching zero-shot voice cloning)
# ============================================================================

def _seg_text(seg: dict) -> str:
    return (seg.get("text_fr") or seg.get("text") or "").strip()


# Inclusive-writing doublets ("conférencier·ère", "professionnel·les") are a
# written-only convention — there is no way to pronounce the median dot, and
# F5-TTS garbles or hallucinates around it. The subtitles keep the inclusive
# form; only the text sent to the TTS is collapsed to the base (masculine)
# form, re-pluralised when the suffix carried the plural "s".
# Covers U+00B7 (median dot), U+2027 (hyphenation point), U+2022 (bullet).
_INCLUSIVE_DOT_RE = re.compile(
    r"([A-Za-zÀ-ÿ]+)[··‧•]((?:[A-Za-zÀ-ÿ]+[··‧•]?)+)"
)
# Written-only symbols the TTS should never see (guillemets read as noise).
_TTS_STRIP_CHARS = str.maketrans({"«": "", "»": "", "“": "", "”": ""})


def _tts_spoken_form(text: str) -> str:
    """Rewrite written-only conventions into something the TTS can pronounce."""
    def _collapse(m: "re.Match") -> str:
        base, suffix = m.group(1), m.group(2)
        plural = suffix.rstrip("··‧•").endswith("s") and not base.endswith(("s", "x"))
        return base + ("s" if plural else "")

    text = _INCLUSIVE_DOT_RE.sub(_collapse, text)
    text = text.translate(_TTS_STRIP_CHARS)
    return re.sub(r"\s{2,}", " ", text).strip()


# F5-TTS does not have XTTS's hard token cap, but very long single-call inputs
# still reduce quality. 250 chars/chunk keeps each call focused while allowing
# longer sentences than XTTS permitted.
_F5TTS_CHUNK_LIMIT = 250

# Runaway / quality guard. F5-TTS is much less prone to repetition loops than
# autoregressive XTTS, but near-silent or oddly-long outputs can still occur on
# unusual input. We retry up to _TTS_MAX_RETRIES times; each call uses seed=-1
# (random), so retries are genuinely independent samples.
_TTS_EXPECTED_CPS = 13.0          # chars/sec of natural synthesized speech
_TTS_RUNAWAY_FACTOR = 2.5         # natural > 2.5x expected → suspect runaway
_TTS_RUNAWAY_MIN_EXCESS_S = 2.0   # …and at least this many seconds over expected
_TTS_MAX_RETRIES = 2


def _split_for_tts(text: str, limit: int) -> List[str]:
    """Break text into ≤limit-char chunks on natural boundaries.

    Always splits at sentence boundaries so the caller's chunk_gap inserts an
    audible inter-sentence pause even for short texts. Further sub-splits long
    sentences on clause punctuation or hard-wraps as a last resort.
    Returns chunks that are each ≤ limit chars.
    """
    text = text.strip()
    if not text:
        return []
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
    """Synthesize every segment with F5-TTS (flow-matching zero-shot voice cloning)."""
    try:
        from f5_tts.api import F5TTS
    except ImportError:
        log.error("f5-tts not installed. Install: pip install f5-tts")
        return [], 24000

    import torch as _torch

    # F5-TTS's max_chars formula: ref_text_bytes / ref_dur_s * (22 - ref_dur_s) * speed
    # When ref_dur_s > 22 the result is negative → chunk_text splits every sentence into
    # its own generation call → cross-fade overlaps accumulate → audio 2-3x too short.
    # Cap all reference clips at 15s regardless of config.
    _F5_REF_MAX_S = 15.0
    _ref_wav_trunc: dict = {}  # original path → safe (possibly truncated) path

    def _prepare_ref_wavs() -> None:
        unique = {
            _raw_pick_wav(s) for s in segments
            if _raw_pick_wav(s) and os.path.exists(_raw_pick_wav(s))
        }
        for wav_path in unique:
            try:
                import soundfile as _sf
                data, rate = _sf.read(wav_path, always_2d=False)
                max_samp = int(_F5_REF_MAX_S * rate)
                if len(data) <= max_samp:
                    _ref_wav_trunc[wav_path] = wav_path
                else:
                    trunc = os.path.join(os.path.dirname(wav_path),
                                         Path(wav_path).stem + "_trunc.wav")
                    _sf.write(trunc, data[:max_samp], rate)
                    _ref_wav_trunc[wav_path] = trunc
                    log.info(
                        f"  Ref clip {Path(wav_path).name}: "
                        f"{len(data)/rate:.1f}s → {_F5_REF_MAX_S:.0f}s (F5-TTS 22s limit)"
                    )
            except Exception as e:
                log.warning(f"  Ref clip truncation failed for {wav_path}: {e}")
                _ref_wav_trunc[wav_path] = wav_path

    def _raw_pick_wav(seg: dict) -> Optional[str]:
        if speaker_profiles:
            profile = speaker_profiles.get(seg.get("speaker", "SPEAKER_00"))
            if profile and os.path.exists(profile):
                return profile
        return speaker_wav

    def _pick_wav(seg: dict) -> Optional[str]:
        raw = _raw_pick_wav(seg)
        return _ref_wav_trunc.get(raw, raw) if raw else raw

    log.info(f"Loading F5-TTS: {config.f5tts_model} …")
    device = "cuda" if _torch.cuda.is_available() else "cpu"

    if "/" in config.f5tts_model:
        # HuggingFace repo ID — download checkpoint + vocab, use F5TTS_Base arch.
        # Community fine-tunes like RASPIAUDIO/F5-French-MixedSpeakers-reduced are
        # based on the original F5TTS_Base (not v1), so we use that config for the
        # model architecture while supplying the custom weights.
        try:
            from huggingface_hub import hf_hub_download
            repo_id = config.f5tts_model
            log.info(f"  Fetching weights from HuggingFace: {repo_id} …")
            # Try the reduced checkpoint first; fall back to the full one.
            try:
                ckpt_file = hf_hub_download(repo_id=repo_id, filename="model_last_reduced.pt")
            except Exception:
                ckpt_file = hf_hub_download(repo_id=repo_id, filename="model_last.pt")
            vocab_file = hf_hub_download(repo_id=repo_id, filename="vocab.txt")
            f5 = F5TTS(model="F5TTS_Base", ckpt_file=ckpt_file, vocab_file=vocab_file, device=device)
        except Exception as e:
            log.error(f"Failed to load HuggingFace model '{config.f5tts_model}': {e}")
            return [], 24000
    else:
        f5 = F5TTS(model=config.f5tts_model, device=device)

    sr = 24000  # F5-TTS/vocos native rate
    log.info(f"✓ F5-TTS ready (output: {sr} Hz, device: {device})")

    # When ref_text="" F5-TTS auto-transcribes its reference clips internally.
    # That path uses torchcodec, which fails on systems missing the matching
    # libnvrtc.so.N (e.g. PyTorch 2.8+cu128 vs CUDA 12.x).
    #
    # Preferred path: ref_text="" — F5-TTS auto-transcribes, no vocabulary
    #   bleeds from the reference clip into the generated speech.
    # Fallback path (torchcodec broken): pre-transcribe with faster-whisper and
    #   pass the result as ref_text so F5-TTS never calls torchcodec.
    #
    # Vocabulary bleed is the cause of artifacts like "vidéo" appearing after
    # sentences that don't contain that word: when the reference clip contains
    # "vidéo" and ref_text is passed, F5-TTS's combined ref+gen sequence can
    # echo that word near sentence boundaries.
    _ref_text_cache: dict = {}

    def _torchcodec_ok() -> bool:
        """Return True if torchcodec's shared library loads successfully.

        Importing torchcodec eagerly loads its native libs and raises OSError when
        the matching libnvrtc.so is missing (the cu128-vs-CUDA13 mismatch), so a
        clean `import torchcodec` + AudioDecoder import is a reliable signal across
        torchcodec versions. (The old `_internally_replaced_utils` symbol was
        removed in torchcodec 0.7+, so the previous check always returned False.)
        """
        try:
            import torchcodec  # noqa: F401
            from torchcodec.decoders import AudioDecoder  # noqa: F401
            return True
        except Exception:
            return False

    def _prefetch_ref_texts() -> None:
        unique_refs = {
            _pick_wav(s)
            for s in segments
            if _pick_wav(s) and os.path.exists(_pick_wav(s))
        }
        if not unique_refs:
            return
        log.info(f"  Pre-transcribing {len(unique_refs)} reference clip(s) with faster-whisper …")
        try:
            from faster_whisper import WhisperModel as _WM
            # "small" on CPU/int8 is fast enough for short reference clips.
            _m = _WM("small", device="cpu", compute_type="int8",
                     download_root=config.models_folder)
            for wav_path in sorted(unique_refs):
                try:
                    segs_iter, _ = _m.transcribe(
                        wav_path, language=None,  # auto-detect: clips are source-language (English)
                        beam_size=1, condition_on_previous_text=False,
                    )
                    _ref_text_cache[wav_path] = " ".join(
                        s.text.strip() for s in segs_iter
                    ).strip()
                    log.info(f"    {Path(wav_path).name}: {_ref_text_cache[wav_path][:80]!r}")
                except Exception as e:
                    log.warning(f"    ref-text transcription failed for {Path(wav_path).name}: {e}")
                    _ref_text_cache[wav_path] = ""
            del _m
        except Exception as e:
            log.warning(f"  faster-whisper ref-text prefetch failed: {e} — "
                        f"F5-TTS will transcribe references internally")

    _prepare_ref_wavs()
    _use_ref_text_cache = not _torchcodec_ok()
    if _use_ref_text_cache:
        log.warning("  torchcodec shared library unavailable — pre-transcribing reference clips "
                    "to avoid internal Whisper (vocabulary-bleed artifacts possible)")
        _prefetch_ref_texts()
    else:
        log.info("  torchcodec OK — using F5-TTS auto-transcription (ref_text='' for clean synthesis)")

    # 180 ms silence between sentence chunks — inter-sentence pause.
    # _split_for_tts always splits on sentence endings so every boundary gets this gap.
    chunk_gap = np.zeros(int(sr * 0.18), dtype=np.float32)

    def _synth_one(text: str, ref_wav: str) -> np.ndarray:
        t = text.rstrip()
        if t and t[-1] not in ".!?…":
            t += "."
        # Use pre-transcribed ref_text only as a torchcodec fallback; otherwise
        # pass "" so F5-TTS auto-transcribes and vocabulary bleed cannot occur.
        ref_text_val = _ref_text_cache.get(ref_wav, "") if _use_ref_text_cache else ""
        wav, _, _ = f5.infer(
            ref_file=ref_wav,
            ref_text=ref_text_val,
            gen_text=t,
            nfe_step=config.f5tts_nfe_step,
            cfg_strength=config.f5tts_cfg_strength,
            speed=config.f5tts_speed,
            remove_silence=False,
            file_wave=None,
            seed=None,            # None → random each call, so retries are independent
        )
        return np.asarray(wav, dtype=np.float32)

    def _synth_guarded(text: str, ref_wav: str, seg_id) -> np.ndarray:
        """Synthesize `text`, regenerating on runaway duration or near-silent output."""
        wav = _synth_one(text, ref_wav)
        expected = max(len(text) / _TTS_EXPECTED_CPS, 0.4)
        natural = len(wav) / sr

        is_runaway = (natural > expected * _TTS_RUNAWAY_FACTOR
                      and natural - expected > _TTS_RUNAWAY_MIN_EXCESS_S)
        if is_runaway:
            best = wav
            for _ in range(_TTS_MAX_RETRIES):
                try:
                    cand = _synth_one(text, ref_wav)
                except Exception as e:
                    log.debug(f"Segment {seg_id}: runaway retry failed: {e}")
                    continue
                if len(cand) < len(best):
                    best = cand
                if len(best) / sr <= expected * _TTS_RUNAWAY_FACTOR:
                    break
            if len(best) < len(wav):
                log.warning(
                    f"Segment {seg_id}: runaway TTS {natural:.1f}s for {len(text)} chars "
                    f"(~{expected:.1f}s expected) — regenerated to {len(best)/sr:.1f}s"
                )
            else:
                log.warning(
                    f"Segment {seg_id}: runaway TTS {natural:.1f}s for {len(text)} chars "
                    f"(~{expected:.1f}s expected) — retries did not improve, keeping original"
                )
            wav = best

        rms = float(np.sqrt(np.mean(wav ** 2))) if len(wav) > 0 else 0.0
        min_dur = max(len(text) / _TTS_EXPECTED_CPS * 0.25, 0.15)
        if len(text) > 10 and (rms < 5e-3 or len(wav) / sr < min_dur):
            reason = (f"near-silent (RMS={rms:.4f})" if rms < 5e-3
                      else f"too short ({len(wav)/sr:.2f}s, expected ≥{min_dur:.2f}s)")
            for _ in range(_TTS_MAX_RETRIES):
                try:
                    cand = _synth_one(text, ref_wav)
                except Exception as e:
                    log.debug(f"Segment {seg_id}: quality retry failed: {e}")
                    continue
                cand_rms = float(np.sqrt(np.mean(cand ** 2))) if len(cand) > 0 else 0.0
                if cand_rms >= 5e-3 and len(cand) / sr >= min_dur:
                    log.warning(f"Segment {seg_id}: {reason} — fixed on retry")
                    wav = cand
                    break
            else:
                log.warning(f"Segment {seg_id}: {reason} — retries did not improve")

        return wav

    synthesized: List[Tuple[np.ndarray, float, float]] = []
    with tqdm(total=len(segments), desc="Synthesizing (F5-TTS)") as pbar:
        for seg in segments:
            text = _tts_spoken_form(_seg_text(seg))
            if not text:
                pbar.update(1)
                continue
            try:
                ref_wav = _pick_wav(seg)
                if not ref_wav or not os.path.exists(ref_wav):
                    log.warning(f"Segment {seg['id']}: no speaker reference, skipping")
                    pbar.update(1)
                    continue
                chunks = _split_for_tts(text, _F5TTS_CHUNK_LIMIT)
                if len(chunks) > 1:
                    log.debug(
                        f"Segment {seg['id']}: split into {len(chunks)} chunks "
                        f"({len(text)} chars, limit={_F5TTS_CHUNK_LIMIT})"
                    )
                parts = [_synth_guarded(c, ref_wav, seg["id"]) for c in chunks]
                wav = parts[0] if len(parts) == 1 else np.concatenate(
                    [p for i, part in enumerate(parts) for p in
                     ((part,) if i == 0 else (chunk_gap, part))]
                )
                synthesized.append((wav, seg["start"], seg["end"]))
            except Exception as e:
                log.warning(f"Segment {seg['id']} F5-TTS failed: {e}")
            pbar.update(1)

    log.info(f"✓ Synthesized {len(synthesized)} segments at {sr} Hz")
    del f5
    free_vram(log)
    return synthesized, sr


# ============================================================================
# Step 7: Audio Assembly + Encoding + Background Re-mix
# ============================================================================

_CROSSFADE_MS = 120.0
_FADE_OUT_MS  = 100.0
_FADE_IN_MS   = 30.0

# Loudness target for delivered audio (EBU R128 single-pass loudnorm).
# -16 LUFS integrated / -1.5 dBTP is the common online-video delivery target;
# consistent run-to-run, unlike peak normalization which depends on the single
# loudest sample. volume_boost_pct shifts the integrated target instead of
# post-gain so the true-peak ceiling still holds (no clipping).
_LOUDNORM_I   = -16.0
_LOUDNORM_TP  = -1.5
_LOUDNORM_LRA = 11.0


def _loudnorm_af(volume_boost_pct: float = 0.0) -> str:
    import math
    target_i = _LOUDNORM_I
    if volume_boost_pct:
        target_i += 20.0 * math.log10(1.0 + max(volume_boost_pct, -99.0) / 100.0)
    target_i = min(max(target_i, -30.0), -8.0)
    return f"loudnorm=I={target_i:.1f}:TP={_LOUDNORM_TP}:LRA={_LOUDNORM_LRA}"


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

    trimmed = [_trim_silence(a, top_db=40) for a, _, _ in ordered]

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
            a = _apply_fade_out(a, fade_out_samples)
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

    # Keep the interim WAV peak-safe (it also feeds the background remix);
    # delivery loudness is handled by loudnorm at encode time.
    peak = np.max(np.abs(assembled))
    if peak > 0:
        assembled *= 0.95 / peak

    sf.write(wav_path, assembled, src_rate)
    log.info(f"✓ WAV assembled: {os.path.getsize(wav_path) / 1e6:.1f} MB")

    loudnorm = _loudnorm_af(volume_boost_pct)
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", wav_path,
                "-af", loudnorm,
                "-ar", str(out_rate), "-ac", "2",
                "-c:a", "aac", "-b:a", "192k",
                aac_path,
            ],
            check=True, capture_output=True, timeout=300,
        )
        log.info(
            f"✓ AAC encoded: {os.path.getsize(aac_path) / 1e6:.1f} MB "
            f"@ 192 kbps {out_rate} Hz stereo ({loudnorm})"
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
    # amix normalize=0: without it amix scales each input by 1/n, dropping the
    # French voice ~6 dB below the voice-only track. The final loudnorm both
    # sets delivery loudness and true-peak-limits the un-normalized sum.
    filt = (
        f"[0:a]highpass=f=80,compand,volume={voice_gain:.3f},asplit=2[v_f][v_s];"
        f"[1:a]volume={bg_gain_db}dB[bg_pre];"
        "[bg_pre][v_s]sidechaincompress=threshold=0.08:ratio=12:attack=15:release=400[bg_ducked];"
        "[v_f][bg_ducked]amix=inputs=2:duration=first:dropout_transition=0:normalize=0[mix];"
        f"[mix]{_loudnorm_af(volume_boost_pct)}[out]"
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
    # The dubbed audio is already AAC (m4a) — stream-copy it instead of paying
    # a second lossy encode generation.
    if burn_subs:
        # Escape the SRT path for the subtitles filter (\, :, ' are special).
        esc = srt_path.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")
        cmd = [
            "ffmpeg", "-y", "-i", video_path, "-i", audio_path,
            "-vf", f"subtitles='{esc}'",
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "libx264", "-preset", "medium", "-crf", "18",
            "-c:a", "copy", "-shortest", output_path,
        ]
    else:
        # No -shortest here: a soft subtitle track whose last cue ends before the
        # video (trailing no-speech footage) would otherwise truncate the video.
        # The source video length stays authoritative; the dubbed audio is already
        # held to ~that length by the anchored timing policy.
        cmd = [
            "ffmpeg", "-y", "-i", video_path, "-i", audio_path, "-i", srt_path,
            "-map", "0:v:0", "-map", "1:a:0", "-map", "2:0",
            "-c:v", "copy", "-c:a", "copy",
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


def create_english_srt(
    segments: List[dict],
    output_path: str,
    log: logging.Logger,
) -> bool:
    """Write a simple SRT from English source segments using their original timing."""
    try:
        subs = pysrt.SubRipFile()
        for idx, seg in enumerate(segments, 1):
            text = (seg.get("text") or "").strip()
            if not text:
                continue
            subs.append(SubRipItem(
                index=idx,
                start=SubRipTime(seconds=max(0.0, seg["start"])),
                end=SubRipTime(seconds=max(0.0, seg["end"])),
                text=text,
            ))
        subs.save(output_path, encoding="utf-8")
        log.info(f"  English SRT: {len(subs)} cues → {Path(output_path).name}")
        return True
    except Exception as e:
        log.warning(f"Failed to write English SRT: {e}")
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

    metrics = MetricsCollector(
        enabled=config.metrics_enabled,
        output_dir=output_dir,
        name=name,
        log=log,
        budget_cps=config.translation_budget_cps,
        cps_split_threshold=config.cps_split_threshold,
        max_stretch=config.tts_max_stretch,
        readability_cap=config.subtitle_max_cps,
    )

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
    if total_duration <= 0:
        log.error("Source duration unreadable — aborting (anchored timing needs it)")
        return False

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
        language=config.whisper_language,
        initial_prompt=config.whisper_initial_prompt,
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

    # Diarize BEFORE merging so speaker changes become merge boundaries —
    # a merged chunk must never span two speakers, or the whole chunk gets
    # dubbed in one cloned voice.
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
                _dump_segments(segments, temp_dir, "02c_diarized", log)
        else:
            log.warning("  Diarization failed — all segments assigned to SPEAKER_00")
            for seg in segments:
                seg["speaker"] = "SPEAKER_00"

    segments = merge_segments(
        segments,
        max_gap=config.segment_merge_gap,
        max_duration=config.segment_merge_max_duration,
        min_duration=config.segment_merge_min_duration,
        log=log,
    )

    if config.keep_temp:
        _dump_segments(segments, temp_dir, "03_merged", log)

    # Source baseline (English only) — the timeline the French must fit.
    metrics.snapshot("merged_source", segments)

    # ── 3. Translate ────────────────────────────────────────────────────────
    log.info(f"\n[3/6] TRANSLATING ({config.translation_model} via Ollama)")
    segments = translate_segments(
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

    segments = _scan_and_fix_hallucinations(
        segments, config.translation_model, log,
        target_lang=config.target_lang, locale=config.locale,
        glossary_section=glossary_section,
    )

    if config.keep_temp:
        _dump_segments(segments, temp_dir, "05_translated", log)

    metrics.snapshot("translated", segments)

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
        metrics.snapshot("reviewed", segments)

    # Glossary BEFORE compression: substitutions can lengthen text
    # (week-end → fin de semaine), so the budget check must measure the
    # final Québécois forms; the compression prompt carries the glossary
    # so rewrites don't undo them.
    if glossary.entries:
        log.info("\n[3c/6] APPLYING GLOSSARY (deterministic substitution)")
        segments = apply_glossary(segments, glossary.entries, log)
        if config.keep_temp:
            _dump_segments(segments, temp_dir, "06_glossary", log)
        metrics.snapshot("glossary", segments)

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
            glossary_section=glossary_section,
        )
        if config.keep_temp:
            _dump_segments(segments, temp_dir, "06b_compressed", log)
        metrics.snapshot("compressed", segments)

    # CPS guard: split segments whose final French text would force the
    # assembler past max_stretch. Splits at sentence boundary; halves
    # are stretched independently and end up closer to natural rate.
    if config.cps_split_threshold > 0:
        segments = split_overflowing_segments(
            segments, log, max_cps=config.cps_split_threshold,
        )
        if config.keep_temp:
            _dump_segments(segments, temp_dir, "07b_cps_split", log)
        metrics.snapshot("cps_split", segments)

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

    # Unload the translation LLM from VRAM before F5-TTS loads.
    _ollama_unload(config.translation_model, log)

    # ── 4. Speaker reference(s) ─────────────────────────────────────────────
    log.info("\n[4/6] PREPARING SPEAKER REFERENCE(S)")
    speaker_wav, speaker_profiles = resolve_speaker_references(
        vocals_wav, segments, temp_dir, output_dir, name, config, log,
        diarization_turns=diarization_turns,
    )

    # ── 5. TTS synthesis ────────────────────────────────────────────────────
    log.info("\n[5/6] SYNTHESIZING FRENCH AUDIO (F5-TTS)")
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

    # Ground-truth fit: did the natural TTS length fit each segment's window?
    metrics.record_synthesis_fit(segments, synthesized, actual_sr)

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

    # Final phase: French text against the timing it actually plays at.
    metrics.snapshot("final_retimed", srt_segments)
    metrics.finalize()

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


def process_video_phase1(
    video_path: str,
    output_dir: str,
    config: PipelineConfig,
    log: logging.Logger,
    force: bool = False,
) -> bool:
    """Steps 1–3: source separation, transcription, translation.

    Writes ``{output_dir}/{name}_segments.json`` and
    ``{output_dir}/{name}_english.srt``.  Temp files are intentionally
    preserved so Phase 2 can reuse ``vocals.wav``.
    """
    name = Path(video_path).stem
    os.makedirs(output_dir, exist_ok=True)
    temp_dir = os.path.join(config.temp_folder, name)
    os.makedirs(temp_dir, exist_ok=True)

    seg_file    = os.path.join(output_dir, f"{name}_segments.json")
    eng_srt     = os.path.join(output_dir, f"{name}_english.srt")

    if not force and os.path.exists(seg_file) and os.path.exists(eng_srt):
        log.info(f"SKIP {name} Phase 1 — segments already exist (use --force to reprocess)")
        return True

    log.info(f"\n{'=' * 60}\nPipeline v1.0 Phase 1: {name}\n{'=' * 60}")

    metrics = MetricsCollector(
        enabled=config.metrics_enabled,
        output_dir=output_dir,
        name=name,
        log=log,
        budget_cps=config.translation_budget_cps,
        cps_split_threshold=config.cps_split_threshold,
        max_stretch=config.tts_max_stretch,
        readability_cap=config.subtitle_max_cps,
    )

    if not check_ollama(config.translation_model, log):
        return False

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

    # ── 1. Source separation ────────────────────────────────────────────────
    vocals_wav:    Optional[str] = None
    no_vocals_wav: Optional[str] = None  # noqa: F841 (not needed by phase 1, but kept for symmetry)

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
        language=config.whisper_language,
        initial_prompt=config.whisper_initial_prompt,
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

    # Diarize BEFORE merging — speaker changes are merge boundaries (see
    # process_video for rationale).
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
                _dump_segments(segments, temp_dir, "02c_diarized", log)
        else:
            log.warning("  Diarization failed — all segments assigned to SPEAKER_00")
            for seg in segments:
                seg["speaker"] = "SPEAKER_00"

    segments = merge_segments(
        segments,
        max_gap=config.segment_merge_gap,
        max_duration=config.segment_merge_max_duration,
        min_duration=config.segment_merge_min_duration,
        log=log,
    )

    if config.keep_temp:
        _dump_segments(segments, temp_dir, "03_merged", log)

    metrics.snapshot("merged_source", segments)

    # ── 3. Translate ────────────────────────────────────────────────────────
    log.info(f"\n[3/6] TRANSLATING ({config.translation_model} via Ollama)")
    segments = translate_segments(
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

    segments = _scan_and_fix_hallucinations(
        segments, config.translation_model, log,
        target_lang=config.target_lang, locale=config.locale,
        glossary_section=glossary_section,
    )

    if config.keep_temp:
        _dump_segments(segments, temp_dir, "05_translated", log)

    metrics.snapshot("translated", segments)

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
        metrics.snapshot("reviewed", segments)

    # Glossary BEFORE compression (see process_video for rationale).
    if glossary.entries:
        log.info("\n[3c/6] APPLYING GLOSSARY (deterministic substitution)")
        segments = apply_glossary(segments, glossary.entries, log)
        if config.keep_temp:
            _dump_segments(segments, temp_dir, "06_glossary", log)
        metrics.snapshot("glossary", segments)

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
            glossary_section=glossary_section,
        )
        if config.keep_temp:
            _dump_segments(segments, temp_dir, "06b_compressed", log)
        metrics.snapshot("compressed", segments)

    if config.cps_split_threshold > 0:
        segments = split_overflowing_segments(
            segments, log, max_cps=config.cps_split_threshold,
        )
        if config.keep_temp:
            _dump_segments(segments, temp_dir, "07b_cps_split", log)
        metrics.snapshot("cps_split", segments)

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

    # Translation is complete — unload the LLM from VRAM now so Phase 2's
    # F5-TTS has the full GPU budget when it starts (possibly minutes later).
    _ollama_unload(config.translation_model, log)

    # Persist translated segments so Phase 2 (and the review editor) can load them.
    import tempfile as _tf
    fd, tmp = _tf.mkstemp(dir=output_dir, prefix=".seg.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(segments, f, ensure_ascii=False, indent=2)
        os.replace(tmp, seg_file)
    except Exception as e:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        log.error(f"Failed to write segments file: {e}")
        return False

    create_english_srt(segments, eng_srt, log)

    log.info(f"\n{'=' * 60}")
    log.info(f"PHASE 1 DONE: {name}")
    log.info(f"  Segments: {seg_file}")
    log.info(f"  English SRT: {eng_srt}")
    log.info(f"  Temp files preserved at: {temp_dir}  (Phase 2 needs vocals.wav)")
    log.info(f"{'=' * 60}\n")
    return True


def process_video_phase2(
    video_path: str,
    output_dir: str,
    config: PipelineConfig,
    log: logging.Logger,
    segments_file: Optional[str] = None,
    force: bool = False,
) -> bool:
    """Steps 4–6: speaker profiles, TTS synthesis, assemble + encode + SRT.

    Reads segments from ``segments_file`` (defaults to
    ``{output_dir}/{name}_segments.json``).  Recovers ``vocals.wav`` from
    the Phase 1 temp directory; re-runs extraction if it is missing.
    """
    name = Path(video_path).stem
    os.makedirs(output_dir, exist_ok=True)
    temp_dir = os.path.join(config.temp_folder, name)
    os.makedirs(temp_dir, exist_ok=True)

    final_aac = os.path.join(output_dir, f"{name}_french.m4a")
    final_srt = os.path.join(output_dir, f"{name}_french.srt")

    if not force and os.path.exists(final_aac) and os.path.exists(final_srt):
        log.info(f"SKIP {name} Phase 2 — outputs exist (use --force to reprocess)")
        return True

    log.info(f"\n{'=' * 60}\nPipeline v1.0 Phase 2: {name}\n{'=' * 60}")

    if segments_file is None:
        segments_file = os.path.join(output_dir, f"{name}_segments.json")

    if not os.path.exists(segments_file):
        log.error(f"Segments file not found: {segments_file}  (run Phase 1 first)")
        return False

    with open(segments_file, encoding="utf-8") as f:
        segments = json.load(f)
    log.info(f"  Loaded {len(segments)} segments from {Path(segments_file).name}")

    total_duration = get_duration(video_path, log)
    if total_duration <= 0:
        log.error("Source duration unreadable — aborting (anchored timing needs it)")
        return False

    # Recover vocals.wav written by Phase 1.
    vocals_wav:    Optional[str] = None
    no_vocals_wav: Optional[str] = None

    vocals_files = sorted(Path(temp_dir).rglob("vocals.wav"))
    if vocals_files:
        vocals_wav = str(vocals_files[-1])

    no_vocals_files = sorted(Path(temp_dir).rglob("no_vocals.wav"))
    if no_vocals_files:
        no_vocals_wav = str(no_vocals_files[-1])

    if not vocals_wav:
        raw = Path(temp_dir) / f"{name}.wav"
        if raw.exists():
            vocals_wav = str(raw)

    if not vocals_wav:
        log.warning("  vocals.wav not found in temp dir — re-extracting audio")
        if config.use_demucs:
            vocals_wav, no_vocals_wav = separate_vocals(
                video_path, temp_dir, config.demucs_model, log
            )
        if not vocals_wav:
            raw_wav = os.path.join(temp_dir, f"{name}.wav")
            if not extract_audio(video_path, raw_wav, config.synthesis_sample_rate, log):
                return False
            vocals_wav = raw_wav

    metrics = MetricsCollector(
        enabled=config.metrics_enabled,
        output_dir=output_dir,
        name=name,
        log=log,
        budget_cps=config.translation_budget_cps,
        cps_split_threshold=config.cps_split_threshold,
        max_stretch=config.tts_max_stretch,
        readability_cap=config.subtitle_max_cps,
    )
    metrics.snapshot("phase2_input", segments)

    # Ensure the translation LLM is not still resident from Phase 1.
    _ollama_unload(config.translation_model, log)

    # ── 4. Speaker reference(s) ─────────────────────────────────────────────
    log.info("\n[4/6] PREPARING SPEAKER REFERENCE(S)")
    speaker_wav, speaker_profiles = resolve_speaker_references(
        vocals_wav, segments, temp_dir, output_dir, name, config, log,
        diarization_turns=None,  # speaker labels already embedded in segments
    )

    # ── 5. TTS synthesis ────────────────────────────────────────────────────
    log.info("\n[5/6] SYNTHESIZING FRENCH AUDIO (F5-TTS)")
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

    metrics.record_synthesis_fit(segments, synthesized, actual_sr)

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

    mux_audio = final_aac
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

    metrics.snapshot("final_retimed", srt_segments)
    metrics.finalize()

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
    log.info(f"PHASE 2 DONE: {name}")
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
@click.option(
    "--phase",
    type=click.Choice(["1", "2"]),
    default=None,
    help="Run only phase 1 (translate+save) or phase 2 (TTS+dub).",
)
@click.option(
    "--segments-file",
    type=click.Path(),
    default=None,
    help="Phase 2: load segments from this path instead of the default location.",
)
def main(
    video: str,
    output_dir: str,
    config_path: str,
    force: bool,
    locale: Optional[str],
    volume_boost: Optional[float],
    keep_temp: bool,
    phase: Optional[str],
    segments_file: Optional[str],
) -> None:
    """Dub a video to French (audio track + SRT subtitles).

    Canadian French:
      --locale fr-ca

    Two-phase (for web review):
      --phase 1   run transcription + translation, then stop
      --phase 2   run TTS + assembly (reads segments file from phase 1)
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
    log = setup_logging(config.logs_folder, Path(video).stem)
    if phase == "1":
        success = process_video_phase1(video, output_dir, config, log, force=force)
    elif phase == "2":
        success = process_video_phase2(
            video, output_dir, config, log,
            segments_file=segments_file, force=force,
        )
    else:
        success = process_video(video, output_dir, config, log, force=force)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
