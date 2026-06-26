"""FastAPI web UI for the dubbing pipeline.

Single-job FIFO queue; one background worker subprocesses 02_pipeline.py for
each job. Live status comes from streaming the subprocess's stdout (which is
also the pipeline's own log) over Server-Sent Events.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from collections import deque
from pathlib import Path
from typing import Dict, Optional
from urllib.parse import urlparse

import yaml
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import (
    FileResponse, HTMLResponse, JSONResponse, StreamingResponse,
)
from fastapi.staticfiles import StaticFiles

from .jobs import (
    Job, JOBS_FILE, STATUS_AWAITING_REVIEW, STATUS_CANCELLED, STATUS_COMPLETED,
    STATUS_FAILED, STATUS_QUEUED, STATUS_RUNNING, TERMINAL,
    load_jobs, new_job_id, safe_stem, save_jobs, sorted_jobs,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
WEB_ROOT     = Path(__file__).resolve().parent
STATIC_DIR   = WEB_ROOT / "static"
WORKSPACE    = Path(os.environ.get("DUBBING_WORKSPACE", "/workspace"))
UPLOAD_DIR   = WORKSPACE / "web" / "uploads"
OUTPUT_DIR   = WORKSPACE / "web" / "outputs"
LOG_DIR      = WORKSPACE / "logs"
CONFIG_PATH  = WORKSPACE / "config.yaml"
PIPELINE_PY  = WORKSPACE / "scripts" / "02_pipeline.py"

# Mirror config writes to the source repo so the two copies stay in sync.
# Any path that exists is written; missing paths are silently skipped.
CONFIG_MIRRORS = [
    Path("/workspace/french-dubbing/config.yaml"),
]

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ── Choices (mirror Click definitions in 02_pipeline.py) ──────────────────────
LOCALE_CHOICES = ["fr", "fr-ca"]

# ── Advanced config schema ────────────────────────────────────────────────────
# Each entry: dotted path → (type, label, help, ui hints).
# UI hints: {"min", "max", "step", "choices", "group"}.
# Only paths declared here are writable via /api/config; everything else in
# config.yaml is preserved verbatim on save.
CONFIG_SCHEMA: dict = {
    # Audio
    "audio.volume_boost_pct":          ("int",   "Volume boost (%)",        "0 = off. Applied after peak normalization; hard-clipped at ±1.0.", {"min": -50, "max": 100, "step": 1, "group": "Audio"}),

    # Diarization
    "diarization.enabled":             ("bool",  "Enable diarization",      "Detect speakers and clone a distinct voice per speaker.", {"group": "Diarization"}),
    "diarization.min_speakers":        ("int",   "Min speakers",            "Set min == max to force an exact count.", {"min": 1, "max": 20, "step": 1, "group": "Diarization"}),
    "diarization.max_speakers":        ("int",   "Max speakers",            "", {"min": 1, "max": 20, "step": 1, "group": "Diarization"}),
    "diarization.profile_duration":    ("int",   "Profile duration (s)",    "Seconds of per-speaker audio for voice cloning. Must stay ≤ 15s — F5-TTS breaks above 22s (pipeline hard-caps at 15s).", {"min": 5, "max": 15, "step": 1, "group": "Diarization"}),

    # Source separation
    "source_separation.enabled":            ("bool",  "Enable source separation", "Separate vocals from music before transcription/cloning.", {"group": "Source separation"}),
    "source_separation.model":              ("str",   "Demucs model",           "", {"choices": ["htdemucs", "htdemucs_ft", "mdx_extra"], "group": "Source separation"}),
    "source_separation.preserve_background":("bool",  "Preserve background",    "Remix original music/ambience under French vocals into *_french_full.m4a.", {"group": "Source separation"}),

    # Whisper
    "whisper.model":                   ("str",   "Whisper model",           "", {"choices": ["large-v3", "large-v2", "medium", "small", "distil-large-v3"], "group": "Whisper"}),
    "whisper.compute_type":            ("str",   "Compute type",            "", {"choices": ["float16", "int8_float16", "int8", "float32"], "group": "Whisper"}),
    "whisper.condition_on_previous_text": ("bool", "Condition on prev text", "Off = strongest anti-hallucination posture; Whisper won't feed its own (possibly looped) output back as context.", {"group": "Whisper"}),
    "whisper.compression_ratio_threshold": ("float", "Compression-ratio max", "Reject segments above this ratio (a classic loop signature). Lower = more aggressive. fw default 2.4.", {"min": 1.5, "max": 4.0, "step": 0.05, "group": "Whisper"}),
    "whisper.no_speech_threshold":     ("float", "No-speech threshold",     "Drop windows whose no-speech probability exceeds this.", {"min": 0.0, "max": 1.0, "step": 0.05, "group": "Whisper"}),

    # Translation
    "translation.model":               ("str",   "Translation model",       "Ollama tag (e.g. qwen3:14b, qwen3:32b).", {"group": "Translation"}),
    "translation.temperature":         ("float", "Temperature",             "Lower = more literal, higher = more creative.", {"min": 0.0, "max": 1.5, "step": 0.05, "group": "Translation"}),
    "translation.batch_size":          ("int",   "Batch size",              "Segments per Ollama call. Big batches risk 4096-token budget.", {"min": 1, "max": 60, "step": 1, "group": "Translation"}),
    "translation.review_pass":         ("bool",  "Self-review pass",        "Qwen rereads and fixes Anglicisms. Roughly doubles translation time.", {"group": "Translation"}),
    "translation.compression_pass":    ("bool",  "Compression fallback",    "Targeted second Qwen pass that only rewrites segments still over budget after the main translation. Cheap and eliminates most remaining speed-ups.", {"group": "Translation"}),
    "translation.budget_cps":          ("int",   "Char budget per second",  "Per-segment character budget passed to Qwen. ~15 keeps French tight enough to fit the source timeline; raise for more headroom (reintroduces drift), lower to force tighter phrasing.", {"min": 10, "max": 25, "step": 1, "group": "Translation"}),
    "translation.compression_rounds":  ("int",   "Compression rounds",      "Max iterative passes that re-compress only the segments still over budget. More rounds = tighter timing fit, slightly more LLM calls.", {"min": 1, "max": 6, "step": 1, "group": "Translation"}),
    "translation.target_lang":         ("str",   "Target language",         "", {"choices": ["fr", "es", "de", "it", "pt", "nl", "pl", "ru", "ja", "ko", "zh", "ar", "tr", "hi", "vi"], "group": "Translation"}),
    "translation.locale":              ("str",   "Locale variant",          "fr-ca triggers the Canadian glossary.", {"choices": LOCALE_CHOICES, "group": "Translation"}),

    # TTS
    "tts.timing_policy":               ("str",   "Timing policy",           "anchored holds the source timeline (speed dense runs up, re-anchor drift at pauses) so the dub stays in sync over a full program. no_drop never speeds up and drifts longer than the video. lock truncates overflow.", {"choices": ["anchored", "no_drop", "lock"], "group": "TTS"}),
    "tts.f5tts_nfe_step":              ("int",   "F5-TTS ODE steps",        "More steps = better quality, slower. 16 = fast draft, 32 = high quality, 64 = max.", {"min": 8, "max": 64, "step": 4, "group": "TTS"}),
    "tts.f5tts_cfg_strength":          ("float", "CFG strength",             "Classifier-free guidance. Higher = more faithful to reference voice.", {"min": 0.5, "max": 5.0, "step": 0.25, "group": "TTS"}),
    "tts.speaker_profile_duration":    ("int",   "Speaker clip duration (s)", "Length of reference clip for voice cloning (single-speaker mode). Must stay ≤ 15s.", {"min": 5, "max": 15, "step": 1, "group": "TTS"}),
    "tts.use_deepfilter":              ("bool",  "Denoise reference",       "Denoise the speaker reference clip before cloning.", {"group": "TTS"}),
    "tts.max_stretch":                 ("float", "Max stretch ratio",       "anchored: per-group speed-up cap used to hold the source timeline (~1.30 is inaudible). lock: above this the tail is truncated instead of sped up further.", {"min": 1.0, "max": 2.0, "step": 0.05, "group": "TTS"}),
    "tts.min_stretch":                 ("float", "Min stretch ratio",       "Floor for slowing down audio to fill long windows.", {"min": 0.3, "max": 1.0, "step": 0.05, "group": "TTS"}),
    "tts.stretcher":                   ("str",   "Time-stretch engine",     "rubberband preserves formants; atempo is the ffmpeg fallback.", {"choices": ["rubberband", "atempo"], "group": "TTS"}),
    "tts.cps_split_threshold":         ("float", "CPS split threshold",     "Translated segments above this French CPS get split at a sentence boundary before TTS. Shorter halves stretch independently. 0 disables.", {"min": 0.0, "max": 40.0, "step": 0.5, "group": "TTS"}),

    # Subtitles
    "subtitles.sync_offset_ms":        ("int",   "Subtitle offset (ms)",    "Positive = later, negative = earlier.", {"min": -10000, "max": 10000, "step": 50, "group": "Subtitles"}),

    # Output
    "output.mux_video":                ("bool",  "Mux final video",         "Also emit {name}_french.mp4 (original video + dubbed audio + subtitles). Audio is held to the source length so they end together.", {"group": "Output"}),
    "output.burn_subs":                ("bool",  "Burn-in subtitles",       "On = render subtitles into the picture (re-encodes video). Off = soft-embed the SRT track and copy the video stream.", {"group": "Output"}),
}


# Curated presets. Each value must also be a valid value per CONFIG_SCHEMA;
# the PUT validator catches drift if someone edits a preset wrong.
CONFIG_PRESETS: dict[str, dict] = {
    "fast_draft": {
        "label": "Fast draft",
        "desc": "Quickest turnaround; lower fidelity. Skip diarization, smaller whisper, no review/compression pass.",
        "values": {
            "whisper.model": "distil-large-v3",
            "whisper.compute_type": "int8_float16",
            "whisper.condition_on_previous_text": False,
            "whisper.compression_ratio_threshold": 2.4,  # less aggressive — speed over accuracy
            "diarization.enabled": False,
            "source_separation.enabled": True,
            "source_separation.model": "mdx_extra",
            "translation.review_pass": False,
            "translation.compression_pass": False,        # skip the extra Qwen call
            "translation.batch_size": 30,
            "translation.budget_cps": 19,                 # accept slightly longer FR
            "tts.use_deepfilter": False,
            "tts.stretcher": "atempo",
            "tts.cps_split_threshold": 24.0,              # less aggressive splitting
        },
    },
    "high_quality": {
        "label": "High quality",
        "desc": "Best fidelity end-to-end. Strong anti-hallucination, review + compression passes, rubberband stretcher.",
        "values": {
            "whisper.model": "large-v3",
            "whisper.compute_type": "float16",
            "whisper.condition_on_previous_text": False,
            "whisper.compression_ratio_threshold": 2.0,   # aggressive loop rejection
            "whisper.no_speech_threshold": 0.6,
            "diarization.enabled": True,
            "source_separation.enabled": True,
            "source_separation.model": "htdemucs",
            "source_separation.preserve_background": True,
            "translation.review_pass": True,
            "translation.compression_pass": True,         # squeeze out remaining overflows
            "translation.compression_rounds": 3,          # iterate until segments fit
            "translation.batch_size": 15,
            "translation.temperature": 0.25,
            "translation.budget_cps": 15,                 # tight FR → fits the source timeline
            "tts.use_deepfilter": True,
            "tts.timing_policy": "anchored",              # hold sync over long programs
            "tts.stretcher": "rubberband",
            "tts.max_stretch": 1.3,
            "tts.cps_split_threshold": 20.0,              # tighter split threshold
        },
    },
    "voice_clone_focus": {
        "label": "Voice-cloning focus",
        "desc": "Optimise speaker fidelity: long reference clip, denoise, tight stretch limits, aggressive compression.",
        "values": {
            "diarization.enabled": True,
            "diarization.profile_duration": 15,
            "translation.compression_pass": True,         # short utterances clone better
            "translation.budget_cps": 16,
            "tts.speaker_profile_duration": 15,
            "tts.use_deepfilter": True,
            "tts.f5tts_nfe_step": 32,
            "tts.f5tts_cfg_strength": 2.5,                # stronger reference adherence
            "tts.max_stretch": 1.2,
            "tts.min_stretch": 0.8,
            "tts.stretcher": "rubberband",
            "tts.cps_split_threshold": 19.0,
        },
    },
}


def _get_dotted(obj: dict, dotted: str):
    cur = obj
    for k in dotted.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def _set_dotted(obj: dict, dotted: str, value) -> None:
    keys = dotted.split(".")
    cur = obj
    for k in keys[:-1]:
        nxt = cur.get(k)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[k] = nxt
        cur = nxt
    cur[keys[-1]] = value


def _coerce(value, typ: str):
    if typ == "bool":
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "on", "yes")
        return bool(value)
    if typ == "int":
        return int(value)
    if typ == "float":
        return float(value)
    if typ == "str":
        return str(value)
    raise ValueError(f"unknown type {typ}")

MAX_UPLOAD_BYTES = 5 * 1024 * 1024 * 1024  # 5 GB
LOG_BUFFER_LINES = 500

PHASE_RE = re.compile(r"\[(\d+)/6\]\s+(.+)")

log = logging.getLogger("dubbing.web")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# ── App state ─────────────────────────────────────────────────────────────────
class State:
    def __init__(self) -> None:
        self.jobs: Dict[str, Job] = load_jobs()
        self.queue: asyncio.Queue[str] = asyncio.Queue()
        # Per-job log subscribers (asyncio queues) for SSE fan-out
        self.subscribers: Dict[str, list] = {}
        # Per-job in-memory log ring buffer for late-joiners
        self.log_buffers: Dict[str, deque] = {}
        # Running process handle (only one at a time)
        self.current_proc: Optional[asyncio.subprocess.Process] = None
        self.current_job_id: Optional[str] = None
        self.worker_task: Optional[asyncio.Task] = None

    def save(self) -> None:
        save_jobs(self.jobs)

    def publish(self, job_id: str, line: str) -> None:
        buf = self.log_buffers.setdefault(job_id, deque(maxlen=LOG_BUFFER_LINES))
        buf.append(line)
        for q in self.subscribers.get(job_id, []):
            try:
                q.put_nowait(line)
            except asyncio.QueueFull:
                pass


state = State()
app = FastAPI(title="Dubbing Web UI")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ── Queue worker ──────────────────────────────────────────────────────────────
async def _run_job(job: Job) -> None:
    """Execute one pipeline subprocess; stream its stdout into the job's log buffer."""
    job.status = STATUS_RUNNING
    job.started_at = time.time()
    state.current_job_id = job.id
    state.save()
    state.publish(job.id, f">>> Starting: {job.video_filename}")

    # Snapshot the exact config that will produce this dub. Lets you diff
    # settings across runs and re-run the same job with identical params.
    try:
        snap = Path(job.output_dir) / "config.snapshot.yaml"
        if CONFIG_PATH.exists():
            snap.write_bytes(CONFIG_PATH.read_bytes())
            state.publish(job.id, f"... snapshot: {snap}")
    except Exception as e:
        state.publish(job.id, f"!!! snapshot failed: {e}")

    # Pending Vimeo job — fetch the video before running the pipeline.
    if not job.video_path and job.source_url:
        if not await _download_vimeo(job):
            return  # _download_vimeo marked the job failed and saved state

    opts = job.options or {}
    _is_phase1 = opts.get("review") and not opts.get("_p2")
    _is_phase2 = bool(opts.get("_p2"))

    cmd = [
        sys.executable, str(PIPELINE_PY),
        "--video",      job.video_path,
        "--output-dir", job.output_dir,
        "--config",     str(CONFIG_PATH),
    ]
    if opts.get("force"):
        cmd.append("--force")
    if opts.get("locale"):
        cmd += ["--locale", opts["locale"]]
    if opts.get("volume_boost") not in (None, ""):
        cmd += ["--volume-boost", str(opts["volume_boost"])]
    if _is_phase1:
        cmd += ["--phase", "1"]
    elif _is_phase2:
        cmd += ["--phase", "2"]

    state.publish(job.id, "$ " + " ".join(cmd))

    try:
        # start_new_session so we can kill the whole process group on cancel
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    except Exception as e:
        job.status = STATUS_FAILED
        job.error = f"failed to launch pipeline: {e}"
        job.ended_at = time.time()
        state.publish(job.id, f"!!! launch failed: {e}")
        state.current_job_id = None
        state.save()
        return

    state.current_proc = proc

    try:
        assert proc.stdout is not None
        while True:
            line_bytes = await proc.stdout.readline()
            if not line_bytes:
                break
            line = line_bytes.decode("utf-8", errors="replace").rstrip()
            if not line:
                continue
            m = PHASE_RE.search(line)
            if m:
                job.phase = f"[{m.group(1)}/6] {m.group(2)}".strip()
            state.publish(job.id, line)
    except Exception as e:
        state.publish(job.id, f"!!! log-stream error: {e}")

    rc = await proc.wait()
    job.returncode = rc
    job.ended_at = time.time()

    # Determine final status
    if job.status == STATUS_CANCELLED:
        pass  # cancel already set status
    elif rc == 0:
        _collect_outputs(job)
        if _is_phase1:
            job.status = STATUS_AWAITING_REVIEW
        else:
            job.status = STATUS_COMPLETED
    else:
        job.status = STATUS_FAILED
        job.error = job.error or f"pipeline exited with code {rc}"

    state.publish(job.id, f"<<< {job.status.upper()} (rc={rc})")
    # Signal SSE subscribers to close cleanly
    for q in state.subscribers.get(job.id, []):
        try:
            q.put_nowait(None)
        except asyncio.QueueFull:
            pass

    state.current_proc = None
    state.current_job_id = None
    state.save()


async def _download_vimeo(job: Job) -> bool:
    """Fetch a Vimeo video into UPLOAD_DIR via yt-dlp, streaming progress to the
    job log. On success sets job.video_path / video_filename and returns True.
    On failure marks the job failed, saves state, and returns False."""
    url = job.source_url
    out_tmpl = str(UPLOAD_DIR / f"{job.id}__%(title).80s.%(ext)s")
    cmd = [
        "yt-dlp",
        "--no-playlist", "--newline", "--restrict-filenames",
        "-f", "bv*+ba/b",
        "--remux-video", "mp4",
        "-o", out_tmpl,
        url,
    ]
    # [0/6] so the existing PHASE_RE renders it as a phase in the UI.
    state.publish(job.id, "[0/6] Downloading from Vimeo…")
    state.publish(job.id, "$ " + " ".join(cmd))

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    except FileNotFoundError:
        return _fail_job(job, "yt-dlp not installed on the server")
    except Exception as e:
        return _fail_job(job, f"failed to launch yt-dlp: {e}")

    state.current_proc = proc
    try:
        assert proc.stdout is not None
        while True:
            line_bytes = await proc.stdout.readline()
            if not line_bytes:
                break
            line = line_bytes.decode("utf-8", errors="replace").rstrip()
            if line:
                state.publish(job.id, line)
    except Exception as e:
        state.publish(job.id, f"!!! download-stream error: {e}")

    rc = await proc.wait()
    if job.status == STATUS_CANCELLED:
        return False
    if rc != 0:
        return _fail_job(job, f"yt-dlp exited with code {rc}")

    # yt-dlp names the file from the title; find what it produced for this job.
    produced = sorted(UPLOAD_DIR.glob(f"{job.id}__*"))
    if not produced:
        return _fail_job(job, "download finished but no file was produced")
    dest = produced[0]
    job.video_path = str(dest)
    job.video_filename = dest.name
    state.publish(job.id, f"... downloaded: {dest.name}")
    state.save()
    return True


def _fail_job(job: Job, msg: str) -> bool:
    """Mark a job failed with a message; returns False for convenient early-return."""
    job.status = STATUS_FAILED
    job.error = msg
    job.ended_at = time.time()
    state.publish(job.id, f"!!! {msg}")
    state.publish(job.id, f"<<< {job.status.upper()}")
    # Signal SSE subscribers to close cleanly (mirrors _run_job's terminal handling)
    for q in state.subscribers.get(job.id, []):
        try:
            q.put_nowait(None)
        except asyncio.QueueFull:
            pass
    state.current_job_id = None
    state.current_proc = None
    state.save()
    return False


def _collect_outputs(job: Job) -> None:
    """Populate job.outputs by globbing the per-job output directory."""
    od = Path(job.output_dir)
    if not od.exists():
        return
    video   = sorted(od.glob("*_french.mp4"))
    audio   = sorted(od.glob("*_french.m4a"))
    srt     = sorted(od.glob("*_french.srt"))
    full    = sorted(od.glob("*_french_full.m4a"))
    eng_srt = sorted(od.glob("*_english.srt"))
    if video:
        job.outputs["video"] = str(video[0])
    if audio:
        job.outputs["audio"] = str(audio[0])
    if srt:
        job.outputs["srt"] = str(srt[0])
    if full:
        job.outputs["full"] = str(full[0])
    if eng_srt:
        job.outputs["english_srt"] = str(eng_srt[0])


async def _queue_worker() -> None:
    """Consume job IDs from the queue and run them one at a time."""
    while True:
        job_id = await state.queue.get()
        job = state.jobs.get(job_id)
        if not job:
            state.queue.task_done()
            continue
        if job.status != STATUS_QUEUED:
            state.queue.task_done()
            continue
        try:
            await _run_job(job)
        except Exception as e:
            job.status = STATUS_FAILED
            job.error = f"worker exception: {e}"
            job.ended_at = time.time()
            state.save()
            log.exception("worker failed for job %s", job_id)
        finally:
            state.queue.task_done()


# ── Lifecycle ────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def _startup() -> None:
    # Backfill outputs for completed jobs so newly-recognized artifacts
    # (e.g. the muxed *_french.mp4) appear for jobs finished before this build.
    dirty = False
    for j in state.jobs.values():
        if j.status == STATUS_COMPLETED and j.output_dir:
            before = dict(j.outputs)
            _collect_outputs(j)
            if j.outputs != before:
                dirty = True
    if dirty:
        state.save()
    # Re-enqueue any queued jobs (preserved across restarts)
    for j in sorted(state.jobs.values(), key=lambda j: j.queued_at):
        if j.status == STATUS_QUEUED:
            await state.queue.put(j.id)
    state.worker_task = asyncio.create_task(_queue_worker())
    log.info("dubbing web UI started")


@app.on_event("shutdown")
async def _shutdown() -> None:
    if state.current_proc and state.current_proc.returncode is None:
        try:
            os.killpg(os.getpgid(state.current_proc.pid), signal.SIGTERM)
        except Exception:
            pass


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    return HTMLResponse(content=html)


@app.get("/api/options")
async def options() -> JSONResponse:
    defaults = {
        "locale": "fr",
        "volume_boost": 0,
    }
    try:
        if CONFIG_PATH.exists():
            with CONFIG_PATH.open(encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            defaults["locale"] = cfg.get("translation", {}).get("locale", defaults["locale"])
            defaults["volume_boost"] = cfg.get("audio", {}).get("volume_boost_pct", defaults["volume_boost"])
    except Exception as e:
        log.warning("failed to read config defaults: %s", e)
    return JSONResponse({
        "locales": LOCALE_CHOICES,
        "defaults": defaults,
        "config_path": str(CONFIG_PATH),
    })


@app.get("/api/config")
async def get_config() -> JSONResponse:
    """Return the editable subset of config.yaml plus the schema."""
    try:
        cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}
    except Exception as e:
        raise HTTPException(500, f"failed to read config: {e}")
    values: dict = {}
    for dotted in CONFIG_SCHEMA:
        v = _get_dotted(cfg, dotted)
        if v is not None:
            values[dotted] = v
    schema_out = [
        {"path": p, "type": t, "label": label, "help": help_, **hints}
        for p, (t, label, help_, hints) in CONFIG_SCHEMA.items()
    ]
    presets_out = [
        {"id": pid, "label": p["label"], "desc": p["desc"], "values": p["values"]}
        for pid, p in CONFIG_PRESETS.items()
    ]
    return JSONResponse({
        "values": values,
        "schema": schema_out,
        "presets": presets_out,
        "path": str(CONFIG_PATH),
        "job_running": state.current_job_id is not None,
    })


@app.put("/api/config")
async def update_config(payload: dict, force: bool = False) -> JSONResponse:
    """Update editable keys in config.yaml and mirror to the source repo.

    Body: {"values": {"<dotted.path>": <value>, ...}}.
    Unknown keys are rejected; values are coerced per schema.
    NOTE: YAML comments are preserved by an in-place line-rewrite (only the
    matched leaf lines are touched), so the heavily commented config.yaml
    keeps its inline documentation intact.
    """
    updates = payload.get("values") or {}
    if not isinstance(updates, dict):
        raise HTTPException(400, "values must be an object")

    # Hot-reload guard: the pipeline reads config.yaml at job start, so a
    # mid-run edit would surprise the user (running job uses old values; next
    # job uses new ones). Refuse unless force=true.
    if state.current_job_id and not force:
        raise HTTPException(
            409,
            f"a job is currently running ({state.current_job_id}); "
            f"changes would only apply to the next job. Pass ?force=true to override.",
        )

    # Validate and coerce
    coerced: dict = {}
    for dotted, raw in updates.items():
        spec = CONFIG_SCHEMA.get(dotted)
        if not spec:
            raise HTTPException(400, f"unknown config key: {dotted}")
        typ, _label, _help, hints = spec
        try:
            val = _coerce(raw, typ)
        except (ValueError, TypeError) as e:
            raise HTTPException(400, f"{dotted}: {e}")
        choices = hints.get("choices")
        if choices and val not in choices:
            raise HTTPException(400, f"{dotted}: must be one of {choices}")
        for bound in ("min", "max"):
            if bound in hints and isinstance(val, (int, float)):
                if bound == "min" and val < hints["min"]:
                    raise HTTPException(400, f"{dotted}: must be ≥ {hints['min']}")
                if bound == "max" and val > hints["max"]:
                    raise HTTPException(400, f"{dotted}: must be ≤ {hints['max']}")
        coerced[dotted] = val

    # Cross-field sanity
    cfg_now = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}
    merged = yaml.safe_load(yaml.safe_dump(cfg_now)) or {}
    for k, v in coerced.items():
        _set_dotted(merged, k, v)
    dmin = _get_dotted(merged, "diarization.min_speakers")
    dmax = _get_dotted(merged, "diarization.max_speakers")
    if isinstance(dmin, int) and isinstance(dmax, int) and dmin > dmax:
        raise HTTPException(400, "diarization.min_speakers must be ≤ max_speakers")
    smin = _get_dotted(merged, "tts.min_stretch")
    smax = _get_dotted(merged, "tts.max_stretch")
    if isinstance(smin, (int, float)) and isinstance(smax, (int, float)) and smin > smax:
        raise HTTPException(400, "tts.min_stretch must be ≤ tts.max_stretch")

    # Write — comment-preserving line rewrite for known leaves
    try:
        written = _rewrite_yaml_leaves(CONFIG_PATH, coerced)
    except Exception as e:
        raise HTTPException(500, f"failed to write config: {e}")

    # Mirror to repo copies (best-effort; non-fatal if absent)
    mirrored: list[str] = []
    for mp in CONFIG_MIRRORS:
        try:
            if mp.exists() and mp.resolve() != CONFIG_PATH.resolve():
                _rewrite_yaml_leaves(mp, coerced)
                mirrored.append(str(mp))
        except Exception as e:
            log.warning("mirror write failed for %s: %s", mp, e)

    return JSONResponse({
        "ok": True,
        "updated": written,
        "mirrored": mirrored,
        "values": coerced,
    })


def _rewrite_yaml_leaves(path: Path, updates: dict) -> list[str]:
    """In-place rewrite of `key: value` lines under their parent section.

    Preserves comments, blank lines, and key order. Only top-level mapping
    sections (translation:, tts:, …) are tracked via indentation depth.
    For any key not found, the line is appended under its section header
    (or at end of file if the section is absent).
    """
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()

    # Group updates by top-level section
    by_section: dict[str, dict[str, object]] = {}
    for dotted, val in updates.items():
        head, _, leaf = dotted.partition(".")
        by_section.setdefault(head, {})[leaf] = val

    def _inline_yaml(v) -> str:
        # Render a single-line inline YAML node. safe_dump appends a "\n...\n"
        # document-end marker to bare scalars (and may fold long values across
        # lines), so .strip() alone would leave a stray "..." on its own line and
        # corrupt the file. Take only the first line of a wide, flow-style dump.
        dumped = yaml.safe_dump(v, default_flow_style=True, width=10**9).strip()
        return dumped.split("\n", 1)[0]

    def fmt(v) -> str:
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, (int, float)):
            return repr(v) if isinstance(v, float) else str(v)
        if isinstance(v, str):
            # Quote if it contains chars that YAML would otherwise mis-parse
            if v == "" or any(c in v for c in ":#'\"\n") or v.strip() != v:
                return _inline_yaml(v)
            return v
        return _inline_yaml(v)

    out: list[str] = []
    written: list[str] = []
    cur_section: Optional[str] = None
    section_end_idx: dict[str, int] = {}  # last line index inside each section

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.lstrip()
        indent = len(line) - len(stripped)

        # Top-level mapping header (indent==0, ends with ':')
        if indent == 0 and stripped and not stripped.startswith("#") and stripped.rstrip().endswith(":"):
            cur_section = stripped.rstrip()[:-1].strip()
        elif indent == 0 and stripped == "":
            pass  # blank — section unchanged
        elif indent == 0 and stripped.startswith("#"):
            pass  # top-level comment

        # If inside a tracked section and this is a `key:` line, see if we update
        replaced = False
        if cur_section in by_section and indent > 0 and not stripped.startswith("#"):
            m = re.match(r"^(\s*)([A-Za-z_][\w-]*)\s*:(.*)$", line)
            if m:
                lead, key, rest = m.group(1), m.group(2), m.group(3)
                if key in by_section[cur_section]:
                    val = by_section[cur_section].pop(key)
                    # Preserve any trailing comment
                    comment = ""
                    if "#" in rest:
                        # Only treat as comment if preceded by space or at start
                        idx = rest.find("#")
                        comment = "  " + rest[idx:].strip()
                    out.append(f"{lead}{key}: {fmt(val)}{comment}")
                    written.append(f"{cur_section}.{key}")
                    replaced = True
        if not replaced:
            out.append(line)

        if cur_section is not None and indent > 0 and stripped and not stripped.startswith("#"):
            section_end_idx[cur_section] = len(out) - 1

        i += 1

    # Any leftover keys: append under their section, or at EOF as a new section
    leftovers = {sec: kvs for sec, kvs in by_section.items() if kvs}
    if leftovers:
        # Rebuild with insertions at recorded section_end_idx
        # Easier: append a new block at end for missing keys, with section: header
        for sec, kvs in leftovers.items():
            if sec in section_end_idx:
                insert_at = section_end_idx[sec] + 1
                block = [f"  {k}: {fmt(v)}" for k, v in kvs.items()]
                out[insert_at:insert_at] = block
                # Shift other section_end_idx entries past this point
                for s2, idx2 in list(section_end_idx.items()):
                    if idx2 >= insert_at:
                        section_end_idx[s2] = idx2 + len(block)
            else:
                out.append("")
                out.append(f"{sec}:")
                for k, v in kvs.items():
                    out.append(f"  {k}: {fmt(v)}")
            written.extend(f"{sec}.{k}" for k in kvs)

    path.write_text("\n".join(out) + ("\n" if text.endswith("\n") else ""), encoding="utf-8")
    return written


@app.post("/api/jobs")
async def submit(
    video: UploadFile = File(None),
    vimeo_url: str = Form(""),
    locale: str = Form(""),
    volume_boost: str = Form(""),
    force: str = Form(""),
    review: str = Form(""),
) -> JSONResponse:
    # Validate options against allow-lists (empty = use config default)
    if locale and locale not in LOCALE_CHOICES:
        raise HTTPException(400, f"invalid locale: {locale}")
    vb: Optional[float] = None
    if volume_boost.strip():
        try:
            vb = float(volume_boost)
        except ValueError:
            raise HTTPException(400, "volume_boost must be a number")

    # Exactly one source: an uploaded file OR a Vimeo URL.
    has_file = bool(video and video.filename)
    vimeo_url = vimeo_url.strip()
    if has_file and vimeo_url:
        raise HTTPException(400, "provide either a file or a Vimeo URL, not both")
    if not has_file and not vimeo_url:
        raise HTTPException(400, "no file uploaded and no Vimeo URL provided")

    job_id = new_job_id()
    options = {
        "locale": locale or None,
        "volume_boost": vb,
        "force": force.lower() in ("1", "true", "on", "yes"),
        "review": review.lower() in ("1", "true", "on", "yes"),
    }
    output_dir = OUTPUT_DIR / job_id
    output_dir.mkdir(parents=True, exist_ok=True)

    if vimeo_url:
        # URL branch — defer the download to the worker so the request stays fast.
        host = (urlparse(vimeo_url).hostname or "").lower()
        if host not in ("vimeo.com", "www.vimeo.com", "player.vimeo.com"):
            raise HTTPException(400, "URL must be a vimeo.com link")
        log_path = LOG_DIR / f"{job_id}.log"
        job = Job(
            id=job_id,
            video_filename=vimeo_url,  # replaced with the real title after download
            video_path="",             # pending — set once downloaded
            output_dir=str(output_dir),
            log_path=str(log_path),
            options=options,
            source_url=vimeo_url,
        )
    else:
        # File branch — stream upload to disk with a hard size cap.
        stem = safe_stem(video.filename) or "video"
        dest = UPLOAD_DIR / f"{job_id}__{stem}.mp4"
        written = 0
        with dest.open("wb") as out:
            while True:
                chunk = await video.read(1024 * 1024)
                if not chunk:
                    break
                written += len(chunk)
                if written > MAX_UPLOAD_BYTES:
                    out.close()
                    dest.unlink(missing_ok=True)
                    raise HTTPException(413, f"file exceeds {MAX_UPLOAD_BYTES // 1024**3} GB cap")
                out.write(chunk)
        log_path = LOG_DIR / f"{dest.stem}.log"
        job = Job(
            id=job_id,
            video_filename=video.filename,
            video_path=str(dest),
            output_dir=str(output_dir),
            log_path=str(log_path),
            options=options,
        )

    state.jobs[job_id] = job
    state.save()
    await state.queue.put(job_id)

    # Position in queue = number of queued jobs ahead + 1 if running, else 1
    queued_ahead = sum(
        1 for j in state.jobs.values()
        if j.status == STATUS_QUEUED and j.queued_at < job.queued_at
    )
    position = queued_ahead + (1 if state.current_job_id else 0)
    return JSONResponse({"id": job_id, "position": position}, status_code=201)


@app.get("/api/jobs")
async def list_jobs() -> JSONResponse:
    return JSONResponse({
        "jobs": [j.to_dict() for j in sorted_jobs(state.jobs)],
        "current": state.current_job_id,
    })


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str) -> JSONResponse:
    job = state.jobs.get(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    return JSONResponse(job.to_dict())


@app.delete("/api/jobs/{job_id}")
async def cancel_job(job_id: str, cleanup: bool = False) -> JSONResponse:
    job = state.jobs.get(job_id)
    if not job:
        raise HTTPException(404, "job not found")

    if job.status == STATUS_RUNNING:
        job.status = STATUS_CANCELLED
        proc = state.current_proc
        if proc and proc.returncode is None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except Exception as e:
                log.warning("kill failed: %s", e)
        job.ended_at = time.time()
        state.save()
        return JSONResponse({"ok": True, "action": "terminated"})

    if job.status == STATUS_QUEUED:
        job.status = STATUS_CANCELLED
        job.ended_at = time.time()
        state.save()
        return JSONResponse({"ok": True, "action": "dequeued"})

    # Terminal — optionally clean files, always remove the record
    if cleanup:
        try:
            if job.video_path and os.path.exists(job.video_path):
                os.unlink(job.video_path)
            if job.output_dir and os.path.isdir(job.output_dir):
                shutil.rmtree(job.output_dir, ignore_errors=True)
        except Exception as e:
            log.warning("cleanup failed for %s: %s", job_id, e)
    state.jobs.pop(job_id, None)
    state.subscribers.pop(job_id, None)
    state.log_buffers.pop(job_id, None)
    state.save()
    return JSONResponse({"ok": True, "action": "removed"})


@app.get("/api/jobs/{job_id}/logs")
async def job_logs(job_id: str, request: Request) -> StreamingResponse:
    job = state.jobs.get(job_id)
    if not job:
        raise HTTPException(404, "job not found")

    q: asyncio.Queue = asyncio.Queue(maxsize=2000)
    state.subscribers.setdefault(job_id, []).append(q)

    async def _stream():
        # Replay buffer first (for clients that join after the job started)
        for line in list(state.log_buffers.get(job_id, [])):
            yield f"data: {line}\n\n"
        # Live tail
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    item = await asyncio.wait_for(q.get(), timeout=15.0)
                except asyncio.TimeoutError:
                    # Keepalive comment — keeps the proxy from idle-timing out
                    yield ": keepalive\n\n"
                    continue
                if item is None:
                    # Job finished — flush a final status line then close
                    yield f"event: done\ndata: {job.status}\n\n"
                    break
                yield f"data: {item}\n\n"
        finally:
            subs = state.subscribers.get(job_id, [])
            if q in subs:
                subs.remove(q)

    return StreamingResponse(_stream(), media_type="text/event-stream")


def _download(job: Job, kind: str) -> FileResponse:
    path = job.outputs.get(kind)
    if not path or not os.path.exists(path):
        raise HTTPException(404, f"no {kind} output for this job")
    stem = safe_stem(job.video_filename) or "dub"
    ext = Path(path).suffix
    base = {
        "video": "_french",
        "audio": "_french",
        "srt": "_french",
        "full": "_french_full",
        "english_srt": "_english",
    }[kind]
    download_name = f"{stem}{base}{ext}"
    return FileResponse(path, filename=download_name)


GLOSSARY_PATH = Path("/workspace/canadian_glossary.yaml")
GLOSSARY_MIRRORS = [Path("/workspace/french-dubbing/canadian_glossary.yaml")]
GLOSSARY_TERM_MODES = ["suggest", "always"]
# The flat glossary file stores two editable top-level maps:
#   glossary: {english: fr_ca}    → mode "suggest" (injected into the prompt)
#   always:   {find_form: fr_ca}  → mode "always"  (deterministic post-rewrite)
# The editor surfaces each entry as a row {en, fr_ca, mode}. This mirrors how
# 02_pipeline.py:load_glossary reads the file, so edits round-trip correctly.
# The acronyms: section and all header/section comments are preserved on save.
GLOSSARY_FIELDS = ("en", "fr_ca", "mode")
_GLOSSARY_SECTION_BY_MODE = {"suggest": "glossary", "always": "always"}


@app.get("/api/glossary")
async def get_glossary() -> JSONResponse:
    try:
        data = yaml.safe_load(GLOSSARY_PATH.read_text(encoding="utf-8")) or {}
    except Exception as e:
        raise HTTPException(500, f"failed to read glossary: {e}")
    terms: list[dict] = []
    for mode, section in (("suggest", "glossary"), ("always", "always")):
        for k, v in (data.get(section) or {}).items():
            terms.append({"en": str(k), "fr_ca": "" if v is None else str(v), "mode": mode})
    return JSONResponse({
        "terms": terms,
        "modes": GLOSSARY_TERM_MODES,
        "path": str(GLOSSARY_PATH),
    })


@app.post("/api/glossary")
async def update_glossary(payload: dict) -> JSONResponse:
    """Rewrite the editable `glossary:` and `always:` maps in canadian_glossary.yaml.

    Body: {"terms": [{"en", "fr_ca", "mode"}, ...]}. Rows are grouped by mode
    into the two flat maps the pipeline reads (mode "suggest" → glossary:,
    mode "always" → always:). The acronyms: section, header comments, and any
    other top-level content are preserved.
    """
    terms = payload.get("terms")
    if not isinstance(terms, list):
        raise HTTPException(400, "terms must be a list")

    by_mode: dict[str, dict[str, str]] = {"suggest": {}, "always": {}}
    for i, raw in enumerate(terms):
        if not isinstance(raw, dict):
            raise HTTPException(400, f"terms[{i}] must be an object")
        en = (raw.get("en") or "").strip()
        fr_ca = (raw.get("fr_ca") or "").strip()
        if not en or not fr_ca:
            raise HTTPException(400, f"terms[{i}]: en and fr_ca are required")
        mode = (raw.get("mode") or "suggest").strip()
        if mode not in GLOSSARY_TERM_MODES:
            raise HTTPException(400, f"terms[{i}].mode must be one of {GLOSSARY_TERM_MODES}")
        by_mode[mode][en] = fr_ca

    try:
        _rewrite_glossary_sections(GLOSSARY_PATH, by_mode["suggest"], by_mode["always"])
    except Exception as e:
        raise HTTPException(500, f"failed to update glossary: {e}")

    mirrored: list[str] = []
    for mp in GLOSSARY_MIRRORS:
        try:
            if mp.exists() and mp.resolve() != GLOSSARY_PATH.resolve():
                _rewrite_glossary_sections(mp, by_mode["suggest"], by_mode["always"])
                mirrored.append(str(mp))
        except Exception as e:
            log.warning("glossary mirror failed for %s: %s", mp, e)

    count = len(by_mode["suggest"]) + len(by_mode["always"])
    return JSONResponse({"ok": True, "count": count, "mirrored": mirrored})


def _yaml_q(v) -> str:
    """Quote a glossary key/value when YAML would otherwise misparse it."""
    s = "" if v is None else str(v)
    needs_quote = (
        s == ""
        or s != s.strip()
        or any(c in s for c in ":#'\"\n\t[]{}|>&*!%@`,")
        or s[:1] in "-?[]{}#&*!|>'\"%@`"
        or s.lower() in ("true", "false", "null", "yes", "no", "on", "off", "~")
    )
    if not needs_quote:
        return s
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _rewrite_glossary_sections(path: Path, suggest_map: dict, always_map: dict) -> None:
    """Regenerate the `glossary:` and `always:` flat maps in place.

    Only those two top-level sections are rewritten from the editor data; the
    file header, the acronyms: section, and the comment blocks that precede each
    section are preserved. Inline grouping comments *inside* a rewritten section
    are not retained (the section body is regenerated from the submitted rows).
    """
    text = path.read_text(encoding="utf-8")
    newline_eof = "\n" if text.endswith("\n") else ""
    lines = text.splitlines()

    def render(section: str, mapping: dict) -> list[str]:
        body = [f"{section}:"]
        for k, v in mapping.items():
            body.append(f"  {_yaml_q(k)}: {_yaml_q(v)}")
        return body

    def splice(lines: list[str], section: str, mapping: dict) -> list[str]:
        start = None
        for i, line in enumerate(lines):
            if re.match(rf"^{re.escape(section)}:\s*$", line):
                start = i
                break
        if start is None:
            if not mapping:
                return lines  # nothing to write and no section to replace
            sep = [""] if lines and lines[-1].strip() else []
            return lines + sep + render(section, mapping)
        # Walk to the last indented body line; trailing blanks and any comment
        # block that precedes the next section are left for that section.
        last_body = start
        j = start + 1
        while j < len(lines):
            line = lines[j]
            if line.strip() == "":
                j += 1
                continue
            if line[0].isspace():            # indented → belongs to this section
                last_body = j
                j += 1
                continue
            break                            # column-0 content → next section/comment
        return lines[:start] + render(section, mapping) + lines[last_body + 1:]

    lines = splice(lines, "glossary", suggest_map)
    lines = splice(lines, "always", always_map)

    path.write_text("\n".join(lines) + newline_eof, encoding="utf-8")

@app.get("/api/jobs/{job_id}/download/{kind}")
async def download(job_id: str, kind: str) -> FileResponse:
    if kind not in ("video", "audio", "srt", "full", "english_srt"):
        raise HTTPException(400, "kind must be video|audio|srt|full|english_srt")
    job = state.jobs.get(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    return _download(job, kind)


@app.get("/api/jobs/{job_id}/segments")
async def get_segments(job_id: str) -> JSONResponse:
    job = state.jobs.get(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    if job.status != STATUS_AWAITING_REVIEW:
        raise HTTPException(409, f"job is not awaiting review (status: {job.status})")
    # Pipeline names outputs after Path(video_path).stem (includes the job-id prefix),
    # not the original upload filename — use the same derivation here.
    name = Path(job.video_path).stem if job.video_path else safe_stem(job.video_filename)
    seg_file = Path(job.output_dir) / f"{name}_segments.json"
    if not seg_file.exists():
        raise HTTPException(404, "segments file not found")
    return JSONResponse(json.loads(seg_file.read_text(encoding="utf-8")))


@app.put("/api/jobs/{job_id}/segments")
async def put_segments(job_id: str, request: Request) -> JSONResponse:
    job = state.jobs.get(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    if job.status != STATUS_AWAITING_REVIEW:
        raise HTTPException(409, f"job is not awaiting review (status: {job.status})")
    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(400, "invalid JSON body")
    if not isinstance(payload, list):
        raise HTTPException(400, "body must be a JSON array")
    name = Path(job.video_path).stem if job.video_path else safe_stem(job.video_filename)
    seg_file = Path(job.output_dir) / f"{name}_segments.json"
    fd, tmp = tempfile.mkstemp(dir=job.output_dir, prefix=".seg.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        os.replace(tmp, str(seg_file))
    except Exception as e:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise HTTPException(500, f"failed to save segments: {e}")
    return JSONResponse({"ok": True, "count": len(payload)})


@app.post("/api/jobs/{job_id}/approve")
async def approve_job(job_id: str) -> JSONResponse:
    job = state.jobs.get(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    if job.status != STATUS_AWAITING_REVIEW:
        raise HTTPException(409, f"job is not awaiting review (status: {job.status})")
    job.options["_p2"] = True
    job.status = STATUS_QUEUED
    job.ended_at = 0.0
    state.save()
    await state.queue.put(job_id)
    return JSONResponse({"ok": True, "queued": job_id})


@app.get("/api/health")
async def health() -> JSONResponse:
    info: dict = {
        "pipeline_present": PIPELINE_PY.exists(),
        "config_present": CONFIG_PATH.exists(),
        "hf_token_present": bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")),
    }
    # Disk free
    try:
        usage = shutil.disk_usage(str(WORKSPACE))
        info["disk_free_gb"] = round(usage.free / 1e9, 1)
    except Exception:
        info["disk_free_gb"] = None
    # GPU / VRAM via nvidia-smi
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode == 0 and r.stdout.strip():
            name, total, free = [x.strip() for x in r.stdout.splitlines()[0].split(",")]
            info["gpu"] = name
            info["vram_total_gb"] = round(int(total) / 1024, 1)
            info["vram_free_gb"]  = round(int(free)  / 1024, 1)
    except Exception:
        pass
    # Ollama reachable?
    try:
        import urllib.request
        urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2)
        info["ollama_up"] = True
    except Exception:
        info["ollama_up"] = False
    return JSONResponse(info)
