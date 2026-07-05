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
import secrets
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

import requests as _rq
import yaml
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import (
    FileResponse, HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse,
)
from fastapi.staticfiles import StaticFiles

from .jobs import (
    Job, JOBS_FILE, STATUS_AWAITING_REVIEW, STATUS_CANCELLED, STATUS_COMPLETED,
    STATUS_FAILED, STATUS_QUEUED, STATUS_RUNNING, TERMINAL,
    load_jobs, new_job_id, safe_stem, save_jobs, sorted_jobs,
)


def _load_dotenv() -> None:
    """Load .env (repo checkout, then /workspace/.env) before reading config.

    Putting secrets — DUBBING_UI_TOKEN, VIMEO_ACCESS_TOKEN, RUNPOD_API_KEY,
    VIMEO_CLIENT_ID/SECRET — in /workspace/.env keeps them on the volume, so
    freshly created pods boot fully configured without template edits.
    Values already set in the environment win (override=False)."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    for candidate in (
        Path(__file__).resolve().parent.parent / ".env",
        Path("/workspace/.env"),
    ):
        if candidate.exists():
            load_dotenv(candidate, override=False)


_load_dotenv()

# ── Paths ─────────────────────────────────────────────────────────────────────
WEB_ROOT     = Path(__file__).resolve().parent
STATIC_DIR   = WEB_ROOT / "static"
WORKSPACE    = Path(os.environ.get("DUBBING_WORKSPACE", "/workspace"))
UPLOAD_DIR   = WORKSPACE / "web" / "uploads"
OUTPUT_DIR   = WORKSPACE / "web" / "outputs"
LOG_DIR      = WORKSPACE / "logs"
TEMP_DIR     = WORKSPACE / "temp"
VOICES_DIR   = WORKSPACE / "voices"   # curated clean reference clips ("preset voices")
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
VOICES_DIR.mkdir(parents=True, exist_ok=True)

# ── Choices (mirror Click definitions in 02_pipeline.py) ──────────────────────
LOCALE_CHOICES = ["fr", "fr-ca"]

# ── Advanced config schema ────────────────────────────────────────────────────
# Each entry: dotted path → (type, label, help, ui hints).
# UI hints: {"min", "max", "step", "choices", "group"}.
# Only paths declared here are writable via /api/config; everything else in
# config.yaml is preserved verbatim on save.
#
# Deliberately small: only settings that genuinely vary per video/client are
# exposed. The timing stack (budget_cps / max_stretch / compression / timing
# policy), the Whisper anti-hallucination thresholds, and the voice-clone
# internals are tuned as one coherent system — changing one alone breaks the
# balance, so they live in config.yaml only.
CONFIG_SCHEMA: dict = {
    # Audio
    "audio.volume_boost_pct":          ("int",   "Volume boost (%)",        "0 = off. Shifts the loudness-normalization target by the equivalent dB (true-peak ceiling still holds).", {"min": -50, "max": 100, "step": 1, "group": "Audio"}),

    # Diarization
    "diarization.enabled":             ("bool",  "Enable diarization",      "Detect speakers and clone a distinct voice per speaker. Disable for solo presenters.", {"group": "Diarization"}),
    "diarization.min_speakers":        ("int",   "Min speakers",            "Set min == max to force an exact count. The Speakers field on the submit form overrides this per job.", {"min": 1, "max": 20, "step": 1, "group": "Diarization"}),
    "diarization.max_speakers":        ("int",   "Max speakers",            "", {"min": 1, "max": 20, "step": 1, "group": "Diarization"}),

    # Source separation
    "source_separation.preserve_background":("bool",  "Preserve background",    "Remix original music/ambience under French vocals into *_french_full.m4a.", {"group": "Source separation"}),

    # Whisper
    "whisper.language":                ("str",   "Source language",         "ISO code of the spoken language (en, fr, …). Empty = auto-detect per file — use for bilingual sources.", {"group": "Whisper"}),
    "whisper.initial_prompt":          ("str",   "Vocabulary hint",         "Optional comma-separated proper nouns/acronyms (e.g. \"CAPS, CSP, keynote\"). Avoid full sentences — they get echoed into the transcript over silence.", {"group": "Whisper"}),

    # Translation
    "translation.model":               ("str",   "Translation model",       "Ollama tag (e.g. mistral-small:22b, qwen3:14b).", {"group": "Translation"}),
    "translation.review_pass":         ("bool",  "Self-review pass",        "The LLM rereads its output and fixes Anglicisms. Roughly doubles translation time — use for premium deliverables.", {"group": "Translation"}),
    "translation.target_lang":         ("str",   "Target language",         "", {"choices": ["fr", "es", "de", "it", "pt", "nl", "pl", "ru", "ja", "ko", "zh", "ar", "tr", "hi", "vi"], "group": "Translation"}),
    "translation.locale":              ("str",   "Locale variant",          "fr-ca triggers the Canadian glossary.", {"choices": LOCALE_CHOICES, "group": "Translation"}),

    # Subtitles
    "subtitles.sync_offset_ms":        ("int",   "Subtitle offset (ms)",    "Positive = later, negative = earlier.", {"min": -10000, "max": 10000, "step": 50, "group": "Subtitles"}),

    # Output
    "output.mux_video":                ("bool",  "Mux final video",         "Also emit {name}_french.mp4 (original video + dubbed audio + subtitles). Audio is held to the source length so they end together.", {"group": "Output"}),
    "output.burn_subs":                ("bool",  "Burn-in subtitles",       "On = render subtitles into the picture (re-encodes video). Off = soft-embed the SRT track and copy the video stream.", {"group": "Output"}),
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
        self.idle_task: Optional[asyncio.Task] = None
        # Last meaningful activity (job start/finish, any mutating request).
        # GET polling doesn't count — the UI footer polls health forever.
        self.last_activity: float = time.time()

    def touch(self) -> None:
        self.last_activity = time.time()

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
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ── Auth ──────────────────────────────────────────────────────────────────────
# Shared-token auth: set DUBBING_UI_TOKEN in the environment and every request
# must carry it (login form → HttpOnly SameSite cookie, or Bearer header, or
# ?token= for scripted access). SameSite=strict also kills cross-origin
# drive-by requests, which is why no CORS middleware is registered — the UI is
# same-origin only. Unset token = auth disabled (local development).
AUTH_TOKEN = os.environ.get("DUBBING_UI_TOKEN", "")
AUTH_COOKIE = "dubbing_token"

_LOGIN_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>Dubbing UI — sign in</title>
<style>
 body{font-family:system-ui,sans-serif;display:flex;justify-content:center;
      align-items:center;min-height:100vh;margin:0;background:#111;color:#eee}
 form{background:#1c1c1e;padding:2rem;border-radius:12px;display:flex;
      flex-direction:column;gap:.75rem;min-width:280px}
 input,button{padding:.6rem .8rem;border-radius:8px;border:1px solid #444;
      font-size:1rem;background:#111;color:#eee}
 button{background:#2563eb;border:none;cursor:pointer}
 .err{color:#f87171;margin:0;font-size:.9rem}
</style></head><body>
<form method="post" action="/auth">
  <strong>Dubbing Web UI</strong>
  <!--msg-->
  <input type="password" name="token" placeholder="Access token" autofocus>
  <button type="submit">Sign in</button>
</form></body></html>"""


def _token_ok(supplied: str) -> bool:
    return bool(supplied) and secrets.compare_digest(supplied, AUTH_TOKEN)


@app.middleware("http")
async def _require_token(request: Request, call_next):
    # Any mutating request counts as activity for the idle auto-stop —
    # submitting, editing segments/config, saving voice refs. GETs don't:
    # the frontend polls jobs/health forever, which would hold the pod open.
    if request.method not in ("GET", "HEAD", "OPTIONS"):
        state.touch()
    if not AUTH_TOKEN:
        return await call_next(request)
    if request.url.path == "/auth" and request.method == "POST":
        return await call_next(request)
    bearer = request.headers.get("authorization") or ""
    if bearer.startswith("Bearer "):
        bearer = bearer[len("Bearer "):].strip()
    supplied = (
        request.cookies.get(AUTH_COOKIE)
        or request.headers.get("x-auth-token")
        or bearer
        or request.query_params.get("token")
        or ""
    )
    if _token_ok(supplied):
        return await call_next(request)
    if request.method == "GET" and "text/html" in (request.headers.get("accept") or ""):
        return HTMLResponse(_LOGIN_HTML, status_code=401)
    return JSONResponse({"detail": "unauthorized"}, status_code=401)


@app.post("/auth")
async def auth_login(token: str = Form("")) -> HTMLResponse:
    if not AUTH_TOKEN:
        return RedirectResponse("/", status_code=303)
    if not _token_ok(token):
        return HTMLResponse(
            _LOGIN_HTML.replace("<!--msg-->", '<p class="err">Invalid token</p>'),
            status_code=401,
        )
    resp = RedirectResponse("/", status_code=303)
    resp.set_cookie(
        AUTH_COOKIE, token,
        httponly=True, samesite="strict", max_age=30 * 24 * 3600,
    )
    return resp


# ── Idle auto-stop (on-demand / budget operation) ─────────────────────────────
# Set DUBBING_IDLE_STOP_MIN (e.g. 10) in the pod template and the server stops
# its own RunPod pod after that many minutes with no running/queued job and no
# mutating request. Everything that matters (models, jobs.json, outputs,
# phase-1 segments awaiting review) lives on the /workspace volume, so a
# stopped pod resumes exactly where it left off. Unset/0 = disabled.
IDLE_STOP_MIN = float(os.environ.get("DUBBING_IDLE_STOP_MIN", "0") or 0)
RUNPOD_POD_ID = os.environ.get("RUNPOD_POD_ID", "")
RUNPOD_API_KEY = os.environ.get("RUNPOD_API_KEY", "")


def _idle_seconds() -> float:
    """Seconds since the last job/mutating-request activity; 0 while busy."""
    busy = state.current_job_id is not None or any(
        j.status == STATUS_QUEUED for j in state.jobs.values()
    )
    if busy:
        state.touch()
        return 0.0
    return time.time() - state.last_activity


def _stop_self() -> bool:
    """Stop this RunPod pod via runpodctl, falling back to the REST API."""
    if not RUNPOD_POD_ID:
        log.warning("idle-stop: RUNPOD_POD_ID not set — cannot stop pod")
        return False
    try:
        r = subprocess.run(
            ["runpodctl", "stop", "pod", RUNPOD_POD_ID],
            capture_output=True, text=True, timeout=30,
        )
        if r.returncode == 0:
            return True
        log.warning("idle-stop: runpodctl failed: %s", (r.stderr or r.stdout).strip()[:200])
    except FileNotFoundError:
        log.debug("idle-stop: runpodctl not installed — trying REST API")
    except Exception as e:
        log.warning("idle-stop: runpodctl error: %s", e)
    if RUNPOD_API_KEY:
        try:
            import urllib.request
            req = urllib.request.Request(
                f"https://rest.runpod.io/v1/pods/{RUNPOD_POD_ID}/stop",
                method="POST",
                headers={"Authorization": f"Bearer {RUNPOD_API_KEY}"},
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                return 200 <= resp.status < 300
        except Exception as e:
            log.warning("idle-stop: REST stop failed: %s", e)
    return False


async def _idle_watchdog() -> None:
    """Stop the pod after IDLE_STOP_MIN minutes of no jobs / no writes."""
    limit_s = IDLE_STOP_MIN * 60
    while True:
        await asyncio.sleep(60)
        idle = _idle_seconds()
        if idle < limit_s:
            continue
        log.info(
            "idle-stop: no activity for %.0f min (limit %.0f) — stopping pod %s",
            idle / 60, IDLE_STOP_MIN, RUNPOD_POD_ID or "(unknown)",
        )
        state.save()
        if _stop_self():
            return  # the pod is going down; nothing left to watch
        # Stop failed — back off a full cycle before retrying, and reset the
        # clock so we don't hammer the API every minute.
        state.touch()


# ── Queue worker ──────────────────────────────────────────────────────────────
async def _run_job(job: Job) -> None:
    """Execute one pipeline subprocess; stream its stdout into the job's log buffer.

    Every exit path — success, failure, cancel (including cancel during a Vimeo
    download), worker exception — goes through the ``finally`` block, which
    publishes the terminal status, closes SSE streams, and clears the
    current-job state. Without that, a cancel during download left
    ``current_job_id`` set (blocking config edits) and SSE streams open forever.
    """
    job.status = STATUS_RUNNING
    job.started_at = time.time()
    state.current_job_id = job.id
    state.touch()
    state.save()
    state.publish(job.id, f">>> Starting: {job.video_filename}")

    try:
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
                return  # job already marked failed/cancelled
            if job.status == STATUS_CANCELLED:
                return  # cancelled between download and launch

        opts = job.options or {}
        _is_phase1 = opts.get("review") and not opts.get("_p2")
        _is_phase2 = bool(opts.get("_p2"))

        cmd = [
            sys.executable, str(PIPELINE_PY),
            "--video",      job.video_path,
            "--output-dir", job.output_dir,
            "--config",     str(CONFIG_PATH),
        ]
        # Phase 2 is always a deliberate "synthesize now" action from the UI, so it must
        # run even when prior outputs exist (re-generation after re-opening review). On
        # the first approve no outputs exist yet, so --force is a harmless no-op there.
        if opts.get("force") or _is_phase2:
            cmd.append("--force")
        if opts.get("locale"):
            cmd += ["--locale", opts["locale"]]
        if opts.get("volume_boost") not in (None, ""):
            cmd += ["--volume-boost", str(opts["volume_boost"])]
        if opts.get("speakers"):
            cmd += ["--speakers", str(opts["speakers"])]
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
            state.publish(job.id, f"!!! launch failed: {e}")
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

    finally:
        if not job.ended_at:
            job.ended_at = time.time()
        rc_note = f" (rc={job.returncode})" if job.returncode is not None else ""
        state.publish(job.id, f"<<< {job.status.upper()}{rc_note}")
        # Signal SSE subscribers to close cleanly
        for q in state.subscribers.get(job.id, []):
            try:
                q.put_nowait(None)
            except asyncio.QueueFull:
                pass
        state.current_proc = None
        state.current_job_id = None
        state.touch()   # the idle clock starts when the job ends, not when it started
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
    """Mark a job failed with a message; returns False for convenient early-return.

    Terminal-status publishing, SSE close, and current-job state cleanup all
    happen in _run_job's ``finally`` block — every failure path funnels there."""
    job.status = STATUS_FAILED
    job.error = msg
    state.publish(job.id, f"!!! {msg}")
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
    if not AUTH_TOKEN:
        log.warning(
            "DUBBING_UI_TOKEN is not set — the web UI is UNAUTHENTICATED. "
            "Anyone who can reach this port can submit jobs and edit config."
        )
    if IDLE_STOP_MIN > 0:
        state.idle_task = asyncio.create_task(_idle_watchdog())
        log.info(
            "idle auto-stop enabled: pod stops after %.0f min without jobs "
            "(pod %s)", IDLE_STOP_MIN, RUNPOD_POD_ID or "unknown — set RUNPOD_POD_ID",
        )
    # Off the event loop — it's a network call to Vimeo.
    await asyncio.to_thread(_vimeo_seed_from_env)
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
    return JSONResponse({
        "values": values,
        "schema": schema_out,
        "presets": [],   # presets removed — defaults are the tuned configuration
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
    local_path: str = Form(""),
    locale: str = Form(""),
    volume_boost: str = Form(""),
    speakers: str = Form(""),
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
    spk: Optional[int] = None
    if speakers.strip():
        try:
            spk = int(speakers)
        except ValueError:
            raise HTTPException(400, "speakers must be a whole number")
        if not 1 <= spk <= 20:
            raise HTTPException(400, "speakers must be between 1 and 20")

    # Exactly one source: a browser upload, a Vimeo URL, or a path to a file
    # already on the pod. The on-pod path bypasses the browser/proxy upload
    # entirely (copy big files in via runpodctl/scp/volume), avoiding the
    # Cloudflare proxy 502 that kills large uploads.
    has_file = bool(video and video.filename)
    vimeo_url = vimeo_url.strip()
    local_path = local_path.strip()
    active = [n for n, on in
              (("file", has_file), ("vimeo", bool(vimeo_url)), ("path", bool(local_path)))
              if on]
    if len(active) > 1:
        raise HTTPException(400, "provide exactly one source: a file, a Vimeo URL, or an on-pod path")
    if not active:
        raise HTTPException(400, "no source provided: upload a file, give a Vimeo URL, or an on-pod path")

    job_id = new_job_id()
    options = {
        "locale": locale or None,
        "volume_boost": vb,
        "speakers": spk,
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
    elif local_path:
        # On-pod path branch — the file is already on the pod (copied in via
        # runpodctl/scp/volume), so no bytes flow through the browser/proxy.
        # Symlink it into UPLOAD_DIR under the usual {job_id}__{stem}.mp4 name so
        # all downstream output naming (which derives from Path(video_path).stem)
        # behaves exactly like a normal upload.
        src = Path(local_path).expanduser()
        try:
            resolved = src.resolve()
        except Exception:
            raise HTTPException(400, f"invalid path: {local_path}")
        root = WORKSPACE.resolve()
        if root not in resolved.parents and resolved != root:
            raise HTTPException(400, f"path must be inside {root}")
        if not resolved.is_file():
            raise HTTPException(400, f"file not found on pod: {resolved}")
        stem = safe_stem(resolved.name) or "video"
        dest = UPLOAD_DIR / f"{job_id}__{stem}.mp4"
        dest.symlink_to(resolved)
        log_path = LOG_DIR / f"{dest.stem}.log"
        job = Job(
            id=job_id,
            video_filename=resolved.name,
            video_path=str(dest),
            output_dir=str(output_dir),
            log_path=str(log_path),
            options=options,
        )
    else:
        # File branch — stream upload to disk with a hard size cap.
        stem = safe_stem(video.filename) or "video"
        dest = UPLOAD_DIR / f"{job_id}__{stem}.mp4"
        written = 0
        try:
            with dest.open("wb") as out:
                while True:
                    chunk = await video.read(1024 * 1024)
                    if not chunk:
                        break
                    written += len(chunk)
                    if written > MAX_UPLOAD_BYTES:
                        raise HTTPException(413, f"file exceeds {MAX_UPLOAD_BYTES // 1024**3} GB cap")
                    out.write(chunk)
        except BaseException:
            # Aborted/oversize upload — don't leave a partial file on disk.
            dest.unlink(missing_ok=True)
            raise
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


@app.post("/api/jobs/{job_id}/reopen")
async def reopen_job(job_id: str) -> JSONResponse:
    """Step a finished job back to the review stage so the user can edit segments
    / voices and re-run Phase 2 (TTS + assembly) WITHOUT redoing Phase 1
    (transcription, diarization, translation). Phase 2 reads only
    ``{name}_segments.json``, so re-opening is safe as long as that file survives."""
    job = state.jobs.get(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    if job.status not in (STATUS_COMPLETED, STATUS_FAILED):
        raise HTTPException(409, f"job cannot be re-opened (status: {job.status})")
    name = Path(job.video_path).stem if job.video_path else safe_stem(job.video_filename)
    seg_file = Path(job.output_dir) / f"{name}_segments.json"
    if not seg_file.exists():
        raise HTTPException(
            409,
            "Phase-1 segments are no longer on disk — re-run the full pipeline to regenerate.",
        )
    # Drop the phase-2 marker so the job re-enters the review stage rather than
    # re-queuing straight into synthesis; approve() will set it again.
    job.options.pop("_p2", None)
    job.options["review"] = True
    job.status = STATUS_AWAITING_REVIEW
    job.error = ""
    job.ended_at = 0.0
    state.save()
    return JSONResponse({"ok": True, "reopened": job_id})


# ── Voice reference selection (review stage) ─────────────────────────────────

def _job_name(job: Job) -> str:
    """The stem the pipeline derives outputs from (matches Path(video_path).stem)."""
    return Path(job.video_path).stem if job.video_path else safe_stem(job.video_filename)


def _locate_vocals(job: Job) -> Optional[Path]:
    """Find the Demucs vocals.wav preserved from Phase 1 for this job."""
    name = _job_name(job)
    matches = sorted((TEMP_DIR / name).glob("**/vocals.wav"))
    return matches[-1] if matches else None


def _audio_duration(path: Path) -> float:
    try:
        import soundfile as sf
        return float(sf.info(str(path)).duration)
    except Exception:
        return 0.0


# Default reference length to suggest, in seconds (mirrors
# diarization.profile_duration / tts.speaker_profile_duration = 12s).
_SUGGESTED_REF_S = 12.0


def _speaker_turns(segs: list, gap: float = 1.0) -> dict:
    """Merge consecutive same-speaker segments into turns, so the UI can show
    *when each speaker actually talks*. Returns {speaker: [(start, end), ...]}
    sorted longest-first."""
    from collections import defaultdict
    turns: dict = defaultdict(list)
    cur = None  # [speaker, start, end]
    for s in sorted(segs, key=lambda x: float(x.get("start", 0.0))):
        spk = s.get("speaker")
        if not spk:
            continue
        st = float(s.get("start", 0.0))
        en = float(s.get("end", st))
        if cur and cur[0] == spk and st - cur[2] <= gap:
            cur[2] = max(cur[2], en)
        else:
            if cur:
                turns[cur[0]].append((cur[1], cur[2]))
            cur = [spk, st, en]
    if cur:
        turns[cur[0]].append((cur[1], cur[2]))
    for spk in turns:
        turns[spk].sort(key=lambda r: r[1] - r[0], reverse=True)
    return turns


def _speaker_ranges(segs: list) -> dict:
    """Per-speaker reference guidance: a suggested start/duration (anchored to the
    speaker's longest turn) plus their longest turns, so the picker defaults to a
    window where THAT speaker actually speaks instead of a shared fixed offset."""
    turns = _speaker_turns(segs)
    ranges: dict = {}
    for spk, tl in turns.items():
        if tl:
            s0, e0 = tl[0]
            sug = {"start": round(s0, 1),
                   "duration": round(max(1.0, min(e0 - s0, _SUGGESTED_REF_S)), 1)}
        else:
            sug = {"start": 0.0, "duration": _SUGGESTED_REF_S}
        ranges[spk] = {
            "suggested": sug,
            "turns": [[round(a, 1), round(b, 1)] for a, b in tl[:6]],
            "n_turns": len(tl),
        }
    return ranges


@app.get("/api/jobs/{job_id}/voices/speakers")
async def voice_speakers(job_id: str) -> JSONResponse:
    job = state.jobs.get(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    name = _job_name(job)
    seg_file = Path(job.output_dir) / f"{name}_segments.json"
    speakers: list = []
    ranges: dict = {}
    if seg_file.exists():
        try:
            segs = json.loads(seg_file.read_text(encoding="utf-8"))
            speakers = sorted({s.get("speaker") for s in segs if s.get("speaker")})
            ranges = _speaker_ranges(segs)
        except Exception:
            speakers = []
    if not speakers:
        speakers = ["default"]
    vocals = _locate_vocals(job)
    # Existing saved selections, if any.
    refs_file = Path(job.output_dir) / f"{name}_voice_refs.json"
    saved = {}
    if refs_file.exists():
        try:
            saved = json.loads(refs_file.read_text(encoding="utf-8"))
        except Exception:
            saved = {}
    return JSONResponse({
        "speakers": speakers,
        "ranges": ranges,
        "vocals_available": bool(vocals),
        "vocals_duration": round(_audio_duration(vocals), 2) if vocals else 0.0,
        "saved": saved,
    })


@app.get("/api/jobs/{job_id}/vocals-clip")
async def vocals_clip(job_id: str, start: float = 0.0, dur: float = 10.0) -> StreamingResponse:
    job = state.jobs.get(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    vocals = _locate_vocals(job)
    if not vocals:
        raise HTTPException(404, "vocals not found (job has no preserved Phase 1 audio)")
    start = max(0.0, float(start))
    dur = max(0.5, min(float(dur), 30.0))  # cap preview length
    try:
        proc = subprocess.run(
            ["ffmpeg", "-y", "-ss", str(start), "-t", str(dur), "-i", str(vocals),
             "-ac", "1", "-ar", "24000", "-f", "wav", "pipe:1"],
            check=True, capture_output=True, timeout=60,
        )
    except Exception as e:
        raise HTTPException(500, f"clip extraction failed: {e}")
    return StreamingResponse(iter([proc.stdout]), media_type="audio/wav")


@app.get("/api/voices/library")
async def voices_library() -> JSONResponse:
    exts = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}
    clips = sorted(
        p.name for p in VOICES_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in exts
    ) if VOICES_DIR.exists() else []
    return JSONResponse({"voices": clips, "dir": str(VOICES_DIR)})


@app.get("/api/voices/library/{filename}")
async def voices_library_clip(filename: str) -> FileResponse:
    target = (VOICES_DIR / filename).resolve()
    if VOICES_DIR.resolve() not in target.parents or not target.is_file():
        raise HTTPException(404, "voice clip not found")
    return FileResponse(str(target))


@app.put("/api/jobs/{job_id}/voice-refs")
async def put_voice_refs(job_id: str, request: Request) -> JSONResponse:
    job = state.jobs.get(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    if job.status != STATUS_AWAITING_REVIEW:
        raise HTTPException(409, f"job is not awaiting review (status: {job.status})")
    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(400, "invalid JSON body")
    if not isinstance(payload, dict):
        raise HTTPException(400, "body must be a JSON object keyed by speaker")

    vocals = _locate_vocals(job)
    vdur = _audio_duration(vocals) if vocals else 0.0
    cleaned: dict = {}
    for spk, ov in payload.items():
        if not isinstance(ov, dict):
            raise HTTPException(400, f"{spk}: selection must be an object")
        src = ov.get("source")
        if src == "auto":
            continue  # auto = no override; omit from the sidecar
        if src == "range":
            try:
                start = float(ov.get("start", 0.0)); dur = float(ov.get("duration", 0.0))
            except (TypeError, ValueError):
                raise HTTPException(400, f"{spk}: start/duration must be numbers")
            if dur <= 0:
                raise HTTPException(400, f"{spk}: duration must be > 0")
            if vdur and start + dur > vdur + 0.5:
                raise HTTPException(400, f"{spk}: range exceeds vocals length ({vdur:.1f}s)")
            cleaned[spk] = {"source": "range", "start": start, "duration": dur,
                            "denoise": bool(ov.get("denoise", False))}
        elif src == "library":
            fname = os.path.basename(ov.get("path") or ov.get("file") or "")
            target = (VOICES_DIR / fname).resolve()
            if VOICES_DIR.resolve() not in target.parents or not target.is_file():
                raise HTTPException(400, f"{spk}: library clip not found: {fname}")
            cleaned[spk] = {"source": "library", "path": str(target),
                            "denoise": bool(ov.get("denoise", False))}
        else:
            raise HTTPException(400, f"{spk}: unknown source '{src}'")

    name = _job_name(job)
    refs_file = Path(job.output_dir) / f"{name}_voice_refs.json"
    if cleaned:
        fd, tmp = tempfile.mkstemp(dir=job.output_dir, prefix=".vref.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(cleaned, f, ensure_ascii=False, indent=2)
            os.replace(tmp, str(refs_file))
        except Exception as e:
            try: os.unlink(tmp)
            except OSError: pass
            raise HTTPException(500, f"failed to save voice refs: {e}")
    else:
        refs_file.unlink(missing_ok=True)  # all auto → clear any prior selection
    return JSONResponse({"ok": True, "overrides": len(cleaned)})


# ── Vimeo integration ─────────────────────────────────────────────────────────
# Push a finished dub straight onto the source Vimeo video: the SRT as an
# active French text track, and the full-mix M4A as a dubbed audio track.
#
# Two ways to connect:
#   * "Connect to Vimeo" (OAuth code flow) — requires a Vimeo API app; set
#     VIMEO_CLIENT_ID / VIMEO_CLIENT_SECRET and register
#     {pod-url}/api/vimeo/callback as the app's redirect URL.
#   * Paste a personal access token (scopes: public private edit upload) —
#     zero app setup; recommended for a single-user pod.
# The token is stored on the /workspace volume so it survives pod stops.
VIMEO_API = "https://api.vimeo.com"
VIMEO_ACCEPT = "application/vnd.vimeo.*+json;version=3.4"
VIMEO_TOKEN_FILE = WORKSPACE / "web" / "vimeo_token.json"
VIMEO_CLIENT_ID = os.environ.get("VIMEO_CLIENT_ID", "")
VIMEO_CLIENT_SECRET = os.environ.get("VIMEO_CLIENT_SECRET", "")
VIMEO_REDIRECT_URL = os.environ.get("VIMEO_REDIRECT_URL", "")  # optional override
# Personal access token supplied via the environment (pod template or
# /workspace/.env) — the app connects with it automatically at startup, so a
# fresh pod boots ready to push without touching the UI. A token pasted in
# the UI takes precedence; the env seed only (re)applies when the stored
# connection is absent or itself env-sourced.
VIMEO_ACCESS_TOKEN_ENV = os.environ.get("VIMEO_ACCESS_TOKEN", "")
VIMEO_OAUTH_SCOPES = "public private edit upload"

_vimeo_oauth_states: Dict[str, float] = {}   # state -> expiry ts

_VIMEO_ID_RES = [
    re.compile(r"vimeo\.com/(?:video/|manage/videos/)?(\d+)"),
    re.compile(r"^(\d{6,})$"),
]


def _vimeo_video_id(target: str) -> Optional[str]:
    target = (target or "").strip()
    for rx in _VIMEO_ID_RES:
        m = rx.search(target)
        if m:
            return m.group(1)
    return None


def _vimeo_load_token() -> dict:
    try:
        return json.loads(VIMEO_TOKEN_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _vimeo_save_token(data: dict) -> None:
    VIMEO_TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
    VIMEO_TOKEN_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.chmod(VIMEO_TOKEN_FILE, 0o600)


def _vimeo_headers(token: str) -> dict:
    return {"Authorization": f"Bearer {token}", "Accept": VIMEO_ACCEPT}


def _vimeo_err(r: "_rq.Response") -> str:
    try:
        j = r.json()
        return f"HTTP {r.status_code}: {j.get('error') or j.get('developer_message') or r.text[:200]}"
    except Exception:
        return f"HTTP {r.status_code}: {r.text[:200]}"


def _vimeo_me(token: str) -> Optional[dict]:
    try:
        r = _rq.get(f"{VIMEO_API}/me", headers=_vimeo_headers(token), timeout=15)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None


def _vimeo_seed_from_env() -> None:
    """Connect with VIMEO_ACCESS_TOKEN at startup (see its comment above)."""
    if not VIMEO_ACCESS_TOKEN_ENV:
        return
    stored = _vimeo_load_token()
    if stored.get("access_token"):
        if stored.get("via") != "environment":
            return  # user connected explicitly — don't override
        if stored.get("access_token") == VIMEO_ACCESS_TOKEN_ENV:
            return  # same env token already stored
    me = _vimeo_me(VIMEO_ACCESS_TOKEN_ENV)
    if not me:
        log.warning("VIMEO_ACCESS_TOKEN set but Vimeo rejected it "
                    "(check scopes: public private edit upload)")
        return
    _vimeo_save_token({
        "access_token": VIMEO_ACCESS_TOKEN_ENV,
        "user": me.get("name", ""),
        "via": "environment",
        "connected_at": time.time(),
    })
    log.info("Vimeo connected from environment as %s", me.get("name", "?"))


def _find_upload_link(obj) -> Optional[str]:
    """Recursively find the signed upload URL in a Vimeo create response.

    texttracks return it as top-level "link"; other resources nest it under
    "upload"/"upload_link". Search known shapes before giving up."""
    if isinstance(obj, dict):
        up = obj.get("upload")
        if isinstance(up, dict) and up.get("upload_link"):
            return up["upload_link"]
        for key in ("upload_link", "link"):
            v = obj.get(key)
            if isinstance(v, str) and v.startswith("http"):
                return v
        for v in obj.values():
            if isinstance(v, (dict, list)):
                found = _find_upload_link(v)
                if found:
                    return found
    elif isinstance(obj, list):
        for v in obj:
            found = _find_upload_link(v)
            if found:
                return found
    return None


_SRT_TS_RE = re.compile(r"(\d{2,}):(\d{2}):(\d{2}),(\d{3})")


def _srt_to_vtt(srt_text: str) -> str:
    """Convert SRT to WebVTT — Vimeo's native caption format.

    Vimeo's SRT parser rejects files its VTT parser accepts ("Unable to parse
    captions"), so uploads always go up as VTT: strip index lines, add the
    header, and swap the millisecond comma for a period."""
    out = ["WEBVTT", ""]
    for block in re.split(r"\r?\n\s*\r?\n", srt_text.lstrip("﻿").strip()):
        lines = [ln.rstrip("\r") for ln in block.splitlines() if ln.strip()]
        # Skip anything before the timestamp line (index number, stray BOM).
        ts_i = next((i for i, ln in enumerate(lines[:2]) if "-->" in ln), None)
        if ts_i is None:
            continue
        lines = lines[ts_i:]
        lines[0] = _SRT_TS_RE.sub(lambda m: f"{m.group(1)}:{m.group(2)}:{m.group(3)}.{m.group(4)}", lines[0])
        out.extend(lines)
        out.append("")
    return "\n".join(out)


def _vimeo_push_subtitles(token: str, video_id: str, srt_path: str,
                          language: str, name: str = "Français") -> dict:
    """Create a text track, upload the captions (as WebVTT), activate it."""
    with open(srt_path, encoding="utf-8-sig") as f:
        vtt = _srt_to_vtt(f.read())
    if vtt.count("-->") == 0:
        return {"ok": False, "step": "convert captions", "detail": "no cues found in SRT"}
    r = _rq.post(
        f"{VIMEO_API}/videos/{video_id}/texttracks",
        headers=_vimeo_headers(token),
        json={"type": "subtitles", "language": language, "name": name},
        timeout=30,
    )
    if r.status_code not in (200, 201):
        return {"ok": False, "step": "create texttrack", "detail": _vimeo_err(r)}
    tt = r.json()
    link = _find_upload_link(tt)
    if not link:
        return {"ok": False, "step": "create texttrack",
                "detail": f"no upload link in response: {json.dumps(tt)[:300]}"}
    put = _rq.put(link, data=vtt.encode("utf-8"),
                  headers={"Content-Type": "text/vtt"}, timeout=120)
    if put.status_code not in (200, 201, 204):
        return {"ok": False, "step": "upload captions", "detail": _vimeo_err(put)}
    uri = tt.get("uri")
    if uri:
        act = _rq.patch(f"{VIMEO_API}{uri}", headers=_vimeo_headers(token),
                        json={"active": True}, timeout=30)
        if act.status_code not in (200, 201, 204):
            return {"ok": False, "step": "activate texttrack", "detail": _vimeo_err(act)}
    return {"ok": True, "uri": uri or ""}


_VIMEO_SPEC_CACHE = WORKSPACE / "web" / "vimeo_openapi.json"


def _vimeo_openapi(token: str) -> dict:
    """Fetch (and cache for a day) Vimeo's live OpenAPI spec for this token.

    The docs site is JS-rendered, and plan-gated endpoints 404 rather than
    403 — so the only reliable source for the audio-tracks contract is the
    spec Vimeo serves for the authenticated account."""
    try:
        if (_VIMEO_SPEC_CACHE.exists()
                and time.time() - _VIMEO_SPEC_CACHE.stat().st_mtime < 86400):
            return json.loads(_VIMEO_SPEC_CACHE.read_text(encoding="utf-8"))
    except Exception:
        pass
    try:
        r = _rq.get(f"{VIMEO_API}/?openapi=1", headers=_vimeo_headers(token), timeout=60)
        if r.status_code != 200:
            log.warning("vimeo: openapi fetch failed: %s", _vimeo_err(r))
            return {}
        spec = r.json()
    except Exception as e:
        log.warning("vimeo: openapi fetch failed: %s", e)
        return {}
    try:
        _VIMEO_SPEC_CACHE.write_text(json.dumps(spec), encoding="utf-8")
    except Exception:
        pass
    return spec


def _op_body_props(post: dict, spec: dict) -> List[str]:
    """Body property names of an operation (OpenAPI 3 and Swagger 2 shapes)."""
    props: List[str] = []
    rb = post.get("requestBody") or {}
    if "$ref" in rb:   # resolve one level of components ref
        ref = rb["$ref"].lstrip("#/").split("/")
        node = spec
        for part in ref:
            node = (node or {}).get(part, {})
        rb = node or {}
    for mt in (rb.get("content") or {}).values():
        props += list(((mt.get("schema") or {}).get("properties") or {}).keys())
    for p in post.get("parameters") or []:
        if p.get("in") in ("body", "formData"):
            schema_props = ((p.get("schema") or {}).get("properties") or {})
            props += list(schema_props.keys()) or [p.get("name", "")]
    return sorted({p for p in props if p})


def _vimeo_audio_create_op(spec: dict) -> Optional[Tuple[str, List[str]]]:
    """Locate the create-audio-track operation: (path_template, body_props)."""
    paths = spec.get("paths") or {}
    # Exact operationId first (matches the docs anchor), then any audio POST
    # under a video path.
    for match_exact in (True, False):
        for path, ops in paths.items():
            post = (ops or {}).get("post")
            if not post:
                continue
            opid = (post.get("operationId") or "").lower()
            if match_exact and opid != "create_audio_track":
                continue
            if not match_exact and not ("audio" in path.lower() and "video" in path.lower()):
                continue
            return path, _op_body_props(post, spec)
    return None


def _vimeo_push_audio(token: str, video_id: str, m4a_path: str,
                      language: str, name: str = "Français") -> dict:
    """Create a dubbed audio track and upload the full-mix M4A.

    The endpoint path and body fields are discovered from Vimeo's live
    OpenAPI spec (cached daily). Multi-audio is plan-gated: if the spec for
    this account has no audio-track create operation, say so explicitly."""
    spec = _vimeo_openapi(token)
    found = _vimeo_audio_create_op(spec) if spec else None
    if not found:
        if spec:
            return {"ok": False, "step": "create audio track",
                    "detail": "no audio-track endpoint in the API spec for this "
                              "account — multi-audio may not be available on this "
                              "Vimeo plan (or token scopes)"}
        return {"ok": False, "step": "create audio track",
                "detail": "could not fetch the Vimeo API spec to locate the "
                          "audio-track endpoint — retry, or check token scopes"}

    path_tpl, props = found
    path = re.sub(r"\{[^}]*video[^}]*\}", video_id, path_tpl)
    # Send only fields the operation actually accepts.
    candidates = {"language": language, "type": "dubbed", "name": name, "active": True}
    payload = {k: v for k, v in candidates.items() if not props or k in props}
    log.info("vimeo: audio-track endpoint %s (props: %s) payload %s",
             path_tpl, props or "unknown", sorted(payload))

    r = _rq.post(f"{VIMEO_API}{path}", headers=_vimeo_headers(token),
                 json=payload, timeout=30)
    if r.status_code not in (200, 201):
        hint = " — multi-audio may not be available on this Vimeo plan" if r.status_code in (403, 404) else ""
        return {"ok": False, "step": "create audio track",
                "detail": f"POST {path_tpl} {sorted(payload)} → {_vimeo_err(r)}{hint}"}
    created = r.json()

    link = _find_upload_link(created)
    if not link:
        return {"ok": False, "step": "create audio track",
                "detail": f"created but no upload link in response: {json.dumps(created)[:300]}"}
    with open(m4a_path, "rb") as f:
        put = _rq.put(link, data=f, headers={"Content-Type": "audio/mp4"}, timeout=600)
    if put.status_code not in (200, 201, 204):
        return {"ok": False, "step": "upload audio", "detail": _vimeo_err(put)}
    uri = created.get("uri")
    if uri:
        # Best-effort activation — some shapes auto-activate on upload.
        _rq.patch(f"{VIMEO_API}{uri}", headers=_vimeo_headers(token),
                  json={"active": True}, timeout=30)
    return {"ok": True, "uri": uri or "", "endpoint": path_tpl}


def _external_base(request: Request) -> str:
    proto = request.headers.get("x-forwarded-proto", "https")
    host = request.headers.get("x-forwarded-host") or request.headers.get("host", "")
    return f"{proto}://{host}"


@app.get("/api/vimeo/status")
async def vimeo_status() -> JSONResponse:
    tok = _vimeo_load_token()
    return JSONResponse({
        "connected": bool(tok.get("access_token")),
        "user": tok.get("user", ""),
        "via": tok.get("via", ""),
        "oauth_available": bool(VIMEO_CLIENT_ID and VIMEO_CLIENT_SECRET),
    })


@app.post("/api/vimeo/token")
async def vimeo_set_token(payload: dict) -> JSONResponse:
    """Connect with a pasted personal access token."""
    token = (payload.get("token") or "").strip()
    if not token:
        raise HTTPException(400, "token required")
    me = _vimeo_me(token)
    if not me:
        raise HTTPException(400, "Vimeo rejected the token (check scopes: public private edit upload)")
    _vimeo_save_token({
        "access_token": token,
        "user": me.get("name", ""),
        "via": "personal access token",
        "connected_at": time.time(),
    })
    return JSONResponse({"ok": True, "user": me.get("name", "")})


@app.delete("/api/vimeo/token")
async def vimeo_disconnect() -> JSONResponse:
    VIMEO_TOKEN_FILE.unlink(missing_ok=True)
    return JSONResponse({"ok": True})


@app.get("/api/vimeo/connect")
async def vimeo_connect(request: Request):
    if not (VIMEO_CLIENT_ID and VIMEO_CLIENT_SECRET):
        raise HTTPException(400, "set VIMEO_CLIENT_ID / VIMEO_CLIENT_SECRET (or paste a personal access token instead)")
    st = secrets.token_urlsafe(24)
    now = time.time()
    for k in [k for k, exp in _vimeo_oauth_states.items() if exp < now]:
        _vimeo_oauth_states.pop(k, None)
    _vimeo_oauth_states[st] = now + 600
    redirect = VIMEO_REDIRECT_URL or f"{_external_base(request)}/api/vimeo/callback"
    from urllib.parse import urlencode
    return RedirectResponse(
        "https://api.vimeo.com/oauth/authorize?" + urlencode({
            "response_type": "code",
            "client_id": VIMEO_CLIENT_ID,
            "redirect_uri": redirect,
            "state": st,
            "scope": VIMEO_OAUTH_SCOPES,
        })
    )


@app.get("/api/vimeo/callback")
async def vimeo_callback(request: Request, code: str = "", state: str = ""):
    if not code or _vimeo_oauth_states.pop(state, 0) < time.time():
        raise HTTPException(400, "invalid or expired OAuth state — retry Connect to Vimeo")
    redirect = VIMEO_REDIRECT_URL or f"{_external_base(request)}/api/vimeo/callback"
    r = _rq.post(
        f"{VIMEO_API}/oauth/access_token",
        auth=(VIMEO_CLIENT_ID, VIMEO_CLIENT_SECRET),
        headers={"Accept": VIMEO_ACCEPT},
        json={"grant_type": "authorization_code", "code": code, "redirect_uri": redirect},
        timeout=30,
    )
    if r.status_code not in (200, 201):
        raise HTTPException(502, f"Vimeo token exchange failed: {_vimeo_err(r)}")
    data = r.json()
    _vimeo_save_token({
        "access_token": data.get("access_token", ""),
        "user": (data.get("user") or {}).get("name", ""),
        "via": "oauth",
        "scope": data.get("scope", ""),
        "connected_at": time.time(),
    })
    return RedirectResponse("/", status_code=303)


@app.post("/api/jobs/{job_id}/vimeo-push")
async def vimeo_push(job_id: str, request: Request) -> JSONResponse:
    job = state.jobs.get(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    if job.status != STATUS_COMPLETED:
        raise HTTPException(409, f"job is not completed (status: {job.status})")
    tok = _vimeo_load_token()
    token = tok.get("access_token", "")
    if not token:
        raise HTTPException(400, "not connected to Vimeo")

    try:
        payload = await request.json()
    except Exception:
        payload = {}
    want_subs = bool(payload.get("subtitles", True))
    want_audio = bool(payload.get("audio", True))
    language = (payload.get("language") or "").strip() or "fr-CA"
    video_id = _vimeo_video_id(payload.get("video") or job.source_url or "")
    if not video_id:
        raise HTTPException(400, "no Vimeo video id — provide a vimeo.com URL or numeric id")

    srt = job.outputs.get("srt")
    m4a = job.outputs.get("full") or job.outputs.get("audio")
    results: dict = {}
    if want_subs:
        if srt and os.path.exists(srt):
            results["subtitles"] = await asyncio.to_thread(
                _vimeo_push_subtitles, token, video_id, srt, language)
        else:
            results["subtitles"] = {"ok": False, "step": "locate file", "detail": "no SRT output on this job"}
    if want_audio:
        if m4a and os.path.exists(m4a):
            results["audio"] = await asyncio.to_thread(
                _vimeo_push_audio, token, video_id, m4a, language)
        else:
            results["audio"] = {"ok": False, "step": "locate file", "detail": "no audio output on this job"}

    ok = all(v.get("ok") for v in results.values()) if results else False
    log.info("vimeo push job=%s video=%s → %s", job_id, video_id,
             {k: v.get("ok") for k, v in results.items()})
    return JSONResponse({"ok": ok, "video_id": video_id, "language": language, "results": results})


@app.get("/api/health")
async def health() -> JSONResponse:
    info: dict = {
        "pipeline_present": PIPELINE_PY.exists(),
        "config_present": CONFIG_PATH.exists(),
        "hf_token_present": bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")),
    }
    if IDLE_STOP_MIN > 0:
        info["idle_stop_min"] = IDLE_STOP_MIN
        info["idle_for_s"] = round(_idle_seconds())
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
