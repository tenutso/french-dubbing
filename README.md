# French Dubbing Pipeline

Turn English webinar MP4s into French audio tracks with synchronised SRT subtitles. Designed to run end-to-end on a single RTX 4090 (24 GB) RunPod instance.

**v4.1** swaps the TTS engine to **Coqui XTTS-v2** (Idiap fork) for native French support — the previous VoxCPM2 engine was English-centric and bled an English accent into French output. **v4.0** was the deliberate simplification: one translator, one TTS engine, no character-budget micromanagement, no WhisperX force-alignment. See [What's new](#whats-new) below.

The pipeline does everything from source separation through TTS and subtitle generation, and exposes both a CLI and a small **web UI** for non-technical users to submit jobs, monitor progress, and download results.

---

## What's in the box

| Stage | Component | Notes |
|---|---|---|
| Source separation | [Demucs htdemucs](https://github.com/facebookresearch/demucs) | Splits vocals from background music so the dub can be re-mixed cleanly |
| Transcription | [faster-whisper large-v3](https://github.com/SYSTRAN/faster-whisper) | CTranslate2, float16, word timestamps + VAD |
| Segment merging | sentence-scale chunks (2–12 s) | Sub-second fragments are absorbed into their neighbours |
| Speaker diarization | [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1) | Optional; per-speaker voice cloning when enabled |
| Translation | **Qwen3:14b via Ollama** | Single natural pass; no per-segment character budgets |
| Translation review | Qwen3:14b self-review (off by default) | Optional second pass for Anglicisms / register slips |
| TTS | **Coqui XTTS-v2** (Idiap fork, 24 kHz native) | Multilingual zero-shot voice cloning from a 25-second speaker reference; native French support |
| Speaker denoising | DeepFilterNet → noisereduce → FFmpeg `anlmdn` | Layered fallback chain for the voice-clone reference |
| Assembly | FFmpeg `atempo` + crossfade | Stretches French up to 1.3× into original timing windows; upsamples 24 kHz TTS → 48 kHz output |
| Subtitles | Direct from merged segment timings | No force-alignment; 1 s minimum entry duration |
| Output | AAC 192 kbps 48 kHz stereo (+ optional full-mix with background) + UTF-8 SRT | Vimeo-ready |

### Localisation

- **Languages**: 15 target languages supported by the prompt, set via `translation.target_lang`. French is the default.
- **Canadian French**: `--locale fr-ca` injects a glossary into the translation prompt and runs a deterministic post-processing substitution for "always"-mode terms. Glossary lives in [canadian_glossary.yaml](canadian_glossary.yaml).

---

## What's new

### v4.1 — TTS engine swap

VoxCPM2 is English-trained. When cloning an English speaker and asked to read French, it copied the source's phoneme contour — French came out with an English accent and broken prosody on cluster-heavy words. v4.1 replaces it with **Coqui XTTS-v2** via the actively-maintained [Idiap fork](https://github.com/idiap/coqui-ai-TTS) of Coqui TTS:

| | v4.0 (VoxCPM2) | v4.1 (XTTS-v2) |
|---|---|---|
| Native FR support | No — English-centric | Yes — French is a first-class trained language (17 total) |
| Cross-lingual cloning quality | English accent bleeds through | Native French prosody, voice timbre preserved |
| Model size on disk | ~4 GB | ~1.9 GB |
| Native sample rate | 48 kHz | 24 kHz (upsampled at assembly) |
| Code license | Apache 2.0 | MPL-2.0 |
| **Model-weight license** | **Apache 2.0** (commercial use OK) | **CPML — non-commercial only** ⚠️ |

> **⚠️ Commercial-use note:** XTTS-v2 weights are released under the [Coqui Public Model License](https://coqui.ai/cpml). That license **prohibits commercial use of the model**. If you're dubbing content for a paying audience, you'll need to either obtain a commercial license from Coqui, swap in a permissively-licensed engine, or stay on the v4.0 (VoxCPM2 / Apache-2.0) tag. The setup script auto-accepts CPML via `COQUI_TOS_AGREED=1` for convenience — accepting it on your behalf does not waive the non-commercial restriction.

Also in v4.1:
- One-line **RunPod bootstrap** ([bootstrap.sh](bootstrap.sh)) — set the container start command to `curl … | bash`, get an idempotent clone + setup + web-UI launch.
- **Ollama model cache** persisted to `/workspace/.ollama/` so qwen3:14b (~9 GB) survives container restarts.
- `transformers` pinned to `<5` — required by coqui-tts; `04_setup.sh` and `verify_setup.py` both enforce it.

### v4.0 — the rewrite

The v3 pipeline tried to micromanage every segment with character budgets and ran the LLM up to six times per batch. The results were bad enough to motivate a rewrite:

- The LLM echoed `(20)`, `(25)` character-count annotations into the SRT.
- `cps_safety: 0.85` combined with three retry passes shrank budgets to ~29 % of the natural length, producing nonsensical summaries.
- WhisperX force-alignment occasionally collapsed failed alignments into 0.099 s SRT entries.

v4 fixes all three at the root:

| Change | v3 | v4 |
|---|---|---|
| Translator | EuroLLM-9B / Qwen / Gemini (3 backends, 3 implementations) | Qwen3:14b via Ollama only |
| Translation passes per batch | up to 6 (fitted + 3 retries + natural + review) | 1 (+ optional review) |
| Character budgets | per-segment, shrunk on retry | none — atempo handles overflow |
| TTS engines | 6 (voxcpm2, xtts2, edge-tts, qwen3-tts × 2, gemini-tts) | XTTS-v2 only (v4.0 shipped VoxCPM2; replaced in v4.1) |
| Segment merging | break on any `.!?`, ≤5 s chunks | min 2 s, max 12 s, smart merge |
| SRT timing | WhisperX force-align (could collapse to 0.1 s) | direct from merged segments, min 1 s entry |
| `02_pipeline.py` | ~3000 lines | ~1200 lines |

If you need the multi-backend version, the `v3` tag on the repo preserves it.

---

## Quick start on RunPod

### 1. Launch a pod

1. Sign in at <https://runpod.io> → **Pods → Deploy**.
2. **GPU**: RTX 4090 (24 GB) recommended. RTX A5000 / A6000 / H100 also work.
3. **Template**: `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` (PyTorch 2.8.0 / CUDA 12.8.1 pre-installed).
4. **Storage**: at least 100 GB container disk; persist a 50 GB+ volume at `/workspace` — required so models survive container restarts.
5. **Expose HTTP port** `7860` (for the web UI) — set in the pod's **HTTP Service Ports**.
6. **Env vars**: set `HF_TOKEN=...` (pyannote diarization). Optional: `GITHUB_TOKEN` (private repos), `GIT_REF=<branch-or-tag>`.

### 2a. Hands-off: container start command (recommended)

Paste this as the pod's **Container Start Command**:

```bash
bash -c "curl -fsSL https://raw.githubusercontent.com/tenutso/french-dubbing/main/bootstrap.sh | bash"
```

[bootstrap.sh](bootstrap.sh) clones the repo, runs `04_setup.sh` if `requirements.txt` changed since the last boot (sha256 marker at `/workspace/.setup_done`), restarts Ollama, and `exec`s the web UI. First boot ~10–15 min; subsequent restarts ~30 s.

### 2b. Manual: SSH in and run setup yourself

If you'd rather not use the auto-bootstrap:

```bash
cd /workspace
git clone https://github.com/tenutso/french-dubbing.git
cd french-dubbing
bash 04_setup.sh
```

`04_setup.sh` is idempotent and self-recovering: each step prints a `[hh:mm:ss] >>> …` header, errors are collected and shown in a final summary instead of aborting. It will:

- install system packages (`ffmpeg`, `sox`, audio libs)
- create `/workspace/{videos/input,outputs,models,scripts,logs,temp}`
- pip-install the stack: `faster-whisper`, `demucs`, `pyannote.audio`, `DeepFilterNet`, `noisereduce`, `coqui-tts` (XTTS-v2), `transformers<5`, `pysrt`, plus `fastapi + uvicorn` for the web UI
- prompt for **HF_TOKEN** (required only for the pyannote gated model) and persist to `/workspace/.env`
- install + start `ollama` with `OLLAMA_MODELS=/workspace/.ollama/models` (cache persists across restarts), then pull `qwen3:14b`
- pre-download Whisper large-v3 + XTTS-v2 weights
- copy `02_pipeline.py`, `03_batch_runner.py`, `verify_setup.py`, `05_web.sh`, and the `web/` package into `/workspace/scripts/`

Expect 10–15 minutes the first time — faster than v4.0 since XTTS-v2 weights (~1.9 GB) are about half the size of VoxCPM2 (~4 GB).

### 3. Verify

```bash
python /workspace/scripts/verify_setup.py
```

GO/NO-GO summary with all the package + GPU + token checks.

### 4. Run it

**Option A — Web UI** (drag-drop, no terminal needed)

```bash
bash /workspace/scripts/05_web.sh
```

Open the RunPod-proxied URL for port 7860 (RunPod gives you something like `https://<pod-id>-7860.proxy.runpod.net`). Drop in a video, pick locale and volume boost, hit submit. Live log streams in the page, downloads appear when the job finishes.

**Option B — Single video on the CLI**

```bash
python /workspace/scripts/02_pipeline.py \
    --video /workspace/videos/input/webinar.mp4 \
    --locale fr-ca \
    --volume-boost 15
```

All CLI flags:

| Flag | Choices / Type | Effect |
|---|---|---|
| `--video` | path (required) | Input MP4 |
| `--output-dir` | path | Default `/workspace/outputs` |
| `--config` | path | Default `/workspace/config.yaml` |
| `--locale` | `fr` \| `fr-ca` | Overrides `translation.locale`; `fr-ca` triggers glossary |
| `--volume-boost` | float, % | Boost output loudness after peak-normalise; clipped at ±1.0 |
| `--force` | flag | Re-process even if outputs exist |

**Option C — Batch a folder**

```bash
cp ~/incoming/*.mp4 /workspace/videos/input/
python /workspace/scripts/03_batch_runner.py
```

One job at a time (VRAM-safe). Reports to `/workspace/logs/batch_report.json`.

### 5. Stop the pod

When you're done, **Stop** the pod from the RunPod console — that's what stops billing. Resume later; everything in `/workspace` persists if you allocated a volume.

---

## Web UI

The UI is a single FastAPI app at [web/app.py](web/app.py) with a vanilla-JS frontend in [web/static/](web/static/). Launch it with `bash 05_web.sh` (binds `0.0.0.0:7860`).

**Features**

- Drag-drop MP4 upload with progress bar (5 GB cap).
- Pre-fills locale and volume-boost from `config.yaml` defaults so you only override what you want.
- Single-job FIFO queue — submit while a job is running, the next one starts automatically when VRAM frees.
- Live log panel streams the pipeline's `[N/6] PHASE …` lines via Server-Sent Events.
- Download buttons for `_french.m4a`, `_french.srt`, and the optional `_french_full.m4a` (background remix).
- Crash-safe: if `uvicorn` is killed mid-job, the running job is recovered as `failed` on restart and queued jobs resume automatically.
- Cancel sends SIGTERM to the whole process group, so ffmpeg children get reaped.
- Footer shows live GPU, VRAM-free, disk-free, Ollama status, and whether HF token is present.

**Customise the port:**

```bash
DUBBING_WEB_PORT=8080 bash /workspace/scripts/05_web.sh
```

**No auth.** Per the original design choice — the RunPod proxy URL is unguessable for outsiders, but **don't post the URL publicly**. If you need protection, terminate it behind an authenticated reverse proxy (cloudflared / nginx + Basic auth).

---

## Configuration

Everything that isn't a runtime override lives in [config.yaml](config.yaml). The full file is heavily commented; the most useful knobs:

```yaml
diarization:
  enabled: true
  min_speakers: 2          # set to 1 only for known-single-speaker recordings
  max_speakers: 10

translation:
  model: qwen3:14b
  batch_size: 20
  review_pass: false       # set true for a second self-review pass (~2× slower)
  target_lang: fr          # fr es de it pt nl pl ru ja ko zh ar tr hi vi
  locale: fr               # or fr-ca

tts:
  xtts_model: tts_models/multilingual/multi-dataset/xtts_v2
  xtts_temperature: 0.65   # lower = more monotone, higher = more varied
  xtts_repetition_penalty: 2.0
  max_stretch: 1.3         # atempo cap for overflowing segments
  segment_merge_gap: 2.0
  segment_merge_max_duration: 15.0
  segment_merge_min_duration: 2.0

audio:
  volume_boost_pct: 0      # override at runtime with --volume-boost N

subtitles:
  sync_offset_ms: 0
```

---

## Outputs

For each `webinar.mp4` you'll get, in `/workspace/outputs/`:

- `webinar_french.m4a` — French dub only (vocals + silence in the original gaps), AAC 192 kbps / 48 kHz stereo
- `webinar_french.srt` — UTF-8 SRT, timestamps from the merged sentence chunks
- `webinar_french_full.m4a` — full mix: French vocals + the original background bed at −3 dB (only when Demucs separation succeeded and `source_separation.preserve_background: true`)

For Vimeo: upload `_full.m4a` as the alternate audio track and `.srt` as the French subtitle file.

---

## Running outside RunPod

Anywhere you have an NVIDIA GPU with ≥ 16 GB VRAM (24 GB recommended for Whisper + XTTS-v2 + Qwen3:14b co-resident) and CUDA 12.x:

- Use the [`Dockerfile`](Dockerfile) (PyTorch 2.8 / CUDA 12.8 base) — it mirrors what `04_setup.sh` does on the base RunPod image.
- Or run the setup script directly on Ubuntu 24.04 with PyTorch 2.8 already installed; everything else is pip-installable.

Note: v4 drops the CPU-only fallback paths (edge-tts / cloud TTS) that v3 supported, in exchange for a much simpler codebase. If you need CPU-only inference, stay on v3 or check out the `v3` tag.

---

## Troubleshooting

| Symptom | Most likely cause | Fix |
|---|---|---|
| `Diarization failed: 'DiarizeOutput' object has no attribute 'itertracks'` | pyannote returned a newer wrapped namedtuple shape | Already handled by [02_pipeline.py:`diarize_audio`](02_pipeline.py); upgrade pyannote (`pip install -U pyannote.audio`) and re-run. The log prints actual field names so any future shape change is a one-line patch. |
| Only `SPEAKER_00` detected on a multi-speaker recording | `min_speakers: 1` in config — pyannote collapses to 1 | Set `diarization.min_speakers: 2` (the default). Verify HF token has accepted the model license at <https://huggingface.co/pyannote/speaker-diarization-community-1>. |
| Ollama `Read timed out` during Qwen translation | First-batch cold load (~20 s) + ~2000-token generation on `qwen3:14b` | Already at 600 s + `keep_alive: 30m` in [_ollama_call](02_pipeline.py). Check `ollama ps` — if the model is offloaded to CPU, free VRAM and `ollama run qwen3:14b ""` to warm it. |
| `TRANSLATION FAILURE: N/N segments still in English` | Ollama isn't reachable, or the model name in `config.yaml` doesn't match `ollama list` | Run `ollama list`. Pull the model if missing: `ollama pull qwen3:14b`. |
| Output too quiet | Hard-coded 0.95 peak-normalise | Add `--volume-boost 20` (or set `audio.volume_boost_pct` in config). |
| SRT entry sometimes lasts more than the speech | Each entry is floored at 1 s for readability | Set `subtitles.sync_offset_ms` if you also need a global shift. |
| Diarization disabled but you want it | `use_diarization: false` in config, or HF token missing | Flip to `true` in `config.yaml`, ensure HF_TOKEN is in `/workspace/.env`. |
| Web UI shows "missing FastAPI" / `ModuleNotFoundError` | `04_setup.sh` didn't reach the web-deps step | `pip install fastapi 'uvicorn[standard]' python-multipart`, then re-run `bash /workspace/scripts/05_web.sh`. |
| `ImportError: cannot import name 'isin_mps_friendly' from 'transformers.pytorch_utils'` when synth starts | A later `pip install --upgrade` pulled `transformers>=5`, which removed an internal symbol coqui-tts depends on | Pin back: `pip install 'transformers<5'`. `verify_setup.py` will warn you if the wrong version is present. |
| `qwen3:14b` re-downloads on every container restart | Ollama is reading from `~/.ollama` (ephemeral disk) instead of `/workspace/.ollama/models` | Make sure `OLLAMA_MODELS` is exported before `ollama serve` starts. Setup since v4.1 handles this; for older installs add it to `/workspace/.env` and restart Ollama. |
| Web UI restarts mid-job → job stuck `running` forever | Server crashed before SSE could close | Already auto-recovered: `running` → `failed (server restarted mid-job)` on next launch, queued jobs replayed. |

For deeper guidance see `05_QUICK_REFERENCE.md`, `06_ARCHITECTURE.md`, and the comments inside `02_pipeline.py`.

---

## File layout

```
french-dubbing/
├── 02_pipeline.py            # main CLI (single video)
├── 03_batch_runner.py        # batch folder of MP4s
├── 04_setup.sh               # one-shot installer for RunPod / Ubuntu 24.04
├── 05_web.sh                 # uvicorn launcher (web UI)
├── bootstrap.sh              # RunPod container-start entrypoint (clone + setup + web)
├── verify_setup.py           # GO/NO-GO post-install sanity check
├── config.yaml               # all non-runtime knobs (commented)
├── canadian_glossary.yaml    # fr-ca vocabulary + formatting rules
├── requirements.txt          # pip deps (lower bounds; setup.sh auto-upgrades)
├── Dockerfile                # PyTorch 2.8 / CUDA 12.8 reproducible image
├── web/
│   ├── app.py                # FastAPI app + queue worker + SSE
│   ├── jobs.py               # Job dataclass + atomic jobs.json
│   └── static/
│       ├── index.html
│       ├── app.js
│       └── style.css
└── README.md                 # this file
```

Runtime layout (created by `04_setup.sh`):

```
/workspace/
├── videos/input/             # drop MP4s here for CLI / batch runs
├── outputs/                  # CLI / batch outputs land here
├── scripts/                  # installed copy of the pipeline + web/
├── models/                   # auto-populated (Whisper, XTTS-v2, pyannote)
├── logs/                     # per-video + batch + ollama + bootstrap logs
├── temp/                     # intermediate stems (auto-cleaned)
├── .ollama/models/           # Ollama model cache (qwen3:14b ~9 GB)
├── .setup_done               # bootstrap marker: sha256 of requirements.txt
├── web/
│   ├── uploads/              # web-UI uploads, per-job
│   ├── outputs/<job_id>/     # web-UI per-job output dirs
│   └── jobs.json             # persisted queue + history
├── config.yaml
└── .env                      # HF_TOKEN, OLLAMA_MODELS, COQUI_TOS_AGREED, HF_HUB_ENABLE_HF_TRANSFER
```

---

## Costs

Rough numbers on RunPod (May 2026 pricing; subject to change):

| GPU | $/hr | Time/video (1 h source) | Cost/video |
|---|---|---|---|
| RTX 4090 | ~$0.36 | 15–25 min | **$0.09 – $0.15** |
| RTX A5000 | ~$0.45 | 20–30 min | **$0.15 – $0.23** |

v4 is meaningfully faster than v3 (single translation pass instead of up to six; no WhisperX alignment), so per-video cost is roughly halved on the same hardware.

---

## License & attribution

- Pipeline code, web UI, glue: **MIT**
- faster-whisper (CTranslate2): MIT
- pyannote.audio: MIT
- Coqui XTTS-v2 (Idiap fork): **code** MPL-2.0; **model weights** [CPML — non-commercial only](https://coqui.ai/cpml). Auto-accepted at install via `COQUI_TOS_AGREED=1`. For commercial use, swap engines or stay on the `v4.0` tag (VoxCPM2 / Apache 2.0).
- DeepFilterNet: MIT / Apache
- Demucs: MIT
- Qwen3 model (via Ollama): Apache 2.0
- FFmpeg: LGPL/GPL (depending on build)

No proprietary APIs are required to run the pipeline end-to-end. Note the XTTS-v2 model weights' non-commercial restriction above — every other component is permissively licensed for commercial use.

---

## Contributing / debugging

- Logs per video: `/workspace/logs/<stem>.log`
- Web UI logs: stdout of `05_web.sh`
- Batch summary: `/workspace/logs/batch_report.json`
- Job state across web-UI restarts: `/workspace/web/jobs.json`

Issues and PRs welcome at <https://github.com/tenutso/french-dubbing>.
