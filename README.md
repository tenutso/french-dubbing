# French Dubbing Pipeline

Turn English webinar / presentation MP4s into natural **French (or French‑Canadian) audio tracks** with broadcast‑grade **SRT subtitles**. Runs end‑to‑end on a single RTX 4090 (24 GB) — locally or on RunPod — with no proprietary APIs.

The pipeline does everything from source separation through translation, voice‑cloned TTS, and subtitle generation, and exposes both a **CLI** and a small **web UI** for non‑technical users to submit jobs, watch progress, and download results.

---

## What's in the box

| Stage | Component | Notes |
|---|---|---|
| Source separation | [Demucs `htdemucs`](https://github.com/facebookresearch/demucs) | Splits vocals from background music so the dub can be re‑mixed cleanly |
| Transcription | [faster‑whisper `large-v3`](https://github.com/SYSTRAN/faster-whisper) | CTranslate2, float16, word timestamps + VAD, anti‑hallucination tuned |
| Speaker diarization | [pyannote `speaker-diarization-community-1`](https://huggingface.co/pyannote/speaker-diarization-community-1) | On by default; runs **before** merging so chunks never span two speakers; builds a separate voice clone per speaker |
| Segment merging | sentence‑scale chunks (2–12 s) | Sub‑second fragments are absorbed into neighbours for natural prosody, never across a speaker change |
| Translation | **mistral-small:22b via Ollama** | Single natural pass + targeted compression; glossary prompt‑injection; native French quality |
| English‑echo guard | automatic re‑translation | Detects segments the LLM left in English and re‑translates them individually |
| TTS | **F5‑TTS** (flow‑matching DiT, 24 kHz native) | Multilingual zero‑shot voice cloning from a ~12 s pause‑condensed reference; per‑voice pace calibration; reliable on names and unusual input |
| Speaker denoising | noisereduce → FFmpeg `anlmdn` | Layered fallback for a clean voice‑clone reference |
| Assembly | numpy timeline + Rubber Band time‑stretch + crossfade | **Holds the source timeline** (see timing policy); upsamples 24 kHz → 48 kHz |
| Subtitles | **Hybrid BBC/Netflix shaper** | ≤2 lines, ≤42 cpl, ≤17 CPS reading speed, logical line breaks |
| Output | AAC 192 kbps / 48 kHz stereo (+ optional full‑mix with background) + UTF‑8 SRT + optional muxed MP4 | Vimeo‑ready |

### Features worth calling out

- **Timeline‑anchored timing (`timing_policy: anchored`, default).** Each translated run is fit into its original slot — dense runs are sped up a touch (capped at `tts.max_stretch`, ~1.30×, inaudible on speech) and accumulated drift is re‑anchored at every pause — so the dub stays in sync with the video over a full‑length program. Pairs with length‑aware translation (`translation.budget_cps`, iterated over `translation.compression_rounds`) that keeps the French tight enough to fit. A `no_drop` mode never speeds up (timeline extends, so it drifts longer than the source on dense talks); `lock` preserves exact source timing and truncates overflow for lip‑sync‑sensitive work.
- **Reading‑speed coupling.** Speech runs whose translated text is denser than `tts.reading_cps` (16) are gently *slowed* toward that pace (capped at `tts.max_slowdown`, 1.25×, and never past the slot edge under `anchored`). This de‑rushes the dub and keeps subtitles under the 17 CPS reading‑speed limit.
- **Per‑voice pace management.** F5‑TTS clones each reference clip's speaking rate, so a pause‑heavy or slow‑spoken reference would make that speaker's entire dub run long. Three layers prevent it: reference clips are **pause‑condensed** (gaps capped at 300 ms), each cloned voice is **calibrated** once and slow voices get a gentle generative speed‑up (≤1.25×), and any segment that still can't fit its window is **adaptively re‑synthesized** faster instead of relying on time‑stretch. Per‑speaker pace is logged after synthesis, with a warning when a voice is too slow to fit a dub timeline.
- **Name pronunciation + ASR verification.** A **pronunciation lexicon** (`pronunciations:` in the glossary, editable in the web UI) phonetically respells names/brands for the TTS only — subtitles keep real spellings — and unmapped ALL‑CAPS acronyms are spelled out with French letter names. Then every synthesized segment is **transcribed back** (whisper‑small, CPU) and scored against its intended text; low‑similarity takes are re‑synthesized best‑of‑N (`tts.verify_tts`), catching garbled names, vocabulary bleed, and swallowed words that duration/volume checks miss.

### Localisation

- **Languages**: 15 target languages via `translation.target_lang` (French is the default).
- **French‑Canadian**: `--locale fr-ca` injects [`canadian_glossary.yaml`](canadian_glossary.yaml) into the translation prompt **and** runs a deterministic post‑translation substitution for `always:` terms (e.g. *email → courriel*, *le week‑end → la fin de semaine*, gender/elision‑aware). See [CAPS_French_Style_Guide.md](CAPS_French_Style_Guide.md).

---

## Quick start on RunPod

### 1. Launch a pod

1. <https://runpod.io> → **Pods → Deploy**.
2. **GPU**: RTX 4090 (24 GB) recommended (A5000 / A6000 / H100 also work).
3. **Template**: `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` (PyTorch 2.8 / CUDA 12.8).
4. **Storage**: ≥100 GB container disk; persist a 50 GB+ volume at `/workspace` so models survive restarts.
5. **Expose HTTP port** `7860` (web UI).
6. **Env vars**: `HF_TOKEN=...` (pyannote diarization). Optional: `GITHUB_TOKEN`, `GIT_REF`.

### 2a. Hands‑off — container start command (recommended)

```bash
bash -c "curl -fsSL https://raw.githubusercontent.com/tenutso/french-dubbing/main/bootstrap.sh | bash"
```

[bootstrap.sh](bootstrap.sh) clones the repo, runs `04_setup.sh` when `requirements.txt` changes, restarts Ollama, and launches the web UI. First boot ~10–15 min; restarts ~30 s.

### 2b. Manual

```bash
cd /workspace
git clone https://github.com/tenutso/french-dubbing.git
cd french-dubbing
bash 04_setup.sh
```

`04_setup.sh` is idempotent. It installs system packages (`ffmpeg`, `sox`, audio libs), creates `/workspace/{videos/input,outputs,models,scripts,logs,temp}`, pip‑installs the stack (`faster-whisper`, `demucs`, `pyannote.audio`, `noisereduce`, `f5-tts`, `pysrt`, `fastapi`+`uvicorn`), persists `HF_TOKEN`, starts Ollama with a persisted cache and pulls `mistral-small:22b`, pre‑downloads the Whisper + F5‑TTS weights (~1.5 GB), and copies the pipeline into `/workspace/scripts/`.

### 3. Verify

```bash
python /workspace/scripts/verify_setup.py
```

### 4. Run it

**Web UI** (drag‑drop, no terminal):

```bash
bash /workspace/scripts/05_web.sh
```

Open the RunPod‑proxied URL for port 7860, drop in a video, pick locale, speaker count, and volume boost, submit. Live log streams; downloads appear when done.

**Single video (CLI):**

```bash
python /workspace/scripts/02_pipeline.py \
    --video /workspace/videos/input/webinar.mp4 \
    --locale fr-ca \
    --volume-boost 15
```

| Flag | Choices / Type | Effect |
|---|---|---|
| `--video` | path (required) | Input MP4 |
| `--output-dir` | path | Default `/workspace/outputs` |
| `--config` | path | Default `/workspace/config.yaml` |
| `--locale` | `fr` \| `fr-ca` | `fr-ca` loads the Canadian glossary |
| `--speakers` | int 1–20 | Exact speaker count for this video (overrides diarization min/max). `1` = solo presenter: skips diarization entirely |
| `--volume-boost` | float, % | Boost output loudness (shifts the −16 LUFS loudnorm target) |
| `--phase` | `1` \| `2` | `1` = transcribe+translate then stop (for review); `2` = TTS+assembly from the saved segments |
| `--segments-file` | path | Phase 2: load segments from a custom path |
| `--keep-temp` | flag | Keep intermediate stage JSON for debugging |
| `--force` | flag | Re‑process even if outputs exist |

**Batch a folder:**

```bash
cp ~/incoming/*.mp4 /workspace/videos/input/
python /workspace/scripts/03_batch_runner.py
```

One job at a time (VRAM‑safe). Reports to `/workspace/logs/batch_report.json`.

### 5. Stop the pod

**Stop** the pod from the RunPod console to stop billing. Everything in `/workspace` persists if you allocated a volume.

---

## Web UI

A single FastAPI app at [web/app.py](web/app.py) with a vanilla‑JS frontend in [web/static/](web/static/). Launch with `bash 05_web.sh` (binds `0.0.0.0:7860`).

- Drag‑drop MP4 upload with progress bar (or a Vimeo URL, or an on‑pod path for big files).
- Per‑job options: locale, **speaker count** (blank = auto; `1` = solo presenter), volume boost, and an optional pause‑for‑review stage to edit translations and pick voice references before synthesis.
- Single‑job FIFO queue; live log via Server‑Sent Events.
- Download buttons for `_french.m4a`, `_french.srt`, the optional `_french_full.m4a`, and the muxed `_french.mp4`.
- An **Advanced options** panel edits the per‑video subset of `config.yaml` (source language, vocabulary hint, translation model/review pass, output toggles). The tuned timing/quality internals are deliberately not exposed — edit `config.yaml` directly for experiments.
- **Push to Vimeo**: connect once (personal access token or OAuth), then deliver subtitles + dubbed audio track onto the source video from the job card — see [Push to Vimeo](#push-to-vimeo).
- Crash‑safe: a job interrupted by a server restart is recovered as `failed`; queued jobs resume.
- Footer shows live GPU / VRAM / disk / Ollama / HF‑token status.

```bash
DUBBING_WEB_PORT=8080 bash /workspace/scripts/05_web.sh   # custom port
```

**Auth** — set `DUBBING_UI_TOKEN` in the environment before launching. Every request must then carry the token (the UI shows a sign‑in page once and stores an HttpOnly cookie; scripts can pass `Authorization: Bearer <token>` or `?token=`). If unset, the UI is **unauthenticated** — fine on localhost, not behind a public proxy URL.

```bash
DUBBING_UI_TOKEN=$(openssl rand -hex 24) bash /workspace/scripts/05_web.sh
```

---

## On‑demand operation (budget)

For occasional workloads, don't keep the pod running — pay for GPU seconds only while a job
is processing (a typical video costs $0.15–0.35 on a community‑cloud 4090; the only fixed
cost is volume storage, a few $/month):

1. **Put `/workspace` on a RunPod network volume** so pods can be stopped (or even terminated
   and re‑created) without losing models, jobs, or outputs. Cold start with warm volume ≈ 30 s
   via [bootstrap.sh](bootstrap.sh). Secrets (`DUBBING_UI_TOKEN`, `VIMEO_ACCESS_TOKEN`,
   `RUNPOD_API_KEY`, `HF_TOKEN`) can live in `/workspace/.env` on the volume — both the
   pipeline and the web app load it at startup, so fresh pods boot fully configured without
   template edits.
2. **Enable idle auto‑stop** — set in the pod template:

   ```bash
   DUBBING_IDLE_STOP_MIN=10        # stop the pod after 10 min with no jobs
   ```

   The web app stops its own pod (via `runpodctl`, falling back to the RunPod REST API if
   `RUNPOD_API_KEY` is set) once there has been no running/queued job and no mutating request
   for that long. GET polling (the UI footer) does **not** keep the pod alive; submitting,
   editing translations, or saving voice references does. Jobs paused at the review stage
   persist on the volume and survive a stop. `/api/health` reports `idle_for_s` so you can
   see the countdown.
3. **Start on demand with [trigger.sh](trigger.sh)** from any machine (needs `curl` + `python3`):

   ```bash
   export RUNPOD_API_KEY=... RUNPOD_POD_ID=... DUBBING_UI_TOKEN=...
   ./trigger.sh --vimeo https://vimeo.com/12345 --locale fr-ca --speakers 3 --wait --download ./out
   ```

   It starts the pod, waits for the API, submits the job (Vimeo URL, on‑pod path, or file
   upload), and optionally polls to completion and downloads the outputs. The pod then
   idle‑stops on its own. Any backend can do the same three HTTP calls directly:
   `POST rest.runpod.io/v1/pods/{id}/start` → poll `GET /api/health` → `POST /api/jobs`.

One caveat: a *stopped* pod is not guaranteed to get its GPU back at start time. If
`trigger.sh` reports the pod can't start, create a fresh pod on the same network volume
(same template) — nothing is lost.

---

## Configuration

Everything lives in [config.yaml](config.yaml) (heavily commented). The most useful knobs:

```yaml
diarization:
  enabled: true
  min_speakers: 2          # per‑job override: the Speakers field / --speakers flag
  max_speakers: 10

whisper:
  language: en             # "" = auto‑detect per file (bilingual sources)
  initial_prompt: ""       # optional domain vocabulary hint ("CAPS, CSP, keynote")

translation:
  model: mistral-small:22b
  review_pass: false       # optional self‑review pass (~2× slower)
  compression_pass: true   # tighten only the over‑budget segments
  compression_rounds: 3    # iterate the compression pass until segments fit
  budget_cps: 16           # char/sec budget per segment — tighter = better sync
  target_lang: fr          # fr es de it pt nl pl ru ja ko zh ar tr hi vi
  locale: fr-ca            # loads canadian_glossary.yaml

tts:
  f5tts_model: RASPIAUDIO/F5-French-MixedSpeakers-reduced   # French fine‑tune (HF repo ID or built‑in name)
  f5tts_nfe_step: 32                    # ODE steps: 16 = fast draft, 32 = high quality
  f5tts_cfg_strength: 2.0               # CFG: higher = more faithful to reference voice
  stretcher: rubberband
  timing_policy: anchored   # anchored = hold source timeline (default) | no_drop = never speed up | lock = exact timing + truncate
  max_stretch: 1.3          # per‑group speed‑up cap used to hold the timeline (anchored)
  reading_cps: 16.0         # slow speech runs denser than this toward this pace
  max_slowdown: 1.25        # cap on that slow‑down

output:
  mux_video: true           # also emit _french.mp4 (video + dub audio + subs)
  burn_subs: false          # false = soft SRT track (copy video) | true = burn in (re‑encode)

subtitles:
  standard: netflix         # netflix (≤42 cpl) | bbc (≤37 cpl) | kapwing (legacy karaoke)
  max_cps: 17.0             # reading‑speed cap
  min_duration: 0.833
  max_duration: 7.0
```

The timing/quality stack (`budget_cps`, `max_stretch`, compression, the Whisper anti‑hallucination
thresholds, F5‑TTS internals) is tuned as one coherent system — validated by the per‑run metrics in
`{output}/_dubbing_metrics_runs.csv` (watch `synthesis_pct_unfit`; under ~5% means a healthy run).
Change those together and re‑measure, not one at a time.

---

## Outputs

For each `webinar.mp4`, in `/workspace/outputs/`:

- `webinar_french.m4a` — French dub only, AAC 192 kbps / 48 kHz stereo
- `webinar_french.srt` — UTF‑8 SRT, BBC/Netflix‑shaped cues
- `webinar_french_full.m4a` — full mix: French vocals + original background bed (when Demucs separation succeeded and `source_separation.preserve_background: true`)
- `webinar_french.mp4` — original video + dubbed audio + subtitles, muxed for one‑file review (when `output.mux_video: true`). The dub is held to the source length, so audio, subs, and picture stay in sync end‑to‑end.

For Vimeo, use the built‑in push (below) — or manually upload `_full.m4a` as the alternate audio track and `.srt` as the French subtitle file. The `_french.mp4` is mainly for verifying sync locally.

---

## Push to Vimeo

The web UI can deliver a finished dub straight onto the source Vimeo video: the SRT as an
**active French text track** and the full‑mix M4A as a **dubbed audio track**
(multi‑audio requires a Vimeo plan that supports it; the UI surfaces Vimeo's exact error if not).

**Connect once** (token persists on the volume across pod stops):

- *Hands‑off*: set `VIMEO_ACCESS_TOKEN` (a [personal access token](https://developer.vimeo.com/apps)
  with scopes `public private edit upload`) in the pod template env or in `/workspace/.env` —
  the app verifies and connects automatically at every boot. A token pasted in the UI takes
  precedence over the env seed.
- *UI*: paste the same personal access token into the Vimeo card.
- *Or OAuth*: create a Vimeo API app, set `VIMEO_CLIENT_ID` / `VIMEO_CLIENT_SECRET`, register
  `{pod-url}/api/vimeo/callback` as the redirect URL, and use **Connect to Vimeo**.

**Push**: completed jobs show a *Push to Vimeo* button — target video pre‑filled from the job's
Vimeo URL (editable for uploaded files), language defaults to `fr-CA` for `fr-ca` jobs, with
per‑item results shown inline.

**Auto‑push**: tick *Auto‑push to Vimeo* at submit time (Vimeo‑URL sources; `--auto-push` in
`trigger.sh`) and the push runs as the final step of the job itself — safe to walk away from a
long job on an idle‑stopping pod, since the pod can't power down between the dub finishing and
the outputs reaching Vimeo. Results land in the job log and on the job card; a push failure
never fails the job (re‑push manually from the card).

**Automation** (n8n etc.): the same action is one API call —
`POST /api/jobs/{id}/vimeo-push` with `{"video": "...", "language": "fr-CA", "subtitles": true, "audio": true}`
(Bearer‑auth with `DUBBING_UI_TOKEN`), so a workflow can go submit → wait → push without
downloading any files itself.

---

## Evaluation harness

[`eval/`](eval/) contains the quality tooling used to validate the pipeline:

| Script | Measures |
|---|---|
| `srt_lint.py` | BBC/Netflix compliance (line length, reading speed, durations, gaps) + chrF/BLEU vs a reference SRT |
| `audio_eval.py` | ASR round‑trip word‑recall (dropped words) + duration drift |
| `translate_bench.py` | Head‑to‑head LLM translation scoring (reuses the pipeline's own translate path) |
| `tts_mos.py` | Reference‑free speech quality (SQUIM PESQ / STOI / SI‑SDR) |

---

## Running outside RunPod

Any NVIDIA GPU with ≥16 GB VRAM (24 GB recommended for Whisper + F5‑TTS + mistral‑small:22b co‑resident) and CUDA 12.x:

- Use the [`Dockerfile`](Dockerfile) (PyTorch 2.8 / CUDA 12.8 base), or
- Run `04_setup.sh` directly on Ubuntu 24.04 with PyTorch 2.8 installed.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Only `SPEAKER_00` on a multi‑speaker clip | `min_speakers: 1` | Submit with the exact speaker count (Speakers field / `--speakers N`); confirm HF token accepted the [model license](https://huggingface.co/pyannote/speaker-diarization-community-1) |
| Solo presenter split into two alternating voices | Forced `min_speakers: 2` on a one‑person video | Submit with Speakers = 1 (`--speakers 1`) — skips diarization entirely |
| One speaker's lines lag / sound stretched | Slow or pause‑heavy voice reference (log shows `cloned voice speaks at N chars/s` warning) | Usually auto‑corrected by pause‑condensing + pace calibration; if it persists, pick a denser reference range for that speaker in the review UI |
| Ollama `Read timed out` | First‑batch cold load | Already at 600 s + `keep_alive: 30m`; check `ollama ps` and warm with `ollama run mistral-small:22b ""` |
| `TRANSLATION FAILURE: N/N segments still in English` | Ollama unreachable or wrong model name | `ollama list`; `ollama pull mistral-small:22b` |
| A few words sound English in the dub | Rare stochastic LLM echo | The English‑echo guard re‑translates these automatically; check the log for "Re‑translating … Recovered" |
| Output too quiet | Loudness target (−16 LUFS) | `--volume-boost 20` (shifts the loudnorm target ~+1.6 dB) |
| Dub runs longer than the source / drifts out of sync on long videos | `timing_policy: no_drop` extends the timeline on dense passages | Use the default `timing_policy: anchored` (holds the source timeline); tighten `translation.budget_cps` (≈15) so the French fits, or use `lock` for exact timing |
| A dense line sounds slightly rushed | `anchored` sped that group up to fit its slot | Raise `tts.max_stretch` cap relief is the wrong way — instead *lower* `translation.budget_cps` so the line is shorter; or accept it (capped at `tts.max_stretch`, ~1.30×) |
| Ollama model re‑downloads each restart | Ollama reading ephemeral `~/.ollama` | Export `OLLAMA_MODELS=/workspace/.ollama/models` before `ollama serve` |

See [06_ARCHITECTURE.md](06_ARCHITECTURE.md) and the comments in [02_pipeline.py](02_pipeline.py) for deeper detail.

---

## File layout

```
french-dubbing/
├── 02_pipeline.py            # main CLI (single video)
├── 03_batch_runner.py        # batch a folder of MP4s
├── 04_setup.sh               # one‑shot installer (RunPod / Ubuntu 24.04)
├── 05_web.sh                 # uvicorn launcher (web UI)
├── trigger.sh                # on-demand client: start pod → submit job → download
├── bootstrap.sh              # RunPod container‑start entrypoint
├── verify_setup.py           # GO/NO‑GO post‑install check
├── config.yaml               # all knobs (commented)
├── canadian_glossary.yaml    # fr‑ca vocabulary (suggest + always‑substitute)
├── CAPS_French_Style_Guide.md
├── 06_ARCHITECTURE.md        # pipeline internals
├── eval/                     # quality harness (lint, MOS, LLM/TTS benchmarks)
├── Dockerfile
└── web/                      # FastAPI app + static frontend
```

---

## License & attribution

- Pipeline code, web UI, glue: **MIT**
- faster‑whisper (CTranslate2): MIT · pyannote.audio: MIT · Demucs: MIT · noisereduce: MIT · translation LLM: per‑model licence (check your Ollama tag — e.g. `mistral-small:22b` is Mistral Research License, `mistral-small:24b` and `qwen3:14b` are Apache 2.0) · FFmpeg: LGPL/GPL
- **F5‑TTS**: MIT licence (model weights and code). All pipeline components are now commercial‑friendly.

No proprietary APIs are required to run the pipeline end‑to‑end.

Issues and PRs welcome at <https://github.com/tenutso/french-dubbing>.
