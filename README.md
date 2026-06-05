# French Dubbing Pipeline

Turn English webinar / presentation MP4s into natural **French (or French‑Canadian) audio tracks** with broadcast‑grade **SRT subtitles**. Runs end‑to‑end on a single RTX 4090 (24 GB) — locally or on RunPod — with no proprietary APIs.

The pipeline does everything from source separation through translation, voice‑cloned TTS, and subtitle generation, and exposes both a **CLI** and a small **web UI** for non‑technical users to submit jobs, watch progress, and download results.

---

## What's in the box

| Stage | Component | Notes |
|---|---|---|
| Source separation | [Demucs `htdemucs`](https://github.com/facebookresearch/demucs) | Splits vocals from background music so the dub can be re‑mixed cleanly |
| Transcription | [faster‑whisper `large-v3`](https://github.com/SYSTRAN/faster-whisper) | CTranslate2, float16, word timestamps + VAD, anti‑hallucination tuned |
| Segment merging | sentence‑scale chunks (2–12 s) | Sub‑second fragments are absorbed into neighbours for natural prosody |
| Speaker diarization | [pyannote `speaker-diarization-community-1`](https://huggingface.co/pyannote/speaker-diarization-community-1) | On by default; builds a separate voice clone per speaker |
| Translation | **Qwen3:14b via Ollama** | Single natural pass + targeted compression; glossary prompt‑injection |
| English‑echo guard | automatic re‑translation | Detects segments the LLM left in English and re‑translates them individually |
| TTS | **Coqui XTTS‑v2** (Idiap fork, 24 kHz native) | Multilingual zero‑shot voice cloning from a ~25 s reference; native French |
| Speaker denoising | DeepFilterNet → noisereduce → FFmpeg `anlmdn` | Layered fallback for a clean voice‑clone reference |
| Assembly | numpy timeline + Rubber Band time‑stretch + crossfade | **Never drops words** (see timing policy); upsamples 24 kHz → 48 kHz |
| Subtitles | **Hybrid BBC/Netflix shaper** | ≤2 lines, ≤42 cpl, ≤17 CPS reading speed, logical line breaks |
| Output | AAC 192 kbps / 48 kHz stereo (+ optional full‑mix with background) + UTF‑8 SRT | Vimeo‑ready |

### Two features worth calling out

- **Never‑drop‑words timing (`timing_policy: no_drop`, default).** When a translated run is longer than its slot, the assembler *extends the timeline* instead of speeding up and truncating the tail — so no sentence is ever cut. Output may run slightly longer than the source. A `lock` mode preserves exact source timing for lip‑sync‑sensitive work.
- **Reading‑speed coupling.** Speech runs whose translated text is denser than `tts.reading_cps` (16) are gently *slowed* (never sped up), capped at `tts.max_slowdown` (1.25×). This de‑rushes the dub and keeps subtitles under the 17 CPS reading‑speed limit.

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

`04_setup.sh` is idempotent. It installs system packages (`ffmpeg`, `sox`, audio libs), creates `/workspace/{videos/input,outputs,models,scripts,logs,temp}`, pip‑installs the stack (`faster-whisper`, `demucs`, `pyannote.audio`, `DeepFilterNet`, `noisereduce`, `coqui-tts`, `transformers<5`, `pysrt`, `fastapi`+`uvicorn`), persists `HF_TOKEN`, starts Ollama with a persisted cache and pulls `qwen3:14b`, pre‑downloads the Whisper + XTTS‑v2 weights, and copies the pipeline into `/workspace/scripts/`.

### 3. Verify

```bash
python /workspace/scripts/verify_setup.py
```

### 4. Run it

**Web UI** (drag‑drop, no terminal):

```bash
bash /workspace/scripts/05_web.sh
```

Open the RunPod‑proxied URL for port 7860, drop in a video, pick locale + volume boost, submit. Live log streams; downloads appear when done.

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
| `--volume-boost` | float, % | Boost output loudness after peak‑normalise |
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

- Drag‑drop MP4 upload with progress bar.
- Pre‑fills locale + volume‑boost from `config.yaml`.
- Single‑job FIFO queue; live log via Server‑Sent Events.
- Download buttons for `_french.m4a`, `_french.srt`, and the optional `_french_full.m4a`.
- Crash‑safe: a job interrupted by a server restart is recovered as `failed`; queued jobs resume.
- Footer shows live GPU / VRAM / disk / Ollama / HF‑token status.

```bash
DUBBING_WEB_PORT=8080 bash /workspace/scripts/05_web.sh   # custom port
```

**No auth** — the RunPod proxy URL is unguessable, but don't post it publicly. Put it behind an authenticated reverse proxy if you need protection.

---

## Configuration

Everything lives in [config.yaml](config.yaml) (heavily commented). The most useful knobs:

```yaml
diarization:
  enabled: true
  min_speakers: 2          # set to 1 only for known single‑speaker recordings
  max_speakers: 10

translation:
  model: qwen3:14b
  review_pass: false       # optional self‑review pass (~2× slower)
  compression_pass: true   # tighten only the over‑budget segments
  target_lang: fr          # fr es de it pt nl pl ru ja ko zh ar tr hi vi
  locale: fr-ca            # loads canadian_glossary.yaml

tts:
  xtts_temperature: 0.65
  stretcher: rubberband
  timing_policy: no_drop    # no_drop = never truncate (default) | lock = exact timing
  reading_cps: 16.0         # slow speech runs denser than this (no_drop only)
  max_slowdown: 1.25        # cap on that slow‑down

subtitles:
  standard: netflix         # netflix (≤42 cpl) | bbc (≤37 cpl) | kapwing (legacy karaoke)
  max_cps: 17.0             # reading‑speed cap
  min_duration: 0.833
  max_duration: 7.0
```

---

## Outputs

For each `webinar.mp4`, in `/workspace/outputs/`:

- `webinar_french.m4a` — French dub only, AAC 192 kbps / 48 kHz stereo
- `webinar_french.srt` — UTF‑8 SRT, BBC/Netflix‑shaped cues
- `webinar_french_full.m4a` — full mix: French vocals + original background bed (when Demucs separation succeeded and `source_separation.preserve_background: true`)

For Vimeo: upload `_full.m4a` as the alternate audio track and `.srt` as the French subtitle file.

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

Any NVIDIA GPU with ≥16 GB VRAM (24 GB recommended for Whisper + XTTS‑v2 + Qwen3:14b co‑resident) and CUDA 12.x:

- Use the [`Dockerfile`](Dockerfile) (PyTorch 2.8 / CUDA 12.8 base), or
- Run `04_setup.sh` directly on Ubuntu 24.04 with PyTorch 2.8 installed.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Only `SPEAKER_00` on a multi‑speaker clip | `min_speakers: 1` | Set `diarization.min_speakers: 2`; confirm HF token accepted the [model license](https://huggingface.co/pyannote/speaker-diarization-community-1) |
| Ollama `Read timed out` | First‑batch cold load | Already at 600 s + `keep_alive: 30m`; check `ollama ps` and warm with `ollama run qwen3:14b ""` |
| `TRANSLATION FAILURE: N/N segments still in English` | Ollama unreachable or wrong model name | `ollama list`; `ollama pull qwen3:14b` |
| A few words sound English in the dub | Rare stochastic LLM echo | The English‑echo guard re‑translates these automatically; check the log for "Re‑translating … Recovered" |
| Output too quiet | 0.95 peak‑normalise | `--volume-boost 20` |
| Dub runs longer than the source | `timing_policy: no_drop` is extending dense passages | Expected; switch to `lock` for exact timing, or raise `tts.reading_cps` |
| `ImportError: ... isin_mps_friendly` at synth | `transformers>=5` pulled in | `pip install 'transformers<5'` |
| `qwen3:14b` re‑downloads each restart | Ollama reading ephemeral `~/.ollama` | Export `OLLAMA_MODELS=/workspace/.ollama/models` before `ollama serve` |

See [06_ARCHITECTURE.md](06_ARCHITECTURE.md) and the comments in [02_pipeline.py](02_pipeline.py) for deeper detail.

---

## File layout

```
french-dubbing/
├── 02_pipeline.py            # main CLI (single video)
├── 03_batch_runner.py        # batch a folder of MP4s
├── 04_setup.sh               # one‑shot installer (RunPod / Ubuntu 24.04)
├── 05_web.sh                 # uvicorn launcher (web UI)
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
- faster‑whisper (CTranslate2): MIT · pyannote.audio: MIT · Demucs: MIT · DeepFilterNet: MIT/Apache · Qwen3 (via Ollama): Apache 2.0 · FFmpeg: LGPL/GPL
- **Coqui XTTS‑v2 (Idiap fork): code MPL‑2.0; model weights [CPML — non‑commercial only](https://coqui.ai/cpml).** Auto‑accepted at install via `COQUI_TOS_AGREED=1`. ⚠️ For commercial dubbing you must obtain a commercial license from Coqui or swap in a permissively‑licensed cloning TTS — every other component is already commercial‑friendly.

No proprietary APIs are required to run the pipeline end‑to‑end.

Issues and PRs welcome at <https://github.com/tenutso/french-dubbing>.
