# Architecture

How a video moves through the pipeline, stage by stage. The single source of truth is
[`02_pipeline.py`](02_pipeline.py); this document explains the *why* behind each stage and
the knobs that matter. All settings live in [`config.yaml`](config.yaml).

```
MP4
 │
 ├─▶ 1. Source separation        Demucs htdemucs → vocals.wav + no_vocals.wav
 │
 ├─▶ 2. Transcription            faster-whisper large-v3 (word timestamps, VAD)
 │        └─ dedup / loop-collapse anti-hallucination cleanup
 │
 ├─▶ 3. Diarization (optional)   pyannote community-1 → who spoke when
 │        └─ runs BEFORE merging so speaker changes are merge boundaries
 │
 ├─▶ 4. Segment merging          sentence-scale chunks (2–12 s), never across speakers
 │
 ├─▶ 5. Translation              Ollama LLM (default mistral-small:22b)
 │        ├─ glossary prompt-injection (fr-ca)
 │        ├─ English-echo guard (re-translate leftovers)
 │        ├─ always-substitute glossary (deterministic)
 │        └─ compression pass (over-budget segments only, glossary-aware)
 │
 ├─▶ 6. Voice references         per-speaker ~12 s clips, noisereduce-denoised
 │
 ├─▶ 7. TTS                      F5-TTS flow-matching zero-shot cloning (24 kHz)
 │        └─ spoken-form normalization (inclusive doublets → pronounceable French)
 │
 └─▶ 8. Assembly + subtitles     timeline placement (anchored), Rubber Band stretch,
          background re-mix, hybrid BBC/Netflix SRT, optional video mux,
          EBU R128 loudness-normalized output
            →  _french.m4a / _full.m4a / .srt / _french.mp4
```

---

## 1. Source separation — Demucs `htdemucs`

Splits the original audio into **vocals** and **background** (`no_vocals`). The vocals feed
transcription, diarization, and the voice-clone references — cleaner input means better ASR
and a cleaner cloned voice. The background is kept so the French dub can be re-mixed over the
original music/ambience (`source_separation.preserve_background: true`).

## 2. Transcription — faster‑whisper `large-v3`

CTranslate2 backend, float16, **word‑level timestamps** and VAD. Word timestamps are reused
later to anchor subtitle cue boundaries to the original speech. `whisper.language` pins the
source language (empty = auto‑detect, for bilingual sources); `whisper.initial_prompt` is an
optional vocabulary hint — keep it to a term list, full sentences get echoed over silence.

Whisper can hallucinate looped phrases on silence or music. Four guards are tuned in
`config.yaml → whisper`:

- `condition_on_previous_text: false` — the biggest single switch against runaway repetition.
- `compression_ratio_threshold: 2.2` — rejects the "X. X. X." loop signature.
- `no_speech_threshold: 0.6` — drops windows VAD missed.
- `log_prob_threshold: -1.0` — drops low‑confidence segments.

Two post passes (`dedupe_whisper_segments`, `collapse_intrasegment_loops`) strip residual
overlap and within‑segment repetition.

## 3. Speaker diarization — pyannote `speaker-diarization-community-1` (optional)

When `diarization.enabled: true`, pyannote labels who is speaking when. It runs **before
segment merging** so that each raw Whisper fragment is tagged with the speaker holding the
most overlap, and a speaker change becomes a hard merge boundary — a merged chunk can never
span two speakers (which would dub both sides of an exchange in one cloned voice). A separate
voice‑clone reference is assembled per speaker, so every voice in a panel is dubbed
distinctly. Requires an `HF_TOKEN` that has accepted the gated model license. Set
`min_speakers` for reliable multi‑speaker detection.

## 4. Segment merging

Sub‑second Whisper fragments produce robotic TTS prosody and unreadable subtitles, so
fragments are merged into **sentence‑scale chunks**: keep merging across pauses ≤
`segment_merge_gap` until a chunk crosses `segment_merge_min_duration` and hits sentence
punctuation, never exceeding `segment_merge_max_duration` (2–12 s by default) and never
crossing a speaker change.

## 5. Translation — Ollama LLM (default `mistral-small:22b`)

A single natural pass over the merged segments, called over Ollama's HTTP API in batches
(`/no_think` is sent automatically for `qwen3.*` tags; any Ollama tag works via
`translation.model`). Every call pins an explicit `num_ctx` (8192) — Ollama's small default
context silently truncates the *front* of long prompts, i.e. exactly the instruction block
and glossary. Quality layers, in order:

- **Glossary prompt‑injection** (fr‑ca): mandatory vocabulary, acronyms to keep in English,
  and inclusive‑language rules are injected into the prompt from [`canadian_glossary.yaml`](canadian_glossary.yaml).
- **English‑echo guard** (`_retranslate_leftover_english`): the LLM occasionally returns a
  segment still in English; the old behaviour silently dubbed that English. Suspect segments
  (English‑looking output from an English source) are re‑translated individually with a strict
  prompt. Bilingual source clips (already‑French segments) are left untouched.
- **Always‑substitute glossary** (`apply_glossary`): deterministic rewrite of must‑win
  Québécois forms (`always:` section), gender‑ and elision‑aware
  (e.g. *le week‑end → la fin de semaine*, *la newsletter → l'infolettre*). Runs **before**
  compression so the length budget measures the final (often longer) Québécois wording.
- **Compression pass**: only the segments still over `budget_cps` are re‑prompted to tighten
  phrasing, iterated up to `compression_rounds` times (re‑compressing against the latest text
  until they fit) — cheap, and it keeps the French short enough to stay in sync downstream.
  The compression prompt carries the glossary so rewrites don't undo enforced terms.

## 6. Voice references

A ~12 s reference clip is extracted per speaker (skipping the first ~20 s of intro/music) and
denoised through noisereduce → FFmpeg `anlmdn` (whichever is available). F5‑TTS clones timbre
from this clip, so a clean reference is what preserves the original voice. Clips are hard‑
capped at 15 s: F5‑TTS's duration formula breaks above 22 s reference length.

**Pause condensing** (`_condense_silences`): F5‑TTS also clones the reference's *pace* — a
pause‑heavy reference makes every generated line for that speaker slow (measured: ~9.6 chars/s
vs ~16 for a dense reference, an unfittable 1.7× timing deficit). Silent gaps inside every
reference (auto profiles, single‑speaker samples, and review‑UI range picks) are therefore
capped at 300 ms; extra raw audio is collected first so the condensed clip still reaches the
target length. Curated library clips are used as‑is.

## 7. TTS — F5‑TTS (flow‑matching zero‑shot voice cloning)

Multilingual flow‑matching TTS with native French, 24 kHz output
(`tts.f5tts_model` — a built‑in model name or a HuggingFace repo ID such as the
`RASPIAUDIO/F5-French-MixedSpeakers-reduced` French fine‑tune). Text is split on
sentence/clause boundaries (≤250 chars per call); multi‑speaker jobs select the matching
per‑speaker reference for each segment. Quality knobs: `f5tts_nfe_step` (ODE steps),
`f5tts_cfg_strength`, `f5tts_speed`. Runaway/near‑silent outputs are detected and retried.

Before synthesis, written‑only conventions are normalized to a **spoken form**
(`_tts_spoken_form`): inclusive median‑dot doublets like *conférencier·ère* — required in the
subtitles by the CAPS style guide — are unpronounceable, so the TTS receives the collapsed
base form (re‑pluralised when the suffix carried the plural). Guillemets are stripped. The
subtitles keep the full inclusive written forms.

The same pass applies the **pronunciation lexicon** (`pronunciations:` in
[`canadian_glossary.yaml`](canadian_glossary.yaml), editable in the web UI's glossary editor):
names and brands the voice mispronounces are respelled phonetically for the TTS only
(*Vimeo → Viméo*), and — after the lexicon, so its entries win — unmapped 2–4‑letter
ALL‑CAPS acronyms are spelled out with French letter names (*CSP → cé esse pé*;
`tts.spell_acronyms`).

**ASR round‑trip verification** (`tts.verify_tts`): each synthesized segment is transcribed
back (faster‑whisper *small* on CPU — no VRAM contention with F5‑TTS) and scored against the
exact text the TTS was asked to say (punctuation‑insensitive similarity; lexicon respellings
masked from both sides). Segments below `verify_threshold` are re‑synthesized up to
`verify_retries` times — retries are independent samples since the seed is random — and the
best‑scoring take wins. This catches vocabulary bleed, swallowed/duplicated words, and
garbled‑but‑normal‑length output that the duration/RMS guards can't see. Scores land in
`synthesis_fit.csv` (`asr_score`) and the per‑run CSV (`tts_verify_flagged`).

**Pace management** (two generative layers — F5's speed parameter *speaks* faster, which
sounds far better than Rubber Band at 1.7–2.3×):

1. **Calibration** — one fixed sentence is synthesized per unique reference; a voice measured
   slower than the 14 chars/s target gets its base speed raised (≤ 1.25×). This normalizes a
   slow‑spoken reference at the source, since F5 clones the reference's delivery rate.
2. **Adaptive re‑synthesis** — a segment whose natural audio still can't fit its window at
   `tts.max_stretch` is re‑synthesized once, faster. Total speed‑up (calibration × adaptive)
   is capped at 1.45× of the configured `f5tts_speed`; beyond that F5 slurs.

After synthesis, each cloned voice's measured pace is logged per speaker, with a warning below
12 chars/s (the signature of a slow/pause‑heavy reference; `synthesis_fit.csv` also carries a
`speaker` column so this is diagnosable per run).

## 8. Assembly, timing, and subtitles

Synthesized clips are placed on a numpy timeline, time‑stretched with the Rubber Band FFmpeg
filter (formant‑preserving), and crossfaded. Consecutive segments within `tts.group_gap` share
one stretch ratio so speed changes are spread across a run, not dumped on one segment.

**Timing policy (`tts.timing_policy`):**

- `anchored` (default) — **holds the source timeline** so the dub stays in sync over a
  full‑length program. Each group is fit into its slot bidirectionally: dense runs are sped up
  (capped at `tts.max_stretch`, ~1.30×) instead of extending the timeline, and any small residual
  overrun is **re‑anchored** to the original timeline at the next pause, so drift can't accumulate.
  Crucially, **every line is floored at its own original onset** (isochrony): if the French for a
  run is shorter than the source speech, the dub waits for each line's real onset (a natural pause)
  rather than racing ahead of the picture — so a line can lag briefly in a dense burst and catch up,
  but **never leads** the speaker on screen. A group is only *slowed* toward `tts.reading_cps` when
  it already fits and has spare room in the slot (never past the slot edge). Best paired with a tight
  `translation.budget_cps`. The assembly log reports `output Xs vs source Ys (drift ±Zs)`.
- `no_drop` — audio is **never sped up**: dense runs are only slowed, and overruns **extend** the
  timeline, pushing later groups back. Nothing is cut, but on a dense long talk the output drifts
  progressively longer than the source (the original cause of end‑of‑video desync).
- `lock` — legacy exact‑timing mode: speed up to `max_stretch`, then truncate the overflow tail.

Length‑aware translation is the upstream half of staying in sync: `compress_overflowing_translations`
re‑prompts only the segments still over `translation.budget_cps`, iterating up to
`translation.compression_rounds` times, so the French is short enough to fit before assembly even runs.

**Subtitles (`subtitles.standard`)** are shaped to broadcast convention by `create_srt`:
text is split into cue‑sized chunks at sentence → clause → word boundaries, timed to the audio
via the preserved English word anchors, then polished so every cue meets the line‑length
(`max_chars_per_line`), reading‑speed (`max_cps`), duration (`min/max_duration`) and inter‑cue
gap rules. Lines wrap at logical points (never mid‑word, never stranding an article).
`standard: kapwing` restores the legacy single‑line karaoke behaviour.

**Output:** loudness‑normalised AAC 192 kbps / 48 kHz stereo (EBU R128 `loudnorm`,
−16 LUFS / −1.5 dBTP — consistent delivery level run‑to‑run; `audio.volume_boost_pct` shifts
the target instead of applying raw gain) — `_french.m4a` (dub only) and, when the background
was preserved, `_french_full.m4a` (dub side‑chain‑ducked over the original bed, mixed with
`amix normalize=0` so the voice keeps its level) — plus the UTF‑8 `.srt`. Subtitle timings
follow the actual audio placement (`placements`), so they stay in sync even after
speed‑up/slowdown. When `output.mux_video` is set, `mux_final_video` also produces
`_french.mp4` — the original video stream (copied) with the dubbed audio (stream‑copied, no
second lossy encode) and subtitles (soft `mov_text` track by default, or burned in with
`output.burn_subs`) — for one‑file sync verification; because the dub is held to the source
length, the streams end together.

## 9. Lip-sync — Wav2Lip (optional)

An opt-in final stage (`wav2lip.enabled`, or per-run `--wav2lip` / the web checkbox) re-syncs the
on-screen speaker's mouth to the French dub. `run_wav2lip` feeds the **original video** (as the face
source) and the **dubbed delivery audio** (`mux_audio`) to Wav2Lip, then re-attaches the SRT with
`mux_final_video` (video stream-copied), writing a separate `{name}_french_wav2lip.mp4`. The standard
`{name}_french.mp4` is never modified.

Wav2Lip pins old, conflicting dependencies (librosa 0.9 / numpy 1.x / opencv), so `04_setup.sh`
installs it into a **dedicated virtualenv** under `/workspace/wav2lip`, and `run_wav2lip` invokes
`inference.py` there as a subprocess — it never imports into, or perturbs, the main environment. The
stage runs **last** and is deliberately best-effort: a missing install, a missing checkpoint, or
slide-only footage with no detectable face is logged as a warning and returns without touching the
already-written outputs. It is not emitted as a `[N/6]` phase banner, so the web progress bar (which
parses that pattern) is unaffected while Wav2Lip's own progress streams to the live log.

**VRAM:** the Wav2Lip GAN generator plus the S3FD face detector total only a few hundred MB, and the
stage runs after F5-TTS has freed the GPU, so it fits comfortably on the 20 GB A4000. Because it
processes every frame, throughput — not memory — is the cost; `wav2lip.resize_factor` trades a little
resolution for speed, and `wav2lip.wav2lip_batch_size` / `face_det_batch_size` cap peak VRAM.

---

## VRAM budget (RTX 4090, 24 GB)

Models load and free in sequence (`max_workers: 1`), and the translation LLM is explicitly
unloaded from Ollama before F5‑TTS loads. Peak residency: Whisper large‑v3 (~3 GB) during
transcription, `mistral-small:22b` in Ollama (~13 GB) during translation, F5‑TTS (~2 GB)
during synthesis — each phase fits comfortably within 24 GB. Smaller translation models
(e.g. `qwen3:14b`, ~9 GB) leave more headroom.

## Where to look in the code

| Concern | Function(s) in `02_pipeline.py` |
|---|---|
| Transcription + anti‑hallucination | `transcribe_audio`, `dedupe_whisper_segments`, `collapse_intrasegment_loops` |
| Segment merging / CPS split | `merge_segments`, `split_overflowing_segments` |
| Translation + guards | `translate_segments`, `_retranslate_leftover_english`, `compress_overflowing_translations`, `apply_glossary` |
| Diarization / profiles | `diarize_audio`, `assign_speakers`, `build_speaker_profiles` |
| TTS + spoken-form normalization | `synthesize_all_segments`, `_tts_spoken_form` |
| Timeline / anchored / drift re‑anchoring | `assemble_and_encode` |
| Video mux | `mux_final_video` |
| Lip-sync (optional, isolated env) | `run_wav2lip` |
| Subtitles | `create_srt`, `_split_into_chunks`, `_wrap_two_lines`, `_enforce_subtitle_timing` |
