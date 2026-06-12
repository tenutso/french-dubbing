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
 ├─▶ 3. Segment merging          sentence-scale chunks (2–12 s)
 │
 ├─▶ 4. Diarization (optional)   pyannote community-1 → who spoke when
 │
 ├─▶ 5. Translation              Qwen3:14b via Ollama
 │        ├─ glossary prompt-injection (fr-ca)
 │        ├─ English-echo guard (re-translate leftovers)
 │        ├─ compression pass (over-budget segments only)
 │        └─ always-substitute glossary (deterministic)
 │
 ├─▶ 6. Voice references         per-speaker 25 s clips, noisereduce-denoised
 │
 ├─▶ 7. TTS                      Coqui XTTS-v2 zero-shot cloning (24 kHz)
 │
 └─▶ 8. Assembly + subtitles     timeline placement (anchored), Rubber Band stretch,
          background re-mix, hybrid BBC/Netflix SRT, optional video mux
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
later to anchor subtitle cue boundaries to the original speech.

Whisper can hallucinate looped phrases on silence or music. Four guards are tuned in
`config.yaml → whisper`:

- `condition_on_previous_text: false` — the biggest single switch against runaway repetition.
- `compression_ratio_threshold: 2.2` — rejects the "X. X. X." loop signature.
- `no_speech_threshold: 0.6` — drops windows VAD missed.
- `log_prob_threshold: -1.0` — drops low‑confidence segments.

Two post passes (`dedupe_whisper_segments`, `collapse_intrasegment_loops`) strip residual
overlap and within‑segment repetition.

## 3. Segment merging

Sub‑second Whisper fragments produce robotic TTS prosody and unreadable subtitles, so
fragments are merged into **sentence‑scale chunks**: keep merging across pauses ≤
`segment_merge_gap` until a chunk crosses `segment_merge_min_duration` and hits sentence
punctuation, never exceeding `segment_merge_max_duration` (2–12 s by default).

## 4. Speaker diarization — pyannote `speaker-diarization-community-1` (optional)

When `diarization.enabled: true`, pyannote labels who is speaking when. Each transcript
segment is tagged with the speaker holding the most overlap, and a separate voice‑clone
reference is assembled per speaker — so every voice in a panel is dubbed distinctly. Requires
an `HF_TOKEN` that has accepted the gated model license. Set `min_speakers` for reliable
multi‑speaker detection.

## 5. Translation — Qwen3:14b via Ollama

A single natural pass over the merged segments, called over Ollama's HTTP API in batches
(`/no_think` is sent automatically for `qwen3.*` tags). Quality layers:

- **Glossary prompt‑injection** (fr‑ca): mandatory vocabulary, acronyms to keep in English,
  and inclusive‑language rules are injected into the prompt from [`canadian_glossary.yaml`](canadian_glossary.yaml).
- **English‑echo guard** (`_retranslate_leftover_english`): the LLM occasionally returns a
  segment still in English; the old behaviour silently dubbed that English. Suspect segments
  (English‑looking output from an English source) are re‑translated individually with a strict
  prompt. Bilingual source clips (already‑French segments) are left untouched.
- **Compression pass**: only the segments still over `budget_cps` are re‑prompted to tighten
  phrasing, iterated up to `compression_rounds` times (re‑compressing against the latest text
  until they fit) — cheap, and it keeps the French short enough to stay in sync downstream.
- **Always‑substitute glossary** (`apply_glossary`): deterministic, post‑translation rewrite of
  must‑win Québécois forms (`always:` section), gender‑ and elision‑aware
  (e.g. *le week‑end → la fin de semaine*, *la newsletter → l'infolettre*).

Why Qwen3:14b? It scored higher than Gemma3:27b on FR‑CA translation (chrF 63.7 vs 62.6) at
roughly half the VRAM, leaving headroom for Whisper + XTTS to stay resident. Swap models via
`translation.model` — any Ollama tag works.

## 6. Voice references

A ~25 s reference clip is extracted per speaker (skipping the first ~20 s of intro/music) and
denoised through noisereduce → FFmpeg `anlmdn` (whichever is available). XTTS
clones timbre from this clip, so a clean reference is what preserves the original voice.

## 7. TTS — Coqui XTTS‑v2 (Idiap fork)

Multilingual zero‑shot voice cloning with native French, 24 kHz output. Text is split on
sentence/clause boundaries to respect XTTS's per‑call length limit; multi‑speaker jobs select
the matching per‑speaker reference for each segment. Sampling is controlled by
`tts.xtts_temperature` / `repetition_penalty` / `top_k` / `top_p`.

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

**Output:** peak‑normalised AAC 192 kbps / 48 kHz stereo — `_french.m4a` (dub only) and, when
the background was preserved, `_french_full.m4a` (dub side‑chain‑ducked over the original bed) —
plus the UTF‑8 `.srt`. Subtitle timings follow the actual audio placement (`placements`), so they
stay in sync even after speed‑up/slowdown. When `output.mux_video` is set, `mux_final_video` also
produces `_french.mp4` — the original video stream (copied) with the dubbed audio and subtitles
(soft `mov_text` track by default, or burned in with `output.burn_subs`) — for one‑file sync
verification; because the dub is held to the source length, the streams end together.

---

## VRAM budget (RTX 4090, 24 GB)

Models load and free in sequence (`max_workers: 1`). Peak co‑residency is Whisper large‑v3
(~3 GB) + XTTS‑v2 (~2 GB) + Qwen3:14b in Ollama (~9 GB), comfortably within 24 GB. Larger
translation models (e.g. `gemma3:27b`, ~17 GB) also fit but leave less headroom.

## Where to look in the code

| Concern | Function(s) in `02_pipeline.py` |
|---|---|
| Transcription + anti‑hallucination | `transcribe_audio`, `dedupe_whisper_segments`, `collapse_intrasegment_loops` |
| Segment merging / CPS split | `merge_segments`, `split_overflowing_segments` |
| Translation + guards | `translate_segments_qwen`, `_retranslate_leftover_english`, `compress_overflowing_translations`, `apply_glossary` |
| Diarization / profiles | `diarize_audio`, `assign_speakers`, `build_speaker_profiles` |
| TTS | `synthesize_all_segments` |
| Timeline / anchored / drift re‑anchoring | `assemble_and_encode` |
| Video mux | `mux_final_video` |
| Subtitles | `create_srt`, `_split_into_chunks`, `_wrap_two_lines`, `_enforce_subtitle_timing` |
