#!/usr/bin/env python3
"""
Pipeline metrics — per-phase CSV instrumentation for the French dubbing pipeline.

Goal: make every phase that touches the *text* (and the timeline it must fit)
measurable, so the translation prompt, budget_cps, compression rounds and
timing policy can be tuned against hard numbers instead of spot-checks.

Enabled via `metrics.enabled: true` in config.yaml. All output lands in
    {output_folder}/{video-stem}_metrics/

and is written defensively — a metrics failure logs a warning and never breaks
a dub.

What it captures
----------------
1. segments_<NN>_<phase>.csv  — a per-segment snapshot AFTER each phase
   (merge → translate → compress → glossary → cps_split → final retime).
   Diff two snapshots to see exactly what a phase changed.

   Per-segment columns most useful for tuning translation:
     duration_s        target audio window (seconds)
     en_chars/fr_chars source vs translated length
     en_cps/fr_cps     characters per second (the sync-critical number)
     expansion_ratio   fr_chars / en_chars  (French verbosity vs English)
     budget_chars      duration_s * budget_cps  (what the LLM was asked to hit)
     over_budget       fr_chars > budget_chars
     over_by_pct       how far over budget, in percent
     exceeds_split_cps fr_cps above the cps_split_threshold (will be split)

2. synthesis_fit.csv — the ground truth on whether a translation FIT.
   Compares each segment's natural TTS length against its target window:
     tts_natural_s     length of synthesized French before any stretch
     required_stretch  tts_natural_s / duration_s
     fits              required_stretch <= max_stretch
   required_stretch > max_stretch means the French was too long to fit the
   timeline — i.e. the translation (or budget_cps) needs to be tighter.

3. phase_summary.csv — one row per phase: segment counts, fr_cps distribution,
   mean expansion, and the count/percentage of over-budget segments. The
   trend down this table shows how much compression actually buys you.

4. {output_folder}/_dubbing_metrics_runs.csv — one row per video, appended
   across runs, so prompt/config changes can be compared over time.
"""

from __future__ import annotations

import csv
import logging
import os
import re
import statistics
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


# Signatures of LLM meta-commentary / prompt-instruction leakage (echoed review
# criteria, markdown headers, rewrite arrows) — the kind of text that should
# never appear in a translation. Kept in sync with _LEAK_SIGNATURE_RE in
# 02_pipeline.py (duplicated to keep this module dependency-free).
_LEAK_SIGNATURE_RE = re.compile(
    r"\*\*"
    r"|(?:^|\s)#{2,}\s"
    r"|→"
    r"|timing efficiency"
    r"|voice[\s-]*dubbing"
    r"|natural spoken rhythm"
    r"|grammar (?:and|/) fluency"
    r"|canadian french usage"
    r"|(?:reduce|éviter les) anglicism"
    r"|prefer shorter"
    r"|trim redundan"
    r"|privilégier la concision"
    r"|mots redondants"
    r"|termes québécois"
    r"|éviter les traductions",
    re.IGNORECASE,
)


def _looks_like_instruction_leak(text: str) -> bool:
    return bool(text) and bool(_LEAK_SIGNATURE_RE.search(text))


def _seg_en(seg: dict) -> str:
    return (seg.get("text") or "").replace("\n", " ").strip()


def _seg_fr(seg: dict) -> str:
    return (
        seg.get("text_fr")
        or seg.get("text_fr_natural")
        or ""
    ).replace("\n", " ").strip()


def _safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def _pct(part: int, whole: int) -> float:
    return 100.0 * part / whole if whole else 0.0


# Column order for the per-segment snapshot CSVs.
_SEGMENT_FIELDS = [
    "id", "start_s", "end_s", "duration_s", "speaker",
    "en_chars", "fr_chars", "en_cps", "fr_cps", "expansion_ratio",
    "budget_cps", "budget_chars", "over_budget", "over_by_pct",
    "over_readability", "exceeds_split_cps", "suspect_leak", "text_en", "text_fr",
]


class MetricsCollector:
    """Collects per-phase CSV metrics for a single video run.

    Construct once per video, call snapshot() after each phase, optionally
    record_synthesis_fit() after TTS, then finalize() at the end.
    """

    def __init__(
        self,
        enabled: bool,
        output_dir: str,
        name: str,
        log: logging.Logger,
        budget_cps: float = 15.0,
        cps_split_threshold: float = 21.0,
        max_stretch: float = 1.3,
        readability_cap: float = 17.0,
    ) -> None:
        self.enabled = bool(enabled)
        self.name = name
        self.log = log
        self.budget_cps = float(budget_cps) or 15.0
        self.cps_split_threshold = float(cps_split_threshold)
        self.max_stretch = float(max_stretch)
        # Subtitle reading-speed cap (subtitles.max_cps). budget_cps is the
        # translator's *input* target; readability_cap is the actual quality
        # gate for the final output, where anchored retiming has set the real
        # window. The final_retimed phase should be judged against this.
        self.readability_cap = float(readability_cap) or 17.0
        self.output_dir = output_dir
        self.metrics_dir = os.path.join(output_dir, f"{name}_metrics")
        self._phase_index = 0
        # phase_label -> aggregate dict, in capture order
        self._summaries: List[dict] = []
        self._synthesis_summary: Optional[dict] = None

        if not self.enabled:
            return
        try:
            os.makedirs(self.metrics_dir, exist_ok=True)
            self.log.info(f"  [metrics] enabled → {self.metrics_dir}")
        except Exception as e:
            self.log.warning(f"  [metrics] could not create metrics dir ({e}) — disabling")
            self.enabled = False

    # -- per-segment math ---------------------------------------------------

    def _row(self, seg: dict) -> dict:
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", 0.0))
        dur = max(end - start, 0.0)
        en = _seg_en(seg)
        fr = _seg_fr(seg)
        en_chars = len(en)
        fr_chars = len(fr)
        en_cps = _safe_div(en_chars, dur)
        fr_cps = _safe_div(fr_chars, dur)
        budget_chars = int(dur * self.budget_cps)
        over_budget = bool(fr_chars and budget_chars and fr_chars > budget_chars)
        over_by_pct = _pct(fr_chars - budget_chars, budget_chars) if budget_chars else 0.0
        return {
            "id": seg.get("id", ""),
            "start_s": round(start, 3),
            "end_s": round(end, 3),
            "duration_s": round(dur, 3),
            "speaker": seg.get("speaker", ""),
            "en_chars": en_chars,
            "fr_chars": fr_chars,
            "en_cps": round(en_cps, 2),
            "fr_cps": round(fr_cps, 2),
            "expansion_ratio": round(_safe_div(fr_chars, en_chars), 3),
            "budget_cps": round(self.budget_cps, 2),
            "budget_chars": budget_chars,
            "over_budget": int(over_budget),
            "over_by_pct": round(over_by_pct, 1),
            "over_readability": int(fr_chars and fr_cps > self.readability_cap),
            "exceeds_split_cps": int(
                self.cps_split_threshold > 0 and fr_cps > self.cps_split_threshold
            ),
            "suspect_leak": int(_looks_like_instruction_leak(fr)),
            "text_en": en,
            "text_fr": fr,
        }

    @staticmethod
    def _aggregate(phase: str, rows: List[dict]) -> dict:
        n = len(rows)
        fr_cps = [r["fr_cps"] for r in rows if r["fr_chars"] > 0]
        exp = [r["expansion_ratio"] for r in rows if r["fr_chars"] > 0 and r["en_chars"] > 0]
        translated = sum(1 for r in rows if r["fr_chars"] > 0)
        over = sum(r["over_budget"] for r in rows)
        over_read = sum(r["over_readability"] for r in rows)
        over_split = sum(r["exceeds_split_cps"] for r in rows)
        leaks = sum(r["suspect_leak"] for r in rows)

        def _stat(fn, vals, default=0.0):
            try:
                return round(fn(vals), 2) if vals else default
            except Exception:
                return default

        def _pctl(vals, q):
            if not vals:
                return 0.0
            s = sorted(vals)
            k = min(len(s) - 1, int(round(q * (len(s) - 1))))
            return round(s[k], 2)

        return {
            "phase": phase,
            "n_segments": n,
            "n_translated": translated,
            "total_duration_s": round(sum(r["duration_s"] for r in rows), 1),
            "total_en_chars": sum(r["en_chars"] for r in rows),
            "total_fr_chars": sum(r["fr_chars"] for r in rows),
            "mean_fr_cps": _stat(statistics.mean, fr_cps),
            "median_fr_cps": _stat(statistics.median, fr_cps),
            "p90_fr_cps": _pctl(fr_cps, 0.90),
            "max_fr_cps": _stat(max, fr_cps),
            "mean_expansion": _stat(statistics.mean, exp),
            "n_over_budget": over,
            "pct_over_budget": round(_pct(over, translated), 1),
            "n_over_readability": over_read,
            "pct_over_readability": round(_pct(over_read, translated), 1),
            "n_over_split_cps": over_split,
            "n_suspect_leak": leaks,
        }

    # -- public API ---------------------------------------------------------

    def snapshot(self, phase: str, segments: Sequence[dict]) -> None:
        """Write a per-segment CSV for `phase` and record its aggregate row."""
        if not self.enabled:
            return
        try:
            rows = [self._row(s) for s in segments]
            self._phase_index += 1
            fname = f"segments_{self._phase_index:02d}_{phase}.csv"
            path = os.path.join(self.metrics_dir, fname)
            with open(path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=_SEGMENT_FIELDS)
                w.writeheader()
                w.writerows(rows)
            agg = self._aggregate(phase, rows)
            self._summaries.append(agg)
            leak_note = (
                f", {agg['n_suspect_leak']} SUSPECT LEAK" if agg["n_suspect_leak"] else ""
            )
            self.log.info(
                f"  [metrics] {phase}: {agg['n_segments']} seg, "
                f"mean {agg['mean_fr_cps']} CPS, "
                f"{agg['n_over_budget']} over budget ({agg['pct_over_budget']}%), "
                f"{agg['n_over_readability']} over readability "
                f"({agg['pct_over_readability']}% > {self.readability_cap:g} CPS)"
                f"{leak_note} → {fname}"
            )
        except Exception as e:
            self.log.warning(f"  [metrics] snapshot '{phase}' failed: {e}")

    def record_synthesis_fit(
        self,
        segments: Sequence[dict],
        synthesized: Sequence[Tuple],
        src_rate: int,
    ) -> None:
        """Compare natural TTS length vs target window → did the translation fit?

        `synthesized` is the assembler's list of (audio, start, end). We match
        each segment to its synthesized clip by rounded start time and compute
        the stretch the assembler would have to apply.
        """
        if not self.enabled:
            return
        try:
            by_start: Dict[int, "object"] = {}
            for item in synthesized:
                audio, start = item[0], item[1]
                by_start[round(float(start) * 1000)] = audio

            rows: List[dict] = []
            for seg in segments:
                key = round(float(seg.get("start", 0.0)) * 1000)
                audio = by_start.get(key)
                if audio is None:
                    continue
                try:
                    n_samples = len(audio)
                except TypeError:
                    continue
                natural = _safe_div(n_samples, src_rate)
                start = float(seg.get("start", 0.0))
                end = float(seg.get("end", 0.0))
                window = max(end - start, 0.0)
                required = _safe_div(natural, window) if window else 0.0
                fr = _seg_fr(seg)
                rows.append({
                    "id": seg.get("id", ""),
                    "speaker": seg.get("speaker", ""),
                    "start_s": round(start, 3),
                    "duration_s": round(window, 3),
                    "fr_chars": len(fr),
                    "tts_natural_s": round(natural, 3),
                    "required_stretch": round(required, 3),
                    "max_stretch": round(self.max_stretch, 3),
                    "fits": int(required <= self.max_stretch) if required else 1,
                    "text_fr": fr,
                })

            if not rows:
                return

            path = os.path.join(self.metrics_dir, "synthesis_fit.csv")
            fields = [
                "id", "speaker", "start_s", "duration_s", "fr_chars", "tts_natural_s",
                "required_stretch", "max_stretch", "fits", "text_fr",
            ]
            with open(path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=fields)
                w.writeheader()
                w.writerows(rows)

            n_unfit = sum(1 for r in rows if not r["fits"])
            stretches = [r["required_stretch"] for r in rows if r["required_stretch"]]
            self._synthesis_summary = {
                "phase": "synthesis_fit",
                "n_segments": len(rows),
                "n_unfit": n_unfit,
                "pct_unfit": round(_pct(n_unfit, len(rows)), 1),
                "mean_required_stretch": round(statistics.mean(stretches), 3) if stretches else 0.0,
                "max_required_stretch": round(max(stretches), 3) if stretches else 0.0,
            }
            self.log.info(
                f"  [metrics] synthesis_fit: {n_unfit}/{len(rows)} segment(s) "
                f"need > {self.max_stretch}x stretch "
                f"({self._synthesis_summary['pct_unfit']}% won't fit) → synthesis_fit.csv"
            )
        except Exception as e:
            self.log.warning(f"  [metrics] synthesis_fit failed: {e}")

    def finalize(self) -> None:
        """Write phase_summary.csv and append a run row to the cross-run CSV."""
        if not self.enabled:
            return
        try:
            summary_fields = [
                "phase", "n_segments", "n_translated", "total_duration_s",
                "total_en_chars", "total_fr_chars", "mean_fr_cps", "median_fr_cps",
                "p90_fr_cps", "max_fr_cps", "mean_expansion", "n_over_budget",
                "pct_over_budget", "n_over_readability", "pct_over_readability",
                "n_over_split_cps", "n_suspect_leak",
            ]
            path = os.path.join(self.metrics_dir, "phase_summary.csv")
            with open(path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=summary_fields)
                w.writeheader()
                w.writerows(self._summaries)
            self.log.info(f"  [metrics] phase_summary.csv ({len(self._summaries)} phases)")

            self._append_run_row()
        except Exception as e:
            self.log.warning(f"  [metrics] finalize failed: {e}")

    def _append_run_row(self) -> None:
        """Append one row per video to {output_dir}/_dubbing_metrics_runs.csv."""
        if not self._summaries:
            return
        # Prefer the final retimed phase if present, else the last text phase.
        final = self._summaries[-1]
        first = self._summaries[0]
        translated = next(
            (s for s in self._summaries if s["phase"] in ("translated", "translate")),
            None,
        )
        post = next(
            (s for s in reversed(self._summaries) if s["n_translated"]),
            final,
        )
        row = {
            "video": self.name,
            "n_segments": final["n_segments"],
            "total_duration_s": first["total_duration_s"],
            "budget_cps": round(self.budget_cps, 2),
            "post_translate_mean_cps": translated["mean_fr_cps"] if translated else "",
            "post_translate_over_budget_pct": translated["pct_over_budget"] if translated else "",
            "final_mean_cps": post["mean_fr_cps"],
            "final_over_budget_pct": post["pct_over_budget"],
            "final_over_readability_pct": post["pct_over_readability"],
            "final_mean_expansion": post["mean_expansion"],
            "final_max_cps": post["max_fr_cps"],
            "final_suspect_leak": final["n_suspect_leak"],
        }
        if self._synthesis_summary:
            row["synthesis_pct_unfit"] = self._synthesis_summary["pct_unfit"]
            row["synthesis_mean_required_stretch"] = self._synthesis_summary["mean_required_stretch"]
        else:
            row["synthesis_pct_unfit"] = ""
            row["synthesis_mean_required_stretch"] = ""

        runs_path = os.path.join(self.output_dir, "_dubbing_metrics_runs.csv")
        fields = list(row.keys())
        try:
            exists = os.path.exists(runs_path)
            # If an older file has a different header, don't clobber it; just skip.
            if exists:
                with open(runs_path, newline="", encoding="utf-8") as f:
                    header = f.readline().strip().split(",")
                if header and header != fields:
                    self.log.debug("  [metrics] cross-run CSV header differs — leaving as-is")
            with open(runs_path, "a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=fields)
                if not exists:
                    w.writeheader()
                w.writerow(row)
            self.log.info(f"  [metrics] appended run row → {Path(runs_path).name}")
        except Exception as e:
            self.log.warning(f"  [metrics] could not append cross-run row: {e}")
