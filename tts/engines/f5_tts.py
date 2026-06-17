"""F5-TTS engine adapter (self-hosted Flow-Matching TTS, SWivid/F5-TTS).

Zero-shot voice cloning with high stability and no repetition loops. Runs in its
own venv (/workspace/venvs/f5_tts) because it pins its own torch/transformers/
vocos stack.

The stock ``F5TTS_v1_Base`` checkpoint is trained on English + Chinese only, so
French (or any other language) text comes out mispronounced. To dub another
language, point the adapter at a fine-tuned checkpoint — either pulled from a
HuggingFace repo (``hf_repo``) or loaded from a local path (``ckpt_file`` /
``vocab_file``). A fine-tune usually ships its own vocab and may use an older
architecture, so set ``model`` to the architecture the checkpoint was trained
with (NOT necessarily the default v1).

engine_params (all optional):
    model            str   Architecture config name F5-TTS loads from its bundled
                           ``configs/<model>.yaml`` (e.g. "F5TTS_v1_Base",
                           "F5TTS_Base"). Must match the checkpoint's architecture.
                           Default: "F5TTS_v1_Base".
    hf_repo          str   HuggingFace repo id of a fine-tuned checkpoint. When set,
                           the adapter downloads the checkpoint (+ vocab) from it and
                           ignores the SWivid default weights. Default: "" (off).
    hf_ckpt_file     str   Checkpoint filename within hf_repo.
                           Default: "model_last_reduced.pt".
    hf_vocab_file    str   Vocab filename within hf_repo. Default: "vocab.txt".
                           Set to "" to keep the architecture's default vocab.
    ckpt_file        str   Local checkpoint path. Takes precedence over hf_repo.
                           Default: "" → use hf_repo, else the SWivid base weights.
    vocab_file       str   Local vocab path. Default: "" → architecture default.
    use_ema          bool  Load EMA weights from the checkpoint (default: True).
    ref_text         str   Transcription of the reference clip. Leave "" (default)
                           to let F5-TTS transcribe the reference automatically.
    remove_silence   bool  Trim leading/trailing silence from output (default: False).
    seed             int   Fixed sampling seed (default: -1 → random per call).

Example — European French via the RASPIAUDIO fine-tune:
    engine_params:
      model: F5TTS_Base        # the fine-tune's architecture (v0, not v1)
      hf_repo: RASPIAUDIO/F5-French-MixedSpeakers-reduced
      hf_ckpt_file: model_last_reduced.pt
      hf_vocab_file: vocab.txt
"""

import numpy as np


def _resolve_checkpoint(params: dict) -> tuple[str, str, str]:
    """Return (model_arch, ckpt_file, vocab_file) for F5TTS(...).

    An explicit local ``ckpt_file`` wins; otherwise ``hf_repo`` (if set) is
    downloaded from HuggingFace. Empty strings tell F5-TTS to fall back to the
    SWivid base weights / the architecture's default vocab.
    """
    model = params.get("model", "F5TTS_v1_Base")
    ckpt_file = params.get("ckpt_file", "") or ""
    vocab_file = params.get("vocab_file", "") or ""

    hf_repo = params.get("hf_repo", "") or ""
    if hf_repo and not ckpt_file:
        from huggingface_hub import hf_hub_download

        ckpt_name = params.get("hf_ckpt_file", "model_last_reduced.pt")
        ckpt_file = hf_hub_download(repo_id=hf_repo, filename=ckpt_name)
        vocab_name = params.get("hf_vocab_file", "vocab.txt")
        if vocab_name and not vocab_file:
            vocab_file = hf_hub_download(repo_id=hf_repo, filename=vocab_name)

    return model, ckpt_file, vocab_file


class Adapter:
    sample_rate = 24000  # F5-TTS vocoder (vocos) outputs 24 kHz; refined from model below.

    def __init__(self, params: dict, device: str) -> None:
        # The public API class. ``f5_tts.api`` is the supported entrypoint; the
        # bare ``f5_tts`` package has no top-level F5TTS export.
        from f5_tts.api import F5TTS

        model, ckpt_file, vocab_file = _resolve_checkpoint(params)
        self._tts = F5TTS(
            model=model,
            ckpt_file=ckpt_file,
            vocab_file=vocab_file,
            use_ema=bool(params.get("use_ema", True)),
            device=device,
        )
        self.sample_rate = int(getattr(self._tts, "target_sample_rate", 24000))

        self.ref_text = params.get("ref_text", "")
        self.remove_silence = bool(params.get("remove_silence", False))
        self.seed = int(params.get("seed", -1))

    def synth(self, text: str, ref_wav: str, lang: str) -> np.ndarray:
        # F5-TTS is language-agnostic at the API level, but the *checkpoint* must
        # cover the target language's phonetics (see module docstring — use a
        # fine-tune for non-English/Chinese). ``infer`` chunks long text
        # internally, so no manual splitting is needed here.
        kwargs = dict(
            ref_file=ref_wav,
            ref_text=self.ref_text,
            gen_text=text,
            remove_silence=self.remove_silence,
        )
        # Only pin a seed when the user asked for one; the -1 vs None sentinel for
        # "random" differs across f5-tts versions, so let the library default it.
        if self.seed >= 0:
            kwargs["seed"] = self.seed
        wav, sr, _spec = self._tts.infer(**kwargs)
        wav = np.asarray(wav, dtype=np.float32).reshape(-1)
        return wav

    @staticmethod
    def prefetch(params: dict) -> None:
        # Instantiating the model pulls its weights into cache: the fine-tune from
        # hf_repo (if set) plus the vocoder. Same resolution path as __init__, so
        # setup caches exactly what the pipeline will load.
        from f5_tts.api import F5TTS

        model, ckpt_file, vocab_file = _resolve_checkpoint(params)
        F5TTS(
            model=model,
            ckpt_file=ckpt_file,
            vocab_file=vocab_file,
            use_ema=bool(params.get("use_ema", True)),
            device="cpu",
        )
