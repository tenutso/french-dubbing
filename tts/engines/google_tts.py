"""Google Cloud Text-to-Speech engine adapter (Neural2 / WaveNet / Studio voices).

Cloud API, not a local model: it does NOT clone the reference clip — it speaks
with one of Google's pre-built voices, selected by name (e.g. "fr-FR-Neural2-A").
``ref_wav`` is therefore ignored. Pick a voice per the language/locale you're
dubbing into; the voice name encodes both the BCP-47 language and the tier:

    fr-FR-Neural2-A   European French, Neural2 (recommended — best quality/price)
    fr-FR-Wavenet-C   European French, WaveNet
    fr-CA-Neural2-A   Canadian French, Neural2   (matches locale: fr-ca)
    fr-FR-Studio-D    European French, Studio    (highest quality, pricier)

Browse voices: https://cloud.google.com/text-to-speech/docs/voices

Authentication (either one):
    • api_key            — simplest. An API key string, or the NAME of an env var
                           holding it (default tries GOOGLE_TTS_API_KEY / GOOGLE_API_KEY).
                           Uses the REST endpoint; no extra libraries needed.
    • credentials_path   — path to a service-account JSON (or env var name holding
                           it). Sets GOOGLE_APPLICATION_CREDENTIALS and uses the
                           official google-cloud-texttospeech client. Falls back to
                           ambient Application Default Credentials if neither is set.

engine_params (all optional unless noted):
    voice_name       str    Google voice, e.g. "fr-FR-Neural2-A" (STRONGLY recommended).
    language_code    str    BCP-47 code, e.g. "fr-FR". Derived from voice_name if unset.
    ssml_gender      str    "MALE" | "FEMALE" | "NEUTRAL" — only used when no voice_name.
    speaking_rate    float  0.25–4.0, 1.0 = normal (default 1.0).
    pitch            float  -20.0–20.0 semitones, 0.0 = normal (default 0.0).
    volume_gain_db   float  -96.0–16.0 dB (default 0.0).
    sample_rate      int    Output rate in Hz (default 24000).
    api_key          str    See Authentication above.
    credentials_path str    See Authentication above.

Example (config.yaml):
    tts:
      engine: google_tts
      engine_params:
        voice_name: fr-CA-Neural2-A
        api_key: GOOGLE_TTS_API_KEY     # env var name holding the key
        speaking_rate: 1.0
"""

import io
import os

import numpy as np
import soundfile as sf

# Sensible BCP-47 default region per 2-letter language, for when neither
# voice_name nor language_code is provided. Google needs a region (e.g. "fr-FR"),
# not a bare "fr". Setting voice_name explicitly is always preferred.
_DEFAULT_REGION = {
    "fr": "fr-FR", "es": "es-ES", "de": "de-DE", "it": "it-IT", "pt": "pt-PT",
    "nl": "nl-NL", "pl": "pl-PL", "ru": "ru-RU", "ja": "ja-JP", "ko": "ko-KR",
    "zh": "cmn-CN", "ar": "ar-XA", "tr": "tr-TR", "hi": "hi-IN", "vi": "vi-VN",
    "en": "en-US",
}


def _resolve_secret(value: str | None, *env_fallbacks: str) -> str:
    """Treat ``value`` as a literal, an env var name, or "$VAR"; else try fallbacks."""
    if value:
        if value in os.environ:
            return os.environ[value]
        if value.startswith("$") and value[1:] in os.environ:
            return os.environ[value[1:]]
        return value
    for name in env_fallbacks:
        if os.environ.get(name):
            return os.environ[name]
    return ""


def _language_code(voice_name: str, language_code: str, lang: str) -> str:
    """Full BCP-47 code: explicit > derived from voice_name > default-region map."""
    if language_code:
        return language_code
    if voice_name:
        # "fr-FR-Neural2-A" → "fr-FR"; "cmn-CN-Wavenet-A" → "cmn-CN".
        parts = voice_name.split("-")
        if len(parts) >= 2:
            return "-".join(parts[:2])
    return _DEFAULT_REGION.get(lang, lang)


def _decode_audio(wav_bytes: bytes) -> np.ndarray:
    """Google LINEAR16 audio_content is a WAV container — decode to 1-D float32."""
    data, _sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    return np.asarray(data, dtype=np.float32).reshape(-1)


class Adapter:
    sample_rate = 24000

    def __init__(self, params: dict, device: str) -> None:
        self.voice_name = params.get("voice_name", "") or ""
        self.language_code = params.get("language_code", "") or ""
        self.ssml_gender = (params.get("ssml_gender", "") or "").upper()
        self.speaking_rate = float(params.get("speaking_rate", 1.0))
        self.pitch = float(params.get("pitch", 0.0))
        self.volume_gain_db = float(params.get("volume_gain_db", 0.0))
        self.sample_rate = int(params.get("sample_rate", 24000))

        # Auth: prefer an API key (lightweight REST); else the client library
        # (service account / ADC). Resolved once here so failures surface at load.
        self.api_key = _resolve_secret(
            params.get("api_key"), "GOOGLE_TTS_API_KEY", "GOOGLE_API_KEY"
        )
        self._client = None
        if not self.api_key:
            cred = _resolve_secret(params.get("credentials_path"))
            if cred:
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = cred
            try:
                from google.cloud import texttospeech
            except ImportError as e:
                raise ImportError(
                    "google-cloud-texttospeech is required when no api_key is set. "
                    "Install it in the engine venv (requirements-google_tts.txt)."
                ) from e
            # Raises clearly if neither GOOGLE_APPLICATION_CREDENTIALS nor ADC exist.
            self._client = texttospeech.TextToSpeechClient()
            self._texttospeech = texttospeech

    def _build_voice_kwargs(self, lang: str) -> dict:
        lc = _language_code(self.voice_name, self.language_code, lang)
        voice = {"language_code": lc}
        if self.voice_name:
            voice["name"] = self.voice_name
        elif self.ssml_gender:
            voice["ssml_gender"] = self.ssml_gender
        return voice

    def synth(self, text: str, ref_wav: str, lang: str) -> np.ndarray:
        # ref_wav is intentionally unused: Google TTS speaks a fixed voice.
        voice = self._build_voice_kwargs(lang)

        if self.api_key:
            wav_bytes = self._synth_rest(text, voice)
        else:
            wav_bytes = self._synth_client(text, voice)
        return _decode_audio(wav_bytes)

    def _synth_rest(self, text: str, voice: dict) -> bytes:
        import base64

        import requests

        body = {
            "input": {"text": text},
            "voice": {"languageCode": voice["language_code"]},
            "audioConfig": {
                "audioEncoding": "LINEAR16",
                "sampleRateHertz": self.sample_rate,
                "speakingRate": self.speaking_rate,
                "pitch": self.pitch,
                "volumeGainDb": self.volume_gain_db,
            },
        }
        if "name" in voice:
            body["voice"]["name"] = voice["name"]
        if "ssml_gender" in voice:
            body["voice"]["ssmlGender"] = voice["ssml_gender"]

        r = requests.post(
            "https://texttospeech.googleapis.com/v1/text:synthesize",
            params={"key": self.api_key},
            json=body,
            timeout=60,
        )
        if r.status_code != 200:
            raise Exception(f"Google TTS synthesis failed: {r.status_code} - {r.text}")
        return base64.b64decode(r.json()["audioContent"])

    def _synth_client(self, text: str, voice: dict) -> bytes:
        tts = self._texttospeech
        resp = self._client.synthesize_speech(
            input=tts.SynthesisInput(text=text),
            voice=tts.VoiceSelectionParams(**voice),
            audio_config=tts.AudioConfig(
                audio_encoding=tts.AudioEncoding.LINEAR16,
                sample_rate_hertz=self.sample_rate,
                speaking_rate=self.speaking_rate,
                pitch=self.pitch,
                volume_gain_db=self.volume_gain_db,
            ),
        )
        return resp.audio_content

    @staticmethod
    def prefetch(params: dict) -> None:
        # Cloud API — no weights to cache. Surface an auth hint at setup time
        # without making a (billable) synthesis call.
        has_key = bool(
            _resolve_secret(params.get("api_key"), "GOOGLE_TTS_API_KEY", "GOOGLE_API_KEY")
        )
        has_cred = bool(
            _resolve_secret(params.get("credentials_path"))
            or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        )
        if has_key or has_cred:
            print("google_tts: cloud API — credentials detected, nothing to download")
        else:
            print(
                "google_tts: cloud API — no weights to cache. WARNING: no api_key / "
                "credentials_path / GOOGLE_APPLICATION_CREDENTIALS found; set one "
                "before running the pipeline."
            )
