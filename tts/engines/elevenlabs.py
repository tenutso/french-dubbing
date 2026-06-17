"""ElevenLabs engine adapter (REST API wrapper).

Supports zero-shot Instant Voice Cloning (IVC) or pre-configured voices.
Automatically cleans up dynamically cloned voices from your ElevenLabs account on exit.

engine_params (all optional):
    api_key             str   API key or environment variable name (default: ELEVENLABS_API_KEY)
    voice_id            str   Specific voice ID to use. If not specified, clones the reference WAV on the fly.
    model_id            str   "eleven_multilingual_v2" (default) or other multilingual model.
    stability           float 0.5    stability slider
    similarity_boost    float 0.75   similarity slider
    style               float 0.0    style exaggeration slider
    use_speaker_boost   bool  True   use speaker boost toggle
"""

import os
import requests
import numpy as np

class Adapter:
    sample_rate = 24000

    def __init__(self, params: dict, device: str) -> None:
        # Resolve API key
        api_key = params.get("api_key")
        if api_key:
            if api_key in os.environ:
                api_key = os.environ[api_key]
            elif api_key.startswith("$") and api_key[1:] in os.environ:
                api_key = os.environ[api_key[1:]]
        else:
            api_key = os.environ.get("ELEVEN_API_KEY") or os.environ.get("ELEVENLABS_API_KEY")

        if not api_key:
            raise ValueError(
                "ElevenLabs API key not found. Please set the ELEVENLABS_API_KEY environment variable "
                "or specify 'api_key' in tts.engine_params."
            )
        self.api_key = api_key
        self.voice_id = params.get("voice_id")
        self.model_id = params.get("model_id", "eleven_multilingual_v2")
        self.stability = float(params.get("stability", 0.5))
        self.similarity_boost = float(params.get("similarity_boost", 0.75))
        self.style = float(params.get("style", 0.0))
        self.use_speaker_boost = bool(params.get("use_speaker_boost", True))

        self._voice_cache = {}
        self._cloned_voices = []

    def _clone_voice(self, ref_wav: str) -> str:
        url = "https://api.elevenlabs.io/v1/voices/add"
        headers = {"xi-api-key": self.api_key}
        data = {
            "name": f"Dubbing_{os.path.basename(ref_wav)[:20]}",
            "description": f"Cloned speaker profile from {ref_wav}"
        }
        with open(ref_wav, "rb") as f:
            files = [("files", (os.path.basename(ref_wav), f, "audio/wav"))]
            r = requests.post(url, headers=headers, data=data, files=files)
        
        if r.status_code != 200:
            raise Exception(f"ElevenLabs voice cloning failed: {r.status_code} - {r.text}")
        
        voice_id = r.json()["voice_id"]
        self._cloned_voices.append(voice_id)
        return voice_id

    def synth(self, text: str, ref_wav: str, lang: str) -> np.ndarray:
        voice_id = self.voice_id
        if not voice_id:
            if ref_wav not in self._voice_cache:
                voice_id = self._clone_voice(ref_wav)
                self._voice_cache[ref_wav] = voice_id
            else:
                voice_id = self._voice_cache[ref_wav]

        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}?output_format=pcm_24000"
        headers = {
            "xi-api-key": self.api_key,
            "Content-Type": "application/json"
        }
        body = {
            "text": text,
            "model_id": self.model_id,
            "voice_settings": {
                "stability": self.stability,
                "similarity_boost": self.similarity_boost,
                "style": self.style,
                "use_speaker_boost": self.use_speaker_boost
            }
        }
        r = requests.post(url, headers=headers, json=body)
        if r.status_code != 200:
            raise Exception(f"ElevenLabs synthesis failed: {r.status_code} - {r.text}")

        # Convert raw 16-bit PCM bytes to float32 numpy array
        wav = np.frombuffer(r.content, dtype=np.int16).astype(np.float32) / 32768.0
        return wav

    def __del__(self):
        if hasattr(self, "_cloned_voices") and self._cloned_voices:
            for voice_id in self._cloned_voices:
                try:
                    requests.delete(f"https://api.elevenlabs.io/v1/voices/{voice_id}", headers={"xi-api-key": self.api_key})
                except Exception:
                    pass
