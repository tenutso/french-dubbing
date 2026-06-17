#apt-get update && apt-get install -y ffmpeg libsndfile1-dev
#pip install git+https://github.com/m-bain/whisperX.git
#pip install voxcpm soundfile transformers accelerator sacremoses sentencepiece pydub

import os
import gc
import subprocess
import torch
import whisperx
import soundfile as sf
from transformers import pipeline
from voxcpm import VoxCPM
from whisperx.diarize import DiarizationPipeline
from pydub import AudioSegment

# --- CONFIGURATION ---
INPUT_VIDEO = "webinar.mp4"       # Your original Zoom recording
AUDIO_PATH = "temp_audio.wav"     # Temporary extracted audio
HF_TOKEN = "YOUR_HF_TOKEN"        # Hugging Face token for PyAnnote
OUTPUT_DIR = "./output_segments"

os.makedirs(OUTPUT_DIR, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
compute_type = "float16" if device == "cuda" else "float32"

# --- OQLF FRENCH DICTIONARY PATCH ---
OQLF_PATCHES = {
    "email": "courriel",
    "e-mail": "courriel",
    "webinar": "webinaire",
    "chat": "clavardage",
    "newsletter": "infolettre",
    "podcast": "balado",
    "feedback": "rétroaction",
    "sponsor": "commanditaire",
    "smartphone": "téléphone intelligent",
    "baskets": "escarpins",
    "week-end": "fin de semaine"
}

def apply_oqlf_rules(text: str) -> str:
    cleaned = text.lower()
    for eng_fr_term, oqlf_term in OQLF_PATCHES.items():
        cleaned = cleaned.replace(eng_fr_term, oqlf_term)
    return cleaned.capitalize()

# --- STEP 0: EXTRACT AUDIO FROM MP4 ---
print("[0/5] Extracting audio from video file...")
try:
    subprocess.run([
        "ffmpeg", "-i", INPUT_VIDEO, 
        "-vn", "-acodec", "pcm_s16le", 
        "-ar", "16000", "-ac", "1", 
        AUDIO_PATH, "-y"
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
except subprocess.CalledProcessError as e:
    print(f"Error extracting audio: Ensure FFmpeg is installed. {e}")
    exit(1)

# --- STEP 1: TRANSCRIPTION & DIARIZATION (WHISPERX) ---
print("[1/5] Transcribing and Diarizing Audio...")
model = whisperx.load_model("large-v3", device, compute_type=compute_type)
audio = whisperx.load_audio(AUDIO_PATH)
result = model.transcribe(audio, batch_size=16)

# Align Whisper outputs to audio sample level
model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
result = whisperx.align(result["segments"], model_a, metadata, audio, device)

# Diarize (Assign speaker IDs)
diarize_model = DiarizationPipeline(token=HF_TOKEN, device=device)
diarize_segments = diarize_model(audio)
result = whisperx.assign_word_speakers(diarize_segments, result)

del model, model_a, diarize_model
torch.cuda.empty_cache()
gc.collect()

# --- STEP 2: TRANSLATION (EN -> FR) ---
print("[2/5] Translating Text into OQLF French...")
translator = pipeline("translation_en_to_fr", model="Helsinki-NLP/opus-mt-en-fr", device=0 if device=="cuda" else -1)

processed_segments = []
for seg in result["segments"]:
    if "speaker" not in seg:
        seg["speaker"] = "UNKNOWN"
        
    original_text = seg["text"]
    translation = translator(original_text)[0]['translation_text']
    oqlf_text = apply_oqlf_rules(translation)
    
    processed_segments.append({
        "speaker": seg["speaker"],
        "start": seg["start"],
        "end": seg["end"],
        "text": oqlf_text
    })

del translator
torch.cuda.empty_cache()
gc.collect()

# --- STEP 3: VOICE CLONING & TTS GENERATION (VOXCPM2) ---
print("[3/5] Initializing VoxCPM2 Voice Cloner...")
tts_model = VoxCPM.from_pretrained("openbmb/VoxCPM2", load_denoiser=False)

unique_speakers = set(seg["speaker"] for seg in processed_segments)
speaker_refs = {}

print("Extracting reference samples for each unique speaker profile...")
raw_audio, sr = sf.read(AUDIO_PATH)

for speaker in unique_speakers:
    spk_segs = [s for s in result["segments"] if s.get("speaker") == speaker]
    if not spk_segs:
        continue
    best_seg = max(spk_segs, key=lambda x: x["end"] - x["start"])
    
    start_sample = int(best_seg["start"] * sr)
    end_sample = int(best_seg["end"] * sr)
    ref_clip = raw_audio[start_sample:end_sample]
    
    ref_path = f"{OUTPUT_DIR}/{speaker}_ref.wav"
    sf.write(ref_path, ref_clip, sr)
    speaker_refs[speaker] = ref_path

print("Generating cloned French audio clips...")
for idx, seg in enumerate(processed_segments):
    spk = seg["speaker"]
    ref_audio_path = speaker_refs.get(spk)
    text_to_speak = seg["text"]
    
    try:
        wav = tts_model.generate(
            text=text_to_speak, 
            reference_wav_path=ref_audio_path,
            cfg_value=2.0, 
            inference_timesteps=15
        )
        
        out_filename = f"{OUTPUT_DIR}/seg_{idx:04d}_{spk}.wav"
        sf.write(out_filename, wav, tts_model.tts_model.sample_rate)
        print(f"Generated cloned segment {idx} for speaker {spk}")
    except Exception as e:
        print(f"Failed to generate segment {idx} due to error: {e}")

# --- STEP 4 & 5: ASSEMBLE ALIGNED AUDIO & SYNCED SRT ---
print("[4/5 & 5/5] Assembling timeline, smoothing audio, and syncing subtitles...")

original_audio = AudioSegment.from_wav(AUDIO_PATH)
total_duration_ms = len(original_audio)
aligned_french_track = AudioSegment.silent(duration=total_duration_ms)

def format_timestamp(milliseconds: int) -> str:
    seconds = milliseconds / 1000.0
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds - int(seconds)) * 1000))
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

last_end_time_ms = 0

with open("webinar_french.srt", "w", encoding="utf-8") as srt_file:
    for idx, seg in enumerate(processed_segments, start=1):
        spk = seg["speaker"]
        out_filename = f"{OUTPUT_DIR}/seg_{idx:04d}_{spk}.wav"
        
        if os.path.exists(out_filename):
            # Load clip and apply a 10ms crossfade smooth to prevent audio clicks
            cloned_clip = AudioSegment.from_wav(out_filename).fade_in(10).fade_out(10)
            
            start_ms = int(seg["start"] * 1000)
            
            # Apply overlap safeguard if French translation ran long
            if start_ms < last_end_time_ms:
                start_ms = last_end_time_ms
                
            clip_duration = len(cloned_clip)
            end_ms = start_ms + clip_duration
            
            # Stamp audio onto the canvas
            aligned_french_track = aligned_french_track.overlay(cloned_clip, position=start_ms)
            
            # Write synced SRT entry
            start_str = format_timestamp(start_ms)
            end_str = format_timestamp(end_ms)
            text = f"[{spk}] {seg['text']}"
            
            srt_file.write(f"{idx}\n{start_str} --> {end_str}\n{text}\n\n")
            
            last_end_time_ms = end_ms

final_export_name = "translated_french_webinar.wav"
aligned_french_track.export(final_export_name, format="wav")

print("\n=======================================================")
print("PIPELINE COMPLETE!")
print(f"Subtitles: webinar_french.srt")
print(f"Audio Track: {final_export_name}")
print("=======================================================")