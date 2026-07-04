#!/usr/bin/env python3
"""
French Dubbing Pipeline v1.0 — System Verification
Run after setup.sh to confirm everything is ready before processing videos.
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import Tuple

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
BLUE   = "\033[94m"
RESET  = "\033[0m"


class Checker:
    def __init__(self):
        self.passed   = []
        self.failed   = []
        self.warnings = []

    def check(self, name: str, ok: bool, detail: str = "") -> bool:
        sym = f"{GREEN}✓{RESET}" if ok else f"{RED}✗{RESET}"
        print(f"{sym} {name}")
        if detail:
            print(f"  └─ {detail}")
        (self.passed if ok else self.failed).append((name, detail))
        return ok

    def warn(self, name: str, detail: str = ""):
        print(f"{YELLOW}⚠{RESET} {name}")
        if detail:
            print(f"  └─ {detail}")
        self.warnings.append((name, detail))

    def summary(self) -> bool:
        total = len(self.passed) + len(self.failed)
        pct   = 100 * len(self.passed) / max(1, total)
        print(f"\n{'=' * 60}")
        print("VERIFICATION SUMMARY")
        print(f"{'=' * 60}")
        print(f"{GREEN}Passed:{RESET}   {len(self.passed)}/{total}")
        print(f"{RED}Failed:{RESET}   {len(self.failed)}/{total}")
        print(f"{YELLOW}Warnings:{RESET} {len(self.warnings)}")
        print(f"Score: {pct:.0f}%")
        if self.failed:
            print(f"\n{RED}Failed checks:{RESET}")
            for name, msg in self.failed:
                print(f"  ✗ {name}" + (f": {msg}" if msg else ""))
        if self.warnings:
            print(f"\n{YELLOW}Warnings:{RESET}")
            for name, msg in self.warnings:
                print(f"  ⚠ {name}" + (f": {msg}" if msg else ""))
        print(f"\n{'=' * 60}")
        if not self.failed:
            print(f"{GREEN}✓ GO — system is ready!{RESET}")
        else:
            print(f"{RED}✗ NO-GO — fix issues above then re-run setup.sh{RESET}")
        return len(self.failed) == 0


def run(cmd: str, timeout: int = 15) -> Tuple[bool, str]:
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return r.returncode == 0, (r.stdout + r.stderr).strip()
    except subprocess.TimeoutExpired:
        return False, "timed out"
    except Exception as e:
        return False, str(e)


def pkg(import_name: str) -> Tuple[bool, str]:
    try:
        mod = __import__(import_name)
        ver = getattr(mod, "__version__", "unknown")
        return True, ver
    except ImportError as e:
        return False, str(e)


def main():
    print(f"\n{BLUE}{'=' * 60}")
    print("French Dubbing Pipeline v1.0 — System Verification")
    print(f"{'=' * 60}{RESET}\n")

    c = Checker()

    # ── GPU ───────────────────────────────────────────────────────────────────
    print(f"{BLUE}GPU{RESET}")
    ok, out = run("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader")
    if ok and out:
        line       = out.splitlines()[0]
        name, mem  = (line.split(",") + ["?"])[:2]
        c.check("NVIDIA GPU", True, f"{name.strip()} ({mem.strip()})")
        if not any(x in name for x in ("4090", "A5000", "A100", "A6000", "H100")):
            c.warn("GPU model", f"{name.strip()} — verify 24 GB+ VRAM")
    else:
        c.check("NVIDIA GPU", False, "nvidia-smi not found")

    # ── Python & PyTorch ──────────────────────────────────────────────────────
    print(f"\n{BLUE}Python / PyTorch{RESET}")
    ok, out = run("python3 --version")
    ver = out.split()[-1] if ok else "?"
    c.check("Python 3", ok, ver)
    if ok:
        major, minor = (ver.split(".")[:2] + ["0", "0"])[:2]
        if int(major) == 3 and int(minor) >= 12:
            pass  # F5-TTS has no Python version constraint

    try:
        import torch
        c.check("PyTorch import", True, torch.__version__)
        c.check("PyTorch 2.8.x", torch.__version__.startswith("2.8"), torch.__version__)
        c.check("CUDA available", torch.cuda.is_available())
        if torch.cuda.is_available():
            mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            c.check("GPU VRAM ≥ 24 GB", mem_gb >= 20, f"{mem_gb:.1f} GB")
            try:
                x = torch.randn(8, 8).cuda()
                _ = x @ x
                c.check("GPU compute", True, "tensor ops OK")
            except Exception as e:
                c.check("GPU compute", False, str(e))
        ok2, cuda_ver = run('python3 -c "import torch; print(torch.version.cuda)"')
        c.check("CUDA 12.8", ok2 and "12.8" in (cuda_ver or ""), cuda_ver or "unknown")
    except ImportError:
        c.check("PyTorch import", False, "torch not installed")

    # ── HuggingFace token (required for the gated pyannote model) ─────────────
    print(f"\n{BLUE}HuggingFace Access{RESET}")
    hf_token = os.environ.get("HF_TOKEN", "").strip()
    if not hf_token:
        # Check if saved in workspace .env
        env_file = Path("/workspace/.env")
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                if line.startswith("HF_TOKEN="):
                    hf_token = line.split("=", 1)[1].strip().strip('"\'')
    if hf_token:
        c.check("HF_TOKEN set", True, f"{hf_token[:8]}…  (from env/workspace)")
        # Verify it actually authenticates
        ok2, out2 = run(
            f'python3 -c "from huggingface_hub import HfApi; '
            f'HfApi().whoami(token=\'{hf_token}\')"',
            timeout=15,
        )
        c.check("HuggingFace auth valid", ok2, out2[:80] if not ok2 else "token accepted")
    else:
        c.check(
            "HF_TOKEN set", False,
            "Required for the gated pyannote diarization model.\n"
            "  1. Accept the license: https://huggingface.co/pyannote/speaker-diarization-community-1\n"
            "  2. Get token:          https://huggingface.co/settings/tokens\n"
            "  3. Re-run setup.sh (it will prompt you) or: export HF_TOKEN=hf_xxx"
        )

    # ── Core Python packages ──────────────────────────────────────────────────
    print(f"\n{BLUE}Core packages{RESET}")

    required = [
        ("faster_whisper", "faster-whisper (transcription)"),
        ("librosa",        "librosa"),
        ("soundfile",      "soundfile"),
        ("pydub",          "pydub"),
        ("pysrt",          "pysrt"),
        ("yaml",           "pyyaml"),
        ("click",          "click"),
        ("tqdm",           "tqdm"),
        ("requests",       "requests"),
        ("packaging",      "packaging"),
    ]
    for import_name, label in required:
        ok2, ver = pkg(import_name)
        c.check(label, ok2, f"v{ver}" if ok2 else ver)

    ok2, ver = pkg("numpy")
    if ok2:
        try:
            from packaging.version import Version
            v = Version(ver)
            if Version("1.26.4") <= v < Version("2.0.0"):
                c.check("numpy <2.0.0", True, ver)
            else:
                c.warn("numpy <2.0.0",
                       f"{ver} — run: pip install --force-reinstall 'numpy>=1.26.4,<2.0.0'")
        except Exception:
            c.warn("numpy version", f"could not parse {ver}")

    # ── Source separation ─────────────────────────────────────────────────────
    print(f"\n{BLUE}Source separation{RESET}")
    ok2, ver = pkg("demucs")
    c.check("demucs", ok2, f"v{ver}" if ok2 else "pip install demucs")

    # ── Speaker denoising ─────────────────────────────────────────────────────
    print(f"\n{BLUE}Speaker denoising{RESET}")
    ok_nr, ver_nr = pkg("noisereduce")
    if ok_nr:
        c.check("noisereduce", True, f"v{ver_nr}")
    else:
        c.warn(
            "noisereduce", "not installed — will fall back to FFmpeg anlmdn. "
            "Run: pip install noisereduce"
        )

    # ── transformers ──────────────────────────────────────────────────────────
    print(f"\n{BLUE}transformers{RESET}")
    hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
    ok2, ver = pkg("transformers")
    c.check("transformers", ok2, f"v{ver}" if ok2 else ver)

    # ── TTS — F5-TTS ──────────────────────────────────────────────────────────
    print(f"\n{BLUE}TTS — F5-TTS (flow-matching voice cloning, French support){RESET}")
    ok2, ver = pkg("f5_tts")
    if ok2:
        c.check("f5-tts", True, f"v{ver}")
        f5_cached = any(
            "f5tts" in str(p).lower() or "f5-tts" in str(p).lower()
            for p in hf_cache.rglob("*.safetensors") if hf_cache.exists()
        )
        if f5_cached:
            c.check("F5-TTS weights cached", True)
        else:
            c.warn("F5-TTS weights",
                   "not cached — will download (~1.5 GB) on first run")
    else:
        c.warn("f5-tts",
               "not installed — TTS engine unavailable. "
               "Run: pip install f5-tts")

    # ── System tools ──────────────────────────────────────────────────────────
    print(f"\n{BLUE}System tools{RESET}")
    for tool in ("ffmpeg", "ffprobe", "sox", "curl"):
        ok2, _ = run(f"command -v {tool}")
        c.check(f"tool: {tool}", ok2)

    # ── Ollama (primary translator) ───────────────────────────────────────────
    # Check for the model configured in config.yaml, not a hardcoded tag.
    ollama_model = "mistral-small:22b"
    try:
        import yaml as _yaml
        with open("/workspace/config.yaml") as _f:
            _cfg = _yaml.safe_load(_f) or {}
        ollama_model = (_cfg.get("translation", {}) or {}).get("model", ollama_model)
    except Exception:
        pass
    print(f"\n{BLUE}Ollama — translation ({ollama_model}){RESET}")
    ok2, _ = run("command -v ollama")
    if c.check("Ollama installed", ok2):
        ok3, out = run("curl -s http://localhost:11434/api/tags", timeout=5)
        if c.check("Ollama service running", ok3 and "models" in out.lower()):
            ok4, out = run("ollama list 2>/dev/null")
            model_base = ollama_model.split(":")[0]
            has_model = ok4 and model_base in out
            c.check(f"{ollama_model} model present", has_model,
                    f"run: ollama pull {ollama_model}" if not has_model else "")
        else:
            c.warn("Ollama not running",
                   "start: nohup ollama serve > /workspace/logs/ollama.log 2>&1 &")
    else:
        c.warn("Ollama", "install: curl -fsSL https://ollama.ai/install.sh | sh")

    # ── Cached models ─────────────────────────────────────────────────────────
    print(f"\n{BLUE}Cached models{RESET}")
    whisper_cache = Path("/workspace/models/whisper")
    if whisper_cache.exists():
        models = list(whisper_cache.rglob("*.bin"))
        c.check("Whisper large-v3 cached", bool(models),
                f"{len(models)} file(s)" if models else "run setup.sh to download")
    else:
        c.warn("Whisper cache", "not found — will download on first run")

    # ── Workspace ─────────────────────────────────────────────────────────────
    print(f"\n{BLUE}Workspace{RESET}")
    for d in ("/workspace/videos/input", "/workspace/outputs",
              "/workspace/scripts", "/workspace/logs", "/workspace/models", "/workspace/temp"):
        p = Path(d)
        writable = False
        if p.is_dir():
            try:
                t = p / ".writable_test"
                t.touch()
                t.unlink()
                writable = True
            except OSError:
                pass
        c.check(f"dir: {p.name}", writable, str(p))

    for script in ("/workspace/scripts/02_pipeline.py",
                   "/workspace/scripts/03_batch_runner.py"):
        c.check(f"script: {Path(script).name}", Path(script).is_file())

    cfg = Path("/workspace/config.yaml")
    if c.check("config.yaml", cfg.is_file()):
        try:
            import yaml as _yaml
            with open(cfg) as f:
                data = _yaml.safe_load(f)
            c.check("config.yaml valid", bool(data and "pipeline" in data))
        except Exception as e:
            c.check("config.yaml valid", False, str(e))

    # ── Final ─────────────────────────────────────────────────────────────────
    print()
    ready = c.summary()
    if ready:
        print(f"\n{GREEN}Ready to process videos!{RESET}")
        print("\nNext steps:")
        print("  cp webinar.mp4 /workspace/videos/input/")
        print("  python /workspace/scripts/02_pipeline.py \\")
        print("    --video /workspace/videos/input/webinar.mp4")
    else:
        print(f"\n{RED}Fix the issues above, then re-run: bash 04_setup.sh{RESET}")
    sys.exit(0 if ready else 1)


if __name__ == "__main__":
    main()
