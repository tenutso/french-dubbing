#!/usr/bin/env python3
"""
RunPod Serverless handler for French dubbing pipeline.

Deploy as a serverless endpoint with:
- GPU: RTX 4090 (24 GB)
- Min workers: 0
- Network volume: /workspace with pre-cached models
- Environment: S3_BUCKET, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION, HF_TOKEN

Usage:
  Call the endpoint with:
  {
    "video_url": "https://s3.../input.mp4?sig=...",
    "upload_prefix": "s3://bucket/output/",
    "locale": "fr-ca"
  }

  Returns: { "m4a_url": "...", "srt_url": "..." }
"""

import os
import sys
import time
import tempfile
import subprocess
import logging
import requests
from pathlib import Path
from urllib.parse import urlparse

try:
    import runpod
except ImportError:
    print("ERROR: runpod SDK not installed. pip install runpod")
    sys.exit(1)

try:
    import boto3
except ImportError:
    print("ERROR: boto3 not installed. pip install boto3")
    sys.exit(1)

# ============================================================================
# Logging
# ============================================================================

logger = logging.getLogger("runpod_handler")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
logger.addHandler(ch)


# ============================================================================
# Ollama Startup (runs once per worker instance)
# ============================================================================

def _start_ollama(max_retries: int = 60, timeout_s: float = 2.0) -> None:
    """Start ollama serve and wait until /api/tags responds."""
    logger.info("Starting Ollama service...")

    # Start Ollama in background
    proc = subprocess.Popen(
        ["ollama", "serve"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Poll for readiness
    for attempt in range(max_retries):
        try:
            resp = requests.get(
                "http://localhost:11434/api/tags",
                timeout=timeout_s,
            )
            if resp.status_code == 200:
                logger.info("✓ Ollama is ready")
                return
        except (requests.ConnectionError, requests.Timeout):
            pass

        if attempt % 10 == 0:
            logger.debug(f"Waiting for Ollama... ({attempt}/{max_retries})")
        time.sleep(1)

    logger.error("Ollama failed to start after 60s")
    raise RuntimeError("Ollama startup timeout")


# ============================================================================
# S3/Cloudflare R2 Upload/Download
# ============================================================================

def _get_s3_client():
    """Get a boto3 S3 client from environment variables."""
    return boto3.client(
        "s3",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=os.getenv("AWS_REGION", "us-east-1"),
    )


def _download_video(video_url: str, tmpdir: str) -> str:
    """Download video from presigned URL to tmpdir."""
    logger.info(f"Downloading video from {video_url[:80]}...")

    resp = requests.get(video_url, stream=True, timeout=120)
    resp.raise_for_status()

    # Infer filename from URL or use default
    parsed = urlparse(video_url)
    filename = Path(parsed.path).name or "input.mp4"
    local_path = Path(tmpdir) / filename

    with open(local_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

    logger.info(f"✓ Downloaded {local_path.name} ({local_path.stat().st_size / 1e6:.1f} MB)")
    return str(local_path)


def _upload_file(
    local_path: Path,
    upload_prefix: str,
    filename: str,
) -> str:
    """
    Upload file to S3/R2.

    upload_prefix can be:
    - "s3://bucket/path/" — inferred as S3 bucket URL
    - "https://..." — presigned PUT URL (direct POST)
    """
    logger.info(f"Uploading {filename}...")

    if upload_prefix.startswith("s3://"):
        # S3 URI — extract bucket + key
        parts = upload_prefix.replace("s3://", "").split("/", 1)
        bucket = parts[0]
        key = parts[1] + filename if len(parts) > 1 else filename

        s3 = _get_s3_client()
        s3.upload_file(str(local_path), bucket, key)

        url = f"s3://{bucket}/{key}"
        logger.info(f"✓ Uploaded to {url}")
        return url

    elif upload_prefix.startswith("http"):
        # Presigned PUT URL — POST directly
        with open(local_path, "rb") as f:
            resp = requests.put(upload_prefix, data=f, timeout=120)
        resp.raise_for_status()
        logger.info(f"✓ Uploaded via presigned URL")
        return upload_prefix

    else:
        raise ValueError(f"Invalid upload_prefix: {upload_prefix}")


# ============================================================================
# Pipeline Invocation
# ============================================================================

def _run_pipeline(
    video_path: str,
    output_dir: str,
    locale: str = "fr-ca",
    job = None,  # RunPod job object for progress updates
) -> dict:
    """Run the pipeline on a video file."""

    # Add pipeline repo to path
    sys.path.insert(0, "/workspace/scripts")

    # Import pipeline (handles numeric module name via importlib)
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "pipeline_02", "/workspace/scripts/02_pipeline.py"
    )
    pipeline = importlib.util.module_from_spec(spec)
    sys.modules["pipeline_02"] = pipeline
    spec.loader.exec_module(pipeline)
    load_config = pipeline.load_config
    process_video = pipeline.process_video

    # Load and override config
    cfg = load_config("/workspace/config.yaml")
    cfg.locale = locale
    cfg.output_folder = output_dir
    cfg.temp_folder = os.path.join(output_dir, "temp")
    cfg.keep_temp = False

    os.makedirs(cfg.temp_folder, exist_ok=True)

    # Run pipeline
    try:
        process_video(Path(video_path), cfg, logger)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        raise

    # Collect outputs
    stem = Path(video_path).stem
    m4a_path = Path(output_dir) / f"{stem}_french.m4a"
    srt_path = Path(output_dir) / f"{stem}_french.srt"

    if not m4a_path.exists():
        raise FileNotFoundError(f"Pipeline did not produce {m4a_path.name}")
    if not srt_path.exists():
        raise FileNotFoundError(f"Pipeline did not produce {srt_path.name}")

    return {
        "stem": stem,
        "m4a_path": m4a_path,
        "srt_path": srt_path,
    }


# ============================================================================
# RunPod Handler
# ============================================================================

def handler(job) -> dict:
    """Process a single video job."""
    try:
        job_id = job.get("id", "unknown")
        logger.info(f"\n{'='*60}")
        logger.info(f"Job {job_id} started")
        logger.info(f"{'='*60}")

        # Parse input
        inp = job.get("input", {})
        video_url = inp.get("video_url")
        upload_prefix = inp.get("upload_prefix")
        locale = inp.get("locale", "fr-ca")

        if not video_url:
            raise ValueError("Missing 'video_url' in job input")
        if not upload_prefix:
            raise ValueError("Missing 'upload_prefix' in job input")

        logger.info(f"Locale: {locale}")

        # Create isolated temp directory
        with tempfile.TemporaryDirectory(prefix="dubbing_") as tmpdir:
            logger.info(f"Working in {tmpdir}")

            # 1. Download
            logger.info("\n[1/4] Downloading video...")
            video_path = _download_video(video_url, tmpdir)

            # 2. Run pipeline
            logger.info("\n[2/4] Running pipeline...")
            result = _run_pipeline(
                video_path,
                tmpdir,
                locale=locale,
                job=job,
            )

            # 3. Upload
            logger.info("\n[3/4] Uploading outputs...")
            m4a_url = _upload_file(
                result["m4a_path"],
                upload_prefix,
                f"{result['stem']}_french.m4a",
            )
            srt_url = _upload_file(
                result["srt_path"],
                upload_prefix,
                f"{result['stem']}_french.srt",
            )

            output = {
                "m4a_url": m4a_url,
                "srt_url": srt_url,
                "stem": result["stem"],
            }

            logger.info("\n[4/4] Complete!")
            logger.info(f"✓ M4A: {m4a_url}")
            logger.info(f"✓ SRT: {srt_url}")

            return output

    except Exception as e:
        logger.error(f"Job {job.get('id')} failed: {e}", exc_info=True)
        return {
            "error": str(e),
            "traceback": __import__("traceback").format_exc(),
        }


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    logger.info("Initializing RunPod serverless worker...")
    logger.info(f"GPU: {os.popen('nvidia-smi --query-gpu=name --format=csv,noheader').read().strip()}")

    # Start Ollama once (persists for all jobs on this worker instance)
    _start_ollama()

    # Start serverless handler
    logger.info("Starting job handler...")
    runpod.serverless.start({"handler": handler})
