#!/usr/bin/env python3
"""
RunPod Serverless Client Example

Call a French dubbing endpoint and download the results.

Usage:
  python runpod_client.py \
    --endpoint-id your-endpoint-id \
    --api-key your-runpod-api-key \
    --video-url https://s3.../webinar.mp4 \
    --upload-prefix s3://my-bucket/output/ \
    --output-dir ./downloads \
    --locale fr-ca
"""

import argparse
import sys
import json
from pathlib import Path
from typing import Optional

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

import requests


def download_s3_file(s3_url: str, local_path: Path, timeout_s: int = 300) -> None:
    """Download a file from S3 (presigned URL or public)."""
    print(f"Downloading {s3_url[:80]}...")
    resp = requests.get(s3_url, stream=True, timeout=timeout_s)
    resp.raise_for_status()

    total_size = int(resp.headers.get("content-length", 0))
    downloaded = 0

    with open(local_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total_size:
                    pct = 100 * downloaded / total_size
                    print(f"  {pct:.1f}% ({downloaded / 1e6:.1f} MB)", end="\r")

    print(f"✓ Downloaded to {local_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Call a RunPod Serverless French dubbing endpoint"
    )
    parser.add_argument(
        "--endpoint-id",
        required=True,
        help="RunPod endpoint ID (from console)",
    )
    parser.add_argument(
        "--api-key",
        required=True,
        help="RunPod API key (from account settings)",
    )
    parser.add_argument(
        "--video-url",
        required=True,
        help="Presigned S3 URL to input video",
    )
    parser.add_argument(
        "--upload-prefix",
        required=True,
        help="S3 path prefix for outputs (e.g. s3://bucket/output/)",
    )
    parser.add_argument(
        "--output-dir",
        default="./dubbing_output",
        help="Local directory to save M4A and SRT (default: ./dubbing_output)",
    )
    parser.add_argument(
        "--locale",
        default="fr-ca",
        choices=["fr", "fr-ca"],
        help="Target locale (default: fr-ca for Québécois)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=3600,
        help="Job timeout in seconds (default: 3600 for 1 hour)",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Print output URLs but don't download files",
    )

    args = parser.parse_args()

    # Set up RunPod client
    runpod.api_key = args.api_key
    endpoint = runpod.Endpoint(args.endpoint_id)

    print("=" * 60)
    print("RunPod French Dubbing Client")
    print("=" * 60)
    print(f"Endpoint: {args.endpoint_id}")
    print(f"Video: {args.video_url[:60]}...")
    print(f"Locale: {args.locale}")
    print(f"Timeout: {args.timeout}s")
    print()

    # Submit job
    print("[1/3] Submitting job...")
    job_input = {
        "video_url": args.video_url,
        "upload_prefix": args.upload_prefix,
        "locale": args.locale,
    }

    job = endpoint.run(job_input)
    job_id = job.get("id", "unknown")
    print(f"✓ Job {job_id} submitted")
    print()

    # Wait for completion
    print("[2/3] Waiting for pipeline to complete...")
    print(
        f"(This may take 15–30 minutes depending on video length and job queue)"
    )

    try:
        output = job.output(timeout=args.timeout)
    except TimeoutError:
        print(f"ERROR: Job timed out after {args.timeout}s")
        sys.exit(1)

    if "error" in output:
        print(f"ERROR: Job failed")
        print(f"  Error: {output['error']}")
        if "traceback" in output:
            print(f"  Traceback:\n{output['traceback']}")
        sys.exit(1)

    m4a_url = output.get("m4a_url")
    srt_url = output.get("srt_url")

    print(f"✓ Pipeline complete!")
    print(f"  M4A: {m4a_url}")
    print(f"  SRT: {srt_url}")
    print()

    # Download outputs
    if args.no_download:
        print("✓ URLs ready (skipping download)")
        return

    print("[3/3] Downloading outputs...")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = output.get("stem", "dubbed")

    if m4a_url:
        m4a_path = output_dir / f"{stem}_french.m4a"
        download_s3_file(m4a_url, m4a_path)

    if srt_url:
        srt_path = output_dir / f"{stem}_french.srt"
        download_s3_file(srt_url, srt_path)

    print()
    print("=" * 60)
    print("✓ Complete!")
    print(f"  Output: {output_dir.absolute()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
