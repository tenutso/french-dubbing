# RunPod Serverless Deployment Guide

Deploy this French dubbing pipeline to RunPod Serverless for true scale-to-zero pricing. You only pay for active translation jobs, not for idle compute.

## Architecture

```
┌─────────────────────┐
│  Your Application   │
│  (any language)     │
└──────────┬──────────┘
           │ HTTP POST { video_url, upload_prefix, locale }
           ▼
┌─────────────────────────────────────┐
│  RunPod Serverless Endpoint         │
│  (scales to 0 when idle)            │
└──────────┬──────────────────────────┘
           │ Cold start: ~2-3 min (with network volume)
           │ Run time: ~15-30 min per video
           │
           ▼
┌─────────────────────────────────────┐
│  GPU Worker (RTX 4090)              │
│  ├─ Download video from S3/R2       │
│  ├─ Run pipeline stages 1-6         │
│  │  (Demucs, Whisper, Qwen, XTTS)  │
│  └─ Upload outputs to S3/R2         │
└──────────┬──────────────────────────┘
           │ Returns: { m4a_url, srt_url }
           │
           ▼
       Your App
```

## Prerequisites

1. **AWS Account** (or Cloudflare R2 for S3-compatible storage)
   - Create an S3 bucket for input videos and outputs
   - Generate IAM credentials: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`
   - Note the bucket name and region

2. **HuggingFace Account**
   - Get your API token: https://huggingface.co/settings/tokens
   - Accept the license for pyannote speaker diarization if using it:
     https://huggingface.co/pyannote/speaker-diarization-community-1

3. **RunPod Account**
   - Create account at https://runpod.io
   - Add credits or payment method

## Step 1: Prepare Models on Network Volume

Pre-download all models to a RunPod Network Volume so cold starts are fast (~2–3 min instead of 10–15 min).

### Option A: Pre-cached volume (recommended)

1. In RunPod Console, create a new **Network Volume** (e.g., "dubbing-models", 50 GB)

2. Spin up a temporary **On-Demand Pod** (RTX 4090):
   - Attach the network volume to `/workspace`
   - Docker image: `ubuntu:24.04`
   - Start bash shell

3. In the pod, install and run setup:
   ```bash
   # Install Python and git
   apt-get update && apt-get install -y python3 pip git ffmpeg wget curl sox libsndfile1 build-essential
   cd /workspace
   
   # Clone the repo or copy files
   git clone https://github.com/yourusername/french-dubbing.git
   cd french-dubbing
   
   # Install dependencies
   pip install -r requirements.txt
   
   # Run setup (this downloads all models)
   export HF_TOKEN="your-huggingface-token"
   bash scripts/04_setup.sh
   ```

4. Wait ~10–15 minutes for all models to download. You'll see:
   - Whisper: `/workspace/models/whisper/`
   - XTTS: HuggingFace cache (auto-linked from `~/.cache/huggingface/`)
   - Qwen3: `/workspace/.ollama/models/`
   - pyannote: HuggingFace cache

5. Once done, **delete the on-demand pod** but **keep the network volume** attached.

## Step 2: Build and Push Docker Image

```bash
# Build the Docker image locally
docker build -t french-dubbing:latest .

# Tag for your container registry
docker tag french-dubbing:latest your-docker-registry/french-dubbing:latest

# Push to registry (Docker Hub, GitHub Container Registry, etc.)
docker push your-docker-registry/french-dubbing:latest
```

For RunPod, you can also use a private registry or upload directly to RunPod's registry.

## Step 3: Create RunPod Serverless Endpoint

1. Go to **RunPod Console** → **Serverless** → **Create New Endpoint**

2. Fill in the form:
   - **Name**: `french-dubbing`
   - **Description**: `Translate and dub video to French`
   - **Docker Image**: `your-docker-registry/french-dubbing:latest`
   - **GPU**: `RTX 4090` (or A100 for higher throughput)
   - **Min Workers**: `0` (scale to zero)
   - **Max Workers**: `2–3` (start small, increase if needed)
   - **Idle Timeout**: `30s` (workers shut down after 30s idle)
   - **Job Timeout**: `3600s` (1 hour, enough for long videos)

3. Under **Network Volume**, select the volume created in Step 1 and mount at `/workspace`

4. Under **Environment Variables**, add:
   ```
   HF_TOKEN=your-huggingface-token
   AWS_ACCESS_KEY_ID=your-aws-access-key
   AWS_SECRET_ACCESS_KEY=your-aws-secret-key
   AWS_REGION=us-east-1
   S3_BUCKET=your-bucket-name
   OLLAMA_MODELS=/workspace/.ollama/models
   ```

5. Click **Create Endpoint**

6. Copy the **Endpoint ID** — you'll need it to call the API

## Step 4: Call the Endpoint

### Python Client

```python
import runpod

# Set your API key (get from RunPod account settings)
runpod.api_key = "your-runpod-api-key"

# Create endpoint reference
endpoint_id = "your-endpoint-id"
endpoint = runpod.Endpoint(endpoint_id)

# Call the endpoint
job = endpoint.run({
    "video_url": "https://s3.amazonaws.com/your-bucket/input/webinar.mp4?AWSAccessKeyId=...",
    "upload_prefix": "s3://your-bucket/output/",
    "locale": "fr-ca"
})

# Wait for result (timeout in seconds)
output = job.output(timeout=3600)

if "error" in output:
    print(f"Job failed: {output['error']}")
else:
    print(f"M4A: {output['m4a_url']}")
    print(f"SRT: {output['srt_url']}")
```

### cURL

```bash
curl -X POST https://api.runpod.io/v1/serverless/french-dubbing/run \
  -H "Content-Type: application/json" \
  -d '{
    "api_key": "your-runpod-api-key",
    "input": {
      "video_url": "https://s3.amazonaws.com/your-bucket/input/webinar.mp4?...",
      "upload_prefix": "s3://your-bucket/output/",
      "locale": "fr-ca"
    }
  }'
```

## Step 5: Test Locally

Before deploying to RunPod, test the handler locally:

```bash
# Build the image
docker build -t french-dubbing:test .

# Run with GPU
docker run --gpus all -it \
  -e HF_TOKEN="your-huggingface-token" \
  -e AWS_ACCESS_KEY_ID="..." \
  -e AWS_SECRET_ACCESS_KEY="..." \
  -e AWS_REGION="us-east-1" \
  -v /workspace:/workspace \
  french-dubbing:test \
  python /workspace/scripts/runpod_handler.py

# In another terminal, send a test job
curl -X POST http://localhost:8000/run \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "video_url": "https://...",
      "upload_prefix": "s3://..."
    }
  }'
```

## Cost Breakdown

### Storage
- **Network Volume**: ~$0.10/GB/month → ~$5/month for 50 GB
- **S3 Input/Output**: ~$0.023/GB transferred

### Compute
- **RTX 4090**: ~$0.35/hour
- **Video duration**: ~15–30 min per video
- **Cost per video**: $0.09–$0.18 + storage

### Example: 100 videos/month
- Compute: 100 × 25 min / 60 min × $0.35 = ~$145
- Storage: ~$10
- **Total**: ~$155 (vs. ~$8000/month for a persistent pod!)

## Monitoring & Debugging

### Check endpoint status
```python
endpoint = runpod.Endpoint(endpoint_id)
print(endpoint.health())
```

### View worker logs
RunPod Console → Endpoint → Logs (shows all job logs in real-time)

### Monitor costs
RunPod Console → Billing → Serverless charges are itemized per job

## Troubleshooting

### Cold start is too slow (> 5 min)
- **Cause**: Models not on network volume or network volume too slow
- **Fix**: Pre-download models to the volume using Step 1. Verify volume is attached under Endpoint config.

### Out of VRAM
- **Cause**: GPU is too small (need RTX 4090 = 24 GB)
- **Fix**: Use `RTX 4090` or `A100` only. Do not attempt on T4 or smaller.

### Qwen3 not loading
- **Cause**: Ollama failing to start or network timeout
- **Fix**: Check Ollama startup in logs; add retry logic in `runpod_handler.py`

### Upload fails
- **Cause**: Invalid AWS credentials or S3 bucket doesn't exist
- **Fix**: Verify `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, and `S3_BUCKET` are set correctly in Endpoint environment

## Advanced: Custom Locale or Config

To use a custom locale or override any config value, modify `runpod_handler.py` line ~180:

```python
cfg.locale = locale  # from job input
cfg.target_lang = "es"  # or any language
cfg.translation_temperature = 0.5  # lower = more deterministic
```

You can also accept additional fields in the job input and pass them to the config.

## Advanced: Batch Processing

To process multiple videos in parallel, submit multiple jobs simultaneously. RunPod will scale up to `max_workers` and process them in parallel.

```python
import runpod
runpod.api_key = "..."
endpoint = runpod.Endpoint("your-endpoint-id")

jobs = [
    endpoint.run({"video_url": url1, "upload_prefix": prefix1}),
    endpoint.run({"video_url": url2, "upload_prefix": prefix2}),
    endpoint.run({"video_url": url3, "upload_prefix": prefix3}),
]

for job in jobs:
    output = job.output(timeout=3600)
    print(output)
```

## Cleanup

- **To pause**: Reduce max workers to 0 in Endpoint config
- **To delete**: RunPod Console → Endpoint → Delete
- **To keep the volume**: Detach it before deleting the endpoint

---

## Support

- **RunPod Documentation**: https://docs.runpod.io/serverless
- **RunPod Community**: https://discord.gg/runpod
- **Issue Tracker**: https://github.com/yourusername/french-dubbing/issues
