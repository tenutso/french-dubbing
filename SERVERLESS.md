# Serverless Deployment Overview

This document explains how to run the French dubbing pipeline on RunPod Serverless with true scale-to-zero pricing.

## Quick Start

### For Operators

1. **Build**: `docker build -t french-dubbing:latest .`
2. **Push**: `docker push your-registry/french-dubbing:latest`
3. **Deploy**: Follow [RUNPOD_DEPLOYMENT.md](RUNPOD_DEPLOYMENT.md) steps 1–3
4. **Test**: `python runpod_client.py --endpoint-id xxx --api-key yyy --video-url "s3://..." --upload-prefix "s3://..."`

### For Users

```python
import runpod

runpod.api_key = "your-api-key"
endpoint = runpod.Endpoint("endpoint-id")

job = endpoint.run({
    "video_url": "https://s3.../video.mp4?...",
    "upload_prefix": "s3://bucket/output/",
    "locale": "fr-ca",
})

output = job.output(timeout=3600)  # Wait up to 1 hour
print(f"M4A: {output['m4a_url']}")
print(f"SRT: {output['srt_url']}")
```

## Architecture

### Components

| Component | Role | Lifespan |
|-----------|------|----------|
| `runpod_handler.py` | Serverless entrypoint | Per job |
| `02_pipeline.py` | Core pipeline | Per job |
| Network Volume `/workspace` | Persistent models | Across all workers |
| Ollama | LLM server | Persists per worker instance |
| S3/R2 | Input/output storage | Outside RunPod |

### Flow

```
Job submission (JSON)
   ↓
[RunPod Serverless Gateway]
   ↓
Worker startup (2–3 min cold start)
   ├─ Mount network volume
   ├─ Start Ollama
   └─ Load handler
   ↓
runpod_handler.handler(job)
   ├─ Download video from S3 presigned URL
   ├─ Call process_video() (15–30 min)
   ├─ Upload M4A + SRT to S3
   └─ Return { m4a_url, srt_url }
   ↓
[RunPod returns output]
   ↓
Worker idles → scales down to 0 (within 30s)
```

## Input/Output Format

### Input

```json
{
  "video_url": "https://s3.amazonaws.com/bucket/input.mp4?AWSAccessKeyId=...",
  "upload_prefix": "s3://bucket/output/",
  "locale": "fr-ca"
}
```

- **video_url**: Presigned HTTPS URL (S3, R2, or any URL accessible from RunPod)
- **upload_prefix**: S3 URI or presigned PUT URL where outputs will be written
- **locale**: `"fr"` (standard European) or `"fr-ca"` (Québécois, default)

### Output

```json
{
  "m4a_url": "s3://bucket/output/video_french.m4a",
  "srt_url": "s3://bucket/output/video_french.srt",
  "stem": "video"
}
```

Or on error:

```json
{
  "error": "Failed to download video: ...",
  "traceback": "..."
}
```

## Calling the Endpoint

### Python (Recommended)

```bash
pip install runpod boto3 requests
python runpod_client.py \
  --endpoint-id your-id \
  --api-key your-key \
  --video-url "https://..." \
  --upload-prefix "s3://..."
```

### cURL

```bash
curl -X POST https://api.runpod.io/v1/serverless/{endpoint_id}/run \
  -H "Content-Type: application/json" \
  -d '{
    "api_key": "your-key",
    "input": {
      "video_url": "https://...",
      "upload_prefix": "s3://..."
    }
  }'
```

### RunPod Python SDK

```python
import runpod

runpod.api_key = "your-api-key"
endpoint = runpod.Endpoint("your-endpoint-id")

# Non-blocking submit
job = endpoint.run({
    "video_url": "...",
    "upload_prefix": "...",
})

# Poll for output
output = job.output(timeout=3600)
```

## Cost Model

### One-Time Setup
- Network Volume (50 GB): ~$5–10
- Initial model download: Free (part of setup pod)

### Per-Job (Recurring)
- **RTX 4090**: $0.35/hour
- **Typical job**: 20–30 minutes = 25 min avg = $0.15 per video
- **S3 storage**: ~$0.1 per video (input + output)
- **Network Volume**: ~$0.10/GB/month (negligible)

### Example: 100 videos/month
```
Compute:    100 videos × 25 min / 60 min × $0.35/hour = $145
Storage:    Input (50 GB) + Output (50 GB) = ~$2
Volume:     ~$5
Total:      ~$152/month

vs. Persistent Pod (24/7):
RTX 4090 × 24 × 30 = $8,064/month
```

## Deployment Checklist

- [ ] HuggingFace token created and licenses accepted
- [ ] AWS S3 bucket created with IAM credentials
- [ ] Docker image built and pushed to registry
- [ ] Network Volume created and pre-populated with models
- [ ] RunPod Serverless endpoint configured
- [ ] Environment variables set (AWS, HF, S3)
- [ ] Endpoint tested with sample job
- [ ] Client code (runpod_client.py or custom) ready

## Monitoring

### Job Status

```python
job = endpoint.run(input_dict)
status = job.status()
print(status)  # "IN_QUEUE", "IN_PROGRESS", "COMPLETED"
```

### Logs

RunPod Console → Serverless → Endpoint → Logs

Shows:
- Worker startup/shutdown times
- Pipeline stage progress
- Error messages
- Total execution time

### Cost Tracking

RunPod Console → Billing → Serverless charges

Shows cost per job, total monthly spend.

## Troubleshooting

### Cold start > 5 minutes

**Cause**: Models not cached on network volume
**Fix**: 
1. Check volume is attached to endpoint
2. Re-run Step 1 of RUNPOD_DEPLOYMENT.md to pre-download models

### "Ollama failed to start"

**Cause**: Worker ran out of VRAM during Ollama startup
**Fix**: 
1. Use only RTX 4090 or A100
2. Reduce max_workers if workers are OOMing

### "Out of Memory" during pipeline

**Cause**: GPU is too small
**Fix**: Only RTX 4090 (24 GB) or larger. T4 (16 GB) will fail.

### Upload fails with 403 Forbidden

**Cause**: Invalid AWS credentials
**Fix**: 
1. Verify `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` in endpoint env vars
2. Test credentials locally: `aws s3 ls s3://bucket/`

### Video download timeout

**Cause**: Video URL is not accessible from RunPod's network or presigned URL expired
**Fix**: 
1. Test URL directly: `curl -I "https://..."`
2. Regenerate presigned URLs with longer expiry (≥1 hour)

## Advanced Configuration

### Custom Model

To use a different Qwen model size:

1. In `runpod_handler.py`, after `_start_ollama()`, run:
   ```python
   os.system("ollama pull qwen3:70b")  # or any Qwen variant
   ```

2. In `config.yaml`, change:
   ```yaml
   translation:
     model: qwen3:70b
   ```

### Batch Processing

Submit multiple jobs in parallel:

```python
videos = ["video1.mp4", "video2.mp4", "video3.mp4"]
jobs = []

for video in videos:
    job = endpoint.run({
        "video_url": f"s3://bucket/{video}",
        "upload_prefix": "s3://bucket/output/",
    })
    jobs.append(job)

# Wait for all
for job in jobs:
    output = job.output(timeout=3600)
    print(output)
```

RunPod scales workers up to `max_workers` to handle parallel jobs.

### Custom Locale at Runtime

Modify `runpod_handler.py` to accept additional fields:

```python
def handler(job):
    inp = job["input"]
    locale = inp.get("locale", "fr-ca")
    temperature = inp.get("temperature", 0.3)
    
    # ... set config
    cfg.locale = locale
    cfg.translation_temperature = temperature
```

Then pass from client:

```python
endpoint.run({
    "video_url": "...",
    "upload_prefix": "...",
    "locale": "fr",
    "temperature": 0.5,
})
```

## Performance Tips

1. **Keep workers warm**: Set `idle_timeout` to 60s if running frequent jobs
2. **Batch upload/download**: Use S3 transfer acceleration or multipart uploads for large videos
3. **Monitor GPU**: Check worker logs for GPU utilization; add logging to `02_pipeline.py` if needed
4. **Pre-warm Ollama**: First job loads qwen3:14b (~30s); subsequent jobs reuse it

## Cleanup & Cost Reduction

1. **Pause deployment**: Set `max_workers: 0` in endpoint config
2. **Delete old workers**: If idle timeout is set too high, manually delete workers
3. **Archive outputs**: Move old outputs from S3 to Glacier to reduce storage costs
4. **Monitor volume**: Check if network volume has accumulated temp files; clean if needed

---

For step-by-step deployment, see [RUNPOD_DEPLOYMENT.md](RUNPOD_DEPLOYMENT.md).
