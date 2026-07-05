#!/usr/bin/env bash
# On-demand dubbing trigger: start the (stopped) RunPod pod, wait for the web
# API, submit a job, and optionally wait for completion and download outputs.
# Pairs with the pod-side idle auto-stop (DUBBING_IDLE_STOP_MIN) so the GPU
# only bills while work is running.
#
# Requirements: bash, curl, python3 (for JSON parsing).
#
# Environment:
#   RUNPOD_API_KEY    RunPod API key  (runpod.io → Settings → API Keys)
#   RUNPOD_POD_ID     ID of the pod to start (must have the web UI on :7860)
#   DUBBING_UI_TOKEN  web UI auth token (same value as on the pod)
#   DUBBING_URL       optional override; default https://{POD_ID}-7860.proxy.runpod.net
#
# Usage:
#   ./trigger.sh --vimeo https://vimeo.com/12345 [options]
#   ./trigger.sh --path /workspace/videos/input/foo.mp4 [options]
#   ./trigger.sh --file ~/videos/foo.mp4 [options]
#
# Options:
#   --locale fr|fr-ca     locale for this job
#   --speakers N          exact speaker count (1 = solo presenter)
#   --volume-boost N      loudness boost percent
#   --review              pause for translation review (default off — a job
#                         waiting for review lets the pod idle-stop; state
#                         persists and resumes on next start)
#   --wait                poll until the job finishes; exit 0/1 on success/failure
#   --download DIR        with --wait: download outputs into DIR
set -euo pipefail

: "${RUNPOD_API_KEY:?set RUNPOD_API_KEY}"
: "${RUNPOD_POD_ID:?set RUNPOD_POD_ID}"
: "${DUBBING_UI_TOKEN:?set DUBBING_UI_TOKEN}"
BASE="${DUBBING_URL:-https://${RUNPOD_POD_ID}-7860.proxy.runpod.net}"
AUTH=(-H "Authorization: Bearer ${DUBBING_UI_TOKEN}")

SOURCE_KIND="" SOURCE_VAL="" LOCALE="" SPEAKERS="" VBOOST="" REVIEW="" WAIT=0 DOWNLOAD_DIR=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --vimeo)        SOURCE_KIND=vimeo_url;  SOURCE_VAL="$2"; shift 2 ;;
    --path)         SOURCE_KIND=local_path; SOURCE_VAL="$2"; shift 2 ;;
    --file)         SOURCE_KIND=file;       SOURCE_VAL="$2"; shift 2 ;;
    --locale)       LOCALE="$2";   shift 2 ;;
    --speakers)     SPEAKERS="$2"; shift 2 ;;
    --volume-boost) VBOOST="$2";   shift 2 ;;
    --review)       REVIEW=on;     shift ;;
    --wait)         WAIT=1;        shift ;;
    --download)     WAIT=1; DOWNLOAD_DIR="$2"; shift 2 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done
[[ -n "$SOURCE_KIND" ]] || { echo "provide a source: --vimeo URL | --path POD_PATH | --file LOCAL_FILE" >&2; exit 2; }

jsonget() { python3 -c "import sys,json;d=json.load(sys.stdin);print(d$1)" 2>/dev/null; }

# ── 1. Start the pod (idempotent: already-running is fine) ────────────────────
echo ">> starting pod ${RUNPOD_POD_ID} …"
start_resp=$(curl -sS -X POST "https://rest.runpod.io/v1/pods/${RUNPOD_POD_ID}/start" \
  -H "Authorization: Bearer ${RUNPOD_API_KEY}" -w '\n%{http_code}')
start_code=${start_resp##*$'\n'}
if [[ "$start_code" != 2* ]]; then
  body=${start_resp%$'\n'*}
  # A pod that is already running returns an error from /start — check status.
  status=$(curl -sS "https://rest.runpod.io/v1/pods/${RUNPOD_POD_ID}" \
    -H "Authorization: Bearer ${RUNPOD_API_KEY}" | jsonget "['desiredStatus']" || true)
  if [[ "$status" != "RUNNING" ]]; then
    echo "!! could not start pod (HTTP ${start_code}): ${body}" >&2
    echo "   If the GPU is unavailable, start a new pod on the network volume." >&2
    exit 1
  fi
  echo "   pod already running"
fi

# ── 2. Wait for the web API ───────────────────────────────────────────────────
echo ">> waiting for ${BASE} …"
deadline=$((SECONDS + 600))
until curl -fsS "${AUTH[@]}" "${BASE}/api/health" >/dev/null 2>&1; do
  (( SECONDS < deadline )) || { echo "!! web UI not up after 10 min" >&2; exit 1; }
  sleep 5
done
echo "   up."

# ── 3. Submit the job ─────────────────────────────────────────────────────────
form=()
if [[ "$SOURCE_KIND" == file ]]; then
  [[ -f "$SOURCE_VAL" ]] || { echo "!! no such file: $SOURCE_VAL" >&2; exit 1; }
  form+=(-F "video=@${SOURCE_VAL}")
else
  form+=(-F "${SOURCE_KIND}=${SOURCE_VAL}")
fi
[[ -n "$LOCALE"   ]] && form+=(-F "locale=${LOCALE}")
[[ -n "$SPEAKERS" ]] && form+=(-F "speakers=${SPEAKERS}")
[[ -n "$VBOOST"   ]] && form+=(-F "volume_boost=${VBOOST}")
[[ -n "$REVIEW"   ]] && form+=(-F "review=on")

resp=$(curl -sS "${AUTH[@]}" "${form[@]}" "${BASE}/api/jobs")
job_id=$(echo "$resp" | jsonget "['id']")
[[ -n "$job_id" ]] || { echo "!! submit failed: $resp" >&2; exit 1; }
echo ">> job ${job_id} queued  (${BASE}/?token=…)"

# ── 4. Optionally wait + download ────────────────────────────────────────────
(( WAIT )) || exit 0
echo ">> waiting for job ${job_id} …"
while :; do
  sleep 30
  j=$(curl -fsS "${AUTH[@]}" "${BASE}/api/jobs/${job_id}") || { echo "   (poll failed, retrying)"; continue; }
  status=$(echo "$j" | jsonget "['status']")
  phase=$(echo "$j" | jsonget "['phase']")
  echo "   ${status}  ${phase}"
  case "$status" in
    completed|awaiting_review) break ;;
    failed|cancelled) echo "!! job ${status}: $(echo "$j" | jsonget "['error']")" >&2; exit 1 ;;
  esac
done

if [[ -n "$DOWNLOAD_DIR" && "$status" == completed ]]; then
  mkdir -p "$DOWNLOAD_DIR"
  for kind in video audio full srt; do
    if curl -fsS "${AUTH[@]}" -o "${DOWNLOAD_DIR}/.probe" "${BASE}/api/jobs/${job_id}/download/${kind}" 2>/dev/null; then
      name=$(curl -fsSI "${AUTH[@]}" "${BASE}/api/jobs/${job_id}/download/${kind}" \
        | tr -d '\r' | sed -n 's/.*filename="\{0,1\}\([^";]*\).*/\1/p' | tail -1)
      mv "${DOWNLOAD_DIR}/.probe" "${DOWNLOAD_DIR}/${name:-${job_id}_${kind}}"
      echo "   saved ${DOWNLOAD_DIR}/${name:-${job_id}_${kind}}"
    fi
  done
  rm -f "${DOWNLOAD_DIR}/.probe"
fi
echo ">> done (${status}). Pod will idle-stop on its own."
