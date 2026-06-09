"""Job model and JSON persistence for the dubbing web UI."""
from __future__ import annotations

import json
import os
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

JOBS_FILE = "/workspace/web/jobs.json"

STATUS_QUEUED = "queued"
STATUS_RUNNING = "running"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"
STATUS_CANCELLED = "cancelled"
TERMINAL = {STATUS_COMPLETED, STATUS_FAILED, STATUS_CANCELLED}


@dataclass
class Job:
    id: str
    video_filename: str
    video_path: str
    output_dir: str
    log_path: str
    options: dict
    status: str = STATUS_QUEUED
    queued_at: float = field(default_factory=time.time)
    started_at: float = 0.0
    ended_at: float = 0.0
    error: str = ""
    outputs: dict = field(default_factory=dict)
    phase: str = ""
    returncode: Optional[int] = None
    source_url: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def is_terminal(self) -> bool:
        return self.status in TERMINAL


def new_job_id() -> str:
    return uuid.uuid4().hex[:12]


def load_jobs(path: str = JOBS_FILE) -> Dict[str, Job]:
    """Load jobs.json. Any job left 'running' is recovered as 'failed' since
    the server clearly restarted mid-execution."""
    if not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
    except Exception:
        return {}
    jobs: Dict[str, Job] = {}
    for entry in raw.get("jobs", []):
        try:
            j = Job(**entry)
        except TypeError:
            # Forward-compat: ignore unknown fields on older records.
            allowed = {k: entry[k] for k in entry if k in Job.__dataclass_fields__}
            j = Job(**allowed)
        if j.status == STATUS_RUNNING:
            j.status = STATUS_FAILED
            j.error = j.error or "server restarted mid-job"
            j.ended_at = j.ended_at or time.time()
        jobs[j.id] = j
    return jobs


def save_jobs(jobs: Dict[str, Job], path: str = JOBS_FILE) -> None:
    """Atomic write of jobs.json (tempfile + rename)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {"jobs": [j.to_dict() for j in sorted(
        jobs.values(), key=lambda j: j.queued_at
    )]}
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), prefix=".jobs.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def sorted_jobs(jobs: Dict[str, Job]) -> List[Job]:
    """Newest first (by queued_at) for UI display."""
    return sorted(jobs.values(), key=lambda j: j.queued_at, reverse=True)


def safe_stem(filename: str) -> str:
    """Strip extension and any path separators from a user-supplied filename."""
    return Path(filename).stem.replace("/", "_").replace("\\", "_")
