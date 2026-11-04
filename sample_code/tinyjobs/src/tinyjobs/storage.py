"""Saving and loading the queue as a JSON file."""

from __future__ import annotations

import json
import os
from pathlib import Path

from tinyjobs.jobs import Job, JobQueue, JobStatus

DEFAULT_QUEUE_FILE = ".tinyjobs.json"


def queue_file() -> Path:
    return Path(os.environ.get("TINYJOBS_QUEUE_FILE", DEFAULT_QUEUE_FILE))


def save(queue: JobQueue, path: Path) -> None:
    records = [
        {"id": job.id, "name": job.name, "status": job.status.value, "error": job.error}
        for job in queue
    ]
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(records, indent=2))
    tmp.replace(path)  # atomic on POSIX: a crash never leaves half a file


def load(path: Path) -> JobQueue:
    if not path.exists():
        return JobQueue()
    records = json.loads(path.read_text())
    jobs = [
        Job(
            id=record["id"],
            name=record["name"],
            status=JobStatus(record["status"]),
            error=record["error"],
        )
        for record in records
    ]
    return JobQueue(jobs)
