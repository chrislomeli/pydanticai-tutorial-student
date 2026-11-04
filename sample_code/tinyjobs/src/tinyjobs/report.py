"""A summary of the queue."""

from __future__ import annotations

from collections import Counter

from tinyjobs.jobs import JobQueue, JobStatus


def format_report(queue: JobQueue) -> str:
    """One line per state that has jobs, then the failures with their errors."""
    counts = Counter(job.status for job in queue)
    lines = [f"{status.value}: {counts[status]}" for status in JobStatus if counts[status]]
    lines += [f"  {job.id} {job.name}: {job.error}" for job in queue if job.status is JobStatus.FAILED]
    return "\n".join(lines) if lines else "no jobs"
