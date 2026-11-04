"""The job record and the queue that holds jobs.

A job moves through `PENDING -> RUNNING -> SUCCEEDED | FAILED`, or from
`PENDING` to `CANCELLED`. Once a job has finished, its status never changes.
"""

from __future__ import annotations

import enum
import itertools
import logging
from collections.abc import Callable, Iterator
from dataclasses import dataclass

log = logging.getLogger(__name__)


class JobStatus(enum.StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def finished(self) -> bool:
        return self in (JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED)


class UnknownJob(KeyError):
    """No job has this id."""


@dataclass
class Job:
    id: int
    name: str
    status: JobStatus = JobStatus.PENDING
    error: str | None = None


class JobQueue:
    """Jobs in submission order. Runs them one at a time, oldest first."""

    def __init__(self, jobs: list[Job] | None = None) -> None:
        self._jobs: dict[int, Job] = {job.id: job for job in jobs or []}
        start = max(self._jobs, default=0) + 1
        self._ids = itertools.count(start)

    def submit(self, name: str) -> Job:
        if not name.strip():
            raise ValueError("a job needs a name")
        job = Job(id=next(self._ids), name=name)
        self._jobs[job.id] = job
        log.debug("submitted job %d: %s", job.id, name)
        return job

    def get(self, job_id: int) -> Job:
        try:
            return self._jobs[job_id]
        except KeyError:
            raise UnknownJob(job_id) from None

    def __iter__(self) -> Iterator[Job]:
        return iter(sorted(self._jobs.values(), key=lambda job: job.id))

    def pending(self) -> list[Job]:
        return [job for job in self if job.status is JobStatus.PENDING]

    def cancel(self, job_id: int) -> Job:
        """Cancel a job that has not started. Cancelling a finished job does nothing."""
        job = self.get(job_id)
        if job.status is JobStatus.RUNNING:
            raise RuntimeError(f"job {job_id} is running and cannot be cancelled")
        job.status = JobStatus.CANCELLED
        log.debug("cancelled job %d", job_id)
        return job

    def forget(self, job_id: int) -> None:
        """Remove a finished job from the queue."""
        job = self.get(job_id)
        if not job.status.finished:
            raise RuntimeError(f"job {job_id} has not finished")
        del self._jobs[job_id]

    def run_next(self, work: Callable[[Job], None]) -> Job | None:
        """Run the oldest pending job with `work`. Returns it, or None if none wait."""
        waiting = self.pending()
        if not waiting:
            return None
        job = waiting[0]
        job.status = JobStatus.RUNNING
        try:
            work(job)
        except Exception as error:  # a job's failure is recorded, not raised
            job.status = JobStatus.FAILED
            job.error = str(error)
            log.warning("job %d failed: %s", job.id, error)
        else:
            job.status = JobStatus.SUCCEEDED
        return job
