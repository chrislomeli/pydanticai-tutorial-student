"""A worker that runs the pending jobs, one at a time, and takes its time."""

from __future__ import annotations

import time
from collections.abc import Callable

from tinyjobs.jobs import Job, JobQueue
from tinyjobs.retry import RetryPolicy, with_retries


class Worker:
    """Runs a queue's pending jobs, oldest first, each after `seconds_per_job`.

    A job whose name starts with 'fail' always fails. One whose name starts
    with 'flaky' fails on its first attempt only, so a retry saves it.
    """

    def __init__(
        self,
        queue: JobQueue,
        *,
        seconds_per_job: float,
        policy: RetryPolicy,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._queue = queue
        self._seconds = seconds_per_job
        self._sleep = sleep
        self._tried: set[int] = set()
        self._work = with_retries(self._do, policy=policy, sleep=sleep)

    def _do(self, job: Job) -> None:
        self._sleep(self._seconds)
        first_attempt = job.id not in self._tried
        self._tried.add(job.id)
        if job.name.startswith("fail"):
            raise RuntimeError("asked to fail")
        if job.name.startswith("flaky") and first_attempt:
            raise RuntimeError("flaked once")

    def run(self, limit: int | None = None) -> list[Job]:
        """Run pending jobs, at most `limit` of them. Returns the jobs it ran."""
        ran: list[Job] = []
        while limit is None or len(ran) < limit:
            job = self._queue.run_next(self._work)
            if job is None:
                break
            ran.append(job)
        return ran
