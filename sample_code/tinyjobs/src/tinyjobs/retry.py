"""Retrying work that may fail.

A job is tried up to `max_attempts` times in all. The wait before each retry
starts at `base_delay` seconds and doubles.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass

from tinyjobs.jobs import Job

log = logging.getLogger(__name__)

DEFAULT_MAX_ATTEMPTS = 2


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = DEFAULT_MAX_ATTEMPTS
    base_delay: float = 0.5

    def delay_before(self, attempt: int) -> float:
        """Seconds to wait before `attempt`. The first retry is attempt 2."""
        return self.base_delay * 2 ** (attempt - 2)


def with_retries(
    work: Callable[[Job], None],
    *,
    policy: RetryPolicy,
    sleep: Callable[[float], None] = time.sleep,
) -> Callable[[Job], None]:
    """`work`, tried again on failure. The last failure is raised."""

    def attempt_work(job: Job) -> None:
        for attempt in range(1, policy.max_attempts + 1):
            try:
                work(job)
                return
            except Exception as error:
                if attempt == policy.max_attempts:
                    raise
                log.warning("job %d failed on attempt %d: %s", job.id, attempt, error)
                sleep(policy.delay_before(attempt + 1))

    return attempt_work
