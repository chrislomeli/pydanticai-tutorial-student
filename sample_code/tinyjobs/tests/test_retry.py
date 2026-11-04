import pytest

from tinyjobs.jobs import Job
from tinyjobs.retry import RetryPolicy, with_retries


def test_the_wait_doubles_from_the_first_retry():
    policy = RetryPolicy(base_delay=0.5)
    assert [policy.delay_before(n) for n in (2, 3, 4)] == [0.5, 1.0, 2.0]


def test_work_that_succeeds_is_not_retried():
    calls = []
    with_retries(calls.append, policy=RetryPolicy(), sleep=lambda s: None)(Job(1, "a"))
    assert len(calls) == 1


def test_a_failure_is_retried_after_waiting():
    waits, calls = [], []

    def flaky(job):
        calls.append(job)
        if len(calls) == 1:
            raise RuntimeError("once")

    with_retries(flaky, policy=RetryPolicy(max_attempts=3), sleep=waits.append)(Job(1, "a"))
    assert (len(calls), waits) == (2, [0.5])


def test_the_last_failure_is_raised():
    def explode(job):
        raise RuntimeError("always")

    with pytest.raises(RuntimeError, match="always"):
        with_retries(explode, policy=RetryPolicy(max_attempts=2), sleep=lambda s: None)(Job(1, "a"))
