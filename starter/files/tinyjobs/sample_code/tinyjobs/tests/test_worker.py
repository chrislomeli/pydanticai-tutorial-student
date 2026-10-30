from tinyjobs.jobs import JobQueue, JobStatus
from tinyjobs.retry import RetryPolicy
from tinyjobs.worker import Worker


def worker_for(queue, waits):
    return Worker(queue, seconds_per_job=1.5, policy=RetryPolicy(max_attempts=2), sleep=waits.append)


def test_each_job_takes_its_time():
    queue, waits = JobQueue(), []
    queue.submit("a")
    worker_for(queue, waits).run()
    assert waits == [1.5]


def test_a_flaky_job_is_saved_by_a_retry():
    queue, waits = JobQueue(), []
    job = queue.submit("flaky export")
    worker_for(queue, waits).run()
    assert job.status is JobStatus.SUCCEEDED
    assert waits == [1.5, 0.5, 1.5]  # attempt, backoff, attempt


def test_a_failing_job_fails_after_its_attempts():
    queue, waits = JobQueue(), []
    job = queue.submit("fail on purpose")
    worker_for(queue, waits).run()
    assert (job.status, job.error) == (JobStatus.FAILED, "asked to fail")


def test_the_limit_stops_the_run():
    queue = JobQueue()
    for name in ("a", "b", "c"):
        queue.submit(name)
    ran = worker_for(queue, []).run(limit=2)
    assert [job.name for job in ran] == ["a", "b"]
