import pytest

from tinyjobs.jobs import JobQueue, JobStatus, UnknownJob


def succeed(job):
    pass


def explode(job):
    raise RuntimeError("disk full")


def test_submit_assigns_increasing_ids():
    queue = JobQueue()
    first = queue.submit("a")
    second = queue.submit("b")
    assert (first.id, second.id) == (1, 2)
    assert first.status is JobStatus.PENDING


def test_submit_refuses_a_blank_name():
    with pytest.raises(ValueError):
        JobQueue().submit("   ")


def test_run_next_runs_the_oldest_pending_job():
    queue = JobQueue()
    queue.submit("first")
    queue.submit("second")
    job = queue.run_next(succeed)
    assert job is not None
    assert job.name == "first"
    assert job.status is JobStatus.SUCCEEDED


def test_run_next_records_a_failure_instead_of_raising():
    queue = JobQueue()
    queue.submit("backup")
    job = queue.run_next(explode)
    assert job is not None
    assert job.status is JobStatus.FAILED
    assert job.error == "disk full"


def test_run_next_returns_none_when_nothing_waits():
    assert JobQueue().run_next(succeed) is None


def test_cancel_a_pending_job():
    queue = JobQueue()
    job = queue.submit("report")
    assert queue.cancel(job.id).status is JobStatus.CANCELLED


def test_cancel():
    """Cancelling a finished job does nothing: its status never changes."""
    queue = JobQueue()
    job = queue.submit("report")
    queue.run_next(succeed)
    queue.cancel(job.id)
    assert queue.get(job.id).status is JobStatus.SUCCEEDED


def test_cancel_an_unknown_job():
    with pytest.raises(UnknownJob):
        JobQueue().cancel(99)


def test_forget_removes_a_finished_job():
    queue = JobQueue()
    job = queue.submit("report")
    queue.run_next(succeed)
    queue.forget(job.id)
    with pytest.raises(UnknownJob):
        queue.get(job.id)


def test_forget_refuses_a_pending_job():
    queue = JobQueue()
    job = queue.submit("report")
    with pytest.raises(RuntimeError):
        queue.forget(job.id)


def test_ids_continue_after_reload():
    queue = JobQueue()
    queue.submit("a")
    queue.submit("b")
    reloaded = JobQueue(list(queue))
    assert reloaded.submit("c").id == 3
