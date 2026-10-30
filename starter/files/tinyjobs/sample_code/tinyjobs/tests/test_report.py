from tinyjobs.jobs import JobQueue
from tinyjobs.report import format_report


def explode(job):
    raise RuntimeError("disk full")


def test_an_empty_queue():
    assert format_report(JobQueue()) == "no jobs"


def test_counts_and_failures():
    queue = JobQueue()
    for name in ("a", "b", "c"):
        queue.submit(name)
    queue.run_next(lambda job: None)
    queue.run_next(explode)
    assert format_report(queue).splitlines() == [
        "pending: 1",
        "succeeded: 1",
        "failed: 1",
        "  2 b: disk full",
    ]
