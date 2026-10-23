import pytest

from tinyjobs import storage
from tinyjobs.jobs import JobQueue, JobStatus

# Many small cases on purpose: `pytest -v` on this file prints a long log,
# which is what pack question q10 needs.
NAMES = [f"nightly-export-{n:03d}" for n in range(120)]


def explode(job):
    raise RuntimeError("disk full")


@pytest.mark.parametrize("name", NAMES)
def test_a_job_round_trips(tmp_path, name):
    queue = JobQueue()
    queue.submit(name)
    path = tmp_path / "queue.json"
    storage.save(queue, path)
    [job] = list(storage.load(path))
    assert job.name == name
    assert job.status is JobStatus.PENDING


def test_a_failure_round_trips(tmp_path):
    queue = JobQueue()
    queue.submit("backup")
    queue.run_next(explode)
    path = tmp_path / "queue.json"
    storage.save(queue, path)
    [job] = list(storage.load(path))
    assert (job.status, job.error) == (JobStatus.FAILED, "disk full")


def test_a_missing_file_is_an_empty_queue(tmp_path):
    assert list(storage.load(tmp_path / "absent.json")) == []


def test_save_leaves_no_temporary_file(tmp_path):
    path = tmp_path / "queue.json"
    storage.save(JobQueue(), path)
    assert [p.name for p in tmp_path.iterdir()] == ["queue.json"]


def test_queue_file_reads_the_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("TINYJOBS_QUEUE_FILE", str(tmp_path / "q.json"))
    assert storage.queue_file() == tmp_path / "q.json"
