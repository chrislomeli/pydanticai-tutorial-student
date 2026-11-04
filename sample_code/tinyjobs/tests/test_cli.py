from tinyjobs.cli import main


def run(tmp_path, *args):
    return main(["--queue-file", str(tmp_path / "queue.json"), *args])


def test_submit_then_list(tmp_path, capsys):
    assert run(tmp_path, "submit", "rebuild index") == 0
    assert run(tmp_path, "list") == 0
    out = capsys.readouterr().out.splitlines()
    assert out == ["1\tpending\trebuild index", "1\tpending\trebuild index"]


def test_run_respects_max(tmp_path, capsys):
    for name in ("a", "b", "c"):
        run(tmp_path, "submit", name)
    capsys.readouterr()
    assert run(tmp_path, "run", "--max", "2") == 0
    assert capsys.readouterr().out.splitlines() == ["1\tsucceeded\ta", "2\tsucceeded\tb"]


def test_a_failing_job_shows_its_error(tmp_path, capsys):
    run(tmp_path, "submit", "fail on purpose")
    capsys.readouterr()
    run(tmp_path, "run")
    assert capsys.readouterr().out == "1\tfailed\tfail on purpose\t(asked to fail)\n"


def test_status_of_an_unknown_job(tmp_path, capsys):
    assert run(tmp_path, "status", "7") == 2
    assert "no such job: 7" in capsys.readouterr().err


def test_work_retries_a_flaky_job(tmp_path, capsys):
    run(tmp_path, "submit", "flaky export")
    capsys.readouterr()
    assert run(tmp_path, "work", "--seconds", "0") == 0
    assert capsys.readouterr().out == "1\tsucceeded\tflaky export\n"


def test_report_counts_the_jobs(tmp_path, capsys):
    run(tmp_path, "submit", "a")
    capsys.readouterr()
    assert run(tmp_path, "report") == 0
    assert capsys.readouterr().out == "pending: 1\n"
