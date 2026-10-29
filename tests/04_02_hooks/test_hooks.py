"""Checks that the project from the hooks lesson is in place.

Our checks for the hooks lesson: they verify the lesson, and the student
is not asked to run them. Run them with `uv run pytest`. They fail until
the lesson is done. None of them needs an API key or makes a request:
every run is on a stand-in model.

Two checks that pinned the chat's tool count (fourteen) moved to the memory
lesson (04-05), which adds five tools.

Each test runs in an empty folder of its own, so the project's own `.env`
and `work/` folder are never read. A test that needs the workspace folder
makes `work/tinyjobs` there, from the sample code.

Every import of the project is inside the test that needs it, so that
before the lesson each test fails on its own, instead of the whole file
failing to load.

The check that the log counts the tools actually sent moved to the deferred
lesson (06-02), where a hidden tool is in the request's definitions but is
not sent.

The check that the request log is the chat's last capability moved to the
approval lesson (05-01), whose guardrail declares itself innermost beside it.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic_ai import models

# This file is tests/04_02_hooks/test_hooks.py: two folders below the root.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
PREFIX = "PYDANTIC_TUTOR_"
KEY_VARIABLES = ("OPENAI_API_KEY", "ANTHROPIC_API_KEY")
LOGGER = "pydantic_tutor.capabilities.request_log"


@pytest.fixture(autouse=True)
def no_real_models(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for name in list(os.environ):
        if name.upper().startswith(PREFIX) or name.upper() in KEY_VARIABLES:
            monkeypatch.delenv(name)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(target=models, name="ALLOW_MODEL_REQUESTS", value=False)


@pytest.fixture
def project(tmp_path: Path) -> Path:
    """A stand-in project folder with a fresh `work/tinyjobs`, made by the script."""
    shutil.copytree(src=PROJECT_ROOT / "sample_code", dst=tmp_path / "sample_code")
    shutil.copy(src=PROJECT_ROOT / "pyproject.toml", dst=tmp_path / "pyproject.toml")
    made = run_script(name="fresh_copy.py", folder=tmp_path)
    assert made.returncode == 0, made.stderr
    return tmp_path


def run_script(*, name: str, folder: Path) -> subprocess.CompletedProcess[str]:
    env = {k: v for k, v in os.environ.items() if k not in KEY_VARIABLES}
    return subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "demos" / name)],
        capture_output=True,
        text=True,
        check=False,
        cwd=folder,
        env={
            **env,
            "PYTHONPATH": str(PROJECT_ROOT / "src"),
            "PYDANTIC_AI_NO_BANNER": "1",
        },
    )


def logged(*, caplog: pytest.LogCaptureFixture) -> list[str]:
    """The messages our request log wrote during the test."""
    return [record.getMessage() for record in caplog.records if record.name == LOGGER]


# Point 1: a hook that logs each request, as Hooks(...) and as a class.


# Point 2: where a hook stands decides what it sees; a module only writes.


def test_nothing_is_logged_until_the_program_turns_it_on(project: Path) -> None:
    # Earlier script that runs the chat's agent and configures no logging:
    # its output is what it was before the lesson.
    result = run_script(name="workspace_reach.py", folder=project)
    assert result.returncode == 0, result.stderr
    assert "request step=" not in result.stdout
    assert "request step=" not in result.stderr


def turn(*, step: int, logged: int, sent: int) -> list[str]:
    """The two lines one request prints: the log's count, then the model's."""
    return [
        f"{LOGGER}: request step={step} messages={logged} tools=2",
        f"  the model receives: messages={sent} tools=2",
    ]


def test_hook_forms_shows_why_the_position_matters(project: Path) -> None:
    result = run_script(name="hook_forms.py", folder=project)
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [
        "the log, as a class, before a hook that trims the history:",
        *turn(step=1, logged=1, sent=1),
        *turn(step=2, logged=3, sent=3),
        # It logs the request as it was before the trimmer shortened it.
        *turn(step=3, logged=5, sent=3),
        "the log, as a class, after it:",
        *turn(step=1, logged=1, sent=1),
        *turn(step=2, logged=3, sent=3),
        *turn(step=3, logged=3, sent=3),
    ]
