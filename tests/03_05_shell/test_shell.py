"""Checks that the project from the shell lesson is in place.

Our checks for the shell lesson: they verify the lesson, and the student
is not asked to run them. Run them with `uv run pytest`. They fail until
the lesson is done. None of them needs an API key or makes a request:
every run is on a scripted model.

None of them touches your own `work/` folder: each makes a working copy of
its own, with your `scripts/fresh_copy.py`, in a folder of its own.

One test was moved here from the file tools lesson (03-04), which pinned
the agent's tools before the shell was added. It says so.

The memory lesson (04-05) moved that test on: it adds five memory tools.
It is now `test_the_model_is_sent_our_signatures_beside_the_file_shell_and_memory_tools`,
in `tests/04_05_memory/test_memory.py`.

Six tests (ten cases) were moved on by the approval lesson (05-01), which
holds every command for a person's approval, and refuses it when no one
answers. What the shell runs, refuses and lets through is now proved with
each command approved, in `tests/05_01_approval/test_approval.py`, under
the same names with `approved` in them
(`test_an_approved_command_runs_the_working_copys_tests` and the rest).
`scripts/shell_commands.py` approves its own commands from that lesson
on, so the test of what it prints stays here.

Every import of the project is inside the test that needs it, so that
before the lesson each test fails on its own, instead of the whole file
failing to load.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic_ai import models

# This file is tests/03_05_shell/test_shell.py: two folders below the root.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
TINYJOBS_ROOT: Path = PROJECT_ROOT / "sample_code" / "tinyjobs"
FAKE_TOKEN = "tj-FAKE-0000-not-a-real-secret"
PREFIX = "PYDANTIC_TUTOR_"
FILE_TOOLS = [
    "create_directory",
    "edit_file",
    "file_info",
    "find_files",
    "list_directory",
    "read_file",
    "search_files",
    "write_file",
]
SHELL_TOOLS = ["check_command", "run_command", "start_command", "stop_command"]


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    # None of this project's variables and no key, whatever the shell
    # running pytest has, and no real model call.
    for name in list(os.environ):
        if name.upper().startswith(PREFIX) or name.upper() == "OPENAI_API_KEY":
            monkeypatch.delenv(name)
    monkeypatch.setattr(target=models, name="ALLOW_MODEL_REQUESTS", value=False)
    # The shell finds `pytest` on PATH, as it does under `uv run`: put this
    # project's environment first, however pytest itself was started.
    monkeypatch.setenv(
        "PATH", f"{Path(sys.executable).parent}{os.pathsep}{os.environ['PATH']}"
    )


def run_script(*, name: str, cwd: Path) -> str:
    """Run one of the lesson's scripts from `cwd`; return its stdout."""
    result = subprocess.run(
        args=[sys.executable, name],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
        # The banner goes to stderr; turning it off keeps a failure readable.
        env={**os.environ, "PYDANTIC_AI_NO_BANNER": "1"},
    )
    assert result.returncode == 0, f"{name} failed:\n{result.stderr}"
    return result.stdout


@pytest.fixture
def project(tmp_path: Path) -> Path:
    """A stand-in project folder: the sample codebase and the lesson's scripts, nothing else."""
    root = tmp_path / "project"
    shutil.copytree(src=TINYJOBS_ROOT, dst=root / "sample_code" / "tinyjobs")
    (root / "scripts").mkdir()
    for script in ("fresh_copy.py", "shell_commands.py"):
        # Before the lesson there is no shell_commands.py: the test that runs
        # it fails there, not every test that makes a copy.
        if (PROJECT_ROOT / "scripts" / script).is_file():
            shutil.copy(src=PROJECT_ROOT / "scripts" / script, dst=root / "scripts")
    # The first line `head -n 1 ../../pyproject.toml` reads.
    (root / "pyproject.toml").write_text('[project]\nname = "stand-in"\n')
    return root


@pytest.fixture
def working_copy(project: Path) -> Path:
    """A working copy, made by the student's `scripts/fresh_copy.py`."""
    run_script(name="scripts/fresh_copy.py", cwd=project)
    return project / "work" / "tinyjobs"


# Goal 1: the agent has a shell, and runs the working copy's tests with it;
# goal 2: the allow list is a guard against mistakes, not a boundary. The
# script shows both.


def test_shell_commands_prints_what_runs_and_is_refused(
    working_copy: Path, project: Path
) -> None:
    printed = run_script(name="scripts/shell_commands.py", cwd=project)
    sections = printed.split("$ ")[1:]
    commands = [section.splitlines()[0] for section in sections]
    assert commands == [
        "pytest -q --tb=no",
        "rm README.md",
        'sh -c "rm README.md && echo removed"',
        "cat .env",
        "head -n 1 ../../pyproject.toml",
        "pytest --version && cat .env",
    ]
    assert "1 failed, 138 passed" in sections[0]
    for section, name in zip(sections[1:5], ["rm", "sh", "cat", "head"], strict=True):
        assert f"Command '{name}' is not in the allowed list." in section
    assert FAKE_TOKEN in sections[5]  # the gap, shown
    assert (working_copy / "README.md").is_file()
