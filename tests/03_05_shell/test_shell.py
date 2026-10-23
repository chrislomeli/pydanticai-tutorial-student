"""Checks that the project from the shell lesson is in place.

Our checks for the shell lesson: they verify the lesson, and the student
is not asked to run them. Run them with `uv run pytest`. They fail until
the lesson is done. None of them needs an API key or makes a request:
every run is on a scripted model.

None of them touches your own `work/` folder: each makes a working copy of
its own, with your `scripts/fresh_copy.py`, in a folder of its own.

Three of them check that the shell's gaps are still there: that it checks
only a command's first word, runs code the agent wrote, and reaches past
the working copy. They pass today because the gaps exist. If a release of
the library closes one, its test fails, and the lesson must change.

One test was moved here from the file tools lesson (03-04), which pinned
the agent's tools before the shell was added. It says so.

The memory lesson (04-05) moved that test on: it adds five memory tools.
It is now `test_the_model_is_sent_our_signatures_beside_the_file_shell_and_memory_tools`,
in `tests/04_05_memory/test_memory.py`.

Every import of the project is inside the test that needs it, so that
before the lesson each test fails on its own, instead of the whole file
failing to load.
"""

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest
from pydantic_ai import models
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

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


@dataclass(frozen=True)
class ToolResult:
    """What one tool call gave the model."""

    refused: bool  # True: a retry prompt, asking the model to try again
    text: str


def run_calls(*, root: Path, calls: list[ToolCallPart]) -> list[ToolResult]:
    """Run the student's agent on `root`, with a model that makes `calls`, one per request."""
    from pydantic_tutor.agent import agent
    from pydantic_tutor.codebase_tools import CodebaseDeps

    queue = list(calls)

    def reply(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if queue:
            return ModelResponse(parts=[queue.pop(0)])
        return ModelResponse(parts=[TextPart(content="done")])

    result = agent.run_sync(
        user_prompt="x",
        model=FunctionModel(function=reply),
        deps=CodebaseDeps(root=root),
    )
    results: list[ToolResult] = []
    for message in result.all_messages():
        if not isinstance(message, ModelRequest):
            continue
        for part in message.parts:
            if isinstance(part, ToolReturnPart):
                results.append(ToolResult(refused=False, text=str(part.content)))
            if isinstance(part, RetryPromptPart):
                results.append(ToolResult(refused=True, text=str(part.content)))
    assert len(results) == len(calls), "a tool was never called"
    return results


def run_command(*, root: Path, command: str) -> ToolResult:
    """Run one shell command through the student's agent; what the model got back."""
    call = ToolCallPart(tool_name="run_command", args={"command": command})
    return run_calls(root=root, calls=[call])[0]


# Goal 1: the agent has a shell, and runs the working copy's tests with it.


def test_the_shell_runs_the_working_copys_tests(working_copy: Path) -> None:
    result = run_command(root=working_copy, command="pytest -q --tb=no")
    assert not result.refused, result.text
    # Found and run in the copy: its one planted failure, and the rest.
    assert "FAILED tests/test_jobs.py::test_cancel" in result.text
    assert "1 failed, 138 passed" in result.text
    assert "[exit code: 1]" in result.text


@pytest.mark.parametrize(
    argnames="command",
    argvalues=[
        "rm README.md",
        'sh -c "rm README.md"',
        "cat .env",
        "python -c \"print(open('.env').read())\"",
        "uv run pytest",
    ],
)
def test_the_shell_starts_only_pytest(working_copy: Path, command: str) -> None:
    result = run_command(root=working_copy, command=command)
    assert result.refused, f"{command!r} ran"
    assert "is not in the allowed list" in result.text
    assert (working_copy / "README.md").is_file()
    assert FAKE_TOKEN not in result.text


# Goal 2: the allow list is a guard against mistakes, not a boundary. These
# three pass because the gaps are there; if one fails, the library closed it.


def test_only_the_first_word_is_checked(working_copy: Path) -> None:
    result = run_command(root=working_copy, command="pytest --version && cat .env")
    assert not result.refused, result.text
    assert FAKE_TOKEN in result.text


def test_pytest_runs_code_the_agent_wrote(working_copy: Path) -> None:
    peek = 'def test_peek() -> None:\n    print(open(".env").read())\n'
    wrote, ran = run_calls(
        root=working_copy,
        calls=[
            ToolCallPart(
                tool_name="write_file",
                args={"path": "tests/test_peek.py", "content": peek},
            ),
            ToolCallPart(
                tool_name="run_command",
                args={"command": "pytest -q -s tests/test_peek.py"},
            ),
        ],
    )
    assert not wrote.refused, wrote.text
    assert not ran.refused, ran.text
    assert FAKE_TOKEN in ran.text


def test_the_shell_reaches_past_the_working_copy(working_copy: Path) -> None:
    # The file tools refuse this path (03-04); the shell does not.
    (working_copy.parent / "outside.txt").write_text("outside the working copy\n")
    result = run_command(
        root=working_copy, command="pytest --version && cat ../outside.txt"
    )
    assert not result.refused, result.text
    assert "outside the working copy" in result.text


def test_a_key_in_the_apps_environment_does_not_reach_commands(
    working_copy: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A guard the workspace meets: a local workspace passes on only PATH,
    # HOME and the locale variables.
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-placeholder")
    result = run_command(root=working_copy, command="pytest --version && env")
    assert not result.refused, result.text
    assert "PATH=" in result.text
    assert "sk-test-placeholder" not in result.text


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
