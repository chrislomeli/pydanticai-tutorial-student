"""Checks that the project from the lesson where a run survives a crash is in place.

Our checks for the crash lesson: they verify the lesson, and the student
is not asked to run them. Run them with `uv run pytest`. They fail until
the lesson is done. None of them needs an API key or makes a request:
every run is on a scripted model, and every line "typed" at the keyboard
is scripted too. The crash is real: two tests kill a process with
SIGKILL in the middle of a tool, and the first takes about fifteen
seconds.

None of them touches your own `work/` folder or `pydantic-tutor.db`: each
uses a folder of its own.

Every import of the project is inside the test that needs it, so that
before the lesson each test fails on its own, instead of the whole file
failing to load.
"""

import dataclasses
import os
import shutil
import signal
import sqlite3
import subprocess
import sys
from collections.abc import Callable
from contextlib import closing
from pathlib import Path

import pytest
from pydantic_ai import Agent, Conversation, models
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    TextPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.usage import RunUsage

# This file is tests/05_05_crash/test_crash.py: two folders below the root.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
TINYJOBS_ROOT: Path = PROJECT_ROOT / "sample_code" / "tinyjobs"
PREFIX = "PYDANTIC_TUTOR_"
SESSION = "scripts/crash_session.py"


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    # None of this project's variables and no key, whatever the shell
    # running pytest has, and no real model call.
    for name in list(os.environ):
        if name.upper().startswith(PREFIX) or name.upper() == "OPENAI_API_KEY":
            monkeypatch.delenv(name)
    monkeypatch.setattr(target=models, name="ALLOW_MODEL_REQUESTS", value=False)


# The banner goes to stderr; turning it off keeps a failure readable.
# Unbuffered, so a line is read as soon as it is printed.
CHILD_ENV = {**os.environ, "PYDANTIC_AI_NO_BANNER": "1", "PYTHONUNBUFFERED": "1"}


def run_script(*, name: str, cwd: Path, stdin: str = "") -> str:
    """Run one of the project's scripts from `cwd`, `stdin` piped in; return its stdout."""
    result = subprocess.run(
        args=[sys.executable, name],
        cwd=cwd,
        input=stdin,
        capture_output=True,
        text=True,
        check=False,
        env=CHILD_ENV,
    )
    assert result.returncode == 0, f"{name} failed:\n{result.stderr}"
    return result.stdout


def kill_during(*, name: str, cwd: Path, stdin: str, marker: str) -> str:
    """Run a script, and kill it with SIGKILL once it prints `marker`; return its stdout."""
    process = subprocess.Popen(
        args=[sys.executable, name],
        cwd=cwd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        env=CHILD_ENV,
    )
    assert process.stdin is not None
    assert process.stdout is not None
    process.stdin.write(stdin)
    process.stdin.flush()
    out = ""
    while marker not in out:
        char = process.stdout.read(1)
        assert char, f"{name} ended before printing {marker!r}:\n{out}"
        out += char
    os.kill(process.pid, signal.SIGKILL)
    assert process.wait() == -signal.SIGKILL
    return out


@pytest.fixture
def project(tmp_path: Path) -> Path:
    """A stand-in project folder: the sample codebase and the lesson's scripts, nothing else."""
    root = tmp_path / "project"
    shutil.copytree(src=TINYJOBS_ROOT, dst=root / "sample_code" / "tinyjobs")
    (root / "scripts").mkdir()
    for script in ("fresh_copy.py", "crash_session.py"):
        # Before the lesson there is no crash_session.py: the test that runs
        # it fails there, not the fixture.
        if (PROJECT_ROOT / "scripts" / script).is_file():
            shutil.copy(src=PROJECT_ROOT / "scripts" / script, dst=root / "scripts")
    run_script(name="scripts/fresh_copy.py", cwd=root)
    return root


def said(messages: list[ModelMessage]) -> list[str]:
    """Every prompt typed so far, in order."""
    return [
        str(part.content)
        for message in messages
        for part in message.parts
        if isinstance(part, UserPromptPart)
    ]


def echo(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """A scripted model: says what it has been told so far."""
    return ModelResponse(parts=[TextPart(content=f"I can see: {said(messages)}")])


def tables(*, path: Path) -> set[str]:
    with closing(sqlite3.connect(database=path)) as connection:
        return {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }


class Keyboard:
    """Scripted lines in place of `input`; once they run out, input has ended."""

    def __init__(self, *, lines: list[str]) -> None:
        self.lines = list(lines)

    def __call__(self, prompt: object = "") -> str:
        if not self.lines:
            raise EOFError
        return self.lines.pop(0)


def open_chat(
    *,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    lines: list[str],
    function: Callable[[list[ModelMessage], AgentInfo], ModelResponse] = echo,
) -> None:
    """Open the student's chat on a scripted model, saving and recording in tmp_path/c.db."""
    from pydantic_ai_harness.step_persistence import SqliteStepStore

    from pydantic_tutor.agent import agent
    from pydantic_tutor.chat import chat
    from pydantic_tutor.codebase_tools import CodebaseDeps
    from pydantic_tutor.composition import App
    from pydantic_tutor.conversations import ConversationStore

    monkeypatch.setattr("builtins.input", Keyboard(lines=lines))
    chat(
        app=App(
            agent=agent,
            model=FunctionModel(function=function, model_name="stand-in"),
            deps=CodebaseDeps(root=TINYJOBS_ROOT),
            store=ConversationStore(path=tmp_path / "c.db"),
            steps=SqliteStepStore(database=tmp_path / "c.db"),
        )
    )


def test_a_crashed_run_is_carried_on_without_running_its_finished_tools_again(
    project: Path,
) -> None:
    # The lesson's point. A turn finishes; the next is killed with
    # SIGKILL while its last tool runs. A new process says which tools
    # finished and which may or may not have, and /resume carries the run
    # on from its last settled step: the finished tools do not run again
    # (no second approval of write_file), the one in flight does, and the
    # saved conversation's usage counts on.
    run_script(name=SESSION, cwd=project, stdin="Write some notes\ny\n/exit\n")
    crashed = kill_during(
        name=SESSION, cwd=project, stdin="Update the notes\ny\n", marker="building"
    )
    assert "ran write_file" in crashed
    assert (project / "work" / "tinyjobs" / "NOTES.md").is_file()

    resumed = run_script(name=SESSION, cwd=project, stdin="/resume\n/exit\n")
    assert ": 10 messages, usage.requests=5. Type /new to start another.\n" in resumed
    assert "The last run did not finish: the process stopped during it.\n" in resumed
    assert "  Finished: list_modules, read_docstring, write_file\n" in resumed
    assert (
        "  Finished, but its return was lost (the model may ask for it again): none\n"
        in resumed
    )
    assert (
        "  Started, not finished (may or may not have done its work): wait_for_build\n"
        in resumed
    )
    after = resumed.split("or anything else to leave it.\n")[1]
    assert "Run it?" not in after
    assert after.count("  ran ") == 1
    assert "  ran wait_for_build\n" in after
    assert "Wrote NOTES.md; build passed.\n" in after

    again = run_script(name=SESSION, cwd=project, stdin="/exit\n")
    assert ": 20 messages, usage.requests=7. Type /new to start another.\n" in again
    assert "did not finish" not in again


# A run of the student's recorder in its own process, killed with SIGKILL
# while the second of two tools asked for in one response waits: the first
# completes, but its round never settles. argv: the database file.
PARALLEL = """
import asyncio, os, signal, sys
from pydantic_ai import Agent
from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai_harness.step_persistence import SqliteStepStore
from pydantic_tutor.recovery import recorder

async def quick() -> str:
    return "done"

async def dies() -> str:
    await asyncio.sleep(0.5)
    os.kill(os.getpid(), signal.SIGKILL)
    return "never"

def both(messages, info):
    return ModelResponse(
        parts=[ToolCallPart(tool_name="quick", args={}), ToolCallPart(tool_name="dies", args={})]
    )

Agent(FunctionModel(both), tools=[quick, dies]).run_sync(
    "go", capabilities=[recorder(store=SqliteStepStore(database=sys.argv[1]))]
)
"""


def test_a_tool_finished_in_a_round_the_crash_cut_short_is_not_counted_saved(
    tmp_path: Path,
) -> None:
    # Returns reach a snapshot a round at a time. A tool that completed in
    # a round the process didn't finish did its work, but the model never
    # sees its return when the run is carried on: it is `unsaved`, not
    # `finished`.
    from pydantic_ai_harness.step_persistence import SqliteStepStore

    from pydantic_tutor import recovery

    script = tmp_path / "parallel.py"
    script.write_text(PARALLEL)
    process = subprocess.run(
        args=[sys.executable, str(script), str(tmp_path / "p.db")],
        capture_output=True,
        check=False,
        env=CHILD_ENV,
    )
    assert process.returncode == -signal.SIGKILL, process.stderr
    found = recovery.last_unfinished(store=SqliteStepStore(database=tmp_path / "p.db"))
    assert found is not None
    assert found.finished == ()
    assert found.unsaved == ("quick",)
    assert found.unknown == ("dies",)


# The lesson's module is imported as a name of the package in the tests
# below: before the lesson it does not exist, and Ruff would sort a `from`
# import of it apart from the rest of its block (I001).


def test_the_chat_records_each_run_beside_the_conversations(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from pydantic_ai_harness.step_persistence import SqliteStepStore

    from pydantic_tutor import recovery

    open_chat(monkeypatch=monkeypatch, tmp_path=tmp_path, lines=["one", "two"])
    assert {"chat_conversations", "runs", "events", "snapshots", "tool_effects"} <= (
        tables(path=tmp_path / "c.db")
    )
    store = SqliteStepStore(database=tmp_path / "c.db")
    assert recovery.last_unfinished(store=store) is None
    open_chat(monkeypatch=monkeypatch, tmp_path=tmp_path, lines=["three"])
    out = capsys.readouterr().out
    assert "did not finish" not in out
    assert "I can see: ['one', 'two', 'three']" in out


def test_a_turn_that_fails_is_not_a_crash(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # A failed run records that it failed: it ended, and the chat said so
    # at the time. Only a run with no ending is offered to /resume.
    from pydantic_ai_harness.step_persistence import SqliteStepStore

    from pydantic_tutor import recovery

    def fails(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        raise UnexpectedModelBehavior("the model sent nonsense")

    open_chat(monkeypatch=monkeypatch, tmp_path=tmp_path, lines=["one"], function=fails)
    store = SqliteStepStore(database=tmp_path / "c.db")
    assert recovery.last_unfinished(store=store) is None
    open_chat(monkeypatch=monkeypatch, tmp_path=tmp_path, lines=["/resume"])
    out = capsys.readouterr().out
    assert "did not finish" not in out
    assert "There is no unfinished run to carry on.\n" in out


def test_resume_point_carries_the_usage_and_revision_of_the_same_conversation() -> None:
    from pydantic_tutor import chat, recovery

    first = Agent(model=FunctionModel(function=echo)).run_sync(user_prompt="one")
    carried = Conversation(
        conversation_id=first.conversation.conversation_id,
        messages=first.all_messages(),
        usage=RunUsage(requests=5),
    )
    unfinished = recovery.Unfinished(
        conversation_id=carried.conversation_id,
        finished=("list_modules",),
        unsaved=(),
        unknown=("wait_for_build",),
        messages=first.all_messages(),
    )
    start, revision = chat.resume_point(
        unfinished=unfinished, conversation=carried, revision=3
    )
    assert start.conversation_id == carried.conversation_id
    assert start.usage.requests == 5
    assert start.messages == first.all_messages()
    assert revision == 3
    # A conversation the chat does not carry: never saved, so from zero.
    other = Agent(model=FunctionModel(function=echo)).run_sync(user_prompt="two")
    start, revision = chat.resume_point(
        unfinished=unfinished, conversation=other.conversation, revision=3
    )
    assert start.conversation_id == unfinished.conversation_id
    assert start.usage.requests == 0
    assert revision == 0


def test_the_app_records_its_runs_in_its_database_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from pydantic_tutor.chat import chat
    from pydantic_tutor.composition import build_app
    from pydantic_tutor.settings import Settings

    monkeypatch.setenv("PYDANTIC_TUTOR_DATABASE", str(tmp_path / "x.db"))
    app = build_app(settings=Settings(chat_model="test", codebase_root=TINYJOBS_ROOT))
    assert app.steps is not None
    assert not (tmp_path / "x.db").exists()  # made on first use, by the chat
    monkeypatch.setattr("builtins.input", Keyboard(lines=["hello"]))
    chat(app=dataclasses.replace(app, model=FunctionModel(function=echo)))
    assert {"chat_conversations", "runs", "events"} <= tables(path=tmp_path / "x.db")


def test_git_ignores_the_files_sqlite_keeps_beside_the_database() -> None:
    lines = (PROJECT_ROOT / ".gitignore").read_text().splitlines()
    assert "/pydantic-tutor.db" in lines
    assert "/pydantic-tutor.db-*" in lines
