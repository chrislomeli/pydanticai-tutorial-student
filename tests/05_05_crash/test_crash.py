"""Checks that the project from the crash lesson is in place.

Our checks for the crash lesson: they verify the lesson, and the student
is not asked to run them. Run them with `uv run pytest`. They fail until
the lesson is done. None of them needs an API key or makes a request: every
run is on a stand-in model, and the person at the keyboard is a list of
typed lines. The crash is real: the first test kills a process with
SIGKILL in the middle of a tool, and takes about twenty seconds.

Each test runs in an empty folder of its own, so the project's own `.env`,
`work/` folder and `pydantic-tutor.db` are never touched.

Every import of the project is inside the test that needs it, so that
before the lesson each test fails on its own, instead of the whole file
failing to load.
"""

import dataclasses
import importlib
import os
import shutil
import signal
import sqlite3
import subprocess
import sys
from collections.abc import AsyncIterator
from contextlib import closing
from pathlib import Path
from types import ModuleType

import pytest
from pydantic_ai import models
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo, DeltaToolCalls, FunctionModel

# This file is tests/05_05_crash/test_crash.py: two folders below the root.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
PREFIX = "PYDANTIC_TUTOR_"
KEY_VARIABLES = ("OPENAI_API_KEY", "ANTHROPIC_API_KEY")
SESSION = "demos/crash_session.py"

# The banner goes to stderr; turning it off keeps a failure readable.
# Unbuffered, so a line is read as soon as it is printed.
CHILD_ENV = {**os.environ, "PYDANTIC_AI_NO_BANNER": "1", "PYTHONUNBUFFERED": "1"}


@pytest.fixture(autouse=True)
def no_real_models(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # None of this project's variables and no key, whatever the shell
    # running pytest has, and no real model call.
    for name in list(os.environ):
        if name.upper().startswith(PREFIX) or name.upper() in KEY_VARIABLES:
            monkeypatch.delenv(name)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(target=models, name="ALLOW_MODEL_REQUESTS", value=False)


@pytest.fixture
def project(tmp_path: Path) -> Path:
    """A stand-in project folder: the sample code and the scripts the lesson runs."""
    root = tmp_path / "project"
    shutil.copytree(src=PROJECT_ROOT / "sample_code", dst=root / "sample_code")
    (root / "demos").mkdir()
    for script in ("fresh_copy.py", "chat.py", "crash_session.py"):
        # Before the lesson there is no crash_session.py: the test that runs
        # it fails there, not every test that makes a copy.
        if (PROJECT_ROOT / "demos" / script).is_file():
            shutil.copy(src=PROJECT_ROOT / "demos" / script, dst=root / "demos")
    run_script(name="demos/fresh_copy.py", cwd=root)
    return root


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


def load_chat() -> ModuleType:
    """The chat loop, as a module, without running its main()."""
    return importlib.import_module("pydantic_tutor.chat.loop")


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


def scripted_model(*, fail_on: str | None = None) -> FunctionModel:
    """A model that answers `I can see: [every prompt so far]`, and raises on the prompt `fail_on`.

    It streams, as the chat needs.
    """

    def check(messages: list[ModelMessage]) -> None:
        if fail_on is not None and said(messages)[-1] == fail_on:
            raise UnexpectedModelBehavior("the model sent nonsense")

    def reply(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        check(messages)
        return echo(messages, info)

    async def stream(
        messages: list[ModelMessage], info: AgentInfo
    ) -> AsyncIterator[str | DeltaToolCalls]:
        check(messages)
        yield f"I can see: {said(messages)}"

    return FunctionModel(function=reply, stream_function=stream)


def open_chat(
    *,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    lines: list[str],
    fail_on: str | None = None,
) -> None:
    """Open the project's chat on a scripted model, saving and recording in tmp_path/c.db."""
    from pydantic_ai_harness.step_persistence import SqliteStepStore

    from pydantic_tutor.chat.agent import build_chat_agent
    from pydantic_tutor.composition import App
    from pydantic_tutor.stores.sqlite_conversations import SyncSqliteConversationStore

    workspace_root = tmp_path / "work" / "tinyjobs"
    if not workspace_root.exists():
        shutil.copytree(
            src=PROJECT_ROOT / "sample_code" / "tinyjobs", dst=workspace_root
        )
    monkeypatch.setattr("builtins.input", Keyboard(lines=lines))
    load_chat().main(
        app=App(
            agent=build_chat_agent(web_search=False, workspace_root=workspace_root),
            model=scripted_model(fail_on=fail_on),
            store=SyncSqliteConversationStore(path=tmp_path / "c.db"),
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
    # (no second approval of write_file), the one in flight does.
    run_script(name=SESSION, cwd=project, stdin="Write some notes\ny\n\n")
    crashed = kill_during(
        name=SESSION, cwd=project, stdin="Update the notes\ny\n", marker="building"
    )
    assert "Run it?" in crashed
    assert (project / "work" / "tinyjobs" / "NOTES.md").is_file()

    resumed = run_script(name=SESSION, cwd=project, stdin="/resume\n\n")
    assert ": 10 messages. Type /new to start another.\n" in resumed
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
    assert after.count("building...") == 1
    assert "Wrote NOTES.md; build passed." in after

    again = run_script(name=SESSION, cwd=project, stdin="\n")
    assert ": 20 messages. Type /new to start another.\n" in again
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
from pydantic_tutor.capabilities.recovery import recorder

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

    from pydantic_tutor.capabilities import recovery

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

    from pydantic_tutor.capabilities import recovery

    open_chat(monkeypatch=monkeypatch, tmp_path=tmp_path, lines=["one", "two"])
    assert {"chat_conversations", "runs", "events", "snapshots", "tool_effects"} <= (
        tables(path=tmp_path / "c.db")
    )
    with closing(sqlite3.connect(database=tmp_path / "c.db")) as connection:
        count = connection.execute("SELECT count(*) FROM runs").fetchone()[0]
    assert count == 2  # one record per run
    store = SqliteStepStore(database=tmp_path / "c.db")
    assert recovery.last_unfinished(store=store) is None
    capsys.readouterr()
    open_chat(monkeypatch=monkeypatch, tmp_path=tmp_path, lines=["three"])
    out = capsys.readouterr().out
    assert "did not finish" not in out
    assert "I can see: ['one', 'two', 'three']" in out


def test_a_turn_that_fails_is_not_a_crash(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # A failed run records that it failed: it ended, and the error was
    # raised at the time. Only a run with no ending is offered to /resume.
    from pydantic_ai_harness.step_persistence import SqliteStepStore

    from pydantic_tutor.capabilities import recovery

    with pytest.raises(UnexpectedModelBehavior):
        open_chat(
            monkeypatch=monkeypatch, tmp_path=tmp_path, lines=["one"], fail_on="one"
        )
    store = SqliteStepStore(database=tmp_path / "c.db")
    assert recovery.last_unfinished(store=store) is None
    capsys.readouterr()
    open_chat(monkeypatch=monkeypatch, tmp_path=tmp_path, lines=["/resume"])
    out = capsys.readouterr().out
    assert "did not finish" not in out
    assert "There is no unfinished run to carry on.\n" in out


def test_the_app_records_its_runs_in_its_database_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from pydantic_tutor.composition import build_app
    from pydantic_tutor.settings import Settings

    root = tmp_path / "work" / "tinyjobs"
    shutil.copytree(src=PROJECT_ROOT / "sample_code" / "tinyjobs", dst=root)
    monkeypatch.setenv("PYDANTIC_TUTOR_DATABASE", str(tmp_path / "x.db"))
    app = build_app(settings=Settings(chat_model="test", web_search=False))
    assert app.steps is not None
    assert not (tmp_path / "x.db").exists()  # made on first use, by the chat
    monkeypatch.setattr("builtins.input", Keyboard(lines=["hello"]))
    load_chat().main(app=dataclasses.replace(app, model=scripted_model()))
    assert {"chat_conversations", "runs", "events"} <= tables(path=tmp_path / "x.db")


def test_git_ignores_the_files_sqlite_keeps_beside_the_database() -> None:
    lines = (PROJECT_ROOT / ".gitignore").read_text().splitlines()
    assert "/pydantic-tutor.db" in lines
    assert "/pydantic-tutor.db-*" in lines
