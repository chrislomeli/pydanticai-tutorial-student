"""Checks that the project from the chat-loop lesson is in place.

Our checks for the chat-loop lesson: they verify the lesson, and the student
is not asked to run them. Run them with `uv run pytest`. They fail until
the lesson is done. None of them needs an API key or makes a request: every
run is on a stand-in model, and the person at the keyboard is a list of
typed lines.

Each test runs in an empty folder of its own, so the project's own `.env`
and `work/` folder are never read. A test that needs the workspace folder
makes `work/tinyjobs` there, from the sample code.

Every import of the project is inside the test that needs it, so that
before the lesson each test fails on its own, instead of the whole file
failing to load.
"""

import importlib
import os
import shutil
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from typing import cast

import pytest
from pydantic_ai import RunContext, models
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.tools import DeferredToolRequests, ToolApproved, ToolDenied

# This file is tests/05_02_chat_loop/test_chat_loop.py: two folders below the root.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
PREFIX = "PYDANTIC_TUTOR_"
KEY_VARIABLES = ("OPENAI_API_KEY", "ANTHROPIC_API_KEY")
# The keyboard answerer never reads the run's context.
NO_CONTEXT = cast(RunContext[None], None)


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
def workspace(tmp_path: Path) -> Path:
    """`work/tinyjobs` in the empty folder, made from the sample code."""
    root = tmp_path / "work" / "tinyjobs"
    shutil.copytree(src=PROJECT_ROOT / "sample_code" / "tinyjobs", dst=root)
    return root


def type_lines(*, monkeypatch: pytest.MonkeyPatch, lines: list[str]) -> list[str]:
    """Answer `input()` from `lines`, in order; the prompts shown are returned.

    Running out of lines is the end of input.
    """
    typed: Iterator[str] = iter(lines)
    shown: list[str] = []

    def fake_input(prompt: str = "") -> str:
        shown.append(prompt)
        try:
            return next(typed)
        except StopIteration:
            raise EOFError from None

    monkeypatch.setattr("builtins.input", fake_input)
    return shown


def load_chat() -> ModuleType:
    """The chat loop, as a module, without running its main()."""
    return importlib.import_module("pydantic_tutor.chat.loop")


def chat_on_the_stand_in(
    *, monkeypatch: pytest.MonkeyPatch, workspace: Path, lines: list[str]
) -> list[str]:
    """Run the project's chat on the approval lesson's stand-in model.

    `lines` are what the person types, the chat's questions and the answers
    to held calls alike. Returns the prompts `input()` showed.
    """
    from pydantic_tutor.chat.agent import build_chat_agent
    from pydantic_tutor.composition import App
    from tutor_given.stand_ins import asks_for_a_test_and_its_run

    chat = load_chat()
    app = App(
        agent=build_chat_agent(web_search=False, workspace_root=workspace),
        model=asks_for_a_test_and_its_run(),
    )
    monkeypatch.setattr(chat, "build_app", lambda *, settings: app)
    shown = type_lines(monkeypatch=monkeypatch, lines=lines)
    chat.main()
    return shown


def held(*, names: list[str]) -> DeferredToolRequests:
    return DeferredToolRequests(
        approvals=[
            ToolCallPart(tool_name=name, args={"command": "x"}, tool_call_id=name)
            for name in names
        ]
    )


def test_only_y_approves_a_held_call(monkeypatch: pytest.MonkeyPatch) -> None:
    from pydantic_tutor.capabilities.approval import at_the_keyboard

    type_lines(monkeypatch=monkeypatch, lines=["y", "not now", ""])
    results = at_the_keyboard(
        ctx=NO_CONTEXT,
        requests=held(names=["write_file", "run_command", "edit_file"]),
    )

    assert results.approvals["write_file"] == ToolApproved()
    assert results.approvals["run_command"] == ToolDenied(
        message="Refused at the keyboard: not now"
    )
    assert results.approvals["edit_file"] == ToolDenied(
        message="Refused at the keyboard."
    )


def test_the_end_of_input_refuses(monkeypatch: pytest.MonkeyPatch) -> None:
    from pydantic_tutor.capabilities.approval import at_the_keyboard

    type_lines(monkeypatch=monkeypatch, lines=[])
    results = at_the_keyboard(
        ctx=NO_CONTEXT,
        requests=held(names=["run_command"]),
    )

    assert results.approvals["run_command"] == ToolDenied(
        message="Refused at the keyboard."
    )


def test_the_chat_shows_each_held_call_and_waits_for_you(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    workspace: Path,
) -> None:
    shown = chat_on_the_stand_in(
        monkeypatch=monkeypatch,
        workspace=workspace,
        lines=["Add a --verbose flag.", "y", "not now", ""],
    )
    out = capsys.readouterr().out

    # You approved the write, so it ran; you refused the command, and the
    # model was told your reason.
    assert (workspace / "tests" / "test_verbose.py").exists()
    assert "model got: write_file, success" in out
    assert "model got: run_command, denied: Refused at the keyboard: not now" in out
    assert "  write_file {" in out
    assert "  run_command {" in out
    assert shown.count("  Run it? y, or say why not: ") == 2
    assert "agent> Done." in out


def test_a_call_you_do_not_approve_does_not_run(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    workspace: Path,
) -> None:
    chat_on_the_stand_in(
        monkeypatch=monkeypatch,
        workspace=workspace,
        lines=["Add a --verbose flag.", "no", "no", ""],
    )

    assert not (workspace / "tests" / "test_verbose.py").exists()
    assert "write_file, denied: Refused at the keyboard: no" in capsys.readouterr().out


def test_every_run_of_the_chat_asks_you(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    workspace: Path,
) -> None:
    shown = chat_on_the_stand_in(
        monkeypatch=monkeypatch,
        workspace=workspace,
        lines=["First.", "y", "y", "Second.", "n", "n", ""],
    )

    assert shown.count("  Run it? y, or say why not: ") == 4
    assert capsys.readouterr().out.count("agent> Done.") == 2


def test_outside_the_chat_the_agent_still_refuses(workspace: Path) -> None:
    from pydantic_tutor.chat.agent import build_chat_agent
    from tutor_given.stand_ins import asks_for_a_test_and_its_run

    chat = build_chat_agent(web_search=False, workspace_root=workspace)
    chat.run_sync(user_prompt="x", model=asks_for_a_test_and_its_run())

    # No answerer was given to this run, so the agent's own answered: no.
    assert not (workspace / "tests" / "test_verbose.py").exists()


def test_the_given_session_opens_the_chat_and_waits_for_you(workspace: Path) -> None:
    env = {k: v for k, v in os.environ.items() if k not in KEY_VARIABLES}
    # Piped input: the question, your yes, your refusal, then the end.
    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "demos" / "chat_session.py")],
        input="Add a flag.\ny\nno\n\n",
        capture_output=True,
        text=True,
        check=False,
        cwd=workspace.parents[1],
        env={
            **env,
            "PYTHONPATH": str(PROJECT_ROOT / "src"),
            "PYDANTIC_AI_NO_BANNER": "1",
        },
    )

    assert result.returncode == 0, result.stderr
    assert "model got: write_file, success" in result.stdout
    assert (
        "model got: run_command, denied: Refused at the keyboard: no" in result.stdout
    )
