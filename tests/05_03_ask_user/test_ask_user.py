"""Checks that the project from the lesson where the model asks you a question is in place.

Our checks for the ask-user lesson: they verify the lesson, and the
student is not asked to run them. Run them with `uv run pytest`. They fail
until the lesson is done. None of them needs an API key or makes a
request: every run is on a scripted model, and every line "typed" at the
keyboard is scripted too.

None of them touches your own `work/` folder: each makes a working copy of
its own, with your `scripts/fresh_copy.py`, in a folder of its own.

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
from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai_harness.ask_user import DECLINED, TOOL_NAME

# This file is tests/05_03_ask_user/test_ask_user.py: two folders below the root.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
TINYJOBS_ROOT: Path = PROJECT_ROOT / "sample_code" / "tinyjobs"
PREFIX = "PYDANTIC_TUTOR_"
OPTIONS = [
    {"label": "--verbose", "description": "print each job as it runs"},
    {"label": "--quiet"},
]


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    # None of this project's variables and no key, whatever the shell
    # running pytest has, and no real model call.
    for name in list(os.environ):
        if name.upper().startswith(PREFIX) or name.upper() == "OPENAI_API_KEY":
            monkeypatch.delenv(name)
    monkeypatch.setattr(target=models, name="ALLOW_MODEL_REQUESTS", value=False)


def run_script(*, name: str, cwd: Path, stdin: str = "") -> str:
    """Run one of the project's scripts from `cwd`, `stdin` piped in; return its stdout."""
    result = subprocess.run(
        args=[sys.executable, name],
        cwd=cwd,
        input=stdin,
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
    for script in ("fresh_copy.py", "ask_session.py"):
        # Before the lesson there is no ask_session.py: the test that runs
        # it fails there, not every test that makes a copy.
        if (PROJECT_ROOT / "scripts" / script).is_file():
            shutil.copy(src=PROJECT_ROOT / "scripts" / script, dst=root / "scripts")
    return root


@pytest.fixture
def working_copy(project: Path) -> Path:
    """A working copy, made by the student's `scripts/fresh_copy.py`."""
    run_script(name="scripts/fresh_copy.py", cwd=project)
    return project / "work" / "tinyjobs"


class Keyboard:
    """Scripted lines in place of `input`; once they run out, input has ended."""

    def __init__(self, *, lines: list[str]) -> None:
        self.lines = list(lines)
        self.prompts: list[str] = []

    def __call__(self, prompt: object = "") -> str:
        self.prompts.append(str(prompt))
        if not self.lines:
            raise EOFError
        return self.lines.pop(0)


def type_lines(*, monkeypatch: pytest.MonkeyPatch, lines: list[str]) -> Keyboard:
    keyboard = Keyboard(lines=lines)
    monkeypatch.setattr("builtins.input", keyboard)
    return keyboard


class Model:
    """A scripted model: each turn, ask one question, then say "Done.".

    It records the tools each request offered, and what came back for
    each question. With `multi_select`, the question allows several
    answers.
    """

    def __init__(self, *, multi_select: bool = False) -> None:
        self.multi_select = multi_select
        self.tools_offered: list[list[str]] = []
        self.answers: list[ToolReturnPart] = []

    def __call__(self, messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        self.tools_offered.append(sorted(t.name for t in info.function_tools))
        back = [p for p in messages[-1].parts if isinstance(p, ToolReturnPart)]
        if back:
            self.answers.extend(back)
            return ModelResponse(parts=[TextPart(content="Done.")])
        if TOOL_NAME not in self.tools_offered[-1]:
            return ModelResponse(parts=[TextPart(content="No one to ask.")])
        question = {
            "header": "Flag",
            "question": "Which flag should the CLI gain?",
            "options": OPTIONS,
            "multi_select": self.multi_select,
        }
        call = ToolCallPart(tool_name=TOOL_NAME, args={"questions": [question]})
        return ModelResponse(parts=[call])


def open_chat(*, working_copy: Path, model: Model) -> None:
    """Open the student's chat on `model`, in `working_copy`."""
    from pydantic_tutor.agent import agent
    from pydantic_tutor.chat import chat
    from pydantic_tutor.codebase_tools import CodebaseDeps
    from pydantic_tutor.composition import App

    chat(
        app=App(
            agent=agent,
            model=FunctionModel(function=model, model_name="stand-in"),
            deps=CodebaseDeps(root=working_copy),
        )
    )


def test_you_pick_an_option_at_the_keyboard_and_the_model_gets_its_label(
    monkeypatch: pytest.MonkeyPatch,
    working_copy: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # The lesson's point: the model asks, inside the run; the person picks
    # from a menu; the model gets the label back as the tool's return.
    keyboard = type_lines(monkeypatch=monkeypatch, lines=["go", "2", "/exit"])
    model = Model()
    open_chat(working_copy=working_copy, model=model)
    assert [part.content for part in model.answers] == [{"Flag": ["--quiet"]}]
    out = capsys.readouterr().out
    assert "  Flag: Which flag should the CLI gain?\n" in out
    assert "    1. --verbose (print each job as it runs)\n" in out
    assert "    2. --quiet\n" in out
    assert "  ran ask_user_question\n" in out
    assert "  Pick a number, or type your own answer; Enter declines: " in (
        keyboard.prompts
    )
    # The question is not one of the calls held for approval.
    assert "  Run it? y, or say why not: " not in keyboard.prompts


def test_pressing_enter_alone_declines_and_the_model_is_told(
    monkeypatch: pytest.MonkeyPatch, working_copy: Path
) -> None:
    type_lines(monkeypatch=monkeypatch, lines=["go", "", "/exit"])
    model = Model()
    open_chat(working_copy=working_copy, model=model)
    assert model.answers[0].outcome == "success"
    assert model.answers[0].content == DECLINED


def test_anything_but_an_option_number_is_your_own_answer(
    monkeypatch: pytest.MonkeyPatch, working_copy: Path
) -> None:
    # Words, a number the menu does not have, two numbers on a question
    # that allows one: each is sent as the person typed it.
    type_lines(
        monkeypatch=monkeypatch,
        lines=["go", "both, behind --log-level", "again", "3", "more", "1,2", "/exit"],
    )
    model = Model()
    open_chat(working_copy=working_copy, model=model)
    assert [part.content for part in model.answers] == [
        {"Flag": ["both, behind --log-level"]},
        {"Flag": ["3"]},
        {"Flag": ["1,2"]},
    ]


def test_a_control_character_typed_at_the_menu_never_costs_the_turn(
    monkeypatch: pytest.MonkeyPatch, working_copy: Path
) -> None:
    # `AskUser` fails the whole run on a custom answer with a control
    # character in it: a tab, or an arrow key's escape code. The answerer
    # sends each as a space; one that leaves nothing typed declines.
    type_lines(
        monkeypatch=monkeypatch,
        lines=["go", "both\tflags", "again", "\x1b[A2", "more", "\x1b", "/exit"],
    )
    model = Model()
    open_chat(working_copy=working_copy, model=model)
    assert [part.content for part in model.answers] == [
        {"Flag": ["both flags"]},
        {"Flag": ["[A2"]},
        DECLINED,
    ]


def test_a_question_that_allows_several_answers_takes_numbers_with_commas(
    monkeypatch: pytest.MonkeyPatch, working_copy: Path
) -> None:
    keyboard = type_lines(monkeypatch=monkeypatch, lines=["go", "2, 1, 2", "/exit"])
    model = Model(multi_select=True)
    open_chat(working_copy=working_copy, model=model)
    assert model.answers[0].content == {"Flag": ["--quiet", "--verbose"]}
    assert (
        "  Pick numbers, with commas, or type your own answer; Enter declines: "
        in keyboard.prompts
    )


def test_when_input_ends_at_a_question_it_is_declined_and_the_chat_ends(
    monkeypatch: pytest.MonkeyPatch, working_copy: Path
) -> None:
    # Ctrl-D, or piped input that runs out, at the menu.
    type_lines(monkeypatch=monkeypatch, lines=["go"])
    model = Model()
    open_chat(working_copy=working_copy, model=model)  # returns: no traceback
    assert model.answers[0].content == DECLINED


def test_only_the_chat_offers_the_model_a_question(
    monkeypatch: pytest.MonkeyPatch, working_copy: Path
) -> None:
    # `AskUser` is given to the chat's runs, where someone can answer; the
    # agent itself, which scripts and tests run with no one there, has no
    # question tool.
    from pydantic_tutor.agent import agent
    from pydantic_tutor.codebase_tools import CodebaseDeps

    type_lines(monkeypatch=monkeypatch, lines=["go", "1", "/exit"])
    in_chat = Model()
    open_chat(working_copy=working_copy, model=in_chat)
    assert TOOL_NAME in in_chat.tools_offered[0]
    alone = Model()
    result = agent.run_sync(
        user_prompt="go",
        model=FunctionModel(function=alone),
        deps=CodebaseDeps(root=working_copy),
    )
    assert TOOL_NAME not in alone.tools_offered[0]
    assert result.output == "No one to ask."


def test_ask_session_shows_the_menu_and_what_the_model_was_told(
    project: Path, working_copy: Path
) -> None:
    out = run_script(
        name="scripts/ask_session.py",
        cwd=project,
        stdin="Add a flag to the CLI.\n2\nAdd another one.\n\n/exit\n",
    )
    assert out.startswith("Chatting with a stand-in model. Type /exit to leave.\n")
    assert out.count("    2. --quiet (print only errors)\n") == 2
    assert "I was told: {'Flag': ['--quiet']}\n" in out
    assert f"I was told: {DECLINED}\n" in out
