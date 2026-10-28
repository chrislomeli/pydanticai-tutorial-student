"""Checks that the project from the ask-user lesson is in place.

Our checks for the ask-user lesson: they verify the lesson, and the student
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

import asyncio
import importlib
import os
import shutil
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType

import pytest
from pydantic_ai import models
from pydantic_ai_harness.ask_user import (
    DECLINED,
    AskUserRequest,
    AskUserResponse,
    Question,
    QuestionOption,
)

# This file is tests/05_03_ask_user/test_ask_user.py: two folders below the root.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
PREFIX = "PYDANTIC_TUTOR_"
KEY_VARIABLES = ("OPENAI_API_KEY", "ANTHROPIC_API_KEY")
OPTIONS = (
    QuestionOption(label="--verbose", description="print each job as it runs"),
    QuestionOption(label="--quiet"),
)


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


def flag_request(*, multi_select: bool = False) -> AskUserRequest:
    """One question, `Flag`, with the two options above."""
    return AskUserRequest(
        questions=(
            Question(
                header="Flag",
                question="Which flag should the CLI gain?",
                options=OPTIONS,
                multi_select=multi_select,
            ),
        )
    )


def put_to_the_person(
    *, monkeypatch: pytest.MonkeyPatch, lines: list[str], multi_select: bool = False
) -> tuple[AskUserResponse, list[str]]:
    """Put the Flag question to the person's menu; return the answer and the prompts."""
    from pydantic_tutor.capabilities.ask_user import at_the_menu

    shown = type_lines(monkeypatch=monkeypatch, lines=lines)
    response = asyncio.run(at_the_menu(flag_request(multi_select=multi_select)))
    return response, shown


def load_chat() -> ModuleType:
    """The chat loop, as a module, without running its main()."""
    return importlib.import_module("pydantic_tutor.chat.loop")


def chat_on_the_stand_in(
    *, monkeypatch: pytest.MonkeyPatch, workspace: Path, lines: list[str]
) -> list[str]:
    """Run the project's chat on the lesson's stand-in model; return the prompts shown."""
    from pydantic_tutor.chat.agent import build_chat_agent
    from pydantic_tutor.composition import App
    from tutor_given.stand_ins import asks_which_flag

    chat = load_chat()
    app = App(
        agent=build_chat_agent(web_search=False, workspace_root=workspace),
        model=asks_which_flag(),
    )
    shown = type_lines(monkeypatch=monkeypatch, lines=lines)
    chat.main(app=app)
    return shown


def test_you_pick_a_number_and_the_model_gets_that_option_s_label(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    workspace: Path,
) -> None:
    # The lesson's point, end to end: the model asks inside the run, the
    # menu is shown at the keyboard, and the label comes back as the
    # tool's return.
    shown = chat_on_the_stand_in(
        monkeypatch=monkeypatch, workspace=workspace, lines=["Add a flag.", "2", ""]
    )
    out = capsys.readouterr().out

    assert "  Flag: Which flag should the CLI gain?\n" in out
    assert "    1. --verbose (print each job as it runs)\n" in out
    assert "    2. --quiet (print only errors)\n" in out
    assert "  Pick a number, or type your own answer; Enter declines: " in shown
    assert "agent> I was told: {'Flag': ['--quiet']}\n" in out
    # The question is not one of the calls held for approval.
    assert "  Run it? y, or say why not: " not in shown


def test_pressing_enter_declines_and_the_model_is_told(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    workspace: Path,
) -> None:
    chat_on_the_stand_in(
        monkeypatch=monkeypatch, workspace=workspace, lines=["Add a flag.", "", ""]
    )

    assert f"agent> I was told: {DECLINED}\n" in capsys.readouterr().out


def test_enter_and_the_end_of_input_both_decline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for lines in ([""], []):
        response, _ = put_to_the_person(monkeypatch=monkeypatch, lines=lines)
        assert response == AskUserResponse(cancelled=True)


def test_a_number_on_the_menu_picks_its_label(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response, _ = put_to_the_person(monkeypatch=monkeypatch, lines=["1"])

    assert not response.cancelled
    assert response.answers[0].header == "Flag"
    assert response.answers[0].selected == ("--verbose",)
    assert response.answers[0].custom_answer is None


def test_anything_but_a_number_on_the_menu_is_your_own_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Words, a number the menu does not have, and two numbers on a question
    # that allows one: each is sent as the person typed it.
    for typed in ("both, behind --log-level", "3", "1,2"):
        response, _ = put_to_the_person(monkeypatch=monkeypatch, lines=[typed])
        assert response.answers[0].selected == ()
        assert response.answers[0].custom_answer == typed


def test_a_question_that_allows_several_answers_takes_numbers_with_commas(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response, shown = put_to_the_person(
        monkeypatch=monkeypatch, lines=["2, 1, 2"], multi_select=True
    )

    assert response.answers[0].selected == ("--quiet", "--verbose")
    assert shown == [
        "  Pick numbers, with commas, or type your own answer; Enter declines: "
    ]


def test_a_control_character_typed_at_the_menu_never_costs_the_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # `AskUser` fails the whole run on a custom answer with a control
    # character in it: a tab, or an arrow key's escape code. The answerer
    # sends each as a space; a reply that leaves nothing declines.
    tab, _ = put_to_the_person(monkeypatch=monkeypatch, lines=["both\tflags"])
    arrow, _ = put_to_the_person(monkeypatch=monkeypatch, lines=["\x1b[A2"])
    nothing, _ = put_to_the_person(monkeypatch=monkeypatch, lines=["\x1b"])

    assert tab.answers[0].custom_answer == "both flags"
    assert arrow.answers[0].custom_answer == "[A2"
    assert nothing == AskUserResponse(cancelled=True)


def test_only_the_chat_s_runs_offer_the_model_a_question(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    workspace: Path,
) -> None:
    from pydantic_ai_harness import AskUser

    from pydantic_tutor.chat.agent import build_chat_agent, chat_capabilities
    from tutor_given.stand_ins import asks_which_flag

    # Not on the agent: no capability of the chat's stack is an AskUser...
    stack = chat_capabilities(web_search=False, workspace_root=workspace)
    assert not any(isinstance(capability, AskUser) for capability in stack)
    # ...so a run that is not the chat's has no question tool...
    agent = build_chat_agent(web_search=False, workspace_root=workspace)
    alone = agent.run_sync(user_prompt="Add a flag.", model=asks_which_flag())
    assert alone.output == "No one to ask."
    # ...and the chat's run has it.
    chat_on_the_stand_in(
        monkeypatch=monkeypatch, workspace=workspace, lines=["Add a flag.", "1", ""]
    )
    assert "agent> I was told: {'Flag': ['--verbose']}\n" in capsys.readouterr().out


def test_the_given_session_asks_you_and_tells_the_model(workspace: Path) -> None:
    env = {k: v for k, v in os.environ.items() if k not in KEY_VARIABLES}
    # Piped input: a line, your pick, a line, your Enter, then the end.
    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "demos" / "ask_session.py")],
        input="Add a flag.\n2\nAdd another one.\n\n\n",
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
    assert result.stdout.count("    2. --quiet (print only errors)\n") == 2
    assert "agent> I was told: {'Flag': ['--quiet']}\n" in result.stdout
    assert f"agent> I was told: {DECLINED}\n" in result.stdout
