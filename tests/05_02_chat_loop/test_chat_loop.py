"""Checks that the project from the chat loop lesson is in place.

Our checks for the chat loop lesson: they verify the lesson, and the
student is not asked to run them. Run them with `uv run pytest`. They fail
until the lesson is done. None of them needs an API key or makes a
request: every run is on a scripted model, and every line "typed" at the
keyboard is scripted too.

None of them touches your own `work/` folder: each makes a working copy of
its own, with your `scripts/fresh_copy.py`, in a folder of its own.

One test was moved here, from the file tools lesson (03-04), because the
app now opens our own chat in place of `to_cli_sync`: that the entry point
opens the chat with the app it built, its key read from `.env`.

Every import of the project is inside the test that needs it, so that
before the lesson each test fails on its own, instead of the whole file
failing to load.
"""

import inspect
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic_ai import models
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.openai import OpenAIResponsesModel

# This file is tests/05_02_chat_loop/test_chat_loop.py: two folders below the root.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
TINYJOBS_ROOT: Path = PROJECT_ROOT / "sample_code" / "tinyjobs"
PREFIX = "PYDANTIC_TUTOR_"
TEST_PATH = "tests/test_verbose.py"
COMMAND = f"pytest -q {TEST_PATH}"


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
    for script in ("fresh_copy.py", "chat_session.py"):
        # Before the lesson there is no chat_session.py: the test that runs
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
    """A scripted model: on the first turn, read, then ask to write and run.

    It records the prompts each request carried, and what came back for
    each tool call; on a later turn it answers with the turn's number.
    A prompt of "fail" makes the provider refuse the request; "bug"
    raises an error Pydantic AI does not report (a bug, as in a tool);
    "stop" is Ctrl-C pressed while the request is made.
    """

    def __init__(self) -> None:
        self.prompts_sent: list[list[str]] = []
        self.returned: dict[str, ToolReturnPart] = {}

    def __call__(self, messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        parts = [part for message in messages for part in message.parts]
        prompts = [str(p.content) for p in parts if isinstance(p, UserPromptPart)]
        self.prompts_sent.append(prompts)
        if prompts[-1] == "fail":
            raise ModelHTTPError(
                status_code=401, model_name="stand-in", body={"error": "bad key"}
            )
        if prompts[-1] == "bug":
            raise ValueError("a bug")
        if prompts[-1] == "stop":
            raise KeyboardInterrupt
        if len(prompts) > 1:
            return ModelResponse(parts=[TextPart(content=f"turn {len(prompts)}")])
        back = [p for p in messages[-1].parts if isinstance(p, ToolReturnPart)]
        for part in back:
            self.returned[part.tool_name] = part
        if not back:
            call = ToolCallPart(
                tool_name="read_docstring", args={"module": "tinyjobs.cli"}
            )
            return ModelResponse(parts=[call])
        if [p.tool_name for p in back] == ["read_docstring"]:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="write_file",
                        args={
                            "path": TEST_PATH,
                            "content": "def test_x() -> None: ...\n",
                        },
                    ),
                    ToolCallPart(tool_name="run_command", args={"command": COMMAND}),
                ]
            )
        return ModelResponse(parts=[TextPart(content="Done.")])


def open_chat(*, working_copy: Path, model: Model) -> None:
    """Open the student's chat on `model`, in `working_copy`."""
    # The module, not the function: before the lesson there is no
    # `chat.py`, and Ruff sorts a name it cannot find differently.
    from pydantic_tutor import chat
    from pydantic_tutor.agent import agent
    from pydantic_tutor.codebase_tools import CodebaseDeps
    from pydantic_tutor.composition import App

    chat.chat(
        app=App(
            agent=agent,
            model=FunctionModel(function=model, model_name="stand-in"),
            deps=CodebaseDeps(root=working_copy),
        )
    )


def test_chat_takes_the_built_app_by_keyword() -> None:
    from pydantic_tutor import chat

    parameters = inspect.signature(chat.chat).parameters
    assert list(parameters) == ["app"]
    assert parameters["app"].kind is inspect.Parameter.KEYWORD_ONLY


def test_you_approve_a_write_and_refuse_a_command_and_the_model_is_told(
    monkeypatch: pytest.MonkeyPatch,
    working_copy: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # The lesson's point: a held call is answered at the keyboard, inside
    # the run. Yes runs it; anything else refuses it, with the reason.
    keyboard = type_lines(
        monkeypatch=monkeypatch,
        lines=["Add a --verbose flag", "y", "not until I have read it", "/exit"],
    )
    model = Model()
    open_chat(working_copy=working_copy, model=model)
    assert (working_copy / TEST_PATH).is_file()
    write = model.returned["write_file"]
    command = model.returned["run_command"]
    assert write.outcome == "success"
    assert command.outcome == "denied"
    assert command.content == "Refused at the keyboard: not until I have read it"
    # The person was shown each call, with its arguments, before answering.
    out = capsys.readouterr().out
    assert f"  run_command {{'command': '{COMMAND}'}}" in out
    assert keyboard.prompts.count("  Run it? y, or say why not: ") == 2


def test_approved_a_command_runs(
    monkeypatch: pytest.MonkeyPatch, working_copy: Path
) -> None:
    type_lines(monkeypatch=monkeypatch, lines=["go", "y", "y", "/exit"])
    model = Model()
    open_chat(working_copy=working_copy, model=model)
    command = model.returned["run_command"]
    assert command.outcome == "success"
    assert "1 passed" in str(command.content)


def test_pressing_enter_alone_refuses(
    monkeypatch: pytest.MonkeyPatch, working_copy: Path
) -> None:
    # Fail closed: nothing typed is not a yes.
    type_lines(monkeypatch=monkeypatch, lines=["go", "", "", "/exit"])
    model = Model()
    open_chat(working_copy=working_copy, model=model)
    assert not (working_copy / TEST_PATH).exists()
    for name in ("write_file", "run_command"):
        assert model.returned[name].outcome == "denied"
        assert model.returned[name].content == "Refused at the keyboard."


def test_when_input_ends_the_held_calls_are_refused_and_the_chat_ends(
    monkeypatch: pytest.MonkeyPatch, working_copy: Path
) -> None:
    # Ctrl-D, or piped input that runs out, in the middle of a question.
    type_lines(monkeypatch=monkeypatch, lines=["go"])
    model = Model()
    open_chat(working_copy=working_copy, model=model)  # returns: no traceback
    assert not (working_copy / TEST_PATH).exists()
    assert model.returned["write_file"].outcome == "denied"
    assert model.returned["run_command"].outcome == "denied"


def test_each_tool_that_ran_is_named_and_a_refused_one_is_not(
    monkeypatch: pytest.MonkeyPatch,
    working_copy: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    type_lines(monkeypatch=monkeypatch, lines=["go", "y", "no", "/exit"])
    open_chat(working_copy=working_copy, model=Model())
    out = capsys.readouterr().out
    assert "  ran read_docstring\n" in out
    assert "  ran write_file\n" in out
    assert "ran run_command" not in out


def test_the_conversation_is_carried_from_turn_to_turn(
    monkeypatch: pytest.MonkeyPatch,
    working_copy: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    type_lines(
        monkeypatch=monkeypatch, lines=["first", "", "", "second", "third", "/exit"]
    )
    model = Model()
    open_chat(working_copy=working_copy, model=model)
    assert model.prompts_sent[-1] == ["first", "second", "third"]
    assert "turn 3\n" in capsys.readouterr().out


def test_a_failed_turn_is_shown_and_the_chat_goes_on_without_it(
    monkeypatch: pytest.MonkeyPatch,
    working_copy: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    type_lines(
        monkeypatch=monkeypatch, lines=["first", "", "", "fail", "again", "/exit"]
    )
    model = Model()
    open_chat(working_copy=working_copy, model=model)
    out = capsys.readouterr().out
    assert "ModelHTTPError: status_code: 401" in out
    # The failed turn is not in the conversation the next turn is sent.
    assert model.prompts_sent[-1] == ["first", "again"]
    assert "turn 2\n" in out


def test_a_bug_ends_the_turn_with_its_traceback_and_the_chat_goes_on(
    monkeypatch: pytest.MonkeyPatch,
    working_copy: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Not an AgentRunError: it comes out of the run as itself. The turn
    # ends, not the chat, and the traceback says where.
    type_lines(
        monkeypatch=monkeypatch, lines=["first", "", "", "bug", "again", "/exit"]
    )
    model = Model()
    open_chat(working_copy=working_copy, model=model)
    captured = capsys.readouterr()
    assert "Traceback (most recent call last):" in captured.err
    assert "ValueError: a bug" in captured.err
    assert model.prompts_sent[-1] == ["first", "again"]
    assert "turn 2\n" in captured.out


def test_ctrl_c_during_a_turn_ends_the_turn_and_the_chat_goes_on(
    monkeypatch: pytest.MonkeyPatch,
    working_copy: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    type_lines(
        monkeypatch=monkeypatch, lines=["first", "", "", "stop", "again", "/exit"]
    )
    model = Model()
    try:
        open_chat(working_copy=working_copy, model=model)
    except KeyboardInterrupt:
        # Let out, it would stop pytest itself: fail by name instead.
        pytest.fail("Ctrl-C during a turn ended the chat")
    out = capsys.readouterr().out
    assert "Interrupted.\n" in out
    assert model.prompts_sent[-1] == ["first", "again"]
    assert "turn 2\n" in out


def test_exit_ends_the_chat_and_blank_lines_are_skipped(
    monkeypatch: pytest.MonkeyPatch,
    working_copy: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    type_lines(monkeypatch=monkeypatch, lines=["", "   ", "/exit", "never read"])
    model = Model()
    open_chat(working_copy=working_copy, model=model)
    assert model.prompts_sent == []
    assert capsys.readouterr().out == ("Chatting with stand-in. Type /exit to leave.\n")


def test_outside_the_chat_the_agent_still_refuses(
    monkeypatch: pytest.MonkeyPatch, working_copy: Path
) -> None:
    # The chat's answerer is given to its own runs only: the agent is
    # unchanged, and a run with no one to ask still refuses.
    from pydantic_tutor.agent import agent
    from pydantic_tutor.approval import UNATTENDED
    from pydantic_tutor.codebase_tools import CodebaseDeps

    type_lines(monkeypatch=monkeypatch, lines=["go", "y", "y", "/exit"])
    open_chat(working_copy=working_copy, model=Model())
    model = Model()
    agent.run_sync(
        user_prompt="go",
        model=FunctionModel(function=model),
        deps=CodebaseDeps(root=working_copy),
    )
    assert model.returned["write_file"].outcome == "denied"
    assert model.returned["write_file"].content == UNATTENDED


def test_chat_session_shows_each_call_and_what_the_model_was_told(
    project: Path, working_copy: Path
) -> None:
    out = run_script(
        name="scripts/chat_session.py",
        cwd=project,
        stdin="Add a --verbose flag\ny\nnot until I have read it\nWhat happened?\n/exit\n",
    )
    assert out.startswith("Chatting with a stand-in model. Type /exit to leave.\n")
    assert "  ran read_docstring\n" in out
    assert f"  run_command {{'command': '{COMMAND}'}}\n" in out
    assert "  ran write_file\n" in out
    assert "write_file, success: Wrote 136 chars (5 lines)" in out
    assert (
        "run_command, denied: Refused at the keyboard: not until I have read it\n"
        in out
    )
    assert (
        "Turn 2. So far: read_docstring success, write_file success, "
        "run_command denied.\n" in out
    )
    assert (working_copy / TEST_PATH).is_file()


def test_the_entry_point_opens_our_chat_with_the_built_app(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Moved from 03-04, where the entry point opened `to_cli_sync`; that
    # is now a failure, so it can no longer be what opens.
    import pydantic_tutor.__main__ as entry
    from pydantic_tutor import agent as agent_module
    from pydantic_tutor.composition import App

    monkeypatch.chdir(tmp_path)
    (tmp_path / "work" / "tinyjobs").mkdir(parents=True)
    Path(".env").write_text("OPENAI_API_KEY=sk-from-dotenv\n")
    opened: list[App] = []

    def to_cli_must_not_open(**kwargs: object) -> None:
        raise AssertionError("to_cli_sync opened")

    def fake_chat(*, app: App) -> None:
        opened.append(app)

    monkeypatch.setattr(
        target=agent_module.agent, name="to_cli_sync", value=to_cli_must_not_open
    )
    monkeypatch.setattr(target=entry, name="chat", value=fake_chat)
    entry.main()
    [app] = opened
    assert app.agent is agent_module.agent
    assert isinstance(app.model, OpenAIResponsesModel)
    assert app.model.client.api_key == "sk-from-dotenv"
    assert app.deps.root == Path("work/tinyjobs")
    assert "OPENAI_API_KEY" not in os.environ  # read into Settings only
