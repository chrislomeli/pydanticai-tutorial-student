"""Checks that the project from the shell lesson is in place.

Our checks for the shell lesson: they verify the lesson, and the student is
not asked to run them. Run them with `uv run pytest`. They fail until the
lesson is done, except `test_a_shell_without_a_list_is_stopped_after_its_timeout`,
which is about the library and passes before it too. None of them needs an
API key or makes a request: every run is on a stand-in model.

Two checks were moved here from the file tools lesson (03-04), which this
lesson changed: that the chat is offered eight tools, and that
`workspace_reach.py` prints it. The shell adds four, so both said twelve;
the toolsets lesson (04-01) moved them on again, to fourteen.

Each test runs in an empty folder of its own, so the project's own `.env`
and `work/` folder are never read. A test that needs the workspace folder
makes `work/tinyjobs` there, from the sample code, with a fake `.env`.

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
from pydantic_ai import Agent, models
from pydantic_ai.capabilities import LocalWorkspace
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
from pydantic_ai_harness import Shell

# This file is tests/03_05_shell/test_shell.py: two folders below the root.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
FAKE_TOKEN = "tj-FAKE-0000-not-a-real-secret"
PREFIX = "PYDANTIC_TUTOR_"
KEY_VARIABLES = ("OPENAI_API_KEY", "ANTHROPIC_API_KEY")


@pytest.fixture(autouse=True)
def no_real_models(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # An empty folder with none of this project's variables, whatever the
    # shell running pytest has, and no real model call.
    for name in list(os.environ):
        if name.upper().startswith(PREFIX) or name.upper() in KEY_VARIABLES:
            monkeypatch.delenv(name)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(target=models, name="ALLOW_MODEL_REQUESTS", value=False)


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """`work/tinyjobs` in the empty folder: the sample code, and a fake `.env`."""
    root = tmp_path / "work" / "tinyjobs"
    shutil.copytree(src=PROJECT_ROOT / "sample_code" / "tinyjobs", dst=root)
    (root / ".env").write_text(f"TINYJOBS_RUNNER_TOKEN={FAKE_TOKEN}\n")
    return root


@dataclass(frozen=True)
class ToolResult:
    """What one tool call gave the model."""

    refused: bool  # True: a retry prompt, asking the model to try again
    content: object  # as the model received it

    @property
    def text(self) -> str:
        return str(self.content)


def one_call(*, tool_name: str, args: dict[str, object]) -> FunctionModel:
    """A model that makes one tool call, then ends with "done"."""

    def reply(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        request = messages[-1]
        assert isinstance(request, ModelRequest)
        for part in request.parts:
            if isinstance(part, ToolReturnPart | RetryPromptPart):
                return ModelResponse(parts=[TextPart(content="done")])
        return ModelResponse(parts=[ToolCallPart(tool_name=tool_name, args=args)])

    return FunctionModel(function=reply)


def outcome(
    *, agent: Agent[None, str], tool_name: str, args: dict[str, object]
) -> ToolResult:
    """Run `agent` on a model that makes one call; say what came back."""
    result = agent.run_sync(
        user_prompt="x", model=one_call(tool_name=tool_name, args=args)
    )
    for message in result.all_messages():
        for part in message.parts:
            if isinstance(part, ToolReturnPart):
                return ToolResult(refused=False, content=part.content)
            if isinstance(part, RetryPromptPart):
                # A tool the agent does not have is refused too; that is not
                # the refusal these tests are about.
                assert "Unknown tool name" not in str(part.content), part.content
                return ToolResult(refused=True, content=part.content)
    raise AssertionError("the tool was never called")


def command(*, root: Path, line: str, tool_name: str = "run_command") -> ToolResult:
    """Run one command line through the project's chat agent, working in `root`."""
    from pydantic_tutor.chat.agent import build_chat_agent

    agent = build_chat_agent(web_search=False, workspace_root=root)
    return outcome(agent=agent, tool_name=tool_name, args={"command": line})


def file_tool(*, root: Path, tool_name: str, args: dict[str, object]) -> ToolResult:
    """Call a file tool of the project's chat agent, working in `root`."""
    from pydantic_tutor.chat.agent import build_chat_agent

    agent = build_chat_agent(web_search=False, workspace_root=root)
    return outcome(agent=agent, tool_name=tool_name, args=args)


# Point 1: the chat has a shell that runs only pytest, each command stopped
# after a timeout.


def test_pytest_runs_in_the_workspace_folder(workspace: Path) -> None:
    # The sample's one failing test is on purpose; pytest exits with 1.
    result = command(root=workspace, line="pytest -q --tb=no")
    assert not result.refused
    assert "test_cancel" in result.text
    assert "[exit code: 1]" in result.text


@pytest.mark.parametrize(
    argnames="line",
    argvalues=["rm README.md", "cat .env", "uv run pytest", 'sh -c "rm README.md"'],
)
def test_a_command_that_does_not_start_with_pytest_is_refused(
    workspace: Path, line: str
) -> None:
    result = command(root=workspace, line=line)
    assert result.refused
    assert "is not in the allowed list" in result.text
    assert (workspace / "README.md").exists()
    assert FAKE_TOKEN not in result.text


def test_a_refusal_reaches_the_model_as_text_it_can_read(workspace: Path) -> None:
    result = command(root=workspace, line="rm README.md")
    assert isinstance(result.content, str)
    assert result.content == "Command 'rm' is not in the allowed list."


def test_the_allow_list_is_a_named_constant() -> None:
    from pydantic_tutor.chat.agent import SHELL_COMMANDS

    assert SHELL_COMMANDS == ("pytest",)


def test_a_shell_without_a_list_is_stopped_after_its_timeout(workspace: Path) -> None:
    # About the library: the shell's own limit, not the agent's tool timeout.
    agent: Agent[None, str] = Agent(
        capabilities=[
            LocalWorkspace(working_dir=workspace),
            Shell(allowed_commands=("sleep",), default_timeout=1.0),
        ]
    )
    result = outcome(agent=agent, tool_name="run_command", args={"command": "sleep 5"})
    assert not result.refused
    assert "timed out after 1.0s" in result.text


# Point 2: the allow list checks a first word, so it is not a boundary.


def test_the_file_tools_refuse_the_env_file_the_shell_prints(workspace: Path) -> None:
    read = file_tool(root=workspace, tool_name="read_file", args={"path": ".env"})
    assert read.refused
    assert FAKE_TOKEN not in read.text

    printed = command(root=workspace, line="pytest --version && cat .env")
    assert not printed.refused
    assert FAKE_TOKEN in printed.text


def test_the_shell_reaches_outside_the_workspace_folder(workspace: Path) -> None:
    outside = workspace.parent / "outside.txt"
    outside.write_text("outside the workspace\n")
    result = command(root=workspace, line="pytest --version; cat ../outside.txt")
    assert "outside the workspace" in result.text


def test_pytest_runs_any_test_file_it_is_given(workspace: Path) -> None:
    (workspace / "test_peek.py").write_text(
        "def test_peek():\n    print(open('.env').read())\n"
    )
    result = command(root=workspace, line="pytest -q -s test_peek.py")
    assert FAKE_TOKEN in result.text


# The scripts.


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


@pytest.fixture
def project(tmp_path: Path) -> Path:
    """A stand-in project folder, with a fresh working copy made by the script."""
    shutil.copytree(src=PROJECT_ROOT / "sample_code", dst=tmp_path / "sample_code")
    shutil.copy(src=PROJECT_ROOT / "pyproject.toml", dst=tmp_path / "pyproject.toml")
    made = run_script(name="fresh_copy.py", folder=tmp_path)
    assert made.returncode == 0, made.stderr
    return tmp_path


def test_shell_commands_prints_what_runs_and_what_is_refused(project: Path) -> None:
    result = run_script(name="shell_commands.py", folder=project)
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert "$ pytest -q --tb=no" in lines
    assert "  Command 'rm' is not in the allowed list." in lines
    assert "  Command 'cat' is not in the allowed list." in lines
    assert "$ pytest --version && cat .env" in lines
    assert "$ sleep 5" in lines
    assert any("timed out after 1.0s" in line for line in lines)
    # The one command that starts with `pytest` and then reads the secret.
    assert result.stdout.count(FAKE_TOKEN) == 1
    assert (project / "work" / "tinyjobs" / "README.md").exists()
