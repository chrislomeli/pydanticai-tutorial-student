"""Checks that the project from the file tools lesson is in place.

Our checks for the file tools lesson: they verify the lesson, and the
student is not asked to run them. Run them with `uv run pytest`. They fail
until the lesson is done, except
`test_the_default_patterns_protect_a_secret_from_writes_only`, which is about
the library and passes before it too. None of them needs an
API key or makes a request: every run is on a stand-in model.

Two checks were moved here from the workspace lesson (03-03), which this
lesson changes: that the chat is offered no tool, and that
`workspace_reach.py` prints it. Both then moved on to the shell lesson
(03-05), which adds four tools, and are proved in
`tests/03_05_shell/test_shell.py`.

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
from pydantic_ai_harness import FileSystem

# This file is tests/03_04_files/test_files.py: two folders below the root.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
FAKE_TOKEN = "tj-FAKE-0000-not-a-real-secret"
PREFIX = "PYDANTIC_TUTOR_"
KEY_VARIABLES = ("OPENAI_API_KEY", "ANTHROPIC_API_KEY")


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
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
    """What one file tool call gave the model."""

    refused: bool  # True: a retry prompt, asking the model to try again
    content: object  # as the model received it
    # Every message of the run, to look for a secret anywhere in it.
    messages: list[ModelMessage]

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
    messages = result.all_messages()
    for message in messages:
        for part in message.parts:
            if isinstance(part, ToolReturnPart):
                return ToolResult(
                    refused=False, content=part.content, messages=messages
                )
            if isinstance(part, RetryPromptPart):
                # A tool the agent does not have is refused too; that is not
                # the refusal these tests are about.
                assert "Unknown tool name" not in str(part.content), part.content
                return ToolResult(refused=True, content=part.content, messages=messages)
    raise AssertionError("the tool was never called")


def call(*, root: Path, tool_name: str, args: dict[str, object]) -> ToolResult:
    """Call a tool of the project's chat agent, working in `root`."""
    from pydantic_tutor.chat.agent import build_chat_agent

    agent = build_chat_agent(web_search=False, workspace_root=root)
    return outcome(agent=agent, tool_name=tool_name, args=args)


def mentions(*, messages: list[ModelMessage], text: str) -> bool:
    """Whether `text` appears in any part of any message, to or from the model."""
    return any(text in str(part) for message in messages for part in message.parts)


# Point 1: the chat has the library's file tools, working in its workspace.


def test_a_file_tool_reads_in_the_workspace_folder(workspace: Path) -> None:
    read = call(root=workspace, tool_name="read_file", args={"path": "README.md"})
    assert not read.refused
    assert "# tinyjobs" in read.text


def test_a_file_tool_writes_in_the_workspace_folder(workspace: Path) -> None:
    wrote = call(
        root=workspace,
        tool_name="write_file",
        args={"path": "notes.txt", "content": "from the agent\n"},
    )
    assert not wrote.refused, wrote.text
    assert (workspace / "notes.txt").read_text() == "from the agent\n"


@pytest.mark.parametrize(
    argnames="path", argvalues=["../outside.txt", "ABSOLUTE", "escape.txt"]
)
def test_the_file_tools_stay_in_the_folder(workspace: Path, path: str) -> None:
    # `..`, an absolute path, and a link out of the folder, all leading to the
    # same file outside it: 03-03's workspace reads it, the file tools refuse.
    outside = workspace.parent / "outside.txt"
    outside.write_text("outside the workspace\n")
    (workspace / "escape.txt").symlink_to(target=outside)
    target = str(outside) if path == "ABSOLUTE" else path
    result = call(root=workspace, tool_name="read_file", args={"path": target})
    assert result.refused
    assert "outside" in result.text
    assert not mentions(messages=result.messages, text="outside the workspace")


def test_a_refusal_reaches_the_model_as_text_it_can_read(workspace: Path) -> None:
    result = call(root=workspace, tool_name="read_file", args={"path": "../x"})
    assert result.refused
    assert isinstance(result.content, str)


# Point 2: protected is not private; a deny list stops the read.


def test_the_default_patterns_protect_a_secret_from_writes_only(
    workspace: Path,
) -> None:
    # The library's own default, with no deny list: what the lesson starts from.
    agent: Agent[None, str] = Agent(
        capabilities=[LocalWorkspace(working_dir=workspace), FileSystem()]
    )
    read = outcome(agent=agent, tool_name="read_file", args={"path": ".env"})
    assert not read.refused
    assert FAKE_TOKEN in read.text
    write = outcome(
        agent=agent, tool_name="write_file", args={"path": ".env", "content": "x\n"}
    )
    assert write.refused
    assert "protected" in write.text


def test_the_env_file_cannot_be_read(workspace: Path) -> None:
    result = call(root=workspace, tool_name="read_file", args={"path": ".env"})
    assert result.refused
    assert "denied" in result.text
    assert not mentions(messages=result.messages, text=FAKE_TOKEN)


@pytest.mark.parametrize(
    argnames="path",
    argvalues=[
        "src/tinyjobs/.env",  # in a folder below the top one
        ".env.local",
        "deploy/runner.pem",
        "deploy/runner.key",
        "config/secrets.toml",
    ],
)
def test_other_secret_files_cannot_be_read(workspace: Path, path: str) -> None:
    secret = workspace / path
    secret.parent.mkdir(parents=True, exist_ok=True)
    secret.write_text(f"TOKEN={FAKE_TOKEN}\n")
    result = call(root=workspace, tool_name="read_file", args={"path": path})
    assert result.refused, f"{path} was read"
    assert not mentions(messages=result.messages, text=FAKE_TOKEN)


def test_a_link_to_the_env_file_cannot_be_read(workspace: Path) -> None:
    (workspace / "settings.txt").symlink_to(target=".env")
    result = call(root=workspace, tool_name="read_file", args={"path": "settings.txt"})
    assert result.refused
    assert not mentions(messages=result.messages, text=FAKE_TOKEN)


def test_a_search_does_not_reveal_a_secret(workspace: Path) -> None:
    # `.env` is a hidden file, which a search skips anyway; a key file is not.
    (workspace / "deploy").mkdir()
    (workspace / "deploy" / "runner.pem").write_text(f"TOKEN={FAKE_TOKEN}\n")
    result = call(root=workspace, tool_name="search_files", args={"pattern": "tj-FAKE"})
    assert not mentions(messages=result.messages, text=FAKE_TOKEN)


def test_the_default_write_protection_is_kept(workspace: Path) -> None:
    # Setting `read_only_patterns` yourself would replace this list, and
    # `.git/` would be writable.
    (workspace / ".git").mkdir()
    (workspace / ".git" / "config").write_text("[core]\n")
    for path in (".git/config", ".env"):
        result = call(
            root=workspace,
            tool_name="write_file",
            args={"path": path, "content": "x\n"},
        )
        assert result.refused, f"{path} was written"
    assert (workspace / ".git" / "config").read_text() == "[core]\n"
    assert FAKE_TOKEN in (workspace / ".env").read_text()


def test_the_deny_list_is_a_named_constant() -> None:
    from pydantic_tutor.chat.agent import SECRET_PATTERNS

    assert "**/.env" in SECRET_PATTERNS
    assert ".env" not in SECRET_PATTERNS  # a bare name misses `src/.env`


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


def test_file_tools_prints_what_is_read_and_refused(project: Path) -> None:
    result = run_script(name="file_tools.py", folder=project)
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert "read_file README.md:" in lines
    assert "read_file .env:" in lines
    assert "  Path '.env' is denied by pattern '**/.env'." in lines
    assert "  Path '.env' is protected (matches '**/.env')." in lines
    assert any("is outside the project root" in line for line in lines)
    assert FAKE_TOKEN not in result.stdout
