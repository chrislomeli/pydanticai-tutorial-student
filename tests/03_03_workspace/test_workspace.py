"""Checks that the project from the workspace lesson is in place.

Our checks for the workspace lesson: they verify the lesson, and the
student is not asked to run them. Run them with `uv run pytest`. They fail
until the lesson is done. None of them needs an API key or makes a request:
every run is on a stand-in model.

Each test runs in an empty folder of its own, so the project's own `.env`
and `work/` folder are never read. A test that needs the workspace folder
makes it there.

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
from pydantic_ai import RunContext, models
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.toolsets import FunctionToolset

# This file is tests/03_03_workspace/test_workspace.py: two folders below the root.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
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


async def first_line(ctx: RunContext[None], *, path: str) -> str:
    """Read the first line of a file through the run's workspace."""
    text = await ctx.workspace.read_text(path=path)
    return text.splitlines()[0]


def read_first_line_of(*, prompt: str) -> FunctionModel:
    """A model that calls `first_line` on the prompt, then answers with the line."""

    def reply(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        for part in messages[-1].parts:
            if isinstance(part, ToolReturnPart):
                return ModelResponse(parts=[TextPart(content=str(part.content))])
            if isinstance(part, UserPromptPart):
                call = ToolCallPart(tool_name="first_line", args={"path": prompt})
                return ModelResponse(parts=[call])
        raise ValueError("the request has neither a prompt nor a tool result")

    return FunctionModel(function=reply)


def make_workspace(*, root: Path) -> Path:
    """A workspace folder with a file inside it and one beside it, outside."""
    root.mkdir(parents=True)
    (root / "README.md").write_text("# inside\n")
    (root.parent / "outside.txt").write_text("outside the folder\n")
    return root


def read_through_workspace(*, root: Path, path: str) -> str:
    from pydantic_tutor.chat.agent import build_chat_agent

    agent = build_chat_agent(web_search=False, workspace_root=root)
    result = agent.run_sync(
        user_prompt=path,
        model=read_first_line_of(prompt=path),
        toolsets=[FunctionToolset(tools=[first_line])],
    )
    return result.output


# Point 1: the workspace is attached once, at the composition root, from a
# setting; it gives the chat a place, and no tools.


def test_the_workspace_folder_is_a_setting() -> None:
    from pydantic_tutor.settings import Settings

    assert Settings().workspace_root == Path("work/tinyjobs")


def test_the_workspace_folder_is_read_from_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pydantic_tutor.settings import Settings

    monkeypatch.setenv("PYDANTIC_TUTOR_WORKSPACE_ROOT", "somewhere/else")
    assert Settings().workspace_root == Path("somewhere/else")


def test_a_tool_reads_from_the_workspace_folder(tmp_path: Path) -> None:
    # A relative path starts in the workspace's folder.
    root = make_workspace(root=tmp_path / "ws")
    assert read_through_workspace(root=root, path="README.md") == "# inside"


def test_build_app_attaches_a_workspace_in_the_setting_folder(tmp_path: Path) -> None:
    from pydantic_tutor.composition import build_app
    from pydantic_tutor.settings import Settings

    make_workspace(root=tmp_path / "ws")
    app = build_app(
        settings=Settings(
            chat_model="test", web_search=False, workspace_root=Path("ws")
        )
    )
    result = app.agent.run_sync(
        user_prompt="README.md",
        model=read_first_line_of(prompt="README.md"),
        toolsets=[FunctionToolset(tools=[first_line])],
    )
    assert result.output == "# inside"


def test_the_workspace_gives_the_chat_no_tools(tmp_path: Path) -> None:
    from pydantic_tutor.composition import build_app
    from pydantic_tutor.settings import Settings

    (tmp_path / "ws").mkdir()
    offered: list[str] = []

    def reply(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        offered.extend(tool.name for tool in info.function_tools)
        return ModelResponse(parts=[TextPart(content="ok")])

    app = build_app(
        settings=Settings(
            chat_model="test", web_search=False, workspace_root=Path("ws")
        )
    )
    app.agent.run_sync(user_prompt="hi", model=FunctionModel(function=reply))
    assert offered == []


def test_build_app_refuses_a_missing_folder_by_the_setting() -> None:
    from pydantic_tutor.composition import build_app
    from pydantic_tutor.settings import Settings

    settings = Settings(chat_model="test", web_search=False)  # work/tinyjobs: absent
    with pytest.raises(UserError) as excinfo:
        build_app(settings=settings)
    assert "PYDANTIC_TUTOR_WORKSPACE_ROOT" in str(excinfo.value)
    assert "work/tinyjobs" in str(excinfo.value)


def test_a_file_is_not_a_workspace_folder(tmp_path: Path) -> None:
    from pydantic_tutor.composition import build_app
    from pydantic_tutor.settings import Settings

    (tmp_path / "notes.txt").write_text("a file\n")
    settings = Settings(
        chat_model="test", web_search=False, workspace_root=Path("notes.txt")
    )
    with pytest.raises(UserError) as excinfo:
        build_app(settings=settings)
    assert "PYDANTIC_TUTOR_WORKSPACE_ROOT" in str(excinfo.value)


def chat_in_empty_folder(*, dotenv: str) -> subprocess.CompletedProcess[str]:
    Path(".env").write_text(dotenv + "\n")
    env = {k: v for k, v in os.environ.items() if k not in KEY_VARIABLES}
    return subprocess.run(
        [sys.executable, "-m", "pydantic_tutor"],
        input="",
        capture_output=True,
        text=True,
        check=False,
        env={**env, "PYTHONPATH": str(PROJECT_ROOT / "src")},
    )


def test_the_chat_stops_by_name_when_the_folder_is_missing() -> None:
    result = chat_in_empty_folder(
        dotenv="PYDANTIC_TUTOR_CHAT_MODEL=test\nPYDANTIC_TUTOR_WEB_SEARCH=false"
    )
    assert result.returncode != 0
    assert result.stderr.startswith("pydantic-tutor cannot start:")
    assert "PYDANTIC_TUTOR_WORKSPACE_ROOT" in result.stderr
    assert "Traceback" not in result.stderr


def test_the_chat_opens_when_the_folder_is_there(tmp_path: Path) -> None:
    (tmp_path / "work" / "tinyjobs").mkdir(parents=True)
    result = chat_in_empty_folder(
        dotenv="PYDANTIC_TUTOR_CHAT_MODEL=test\nPYDANTIC_TUTOR_WEB_SEARCH=false"
    )
    assert result.returncode == 0
    assert "Chat with the agent" in result.stdout


# Point 2: a local workspace is where relative paths start, not a boundary.


def test_dot_dot_reaches_past_the_workspace_folder(tmp_path: Path) -> None:
    root = make_workspace(root=tmp_path / "ws")
    assert read_through_workspace(root=root, path="../outside.txt") == (
        "outside the folder"
    )


def test_an_absolute_path_ignores_the_workspace_folder(tmp_path: Path) -> None:
    root = make_workspace(root=tmp_path / "ws")
    path = str((tmp_path / "outside.txt").resolve())
    assert read_through_workspace(root=root, path=path) == "outside the folder"


# The given files and the script.


def project_copy(*, into: Path) -> Path:
    """A stand-in project folder: the sample code, and a `pyproject.toml`."""
    shutil.copytree(src=PROJECT_ROOT / "sample_code", dst=into / "sample_code")
    shutil.copy(src=PROJECT_ROOT / "pyproject.toml", dst=into / "pyproject.toml")
    return into


def run_script(*, name: str, folder: Path) -> subprocess.CompletedProcess[str]:
    env = {k: v for k, v in os.environ.items() if k not in KEY_VARIABLES}
    return subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "demos" / name)],
        capture_output=True,
        text=True,
        check=False,
        cwd=folder,
        env={**env, "PYTHONPATH": str(PROJECT_ROOT / "src")},
    )


def test_fresh_copy_makes_a_working_copy_of_tinyjobs(tmp_path: Path) -> None:
    folder = project_copy(into=tmp_path)
    result = run_script(name="fresh_copy.py", folder=folder)
    assert result.returncode == 0, result.stderr
    copy = folder / "work" / "tinyjobs"
    assert (copy / "src" / "tinyjobs" / "jobs.py").is_file()
    assert (copy / "tests" / "test_jobs.py").is_file()


def test_work_is_not_committed() -> None:
    ignore = (PROJECT_ROOT / ".gitignore").read_text().splitlines()
    assert "/work/" in ignore


def test_workspace_reach_prints_what_the_workspace_is_and_is_not(
    tmp_path: Path,
) -> None:
    # The script reads `work/tinyjobs`, two folders below the project's
    # `pyproject.toml`: a stand-in project folder has the same layout.
    folder = project_copy(into=tmp_path)
    assert run_script(name="fresh_copy.py", folder=folder).returncode == 0
    result = run_script(name="workspace_reach.py", folder=folder)
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert "tools the model is offered: 0" in lines
    assert "README.md: # tinyjobs" in lines
    assert "../../pyproject.toml: [project]" in lines
    assert "absolute path: [project]" in lines
