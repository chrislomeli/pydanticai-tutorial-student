"""Checks that the project from the hooks lesson is in place.

Our checks for the hooks lesson: they verify the lesson, and the student
is not asked to run them. Run them with `uv run pytest`. They fail until
the lesson is done. None of them needs an API key or makes a request:
every run is on a stand-in model.

Each test runs in an empty folder of its own, so the project's own `.env`
and `work/` folder are never read. A test that needs the workspace folder
makes `work/tinyjobs` there, from the sample code.

Every import of the project is inside the test that needs it, so that
before the lesson each test fails on its own, instead of the whole file
failing to load.
"""

import json
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
    RetryPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

# This file is tests/04_02_hooks/test_hooks.py: two folders below the root.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
PREFIX = "PYDANTIC_TUTOR_"
KEY_VARIABLES = ("OPENAI_API_KEY", "ANTHROPIC_API_KEY")
SECRET_TEXT = "the contents of a file, which the log must not hold"


@pytest.fixture(autouse=True)
def no_real_models(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for name in list(os.environ):
        if name.upper().startswith(PREFIX) or name.upper() in KEY_VARIABLES:
            monkeypatch.delenv(name)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(target=models, name="ALLOW_MODEL_REQUESTS", value=False)


@pytest.fixture
def project(tmp_path: Path) -> Path:
    """A stand-in project folder with a fresh `work/tinyjobs`, made by the script."""
    shutil.copytree(src=PROJECT_ROOT / "sample_code", dst=tmp_path / "sample_code")
    shutil.copy(src=PROJECT_ROOT / "pyproject.toml", dst=tmp_path / "pyproject.toml")
    made = run_script(name="fresh_copy.py", folder=tmp_path, env={})
    assert made.returncode == 0, made.stderr
    return tmp_path


def run_script(
    *, name: str, folder: Path, env: dict[str, str]
) -> subprocess.CompletedProcess[str]:
    kept = {k: v for k, v in os.environ.items() if k not in KEY_VARIABLES}
    return subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "demos" / name)],
        capture_output=True,
        text=True,
        check=False,
        cwd=folder,
        env={
            **kept,
            **env,
            "PYTHONPATH": str(PROJECT_ROOT / "src"),
            "PYDANTIC_AI_NO_BANNER": "1",
            # The scripts' paths come from `project_root()`: keep them in this folder.
            "PYDANTIC_TUTOR_PROJECT_ROOT": str(folder),
        },
    )


class Scripted:
    """A model that asks for the given calls, one response each, then answers."""

    def __init__(self, *, calls: list[tuple[str, dict[str, str]]]) -> None:
        self.calls = calls
        self.returns: list[ToolReturnPart | RetryPromptPart] = []

    def reply(self, messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        self.returns = [
            part
            for message in messages
            for part in message.parts
            if isinstance(part, ToolReturnPart | RetryPromptPart)
        ]
        if len(self.returns) < len(self.calls):
            name, args = self.calls[len(self.returns)]
            return ModelResponse(parts=[ToolCallPart(tool_name=name, args=args)])
        return ModelResponse(parts=[TextPart(content="Done.")])


def run_chat(
    *, project: Path, calls: list[tuple[str, dict[str, str]]]
) -> tuple[list[dict[str, object]], Scripted]:
    """Run the chat's agent with the log added; return the log's lines and the model."""
    from pydantic_tutor.capabilities.activity_log import build_activity_hooks
    from pydantic_tutor.chat.agent import build_chat_agent

    path = project / "work" / "activity.jsonl"
    agent = build_chat_agent(
        web_search=False, workspace_root=project / "work" / "tinyjobs"
    )
    model = Scripted(calls=calls)
    agent.run_sync(
        user_prompt="x",
        model=FunctionModel(function=model.reply),
        capabilities=[build_activity_hooks(path=path)],
    )
    return [json.loads(line) for line in path.read_text().splitlines()], model


# Point 1: a run is a loop with named hooks.


def test_the_log_reads_as_the_loop(project: Path) -> None:
    entries, _ = run_chat(project=project, calls=[("list_modules", {})])
    assert all(entry.pop("tools", 1) for entry in entries)
    assert entries == [
        {"hook": "before_model_request", "step": 1, "messages": 1},
        {"hook": "before_tool_execute", "step": 1, "tool": "list_modules"},
        {
            "hook": "after_tool_execute",
            "step": 1,
            "tool": "list_modules",
            "result_chars": 137,
        },
        {"hook": "before_model_request", "step": 2, "messages": 3},
    ]


# Point 2: the run carries on with what a hook returns, so these only watch.


def test_the_hooks_change_nothing_the_model_or_the_tool_sees(project: Path) -> None:
    from pydantic_tutor.chat.agent import build_chat_agent

    # The same run with the log and without: the model gets the same return.
    _, with_log = run_chat(project=project, calls=[("list_modules", {})])
    without_log = Scripted(calls=[("list_modules", {})])
    build_chat_agent(
        web_search=False, workspace_root=project / "work" / "tinyjobs"
    ).run_sync(user_prompt="x", model=FunctionModel(function=without_log.reply))
    assert [part.content for part in with_log.returns] == [
        part.content for part in without_log.returns
    ]


def test_a_log_that_cannot_be_written_stops_the_tool(project: Path) -> None:
    from pydantic_tutor.capabilities.activity_log import build_activity_hooks
    from pydantic_tutor.chat.agent import build_chat_agent

    written = project / "work" / "tinyjobs" / "written.txt"
    model = Scripted(calls=[("write_file", {"path": "written.txt", "content": "x"})])
    agent = build_chat_agent(
        web_search=False, workspace_root=project / "work" / "tinyjobs"
    )
    with pytest.raises(OSError):
        agent.run_sync(
            user_prompt="x",
            model=FunctionModel(function=model.reply),
            # A folder that does not exist: the first write fails.
            capabilities=[
                build_activity_hooks(path=project / "missing" / "activity.jsonl")
            ],
        )
    assert not written.exists()


def test_a_file_written_is_logged_by_path_and_never_by_content(project: Path) -> None:
    entries, _ = run_chat(
        project=project,
        calls=[("write_file", {"path": "notes.txt", "content": SECRET_TEXT})],
    )
    assert {
        "hook": "before_tool_execute",
        "step": 1,
        "tool": "write_file",
        "target": "notes.txt",
    } in entries
    assert SECRET_TEXT in (project / "work" / "tinyjobs" / "notes.txt").read_text()
    assert SECRET_TEXT not in (project / "work" / "activity.jsonl").read_text()


def test_watch_run_prints_the_log(project: Path) -> None:
    result = run_script(name="watch_run.py", folder=project, env={})
    assert result.returncode == 0, result.stderr
    entries = [json.loads(line) for line in result.stdout.splitlines() if line]
    assert all(entry.pop("tools", 1) for entry in entries)
    assert entries == [
        {"hook": "before_model_request", "step": 1, "messages": 1},
        {"hook": "before_tool_execute", "step": 1, "tool": "list_modules"},
        {
            "hook": "before_tool_execute",
            "step": 1,
            "tool": "write_file",
            "target": "a",
        },
        {
            "hook": "after_tool_execute",
            "step": 1,
            "tool": "list_modules",
            "result_chars": 137,
        },
        {
            "hook": "after_tool_execute",
            "step": 1,
            "tool": "write_file",
            "result_chars": 49,
        },
        {"hook": "before_model_request", "step": 2, "messages": 3},
    ]
