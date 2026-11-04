"""Checks that the project from the activity log lesson is in place.

Our checks for the activity log lesson: they verify the lesson, and the student
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
from pydantic import ValidationError
from pydantic_ai import models
from pydantic_ai.capabilities import Hooks
from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    RetryPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

# This file is tests/04_03_activity_log/test_activity_log.py: two folders below the root.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
PREFIX = "PYDANTIC_TUTOR_"
KEY_VARIABLES = ("OPENAI_API_KEY", "ANTHROPIC_API_KEY")


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
    *,
    project: Path,
    calls: list[tuple[str, dict[str, str]]],
    hooks: frozenset[str] | None = None,
) -> tuple[list[dict[str, object]], Scripted]:
    """Run the chat's agent with the log on; return the log's lines and the model."""
    from pydantic_tutor.capabilities.activity_log import HookName, build_activity_hooks
    from pydantic_tutor.chat.agent import build_chat_agent

    path = project / "work" / "activity.jsonl"
    chosen = (
        frozenset(HookName)
        if hooks is None
        else frozenset(HookName(name) for name in hooks)
    )
    agent = build_chat_agent(
        web_search=False,
        workspace_root=project / "work" / "tinyjobs",
        activity_log=build_activity_hooks(path=path, hooks=chosen),
    )
    model = Scripted(calls=calls)
    agent.run_sync(user_prompt="x", model=FunctionModel(function=model.reply))
    lines = path.read_text().splitlines() if path.exists() else []
    return [json.loads(line) for line in lines], model


# Point 1: the chat keeps the record, through the composition root.


def test_the_chat_gets_the_log_from_the_composition_root(project: Path) -> None:
    from pydantic_tutor.composition import build_app
    from pydantic_tutor.settings import Settings

    settings = Settings(
        chat_model="test",
        web_search=True,
        workspace_root=project / "work" / "tinyjobs",
    )
    capabilities = build_app(settings=settings).agent.root_capability.capabilities
    assert any(isinstance(capability, Hooks) for capability in capabilities)


def test_an_earlier_script_still_runs_and_writes_no_log(project: Path) -> None:
    # Earlier scripts build the agent without a log: nothing changes for them.
    result = run_script(name="workspace_reach.py", folder=project, env={})
    assert result.returncode == 0, result.stderr
    assert "tools the model is offered:" in result.stdout
    assert not (project / "work" / "activity.jsonl").exists()


# Point 2: a setting decides which hooks the agent gets.


def test_only_the_chosen_moments_are_recorded(project: Path) -> None:
    entries, _ = run_chat(
        project=project,
        calls=[("list_modules", {})],
        hooks=frozenset({"before_tool_execute"}),
    )
    assert entries == [
        {"hook": "before_tool_execute", "step": 1, "tool": "list_modules"}
    ]


def test_an_empty_choice_records_nothing_and_makes_no_file(project: Path) -> None:
    entries, _ = run_chat(
        project=project, calls=[("list_modules", {})], hooks=frozenset()
    )
    assert entries == []
    assert not (project / "work" / "activity.jsonl").exists()


def test_settings_default_to_every_moment_and_refuse_a_typo(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pydantic_tutor.capabilities.activity_log import HookName
    from pydantic_tutor.settings import Settings

    assert Settings().activity_log == frozenset(HookName)
    assert Settings().activity_log_path.parts[-2:] == ("work", "activity.jsonl")
    monkeypatch.setenv(name=f"{PREFIX}ACTIVITY_LOG", value='["before_tool"]')
    with pytest.raises(ValidationError):
        Settings()


def test_watch_run_follows_the_setting(project: Path) -> None:
    result = run_script(
        name="watch_run.py",
        folder=project,
        env={f"{PREFIX}ACTIVITY_LOG": "[]"},
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "(nothing recorded)"


# Point 3: the log shows what was tried, not only what finished.


def test_a_refused_command_has_a_before_and_no_after(project: Path) -> None:
    entries, model = run_chat(
        project=project, calls=[("run_command", {"command": "rm -rf build"})]
    )
    tool_lines = [entry for entry in entries if "tool" in entry]
    assert tool_lines == [
        {
            "hook": "before_tool_execute",
            "step": 1,
            "tool": "run_command",
            "target": "rm -rf build",
        }
    ]
    assert isinstance(model.returns[0], RetryPromptPart)
