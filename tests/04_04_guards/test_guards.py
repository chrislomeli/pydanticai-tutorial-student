"""Checks that the project from the guards lesson is in place.

Our checks for the guards lesson: they verify the lesson, and the student
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

# This file is tests/04_04_guards/test_guards.py: two folders below the root.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
PREFIX = "PYDANTIC_TUTOR_"
KEY_VARIABLES = ("OPENAI_API_KEY", "ANTHROPIC_API_KEY")
TOKEN = "tj-FAKE-0000-not-a-real-secret"


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
    """Run the chat's agent with the log on; return the log's lines and the model."""
    from pydantic_tutor.capabilities.activity_log import build_activity_hooks
    from pydantic_tutor.chat.agent import build_chat_agent

    path = project / "work" / "activity.jsonl"
    agent = build_chat_agent(
        web_search=False,
        workspace_root=project / "work" / "tinyjobs",
        activity_log=build_activity_hooks(path=path),
    )
    model = Scripted(calls=calls)
    agent.run_sync(user_prompt="x", model=FunctionModel(function=model.reply))
    return [json.loads(line) for line in path.read_text().splitlines()], model


def with_a_token(*, project: Path) -> None:
    (project / "work" / "tinyjobs" / "deploy.toml").write_text(f'token = "{TOKEN}"\n')


# Point 1: a hook changes a run by returning something else, or refuses.


def test_a_token_in_a_tool_result_never_reaches_the_model(project: Path) -> None:
    with_a_token(project=project)
    _, model = run_chat(project=project, calls=[("read_file", {"path": "deploy.toml"})])
    assert isinstance(model.returns[0], ToolReturnPart)
    assert TOKEN not in str(model.returns[0].content)
    assert "[token removed]" in str(model.returns[0].content)


def test_the_file_itself_is_left_as_it_was(project: Path) -> None:
    with_a_token(project=project)
    run_chat(project=project, calls=[("read_file", {"path": "deploy.toml"})])
    assert TOKEN in (project / "work" / "tinyjobs" / "deploy.toml").read_text()


@pytest.mark.parametrize(
    "command",
    [
        "pytest --version && cat .env",
        "pytest --version; cat .env",
        "pytest --version | cat",
        "pytest $(cat .env)",
    ],
)
def test_a_chained_command_is_refused_and_the_model_is_told(
    project: Path, command: str
) -> None:
    _, model = run_chat(project=project, calls=[("run_command", {"command": command})])
    assert isinstance(model.returns[0], RetryPromptPart)
    assert "one command at a time" in str(model.returns[0].content)
    assert "TINYJOBS_RUNNER_TOKEN" not in str(model.returns[0].content)


def test_a_single_allowed_command_still_runs(project: Path) -> None:
    _, model = run_chat(
        project=project, calls=[("run_command", {"command": "pytest --version"})]
    )
    assert isinstance(model.returns[0], ToolReturnPart)
    assert "pytest" in str(model.returns[0].content)


# Point 2: a hook's place decides what it sees.


def test_the_log_records_a_refused_attempt(project: Path) -> None:
    entries, _ = run_chat(
        project=project,
        calls=[("run_command", {"command": "pytest --version && cat .env"})],
    )
    tool_lines = [entry for entry in entries if "tool" in entry]
    assert tool_lines == [
        {
            "hook": "before_tool_execute",
            "step": 1,
            "tool": "run_command",
            "target": "pytest --version && cat .env",
        }
    ]


def test_the_log_counts_the_result_the_model_gets(project: Path) -> None:
    with_a_token(project=project)
    entries, model = run_chat(
        project=project, calls=[("read_file", {"path": "deploy.toml"})]
    )
    after = [e for e in entries if e["hook"] == "after_tool_execute"]
    assert after[0]["result_chars"] == len(str(model.returns[0].content))


def test_guard_check_prints_what_the_page_shows(project: Path) -> None:
    result = run_script(name="guard_check.py", folder=project, env={})
    assert result.returncode == 0, result.stderr
    assert 'runner_token = "[token removed]"' in result.stdout
    assert "Run one command at a time, with no chaining." in result.stdout
    assert "TINYJOBS_RUNNER_TOKEN" not in result.stdout
    assert '"target":"pytest --version && cat .env"' in result.stdout
