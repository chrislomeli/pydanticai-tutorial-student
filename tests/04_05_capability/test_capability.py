"""Checks that the project from the capability lesson is in place.

Our checks for the capability lesson: they verify the lesson, and the student
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

import os
import re
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

# This file is tests/04_05_capability/test_capability.py: two folders below the root.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
PREFIX = "PYDANTIC_TUTOR_"
KEY_VARIABLES = ("OPENAI_API_KEY", "ANTHROPIC_API_KEY")
TOKEN_PATTERN = r"tj-[A-Za-z0-9-]+"


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


# Point 1: a capability is one object that gives an agent part of what it does.


def test_redact_tokens_is_a_capability_any_agent_can_take(project: Path) -> None:
    from pydantic_ai import Agent
    from pydantic_ai.capabilities import AbstractCapability

    from pydantic_tutor.capabilities.redact_tokens import RedactTokens

    capability = RedactTokens[int](pattern=TOKEN_PATTERN)
    assert isinstance(capability, AbstractCapability)
    # A different deps type: it reads none, so this agent can take it too.
    Agent[int, str](deps_type=int, capabilities=[capability])


def test_show_capabilities_lists_what_each_entry_brings(project: Path) -> None:
    result = run_script(name="show_capabilities.py", folder=project, env={})
    assert result.returncode == 0, result.stderr
    assert "RedactTokens: instructions, hooks" in result.stdout
    assert "LocalWorkspace: a workspace" in result.stdout
    assert "Hooks: hooks" in result.stdout


# Point 2: a class keeps a hook, an instruction and a setting together.


def test_the_instruction_and_the_hook_travel_together(project: Path) -> None:
    from pydantic_ai import Agent
    from pydantic_ai.tools import RunContext

    from pydantic_tutor.capabilities.redact_tokens import RedactTokens

    seen: list[str | None] = []

    def reply(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.append(info.instructions)
        returns = [
            part
            for message in messages
            for part in message.parts
            if isinstance(part, ToolReturnPart)
        ]
        if returns:
            seen.append(str(returns[0].content))
            return ModelResponse(parts=[TextPart(content="ok")])
        return ModelResponse(parts=[ToolCallPart(tool_name="secret", args={})])

    agent: Agent[None, str] = Agent(capabilities=[RedactTokens(pattern=TOKEN_PATTERN)])

    @agent.tool
    def secret(ctx: RunContext[None]) -> str:
        return "the token is tj-abc-123"

    agent.run_sync(user_prompt="x", model=FunctionModel(function=reply))
    assert "[token removed]" in str(seen[0])
    assert seen[-1] == "the token is [token removed]"


def test_a_result_that_is_not_text_passes_through(project: Path) -> None:
    from pydantic_ai import Agent

    from pydantic_tutor.capabilities.redact_tokens import RedactTokens

    seen: list[object] = []

    def reply(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        returns = [
            part
            for message in messages
            for part in message.parts
            if isinstance(part, ToolReturnPart)
        ]
        if returns:
            seen.append(returns[0].content)
            return ModelResponse(parts=[TextPart(content="ok")])
        return ModelResponse(parts=[ToolCallPart(tool_name="numbers", args={})])

    agent: Agent[None, str] = Agent(capabilities=[RedactTokens(pattern=TOKEN_PATTERN)])

    @agent.tool_plain
    def numbers() -> list[int]:
        return [1, 2]

    agent.run_sync(user_prompt="x", model=FunctionModel(function=reply))
    assert seen == [[1, 2]]


def test_a_bad_pattern_fails_when_the_capability_is_made() -> None:
    from pydantic_tutor.capabilities.redact_tokens import RedactTokens

    with pytest.raises(re.error):
        RedactTokens[None](pattern="(")


def test_guard_check_still_removes_tokens_and_refuses(project: Path) -> None:
    result = run_script(name="guard_check.py", folder=project, env={})
    assert result.returncode == 0, result.stderr
    assert 'runner_token = "[token removed]"' in result.stdout
    assert "Run one command at a time, with no chaining." in result.stdout


def test_the_chat_tells_the_model_what_the_replacement_means(project: Path) -> None:
    result = run_script(name="show_chat_request.py", folder=project, env={})
    assert result.returncode == 0, result.stderr
    assert "[token removed] in a tool result" in result.stdout
