"""Checks that the project from the toolsets lesson is in place.

Our checks for the toolsets lesson: they verify the lesson, and the student
is not asked to run them. Run them with `uv run pytest`. They fail until
the lesson is done. None of them needs an API key or makes a request:
every run is on a scripted model.

One test was moved here from the limits lesson (01-08), which pinned the
agent's `tool_timeout`. The time limit on our two tools is now the
toolset's. It says so.

Three tests were moved by the memory lesson (04-05), which adds the
memory's guidance to the instructions and five tools to the agent: that
the agent takes the toolset and has no tools of its own, that the
toolset's text is sent once, and what `toolset_views.py` prints, are now
proved in `tests/04_05_memory/test_memory.py`.

Every import of the project is inside the test that needs it, so that
before the lesson each test fails on its own, instead of the whole file
failing to load.
"""

import asyncio
import os
import subprocess
import sys
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
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai.workspaces import Workspace

# This file is tests/04_01_toolsets/test_toolsets.py: two folders below the root.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
TINYJOBS_ROOT: Path = PROJECT_ROOT / "sample_repos" / "tinyjobs"
STATIC = "You answer questions about a Python code repository. Answer briefly."


@pytest.fixture(autouse=True)
def no_real_model(monkeypatch: pytest.MonkeyPatch) -> None:
    # Any request to a real model raises instead of reaching a provider.
    monkeypatch.setattr(target=models, name="ALLOW_MODEL_REQUESTS", value=False)


def run_script(name: str) -> str:
    """Run one of the lesson's scripts, by its path from the project root; return its stdout."""
    result = subprocess.run(
        args=[sys.executable, name],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
        # The banner goes to stderr; turning it off keeps a failure readable.
        env={**os.environ, "PYDANTIC_AI_NO_BANNER": "1"},
    )
    assert result.returncode == 0, f"{name} failed:\n{result.stderr}"
    return result.stdout


class Recorder:
    """A model that records each request, making the calls it is given first."""

    def __init__(self, *, calls: list[ToolCallPart] | None = None) -> None:
        self.calls = list(calls or [])
        self.infos: list[AgentInfo] = []
        self.requests: list[ModelRequest] = []

    def reply(self, messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        self.infos.append(info)
        request = messages[-1]
        assert isinstance(request, ModelRequest)
        self.requests.append(request)
        if self.calls:
            return ModelResponse(parts=[self.calls.pop(0)])
        return ModelResponse(parts=[TextPart(content="ok")])


def test_the_repo_tools_are_one_function_toolset() -> None:
    from pydantic_tutor import repo_tools

    toolset = repo_tools.repo_toolset
    assert isinstance(toolset, FunctionToolset)
    assert sorted(toolset.tools) == ["list_modules", "read_docstring"]
    # The same functions, moved as they are: not copies, not wrappers.
    assert toolset.tools["list_modules"].function is repo_tools.list_modules
    assert toolset.tools["read_docstring"].function is repo_tools.read_docstring


def test_the_toolset_brings_its_tools_and_instructions_to_another_agent() -> None:
    from pydantic_tutor.repo_tools import RepoDeps, repo_toolset

    # A second agent with no instructions and no tools of its own: only a
    # workspace, which the tools read through and the toolset does not bring.
    other = Agent(
        deps_type=RepoDeps,
        toolsets=[repo_toolset],
        capabilities=[LocalWorkspace(working_dir=TINYJOBS_ROOT)],
    )
    recorder = Recorder(
        calls=[ToolCallPart(tool_name="list_modules", args={}, tool_call_id="c1")]
    )
    other.run_sync(
        user_prompt="x",
        model=FunctionModel(function=recorder.reply),
        deps=RepoDeps(root=TINYJOBS_ROOT),
    )

    assert recorder.requests[0].instructions == "The repository is tinyjobs."
    names = [tool.name for tool in recorder.infos[0].function_tools]
    assert names == ["list_modules", "read_docstring"]
    # And the tools run there: the call came back with the modules.
    [returned] = recorder.requests[1].parts
    assert isinstance(returned, ToolReturnPart)
    assert "tinyjobs.jobs" in str(returned.content)


def test_the_time_limits_are_on_the_model_and_the_toolset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Moved from the limits lesson (01-08), which pinned the agent's
    # `tool_timeout=10`: the limit on our tools is now the toolset's.
    from pydantic_tutor import repo, repo_tools
    from pydantic_tutor.agent import agent

    assert agent.model_settings == {"timeout": 60}
    assert repo_tools.repo_toolset.timeout == 10
    # 2.56 keeps the agent's tool timeout on a private attribute; there is no
    # public one to read. It is gone: it would cover no tool.
    assert agent._tool_timeout is None

    # The agent's runs use the toolset's limit: shorten it, slow the tool
    # down, and the call is abandoned with the library's message.
    async def slow_modules(*, workspace: Workspace) -> dict[str, str]:
        await asyncio.sleep(delay=2)
        return {}

    # The tools find modules through the repo API: slow that down.
    monkeypatch.setattr(target=repo, name="_workspace_modules", value=slow_modules)
    monkeypatch.setattr(target=repo_tools.repo_toolset, name="timeout", value=0.2)
    recorder = Recorder(calls=[ToolCallPart(tool_name="list_modules", args={})])
    agent.run_sync(
        user_prompt="x",
        model=FunctionModel(function=recorder.reply),
        deps=repo_tools.RepoDeps(root=TINYJOBS_ROOT),
    )
    [retry] = recorder.requests[1].parts
    assert isinstance(retry, RetryPromptPart)
    assert retry.content == "Timed out after 0.2 seconds."
