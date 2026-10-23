"""Checks that the project from the toolsets lesson is in place.

Our checks for the toolsets lesson: they verify the lesson, and the student
is not asked to run them. Run them with `uv run pytest`. They fail until
the lesson is done. None of them needs an API key or makes a request:
every run is on a scripted model.

One test was moved here from the limits lesson (01-08), which pinned the
agent's `tool_timeout`. The time limit on our two tools is now the
toolset's. It says so.

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
TINYJOBS_ROOT: Path = PROJECT_ROOT / "sample_code" / "tinyjobs"
STATIC = "You answer questions about a Python codebase. Answer briefly."


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


def test_the_codebase_tools_are_one_function_toolset() -> None:
    from pydantic_tutor import codebase_tools

    toolset = codebase_tools.codebase_toolset
    assert isinstance(toolset, FunctionToolset)
    assert sorted(toolset.tools) == ["list_modules", "read_docstring"]
    # The same functions, moved as they are: not copies, not wrappers.
    assert toolset.tools["list_modules"].function is codebase_tools.list_modules
    assert toolset.tools["read_docstring"].function is codebase_tools.read_docstring


def test_the_agent_takes_the_toolset_and_has_no_tools_of_its_own() -> None:
    from pydantic_tutor.agent import agent
    from pydantic_tutor.codebase_tools import CodebaseDeps, codebase_toolset

    assert any(toolset is codebase_toolset for toolset in agent.toolsets)
    # The agent's own toolset, the one `tools=` fills, comes first, and is
    # empty: the tools are not registered twice. (Twice would fail anyway:
    # a run refuses two tools of one name.)
    own = agent.toolsets[0]
    assert isinstance(own, FunctionToolset)
    assert own.tools == {}

    recorder = Recorder()
    agent.run_sync(
        user_prompt="x",
        model=FunctionModel(function=recorder.reply),
        deps=CodebaseDeps(root=TINYJOBS_ROOT),
    )
    names = [tool.name for tool in recorder.infos[0].function_tools]
    assert names[:2] == ["list_modules", "read_docstring"]
    assert len(names) == 14


def test_the_instructions_are_the_toolsets_and_sent_once() -> None:
    from pydantic_tutor import agent as agent_module
    from pydantic_tutor.agent import agent
    from pydantic_tutor.codebase_tools import CodebaseDeps, name_the_codebase

    # The function moved to the toolset's module, and the agent's module no
    # longer has it. The request carries its text once, after the agent's
    # own, exactly as before the lesson.
    assert name_the_codebase.__module__ == "pydantic_tutor.codebase_tools"
    assert not hasattr(agent_module, "name_the_codebase")
    recorder = Recorder()
    agent.run_sync(
        user_prompt="x",
        model=FunctionModel(function=recorder.reply),
        deps=CodebaseDeps(root=TINYJOBS_ROOT),
    )
    instructions = recorder.requests[0].instructions
    assert instructions == f"{STATIC}\n\nThe codebase is tinyjobs."
    assert instructions.count("The codebase is") == 1


def test_the_toolset_brings_its_tools_and_instructions_to_another_agent() -> None:
    from pydantic_tutor.codebase_tools import CodebaseDeps, codebase_toolset

    # A second agent with no instructions and no tools of its own: only a
    # workspace, which the tools read through and the toolset does not bring.
    other = Agent(
        deps_type=CodebaseDeps,
        toolsets=[codebase_toolset],
        capabilities=[LocalWorkspace(working_dir=TINYJOBS_ROOT)],
    )
    recorder = Recorder(
        calls=[ToolCallPart(tool_name="list_modules", args={}, tool_call_id="c1")]
    )
    other.run_sync(
        user_prompt="x",
        model=FunctionModel(function=recorder.reply),
        deps=CodebaseDeps(root=TINYJOBS_ROOT),
    )

    assert recorder.requests[0].instructions == "The codebase is tinyjobs."
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
    from pydantic_tutor import codebase, codebase_tools
    from pydantic_tutor.agent import agent

    assert agent.model_settings == {"timeout": 60}
    assert codebase_tools.codebase_toolset.timeout == 10
    # 2.56 keeps the agent's tool timeout on a private attribute; there is no
    # public one to read. It is gone: it would cover no tool.
    assert agent._tool_timeout is None

    # The agent's runs use the toolset's limit: shorten it, slow the tool
    # down, and the call is abandoned with the library's message.
    async def slow_modules(*, workspace: Workspace) -> dict[str, str]:
        await asyncio.sleep(delay=2)
        return {}

    # The tools find modules through the codebase API: slow that down.
    monkeypatch.setattr(target=codebase, name="_workspace_modules", value=slow_modules)
    monkeypatch.setattr(
        target=codebase_tools.codebase_toolset, name="timeout", value=0.2
    )
    recorder = Recorder(calls=[ToolCallPart(tool_name="list_modules", args={})])
    agent.run_sync(
        user_prompt="x",
        model=FunctionModel(function=recorder.reply),
        deps=codebase_tools.CodebaseDeps(root=TINYJOBS_ROOT),
    )
    [retry] = recorder.requests[1].parts
    assert isinstance(retry, RetryPromptPart)
    assert retry.content == "Timed out after 0.2 seconds."


def test_toolset_views_shows_the_toolset_given_prefixed_and_filtered() -> None:
    printed = run_script("scripts/toolset_views.py")

    lines = printed.splitlines()
    assert lines[0] == "the codebase assistant:"
    assert lines[1].startswith("  tools: list_modules, read_docstring, read_file,")
    assert lines[2] == f"  instructions: {f'{STATIC}\n\nThe codebase is tinyjobs.'!r}"
    assert lines[3:] == [
        "a second agent, given the toolset:",
        "  tools: list_modules, read_docstring",
        "  instructions: 'The codebase is tinyjobs.'",
        "prefixed with 'codebase':",
        "  tools: codebase_list_modules, codebase_read_docstring",
        "  instructions: 'The codebase is tinyjobs.'",
        "prefixed, calling read_docstring with a wrong module:",
        # The prefix renames the tools the model sees, not the text that
        # names them: this message still says `list_modules`.
        "  asked to retry: There is no module 'jobs'. Call list_modules to see them.",
        "filtered to list_modules:",
        "  tools: list_modules",
        "  instructions: 'The codebase is tinyjobs.'",
    ]
