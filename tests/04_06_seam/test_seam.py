"""Checks that the project from the seam lesson is in place.

Our checks for the seam lesson: they verify the lesson, and the student
is not asked to run them. Run them with `uv run pytest`. They fail until
the lesson is done. None of them needs an API key or makes a request:
every run is on a scripted model.

Two tests were moved here: one pinned the repository toolset on the
constructor's `toolsets=` (04-05), one pinned the prompt behind the two
capabilities the library adds of its own (04-04). Each says where it came
from, and proves its point as the agent is now built.

Every import of the project is inside the test that needs it, so that
before the lesson each test fails on its own, instead of the whole file
failing to load.
"""

import logging
import os
import random
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
from pydantic_ai import ModelRequestContext, RunContext, models
from pydantic_ai.capabilities import AbstractCapability, AgentCapability
from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

# This file is tests/04_06_seam/test_seam.py: two folders below the root.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
TINYJOBS_ROOT: Path = PROJECT_ROOT / "sample_repos" / "tinyjobs"
PROMPT_FILE: Path = (
    PROJECT_ROOT / "src" / "pydantic_tutor" / "prompts" / "repo_assistant.md"
)
# The agent's prompt since the agent lesson, word for word.
PROMPT = "You answer questions about a Python code repository. Answer briefly."
LOGGER = "pydantic_tutor.request_log"
ALL_TOOLS = 19  # ours 2, the file tools 8, the shell's 4, the memory's 5
# How many orders of the stack the order test tries, beyond the two the
# lesson shows: enough to move every capability, few enough to be quick.
SHUFFLES = 12


@pytest.fixture(autouse=True)
def no_real_model(monkeypatch: pytest.MonkeyPatch) -> None:
    # Any request to a real model raises instead of reaching a provider.
    monkeypatch.setattr(target=models, name="ALLOW_MODEL_REQUESTS", value=False)


def run_script(name: str) -> str:
    """Run one of the project's scripts from the project root; return its stdout."""
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


class KeepLastTwo(AbstractCapability[object]):
    """Send the model only the last two messages: a capability that changes
    the request, as the lesson's stand-in does."""

    async def before_model_request(
        self, ctx: RunContext[object], request_context: ModelRequestContext
    ) -> ModelRequestContext:
        return replace(request_context, messages=request_context.messages[-2:])


class Recorder:
    """A model that lists the modules once, then answers; records each request."""

    def __init__(self) -> None:
        self.infos: list[AgentInfo] = []
        self.sizes: list[int] = []

    def reply(self, messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        self.infos.append(info)
        self.sizes.append(len(messages))
        if any(isinstance(part, ToolReturnPart) for part in messages[-1].parts):
            return ModelResponse(parts=[TextPart(content="ok")])
        return ModelResponse(parts=[ToolCallPart(tool_name="list_modules", args={})])

    def instructions(self) -> list[str]:
        """The pieces of the first request's instructions, in order."""
        parts = self.infos[0].model_request_parameters.instruction_parts or []
        return [part.content for part in parts]


def run_built(
    *,
    capabilities: list[AgentCapability[object]] | None,
    caplog: pytest.LogCaptureFixture,
) -> tuple[Recorder, list[int]]:
    """Build the agent with `capabilities`, run it once; return the model's
    view and the message counts the request log logged."""
    from pydantic_tutor.agent import build_agent
    from pydantic_tutor.repo_tools import RepoDeps

    caplog.clear()
    recorder = Recorder()
    build_agent(capabilities=capabilities).run_sync(  # type: ignore[arg-type]
        user_prompt="What does this repo do?",
        model=FunctionModel(function=recorder.reply),
        deps=RepoDeps(root=TINYJOBS_ROOT),
    )
    logged = [
        int(record.getMessage().split("messages=")[1].split()[0])
        for record in caplog.records
        if record.name == LOGGER
    ]
    return recorder, logged


def test_build_agent_builds_a_new_agent_on_what_only_the_constructor_sets() -> None:
    from pydantic_tutor.agent import agent, build_agent
    from pydantic_tutor.repo_tools import RepoDeps

    first, second = build_agent(), build_agent()
    assert first is not second
    for built in (first, second, agent):
        assert built.name == "repo_assistant"
        assert built.deps_type is RepoDeps
        assert built.output_type is str
        assert built.model_settings == {"timeout": 60}


def test_with_no_capabilities_the_agent_gives_the_model_nothing() -> None:
    # The lesson's first point: everything the model is given comes from
    # the stack. The constructor alone sends no instructions and no tools,
    # but its request settings still apply.
    from pydantic_tutor.agent import build_agent
    from pydantic_tutor.repo_tools import RepoDeps

    seen: list[AgentInfo] = []

    def reply(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.append(info)
        return ModelResponse(parts=[TextPart(content="ok")])

    build_agent(capabilities=[]).run_sync(
        user_prompt="hello",
        model=FunctionModel(function=reply),
        deps=RepoDeps(root=TINYJOBS_ROOT),
    )
    assert not seen[0].model_request_parameters.instruction_parts
    assert seen[0].function_tools == []
    assert (seen[0].model_settings or {}).get("timeout") == 60


def test_repo_capabilities_are_new_on_each_call_and_hold_the_whole_stack() -> None:
    from pydantic_ai.capabilities import Toolset
    from pydantic_ai_harness import FileSystem, Shell

    from pydantic_tutor.agent import repo_capabilities, repo_workspace
    from pydantic_tutor.batch_limit import BatchLimit
    from pydantic_tutor.listable_memory import ListableMemory
    from pydantic_tutor.prompt_file import PromptFile
    from pydantic_tutor.repo_tools import repo_toolset
    from pydantic_tutor.request_log import request_log

    first, second = repo_capabilities(), repo_capabilities()
    kinds = (PromptFile, Toolset, FileSystem, Shell, ListableMemory, BatchLimit)
    for kind in kinds:
        ours = [c for c in first if isinstance(c, kind)]
        assert len(ours) == 1, kind
        theirs = [c for c in second if isinstance(c, kind)]
        assert ours[0] is not theirs[0], kind  # built anew, not shared
    assert repo_workspace in first
    assert request_log in first
    toolsets = [c for c in first if isinstance(c, Toolset)]
    assert toolsets[0].toolset is repo_toolset
    assert len(first) == 8


def test_the_prompt_and_the_log_declare_their_places() -> None:
    from pydantic_tutor.agent import PROMPT_PATH
    from pydantic_tutor.prompt_file import PromptFile
    from pydantic_tutor.request_log import request_log

    prompt_order = PromptFile(path=PROMPT_PATH).get_ordering()
    assert prompt_order is not None
    assert prompt_order.position == "outermost"
    log_order = request_log.get_ordering()
    assert log_order is not None
    assert log_order.position == "innermost"


def test_any_order_of_the_stack_sends_the_same_and_logs_what_is_sent(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # The lesson's second point. With a capability that changes the
    # request added last, in every order tried: the instructions are the
    # prompt, the memory's guidance, then the repository's name; the same
    # 19 tools; and the log counts the messages the model is sent.
    from pydantic_tutor.agent import repo_capabilities
    from pydantic_tutor.listable_memory import LISTING_GUIDANCE

    caplog.set_level(level=logging.INFO, logger=LOGGER)
    shuffle = random.Random(4).shuffle  # fixed: the same orders every run
    orders: list[list[AgentCapability[object]]] = []
    for index in range(2 + SHUFFLES):
        stack: list[AgentCapability[object]] = [*repo_capabilities(), KeepLastTwo()]  # type: ignore[list-item]
        if index == 1:
            stack.reverse()
        elif index > 1:
            shuffle(stack)
        orders.append(stack)
    for stack in orders:
        recorder, logged = run_built(capabilities=stack, caplog=caplog)
        assert recorder.instructions() == [
            PROMPT,
            LISTING_GUIDANCE,
            "The repository is tinyjobs.",
        ]
        names = [
            sorted(tool.name for tool in info.function_tools) for info in recorder.infos
        ]
        assert all(len(n) == ALL_TOOLS for n in names)
        assert names[0] == names[1]
        assert recorder.sizes == [1, 2]
        assert logged == recorder.sizes


def test_stack_order_prints_two_orders_alike() -> None:
    out = run_script("scripts/stack_order.py").splitlines()
    request_lines = [
        f"{LOGGER}: request step=1 messages=1 tools=19",
        "  model: sent 1 message(s)",
        "    instructions: You answer questions about ...",
        "    instructions: You have a persistent ...",
        "    instructions: The repository is tinyjobs. ...",
        f"{LOGGER}: request step=2 messages=2 tools=19",
        "  model: sent 2 message(s)",
    ]
    assert out == ["forwards", *request_lines, "reversed", *request_lines]


def test_the_agent_takes_the_repo_toolset_as_a_capability() -> None:
    # Moved from 04-01, by way of 04-05: both found the toolset on the
    # constructor's `toolsets=`. It now reaches the agent through `Toolset(...)` in the
    # stack, and the model is still offered our two tools first.
    from pydantic_ai.capabilities import Toolset
    from pydantic_ai.toolsets import FunctionToolset

    from pydantic_tutor.agent import agent
    from pydantic_tutor.repo_tools import RepoDeps, repo_toolset

    assert not any(toolset is repo_toolset for toolset in agent.toolsets)
    own = agent.toolsets[0]
    assert isinstance(own, FunctionToolset)
    assert own.tools == {}  # no tools of its own: every one is from the stack
    capabilities = agent.root_capability.capabilities
    assert any(
        isinstance(c, Toolset) and c.toolset is repo_toolset for c in capabilities
    )
    seen: list[AgentInfo] = []

    def reply(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.append(info)
        return ModelResponse(parts=[TextPart(content="ok")])

    agent.run_sync(
        user_prompt="hello",
        model=FunctionModel(function=reply),
        deps=RepoDeps(root=TINYJOBS_ROOT),
    )
    names = [tool.name for tool in seen[0].function_tools]
    assert names[:2] == ["list_modules", "read_docstring"]
    assert len(names) == ALL_TOOLS


def test_the_agent_takes_its_prompt_from_the_file_first() -> None:
    # Moved from 04-04, which found the prompt behind the two capabilities
    # the library adds of its own, ToolSearch and
    # PendingMessageDrainCapability: declared outermost, it now sorts ahead
    # of them, and is the first capability of all.
    from pydantic_tutor import prompt_file
    from pydantic_tutor.agent import PROMPT_PATH, agent

    assert PROMPT_PATH.resolve() == PROMPT_FILE.resolve()
    capabilities = agent.root_capability.capabilities
    prompts = [c for c in capabilities if isinstance(c, prompt_file.PromptFile)]
    assert len(prompts) == 1
    assert prompts[0].path == PROMPT_PATH
    assert [type(c).__name__ for c in capabilities[:3]] == [
        "PromptFile",
        "ToolSearch",
        "PendingMessageDrainCapability",
    ]
    # And the log, declared innermost, is last.
    assert type(capabilities[-1]).__name__ == "Hooks"
