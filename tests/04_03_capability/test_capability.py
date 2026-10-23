"""Checks that the project from the capability lesson is in place.

Our checks for the capability lesson: they verify the lesson, and the
student is not asked to run them. Run them with `uv run pytest`. They fail
until the lesson is done. None of them needs an API key or makes a
request: every run is on a scripted model.

Every import of the project is inside the test that needs it, so that
before the lesson each test fails on its own, instead of the whole file
failing to load.
"""

import asyncio
import os
import subprocess
import sys
from pathlib import Path
from typing import cast

import pytest
from pydantic_ai import Agent, ModelRequestContext, RunContext, models
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

# This file is tests/04_03_capability/test_capability.py: two folders below the root.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
TINYJOBS_ROOT: Path = PROJECT_ROOT / "sample_code" / "tinyjobs"
MODULES = [
    "tinyjobs",
    "tinyjobs.cli",
    "tinyjobs.jobs",
    "tinyjobs.log",
    "tinyjobs.storage",
]


@pytest.fixture(autouse=True)
def no_real_model(monkeypatch: pytest.MonkeyPatch) -> None:
    # Any request to a real model raises instead of reaching a provider.
    monkeypatch.setattr(target=models, name="ALLOW_MODEL_REQUESTS", value=False)


def run_script(name: str) -> subprocess.CompletedProcess[str]:
    """Run one of the project's scripts, by its path from the project root."""
    result = subprocess.run(
        args=[sys.executable, name],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "PYDANTIC_AI_NO_BANNER": "1"},
    )
    assert result.returncode == 0, f"{name} failed:\n{result.stderr}"
    return result


class FiveReads:
    """A model that asks for five docstrings in one response, with a word beside them, then answers."""

    def __init__(self) -> None:
        self.returns_received: list[int] = []

    def reply(self, messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        returns = [p for p in messages[-1].parts if isinstance(p, ToolReturnPart)]
        if returns:
            self.returns_received.append(len(returns))
            return ModelResponse(parts=[TextPart(content="Done.")])
        calls = [
            ToolCallPart(tool_name="read_docstring", args={"module": module})
            for module in MODULES
        ]
        return ModelResponse(parts=[TextPart(content="Reading them."), *calls])


def response(*, calls: int) -> ModelResponse:
    """A response of a text part and `calls` tool calls."""
    return ModelResponse(
        parts=[
            TextPart(content="thinking"),
            *[ToolCallPart(tool_name="read_docstring", args={}) for _ in range(calls)],
        ]
    )


def capped(*, limit: int, given: ModelResponse) -> ModelResponse:
    """Call BatchLimit's hook directly; it reads neither the run nor the request."""
    from pydantic_tutor import batch_limit

    cap: batch_limit.BatchLimit[None] = batch_limit.BatchLimit(max_tool_calls=limit)
    return asyncio.run(
        cap.after_model_request(
            cast(RunContext[None], None),
            request_context=cast(ModelRequestContext, None),
            response=given,
        )
    )


def test_batch_limit_is_a_capability_class_with_a_cap() -> None:
    from pydantic_tutor import batch_limit

    cap: batch_limit.BatchLimit[None] = batch_limit.BatchLimit(max_tool_calls=3)
    assert isinstance(cap, AbstractCapability)
    assert cap.max_tool_calls == 3
    # The hook is a method of the class, not a `Hooks` object.
    assert "after_model_request" in vars(batch_limit.BatchLimit)


def test_a_cap_below_one_is_refused() -> None:
    from pydantic_tutor import batch_limit

    with pytest.raises(ValueError, match="at least 1"):
        batch_limit.BatchLimit(max_tool_calls=0)


def test_the_first_calls_are_kept_in_order_and_text_stays() -> None:
    given = response(calls=5)
    out = capped(limit=3, given=given)
    assert [p.part_kind for p in out.parts] == [
        "text",
        "tool-call",
        "tool-call",
        "tool-call",
    ]
    assert list(out.parts) == list(given.parts[:4])
    # The rest of the response is the model's, unchanged.
    assert out.model_name == given.model_name
    assert out.timestamp == given.timestamp


def test_a_response_within_the_cap_is_passed_on_as_it_came() -> None:
    given = response(calls=3)
    assert capped(limit=3, given=given) is given


def test_the_agent_has_the_cap_before_the_request_log() -> None:
    from pydantic_tutor import batch_limit
    from pydantic_tutor.agent import agent
    from pydantic_tutor.request_log import request_log

    capabilities = agent.root_capability.capabilities
    caps = [c for c in capabilities if isinstance(c, batch_limit.BatchLimit)]
    assert len(caps) == 1
    assert caps[0].max_tool_calls == 3
    assert capabilities[-1] is request_log
    assert capabilities[-2] is caps[0]


def test_five_calls_in_one_response_reach_the_tools_as_three() -> None:
    from pydantic_tutor.agent import agent
    from pydantic_tutor.codebase_tools import CodebaseDeps

    model = FiveReads()
    result = agent.run_sync(
        user_prompt="Read every module.",
        model=FunctionModel(function=model.reply),
        deps=CodebaseDeps(root=TINYJOBS_ROOT),
    )
    # Three tools ran, and the model was sent their three results.
    assert result.usage.tool_calls == 3
    assert model.returns_received == [3]
    # The history holds the response as the cap left it: no call without
    # its return, and the model's text kept.
    first = next(m for m in result.all_messages() if isinstance(m, ModelResponse))
    assert [p.part_kind for p in first.parts] == ["text", *["tool-call"] * 3]
    second = [m for m in result.all_messages() if isinstance(m, ModelRequest)][1]
    assert sum(isinstance(p, ToolReturnPart) for p in second.parts) == 3


def test_without_the_cap_all_five_would_run() -> None:
    # A guard on the test above: the same model, on an agent without the
    # cap, runs all five. Passes before the lesson too.
    from pydantic_tutor.agent import codebase_workspace
    from pydantic_tutor.codebase_tools import CodebaseDeps, codebase_toolset

    model = FiveReads()
    bare = Agent(
        deps_type=CodebaseDeps,
        toolsets=[codebase_toolset],
        capabilities=[codebase_workspace],
    )
    result = bare.run_sync(
        user_prompt="Read every module.",
        model=FunctionModel(function=model.reply),
        deps=CodebaseDeps(root=TINYJOBS_ROOT),
    )
    assert result.usage.tool_calls == 5
    assert model.returns_received == [5]


def test_read_every_module_asks_again_for_the_dropped_reads() -> None:
    lines = run_script("scripts/read_every_module.py").stdout.splitlines()
    assert lines[:9] == [
        "the model asks for the module list",
        "the model asks to read 5 modules: " + ", ".join(MODULES),
        "the model asks to read 2 modules: tinyjobs.log, tinyjobs.storage",
        "the model answers",
        "the run's history, response by response:",
        "  tool calls: list_modules",
        "  tool calls: read_docstring, read_docstring, read_docstring",
        "  tool calls: read_docstring, read_docstring",
        "  tool calls: none, the answer",
    ]
    assert lines[9] == "output: I read 5 modules."
    assert lines[10].startswith("usage: ")
    assert "requests=4, tool_calls=6" in lines[10]


def test_middleware_order_shows_after_hooks_in_reverse() -> None:
    lines = run_script("scripts/middleware_order.py").stdout.splitlines()
    assert lines == [
        "first: before the request",
        "last: before the request",
        "  the model asks for 5 tool calls",
        # After the response, the last in the list runs first: it sees the
        # five calls the model made; the first sees what the cap left.
        "last: after, the response has 5 tool calls",
        "first: after, the response has 3 tool calls",
        "first: before the request",
        "last: before the request",
        "  the model answers",
        "last: after, the response has 0 tool calls",
        "first: after, the response has 0 tool calls",
    ]
