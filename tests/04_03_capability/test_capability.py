"""Checks that the project from the capability lesson is in place.

Our checks for the capability lesson: they verify the lesson, and the
student is not asked to run them. Run them with `uv run pytest`. They fail
until the lesson is done. None of them needs an API key or makes a
request: every run is on a stand-in model.

Each test runs in an empty folder of its own, so the project's own `.env`
and `work/` folder are never read. A test that needs the workspace folder
makes `work/tinyjobs` there, from the sample code.

Every import of the project is inside the test that needs it, so that
before the lesson each test fails on its own, instead of the whole file
failing to load.
"""

import asyncio
import logging
import os
import shutil
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
PREFIX = "PYDANTIC_TUTOR_"
KEY_VARIABLES = ("OPENAI_API_KEY", "ANTHROPIC_API_KEY")
LOGGER = "pydantic_tutor.capabilities.request_log"
MODULES = [
    "tinyjobs",
    "tinyjobs.cli",
    "tinyjobs.jobs",
    "tinyjobs.log",
    "tinyjobs.storage",
]


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
    made = run_script(name="fresh_copy.py", folder=tmp_path)
    assert made.returncode == 0, made.stderr
    return tmp_path


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
    from pydantic_tutor.capabilities import batch_limit

    cap: batch_limit.BatchLimit[None] = batch_limit.BatchLimit(max_tool_calls=limit)
    return asyncio.run(
        cap.after_model_request(
            cast(RunContext[None], None),
            request_context=cast(ModelRequestContext, None),
            response=given,
        )
    )


# Point 1: a capability, as a class with a setting, changes the response.


def test_batch_limit_is_a_capability_class_with_a_cap() -> None:
    from pydantic_tutor.capabilities import batch_limit

    cap: batch_limit.BatchLimit[None] = batch_limit.BatchLimit(max_tool_calls=3)
    assert isinstance(cap, AbstractCapability)
    assert cap.max_tool_calls == 3
    # The hook is a method of the class, not a `Hooks` object.
    assert "after_model_request" in vars(batch_limit.BatchLimit)


def test_a_cap_below_one_is_refused() -> None:
    from pydantic_tutor.capabilities import batch_limit

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


def test_the_chat_has_the_cap_before_the_request_log(project: Path) -> None:
    from pydantic_tutor.capabilities import batch_limit
    from pydantic_tutor.capabilities.request_log import request_log
    from pydantic_tutor.chat.agent import build_chat_agent

    for web_search in (False, True):
        agent = build_chat_agent(
            web_search=web_search, workspace_root=project / "work" / "tinyjobs"
        )
        capabilities = agent.root_capability.capabilities
        caps = [c for c in capabilities if isinstance(c, batch_limit.BatchLimit)]
        assert len(caps) == 1
        assert caps[0].max_tool_calls == 3
        assert capabilities[-1] is request_log
        assert capabilities[-2] is caps[0]


def test_five_calls_in_one_response_reach_the_tools_as_three(project: Path) -> None:
    from pydantic_tutor.chat.agent import build_chat_agent

    model = FiveReads()
    chat = build_chat_agent(
        web_search=False, workspace_root=project / "work" / "tinyjobs"
    )
    result = chat.run_sync(
        user_prompt="Read every module.", model=FunctionModel(function=model.reply)
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


def test_without_the_cap_all_five_would_run(project: Path) -> None:
    # A guard on the test above: the same model, on an agent with the same
    # tools and no cap, runs all five. Passes before the lesson too.
    from pydantic_tutor.tools.workspace_tools import build_workspace_toolset

    model = FiveReads()
    bare: Agent[None, str] = Agent(
        toolsets=[build_workspace_toolset(root=project / "work" / "tinyjobs")]
    )
    result = bare.run_sync(
        user_prompt="Read every module.", model=FunctionModel(function=model.reply)
    )
    assert result.usage.tool_calls == 5
    assert model.returns_received == [5]


def test_read_every_module_asks_again_for_the_dropped_reads(project: Path) -> None:
    result = run_script(name="read_every_module.py", folder=project)
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
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


# Point 2: capabilities nest; the list decides what each hook sees.


def test_middleware_order_shows_after_hooks_in_reverse(project: Path) -> None:
    result = run_script(name="middleware_order.py", folder=project)
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [
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


def test_the_cap_and_the_request_log_may_stand_in_either_order(
    project: Path, caplog: pytest.LogCaptureFixture
) -> None:
    from pydantic_tutor.capabilities.batch_limit import BatchLimit

    from pydantic_tutor.capabilities.request_log import request_log
    from pydantic_tutor.tools.workspace_tools import build_workspace_toolset

    caplog.set_level(level=logging.INFO, logger=LOGGER)
    seen: list[list[str]] = []
    for capabilities in (
        [BatchLimit(max_tool_calls=3), request_log],
        [request_log, BatchLimit(max_tool_calls=3)],
    ):
        caplog.clear()
        model = FiveReads()
        agent: Agent[None, str] = Agent(
            toolsets=[build_workspace_toolset(root=project / "work" / "tinyjobs")],
            capabilities=capabilities,
        )
        result = agent.run_sync(
            user_prompt="Read every module.", model=FunctionModel(function=model.reply)
        )
        assert result.usage.tool_calls == 3
        seen.append([r.getMessage() for r in caplog.records if r.name == LOGGER])
    # A request hook and a response hook never see each other's work.
    assert seen[0] == seen[1]
    assert len(seen[0]) == 2
