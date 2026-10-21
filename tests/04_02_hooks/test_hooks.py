"""Checks that the project from the hooks lesson is in place.

Our checks for the hooks lesson: they verify the lesson, and the student
is not asked to run them. Run them with `uv run pytest`. They fail until
the lesson is done. None of them needs an API key or makes a request:
every run is on a scripted model.

Two tests were moved by the memory lesson (04-05), which adds five tools:
the request log's lines with the agent's tool count, and what
`watch_requests.py` prints, are now proved in
`tests/04_05_memory/test_memory.py`, by the tests of the same names.

Every import of the project is inside the test that needs it, so that
before the lesson each test fails on its own, instead of the whole file
failing to load.
"""

import logging
import os
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic_ai import RunContext, models
from pydantic_ai.capabilities import Hooks
from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    TextPart,
    ToolCallPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.tools import ToolDefinition

# This file is tests/04_02_hooks/test_hooks.py: two folders below the root.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
TINYJOBS_ROOT: Path = PROJECT_ROOT / "sample_repos" / "tinyjobs"
LOGGER = "pydantic_tutor.request_log"


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
        # The banner goes to stderr; turning it off keeps stderr for the log.
        env={**os.environ, "PYDANTIC_AI_NO_BANNER": "1"},
    )
    assert result.returncode == 0, f"{name} failed:\n{result.stderr}"
    return result


class Recorder:
    """A model that records how many tools each request offers, calling list_modules first."""

    def __init__(self) -> None:
        self.tool_counts: list[int] = []
        self.message_counts: list[int] = []

    def reply(self, messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        self.tool_counts.append(len(info.function_tools))
        self.message_counts.append(len(messages))
        if len(self.tool_counts) == 1:
            return ModelResponse(
                parts=[ToolCallPart(tool_name="list_modules", args={})]
            )
        return ModelResponse(parts=[TextPart(content="ok")])


def logged(*, caplog: pytest.LogCaptureFixture) -> list[str]:
    """The messages our request log wrote during the test."""
    return [record.getMessage() for record in caplog.records if record.name == LOGGER]


def test_the_request_log_is_a_hooks_capability_last_on_the_agent() -> None:
    from pydantic_tutor import request_log as module
    from pydantic_tutor.agent import agent

    assert isinstance(module.request_log, Hooks)
    # Last of the agent's own capabilities, so it sees each request as the
    # others have left it.
    assert agent.root_capability.capabilities[-1] is module.request_log
    assert module.logger.name == LOGGER


def test_the_log_counts_the_tools_actually_sent(
    caplog: pytest.LogCaptureFixture,
) -> None:
    from pydantic_tutor.agent import agent
    from pydantic_tutor.repo_tools import RepoDeps

    # Hide every tool but ours for one run: the log must count what is
    # sent, not a number fixed when the agent was built.
    def only_ours(
        ctx: RunContext[RepoDeps], tool_defs: list[ToolDefinition]
    ) -> list[ToolDefinition]:
        return [
            tool
            for tool in tool_defs
            if tool.name in {"list_modules", "read_docstring"}
        ]

    caplog.set_level(level=logging.INFO, logger=LOGGER)
    recorder = Recorder()
    agent.run_sync(
        user_prompt="x",
        model=FunctionModel(function=recorder.reply),
        deps=RepoDeps(root=TINYJOBS_ROOT),
        capabilities=[Hooks(prepare_tools=only_ours)],
    )
    assert recorder.tool_counts == [2, 2]
    assert logged(caplog=caplog) == [
        "request step=1 messages=1 tools=2",
        "request step=2 messages=3 tools=2",
    ]


def test_nothing_is_logged_until_the_program_turns_it_on() -> None:
    # An earlier script, which does not configure logging: its output is
    # what it was before the lesson.
    result = run_script("scripts/inspect_run.py")
    assert "request step=" not in result.stdout
    assert "request step=" not in result.stderr


def turn(*, step: int, logged: int, sent: int) -> list[str]:
    """The two lines one request prints: the log's count, then the model's."""
    return [
        f"{LOGGER}: request step={step} messages={logged} tools=2",
        f"  the model receives: messages={sent} tools=2",
    ]


def test_hook_forms_shows_the_class_and_why_the_position_matters() -> None:
    lines = run_script("scripts/hook_forms.py").stdout.splitlines()
    assert lines == [
        "the request log as a class:",
        *turn(step=1, logged=1, sent=1),
        *turn(step=2, logged=3, sent=3),
        *turn(step=3, logged=5, sent=5),
        "the log before a hook that trims the history:",
        *turn(step=1, logged=1, sent=1),
        *turn(step=2, logged=3, sent=3),
        # It logs the request as it was before the trimmer shortened it.
        *turn(step=3, logged=5, sent=3),
        "the log after it:",
        *turn(step=1, logged=1, sent=1),
        *turn(step=2, logged=3, sent=3),
        *turn(step=3, logged=3, sent=3),
    ]
