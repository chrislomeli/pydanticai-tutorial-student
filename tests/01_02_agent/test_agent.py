"""Checks that the project from the agent lesson is in place.

Our checks for the agent lesson: they verify the lesson, and the student is
not asked to run them. Run them with `uv run pytest`. They fail until the
lesson is done.

Every import of the project is inside the test that needs it, so that
before the lesson each test fails on its own, instead of the whole file
failing to load.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic_ai import Agent, models
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

# This file is tests/01_02_agent/test_agent.py: two folders below the root.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def no_real_models(monkeypatch: pytest.MonkeyPatch) -> None:
    # Any request to a real model raises instead of reaching a provider, so
    # every test here passes only if it needs no key and costs nothing.
    monkeypatch.setattr(models, "ALLOW_MODEL_REQUESTS", False)


def run_script(name: str) -> str:
    """Run one of the lesson's scripts, by its path from the project root; return its stdout."""
    result = subprocess.run(
        [sys.executable, name],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
        # The banner goes to stderr; turning it off keeps a failure readable.
        env={**os.environ, "PYDANTIC_AI_NO_BANNER": "1"},
    )
    assert result.returncode == 0, f"{name} failed:\n{result.stderr}"
    return result.stdout


def test_the_agent_is_in_the_package_without_a_model() -> None:
    from pydantic_tutor.agent import agent

    assert isinstance(agent, Agent)
    # The model is chosen where the agent is run, not where it is built.
    assert agent.model is None


def test_the_instructions_travel_on_the_request_not_in_a_part() -> None:
    from pydantic_tutor.agent import agent

    sent: list[ModelMessage] = []

    def reply(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        sent.extend(messages)
        return ModelResponse(parts=[TextPart(content="ok")])

    agent.run_sync(
        user_prompt="What does this repo do?", model=FunctionModel(function=reply)
    )

    [request] = sent
    assert isinstance(request, ModelRequest)
    assert request.instructions, "the agent has no instructions"
    assert [part.part_kind for part in request.parts] == ["user-prompt"]


def test_inspect_run_shows_the_instructions_apart_from_the_parts() -> None:
    from pydantic_tutor.agent import agent

    instructions: list[str | None] = []

    def reply(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        instructions.append(info.instructions)
        return ModelResponse(parts=[TextPart(content="ok")])

    agent.run_sync(user_prompt="anything", model=FunctionModel(function=reply))
    printed = run_script("scripts/inspect_run.py")

    assert instructions[0] is not None and instructions[0] in printed
    assert "user-prompt" in printed
    assert "text" in printed


def test_scripted_run_prints_what_the_function_decided() -> None:
    from pydantic_tutor.agent import agent

    instructions: list[str | None] = []

    def reply(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        instructions.append(info.instructions)
        return ModelResponse(parts=[TextPart(content="ok")])

    agent.run_sync(user_prompt="anything", model=FunctionModel(function=reply))
    printed = run_script("scripts/scripted_run.py")

    # The reply is built from the prompt and the instructions the model was
    # sent, so it can only match if the function, not a canned answer,
    # produced it.
    expected = f"You asked 'What does this repo do?', and my instructions say: {instructions[0]}"
    assert printed.strip().splitlines()[-1] == expected
