"""Checks that the project from the limits lesson is in place.

Our checks for the limits lesson: they verify the lesson, and the student
is not asked to run them. Run them with `uv run pytest`. They fail until
the lesson is done.

Every import of the project is inside the test that needs it, so that
before the lesson each test fails on its own, instead of the whole file
failing to load.
"""

import os
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

# This file is tests/01_08_limits/test_limits.py: two folders below the root.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def no_real_model(monkeypatch: pytest.MonkeyPatch) -> None:
    # Any request to a real model raises instead of reaching a provider.
    monkeypatch.setattr(target=models, name="ALLOW_MODEL_REQUESTS", value=False)


@pytest.fixture
def broken_repo(tmp_path: Path) -> Path:
    """A repository in the src layout whose one module does not parse."""
    package = tmp_path / "src" / "broken"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(data="def half_written(:\n")
    return tmp_path


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


def test_the_agent_has_time_limits() -> None:
    from pydantic_tutor.agent import agent

    assert agent.model_settings == {"timeout": 60}
    # 2.56 keeps the tool timeout on a private attribute; there is no public
    # one to read. slow_tool.py's test proves what a tool timeout does.
    assert agent._tool_timeout == 10


def test_a_module_that_does_not_parse_is_a_failed_tool_result(
    broken_repo: Path,
) -> None:
    from pydantic_tutor.agent import agent
    from pydantic_tutor.repo_tools import RepoDeps

    seen: list[ModelMessage] = []

    def reply(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.extend(messages[len(seen) :])
        if len(messages) == 1:
            call = ToolCallPart(tool_name="read_docstring", args={"module": "broken"})
            return ModelResponse(parts=[call])
        return ModelResponse(parts=[TextPart(content="It does not parse.")])

    result = agent.run_sync(
        user_prompt="x",
        model=FunctionModel(function=reply),
        deps=RepoDeps(root=broken_repo),
    )

    # The run goes on: the model is told the call failed, and answers.
    assert result.output == "It does not parse."
    [part] = seen[-1].parts
    assert isinstance(part, ToolReturnPart)
    assert part.outcome == "failed"
    assert "broken cannot be read" in part.model_response_str()


def test_runaway_is_stopped_by_its_request_limit() -> None:
    printed = run_script("scripts/runaway.py")

    lines = printed.splitlines()
    assert lines[:5] == [
        f"request {number}: the model calls list_modules" for number in range(1, 6)
    ]
    assert lines[5].startswith(
        "stopped: The next request would exceed the request_limit of 5"
    )


def test_broken_module_shows_a_retry_then_a_failure() -> None:
    printed = run_script("scripts/broken_module.py")

    assert "asked to retry: There is no module 'brokn'." in printed
    assert (
        'tool result (failed): {"error":"broken cannot be read: invalid syntax'
        in printed
    )
    assert printed.splitlines()[-1] == "output: The module broken does not parse."


def test_slow_tool_is_abandoned_after_its_timeout() -> None:
    printed = run_script("scripts/slow_tool.py")

    assert "asked to retry: Timed out after 0.5 seconds." in printed
    assert printed.splitlines()[-1] == "output: The test runner did not answer."
