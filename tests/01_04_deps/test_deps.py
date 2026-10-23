"""Checks that the project from the dependencies lesson is in place.

Our checks for the dependencies lesson: they verify the lesson, and the
student is not asked to run them. Run them with `uv run pytest`. They fail
until the lesson is done.

Every import of the project is inside the test that needs it, so that
before the lesson each test fails on its own, instead of the whole file
failing to load.
"""

import ast
import dataclasses
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic_ai import models
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

# This file is tests/01_04_deps/test_deps.py: two folders below the root.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
TINYJOBS_ROOT: Path = PROJECT_ROOT / "sample_code" / "tinyjobs"
TINYJOBS_MODULES = [
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


@pytest.fixture
def other_codebase(tmp_path: Path) -> Path:
    """A second codebase in the src layout, unlike tinyjobs."""
    package = tmp_path / "src" / "otherpkg"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text('"""Another package."""\n')
    (package / "util.py").write_text('"""Helpers for another package."""\n')
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


def calls(tool_name: str, args: dict[str, str]) -> FunctionModel:
    """A model that calls one tool, then answers with what came back to it."""

    def reply(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        request = messages[-1]
        assert isinstance(request, ModelRequest)
        for part in request.parts:
            if isinstance(part, ToolReturnPart):
                return ModelResponse(
                    parts=[TextPart(content=part.model_response_str())]
                )
            if isinstance(part, RetryPromptPart):
                return ModelResponse(parts=[TextPart(content=part.model_response())])
        return ModelResponse(parts=[ToolCallPart(tool_name=tool_name, args=args)])

    return FunctionModel(function=reply)


def test_the_agent_declares_its_dependencies() -> None:
    from pydantic_tutor.agent import agent
    from pydantic_tutor.codebase_tools import CodebaseDeps

    assert agent.deps_type is CodebaseDeps
    assert dataclasses.is_dataclass(CodebaseDeps)
    assert [field.name for field in dataclasses.fields(CodebaseDeps)] == ["root"]
    # Frozen: a run's dependencies are not changed during it.
    deps = CodebaseDeps(root=TINYJOBS_ROOT)
    with pytest.raises(expected_exception=dataclasses.FrozenInstanceError):
        deps.root = PROJECT_ROOT  # type: ignore[misc]


def test_the_tools_read_the_codebase_each_run_is_given(other_codebase: Path) -> None:
    from pydantic_tutor.agent import agent
    from pydantic_tutor.codebase_tools import CodebaseDeps

    model = calls(tool_name="list_modules", args={})
    on_tinyjobs = agent.run_sync(
        user_prompt="x", model=model, deps=CodebaseDeps(root=TINYJOBS_ROOT)
    )
    on_other = agent.run_sync(
        user_prompt="x", model=model, deps=CodebaseDeps(root=other_codebase)
    )

    # One agent, two runs: only the deps differ, and so do the answers.
    assert json.loads(on_tinyjobs.output) == TINYJOBS_MODULES
    assert json.loads(on_other.output) == ["otherpkg", "otherpkg.util"]


def test_read_docstring_reads_the_codebase_it_is_given(other_codebase: Path) -> None:
    from pydantic_tutor.agent import agent
    from pydantic_tutor.codebase_tools import CodebaseDeps

    jobs = agent.run_sync(
        user_prompt="x",
        model=calls(tool_name="read_docstring", args={"module": "tinyjobs.jobs"}),
        deps=CodebaseDeps(root=TINYJOBS_ROOT),
    )
    source = TINYJOBS_ROOT / "src" / "tinyjobs" / "jobs.py"
    assert jobs.output == ast.get_docstring(node=ast.parse(source=source.read_text()))

    util = agent.run_sync(
        user_prompt="x",
        model=calls(tool_name="read_docstring", args={"module": "otherpkg.util"}),
        deps=CodebaseDeps(root=other_codebase),
    )
    assert util.output == "Helpers for another package."


@pytest.mark.parametrize(
    argnames="name",
    argvalues=["jobs", "tinyjobs.nope", "../../pyproject.toml", "tinyjobs/jobs.py"],
)
def test_a_name_that_is_not_a_module_is_sent_back_to_the_model(name: str) -> None:
    # Moved from 01-03's tests, now through the agent: the tool takes the
    # run's context, so it is no longer called directly.
    from pydantic_tutor.agent import agent
    from pydantic_tutor.codebase_tools import CodebaseDeps

    result = agent.run_sync(
        user_prompt="x",
        model=calls(tool_name="read_docstring", args={"module": name}),
        deps=CodebaseDeps(root=TINYJOBS_ROOT),
    )
    # The model got the tool's ModelRetry message, telling it what to do.
    assert "list_modules" in result.output


def test_the_model_never_sees_the_context() -> None:
    from pydantic_tutor.agent import agent
    from pydantic_tutor.codebase_tools import CodebaseDeps

    sent: list[AgentInfo] = []

    def reply(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        sent.append(info)
        return ModelResponse(parts=[TextPart(content="ok")])

    agent.run_sync(
        user_prompt="x",
        model=FunctionModel(function=reply),
        deps=CodebaseDeps(root=TINYJOBS_ROOT),
    )
    tools = {tool.name: tool.parameters_json_schema for tool in sent[0].function_tools}

    assert tools["list_modules"]["properties"] == {}
    assert list(tools["read_docstring"]["properties"]) == ["module"]


def test_a_run_without_deps_fails_when_a_tool_reads_them() -> None:
    from pydantic_tutor.agent import agent

    # Pins what step 2 shows: nothing refuses the run; the tool fails.
    with pytest.raises(
        expected_exception=AttributeError,
        match="'NoneType' object has no attribute 'root'",
    ):
        agent.run_sync(user_prompt="x", model=calls(tool_name="list_modules", args={}))


def test_two_codebases_prints_both_codebases() -> None:
    printed = run_script("scripts/two_codebases.py")

    assert "tinyjobs.storage" in printed
    assert "pydantic_tutor.codebase_tools" in printed
