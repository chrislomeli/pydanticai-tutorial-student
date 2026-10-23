"""Checks that the project from the function tools lesson is in place.

Our checks for the function tools lesson: they verify the lesson, and the
student is not asked to run them. Run them with `uv run pytest`. They fail
until the lesson is done.

Every import of the project is inside the test that needs it, so that
before the lesson each test fails on its own, instead of the whole file
failing to load.
"""

import ast
import os
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest
from pydantic_ai import ModelRetry, models
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.tools import ToolDefinition

# This file is tests/01_03_function_tools/test_function_tools.py: two
# folders below the root.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
TINYJOBS: Path = PROJECT_ROOT / "sample_code" / "tinyjobs" / "src" / "tinyjobs"
MODULES = [
    "tinyjobs",
    "tinyjobs.cli",
    "tinyjobs.jobs",
    "tinyjobs.log",
    "tinyjobs.storage",
]


@pytest.fixture(autouse=True)
def from_the_project_root(monkeypatch: pytest.MonkeyPatch) -> None:
    # The lesson's tools read the sample codebase relative to where they run,
    # which the lesson says is the project root.
    monkeypatch.chdir(path=PROJECT_ROOT)
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


def docstring_of(path: Path) -> str:
    docstring = ast.get_docstring(node=ast.parse(source=path.read_text()))
    assert docstring is not None
    return docstring


def definitions_sent() -> dict[str, ToolDefinition]:
    """The tool definitions the student's agent sends with its first request."""
    from pydantic_tutor.agent import agent

    sent: list[AgentInfo] = []

    def reply(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        sent.append(info)
        return ModelResponse(parts=[TextPart(content="ok")])

    agent.run_sync(user_prompt="anything", model=FunctionModel(function=reply))
    return {tool.name: tool for tool in sent[0].function_tools}


def test_the_tools_read_the_sample_codebase() -> None:
    from pydantic_tutor.codebase_tools import list_modules, read_docstring

    assert list_modules() == MODULES
    assert read_docstring(module="tinyjobs.jobs") == docstring_of(TINYJOBS / "jobs.py")
    assert read_docstring(module="tinyjobs") == docstring_of(TINYJOBS / "__init__.py")


@pytest.mark.parametrize(
    argnames="name",
    argvalues=["jobs", "tinyjobs.nope", "../../pyproject.toml", "tinyjobs/jobs.py"],
)
def test_a_name_that_is_not_a_module_is_sent_back_to_the_model(name: str) -> None:
    from pydantic_tutor.codebase_tools import read_docstring

    # The model chose the argument: the tool looks it up, never opens it as
    # a path, and tells the model what to do instead.
    with pytest.raises(expected_exception=ModelRetry, match="list_modules"):
        read_docstring(module=name)


def test_the_model_is_sent_the_signature_and_docstring() -> None:
    tools = definitions_sent()

    assert sorted(tools) == ["list_modules", "read_docstring"]
    read = tools["read_docstring"]
    assert read.description, "read_docstring has no docstring summary"
    assert "Args" not in read.description, "the Args: section was not parsed"
    schema = read.parameters_json_schema
    assert schema["required"] == ["module"]
    assert schema["properties"]["module"]["type"] == "string"
    assert schema["properties"]["module"].get("description"), (
        "`module` has no Args: line"
    )
    assert tools["list_modules"].parameters_json_schema["properties"] == {}


def test_the_agent_runs_the_tool_and_sends_back_its_result() -> None:
    from pydantic_tutor.agent import agent

    def reply(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        request = messages[-1]
        assert isinstance(request, ModelRequest)
        if any(isinstance(part, ToolReturnPart) for part in request.parts):
            return ModelResponse(parts=[TextPart(content="answered")])
        return ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name="read_docstring", args={"module": "tinyjobs.jobs"}
                )
            ]
        )

    result = agent.run_sync(
        user_prompt="What does the jobs module do?", model=FunctionModel(function=reply)
    )
    messages = result.all_messages()

    assert [[part.part_kind for part in message.parts] for message in messages] == [
        ["user-prompt"],
        ["tool-call"],
        ["tool-return"],
        ["text"],
    ]
    call, returned = messages[1].parts[0], messages[2].parts[0]
    assert isinstance(call, ToolCallPart) and isinstance(returned, ToolReturnPart)
    # The agent, not the model, ran the student's tool on the sample codebase.
    assert returned.content == docstring_of(TINYJOBS / "jobs.py")
    assert returned.tool_call_id == call.tool_call_id
    assert result.usage.requests == 2


def test_show_tools_prints_what_the_model_is_told() -> None:
    printed = run_script("scripts/show_tools.py")
    tools = definitions_sent()

    assert "list_modules" in printed and "read_docstring" in printed
    for tool in tools.values():
        assert tool.description is not None and tool.description in printed
    module = tools["read_docstring"].parameters_json_schema["properties"]["module"]
    assert module["description"] in printed


def test_tool_loop_prints_the_four_messages() -> None:
    printed = run_script("scripts/tool_loop.py")

    lines = printed.splitlines()
    assert "request ['user-prompt']" in lines
    assert "response ['tool-call']" in lines
    assert "request ['tool-return']" in lines
    assert "response ['text']" in lines
    first_line = docstring_of(TINYJOBS / "jobs.py").splitlines()[0]
    assert first_line in printed


def test_pytest_skips_the_sample_codebase() -> None:
    settings = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text())

    # Without this, pytest also collects the sample codebase's tests, which
    # cannot import tinyjobs, and one of which fails on purpose.
    assert "--ignore=sample_code" in settings["tool"]["pytest"]["addopts"]
