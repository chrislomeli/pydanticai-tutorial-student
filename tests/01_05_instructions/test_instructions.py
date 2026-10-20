"""Checks that the project from the instructions lesson is in place.

Our checks for the instructions lesson: they verify the lesson, and the
student is not asked to run them. Run them with `uv run pytest`. They fail
until the lesson is done, except the last one, moved here from the tools
lesson, which passes before it too.

One test was moved by the file tools lesson (03-04), which gives the
agent the library's file tools beside its own two: that the model is sent
each tool's signature and docstring is now proved in
`tests/03_04_files/test_files.py`, by
`test_the_model_is_sent_our_signatures_beside_the_file_tools`.

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
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.tools import ToolDefinition

# This file is tests/01_05_instructions/test_instructions.py: two folders
# below the root.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
TINYJOBS_ROOT: Path = PROJECT_ROOT / "sample_repos" / "tinyjobs"
# The agent's own instructions, from the agent lesson; the lesson adds a
# second paragraph after them.
STATIC = "You answer questions about a Python code repository. Answer briefly."


@pytest.fixture(autouse=True)
def no_real_model(monkeypatch: pytest.MonkeyPatch) -> None:
    # Any request to a real model raises instead of reaching a provider.
    monkeypatch.setattr(target=models, name="ALLOW_MODEL_REQUESTS", value=False)


@pytest.fixture
def other_repo(tmp_path: Path) -> Path:
    """A second repository in the src layout, in a folder named for this test."""
    root = tmp_path / "otherrepo"
    package = root / "src" / "otherpkg"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text('"""Another package."""\n')
    return root


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
    """A model that records what each request gave it, calling `list_modules` first if asked."""

    def __init__(self, *, call_tool: bool) -> None:
        self.call_tool = call_tool
        self.infos: list[AgentInfo] = []
        self.messages: list[list[ModelMessage]] = []

    def reply(self, messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        self.infos.append(info)
        self.messages.append(list(messages))
        returned = any(isinstance(part, ToolReturnPart) for part in messages[-1].parts)
        if self.call_tool and not returned:
            return ModelResponse(
                parts=[ToolCallPart(tool_name="list_modules", args={})]
            )
        return ModelResponse(parts=[TextPart(content="ok")])


def run_recorded(*, root: Path, call_tool: bool = False) -> Recorder:
    """Run the student's agent on `root`; return what its model was given."""
    from pydantic_tutor.agent import agent
    from pydantic_tutor.repo_tools import RepoDeps

    recorder = Recorder(call_tool=call_tool)
    agent.run_sync(
        user_prompt="What does this repo do?",
        model=FunctionModel(function=recorder.reply),
        deps=RepoDeps(root=root),
    )
    return recorder


def test_the_instructions_name_the_repository_each_run_is_given(
    other_repo: Path,
) -> None:
    # Goal 2: one agent, two runs, two renderings, each from its own deps.
    tinyjobs = run_recorded(root=TINYJOBS_ROOT)
    other = run_recorded(root=other_repo)

    assert tinyjobs.infos[0].instructions == f"{STATIC}\n\nThe repository is tinyjobs."
    assert other.infos[0].instructions == f"{STATIC}\n\nThe repository is otherrepo."


def test_the_instructions_are_rendered_for_every_request() -> None:
    recorder = run_recorded(root=TINYJOBS_ROOT, call_tool=True)

    expected = f"{STATIC}\n\nThe repository is tinyjobs."
    assert [info.instructions for info in recorder.infos] == [expected, expected]
    # Each request in the run records the instructions it was sent.
    requests = [m for m in recorder.messages[-1] if isinstance(m, ModelRequest)]
    assert [request.instructions for request in requests] == [expected, expected]


def test_the_agent_sends_instructions_and_no_system_prompt() -> None:
    # Moved from the agent lesson, where it ran without deps: the
    # instructions travel on the request, not as a part, and the agent has
    # no system prompt. Now with this lesson's second paragraph.
    recorder = run_recorded(root=TINYJOBS_ROOT)

    [request] = recorder.messages[0]
    assert isinstance(request, ModelRequest)
    assert [part.part_kind for part in request.parts] == ["user-prompt"]
    assert request.instructions == f"{STATIC}\n\nThe repository is tinyjobs."


def test_a_run_without_deps_fails_before_the_model_is_asked() -> None:
    from pydantic_tutor.agent import agent

    recorder = Recorder(call_tool=False)
    # The instructions read the deps for the first request, so a run that
    # forgets them fails before any request, whatever its model would do.
    with pytest.raises(
        expected_exception=AttributeError,
        match="'NoneType' object has no attribute 'root'",
    ):
        agent.run_sync(user_prompt="x", model=FunctionModel(function=recorder.reply))
    assert recorder.infos == []


def test_prompt_vs_instructions_shows_which_one_replays() -> None:
    # Goal 1: the system prompt comes back from the history; the
    # instructions are only the current agent's.
    printed = run_script("scripts/prompt_vs_instructions.py")

    assert printed.splitlines() == [
        "The first agent, a new run:",
        "  system prompt parts: ['You review Python code.']",
        "  instructions: Answer briefly.",
        "The second agent, continuing the first run's messages:",
        "  system prompt parts: ['You review Python code.']",
        "  instructions: Answer in detail.",
    ]


def test_inspect_run_shows_the_rendered_instructions() -> None:
    # Moved from the agent lesson, where it ran the agent without deps.
    printed = run_script("scripts/inspect_run.py")

    assert f"  instructions: {STATIC}\n\nThe repository is tinyjobs.\n" in printed
    assert "user-prompt" in printed


def test_scripted_run_prints_what_the_function_decided() -> None:
    # Moved from the agent lesson, where it ran the agent without deps. The
    # reply is built from the prompt and the instructions the model was
    # sent, so it can only match if the function, not a canned answer,
    # produced it.
    printed = run_script("scripts/scripted_run.py")

    expected = (
        "You asked 'What does this repo do?', and my instructions say: "
        f"{STATIC}\n\nThe repository is tinyjobs."
    )
    assert printed.strip() == expected


# Moved from the tools lesson, where it ran the agent without deps. Its
# point is that lesson's, so it passes before this lesson too.


def definitions_sent() -> dict[str, ToolDefinition]:
    """The tool definitions the student's agent sends with its first request."""
    recorder = run_recorded(root=TINYJOBS_ROOT)
    return {tool.name: tool for tool in recorder.infos[0].function_tools}


def test_show_tools_prints_what_the_model_is_told() -> None:
    printed = run_script("scripts/show_tools.py")
    tools = definitions_sent()

    assert "list_modules" in printed and "read_docstring" in printed
    for tool in tools.values():
        assert tool.description is not None and tool.description in printed
    module = tools["read_docstring"].parameters_json_schema["properties"]["module"]
    assert module["description"] in printed
