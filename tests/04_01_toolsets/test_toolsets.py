"""Checks that the project from the toolsets lesson is in place.

Our checks for the toolsets lesson: they verify the lesson, and the student
is not asked to run them. Run them with `uv run pytest`. They fail until
the lesson is done. None of them needs an API key or makes a request:
every run is on a stand-in model.

Two checks were moved here from the shell lesson (03-05), which this lesson
changes: that the chat is offered its tools, and that `workspace_reach.py`
prints how many. The toolset adds two, so both now say fourteen.

Each test runs in an empty folder of its own, so the project's own `.env`
and `work/` folder are never read. A test that needs the workspace folder
makes `work/tinyjobs` there, from the sample code.

Every import of the project is inside the test that needs it, so that
before the lesson each test fails on its own, instead of the whole file
failing to load.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic_ai import Agent, RunContext, models
from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    RetryPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.tools import ToolDefinition

# This file is tests/04_01_toolsets/test_toolsets.py: two folders below the root.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
PREFIX = "PYDANTIC_TUTOR_"
KEY_VARIABLES = ("OPENAI_API_KEY", "ANTHROPIC_API_KEY")
MODULES = [
    "tinyjobs",
    "tinyjobs.cli",
    "tinyjobs.jobs",
    "tinyjobs.log",
    "tinyjobs.storage",
]
FILE_AND_SHELL_TOOLS = 12


@pytest.fixture(autouse=True)
def no_real_models(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # An empty folder with none of this project's variables, whatever the
    # shell running pytest has, and no real model call.
    for name in list(os.environ):
        if name.upper().startswith(PREFIX) or name.upper() in KEY_VARIABLES:
            monkeypatch.delenv(name)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(target=models, name="ALLOW_MODEL_REQUESTS", value=False)


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """`work/tinyjobs` in the empty folder: the sample code."""
    root = tmp_path / "work" / "tinyjobs"
    shutil.copytree(src=PROJECT_ROOT / "sample_code" / "tinyjobs", dst=root)
    return root


class Sent:
    """A model that records what it is sent, then makes the calls it is given."""

    def __init__(self, *, calls: list[ToolCallPart] | None = None) -> None:
        self.calls = list(calls or [])
        self.tools: list[str] = []
        self.instructions: str | None = None
        self.returns: list[ToolReturnPart | RetryPromptPart] = []

    def model(self) -> FunctionModel:
        def reply(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            self.tools = [tool.name for tool in info.function_tools]
            self.instructions = info.instructions
            for part in messages[-1].parts:
                if isinstance(part, ToolReturnPart | RetryPromptPart):
                    self.returns.append(part)
            if self.calls:
                return ModelResponse(parts=[self.calls.pop(0)])
            return ModelResponse(parts=[TextPart(content="done")])

        return FunctionModel(function=reply)

    @property
    def result(self) -> ToolReturnPart | RetryPromptPart:
        """What the model's one call gave back."""
        assert self.returns, "the model never got a tool's result back"
        return self.returns[-1]


def call(*, tool_name: str, **args: str) -> ToolCallPart:
    return ToolCallPart(tool_name=tool_name, args=args)


def run_with(*, toolset: object, sent: Sent) -> None:
    from pydantic_ai.toolsets import AbstractToolset

    assert isinstance(toolset, AbstractToolset)
    # A second agent with nothing but the toolset: no workspace, no deps.
    agent: Agent[None, str] = Agent(toolsets=[toolset])
    agent.run_sync(user_prompt="x", model=sent.model())


# Point 1: the tools, their instructions and their time limit are one object.


def test_the_toolset_is_the_two_tools_with_their_instructions(
    workspace: Path,
) -> None:
    from pydantic_tutor.tools.workspace_tools import build_workspace_toolset

    sent = Sent()
    run_with(toolset=build_workspace_toolset(root=workspace), sent=sent)
    assert sent.tools == ["list_modules", "read_docstring"]
    assert sent.instructions == "Your workspace holds the Python project tinyjobs."


def test_list_modules_names_the_modules_under_src(workspace: Path) -> None:
    from pydantic_tutor.tools.workspace_tools import build_workspace_toolset

    sent = Sent(calls=[call(tool_name="list_modules")])
    run_with(toolset=build_workspace_toolset(root=workspace), sent=sent)
    assert isinstance(sent.result, ToolReturnPart)
    assert sent.result.content == MODULES


def test_read_docstring_reads_the_docstring(workspace: Path) -> None:
    from pydantic_tutor.tools.workspace_tools import build_workspace_toolset

    sent = Sent(calls=[call(tool_name="read_docstring", module="tinyjobs.storage")])
    run_with(toolset=build_workspace_toolset(root=workspace), sent=sent)
    assert sent.result.content == "Saving and loading the queue as a JSON file."


@pytest.mark.parametrize(
    argnames="module", argvalues=["jobs", "../../.env", "/etc/passwd"]
)
def test_a_name_that_is_not_listed_is_a_retry_that_reads_nothing(
    workspace: Path, module: str
) -> None:
    from pydantic_tutor.tools.workspace_tools import build_workspace_toolset

    (workspace / ".env").write_text(data="TOKEN=tj-FAKE\n")
    sent = Sent(calls=[call(tool_name="read_docstring", module=module)])
    run_with(toolset=build_workspace_toolset(root=workspace), sent=sent)
    assert isinstance(sent.result, RetryPromptPart)
    assert f"There is no module {module!r}." in str(sent.result.content)
    assert "Call list_modules to see them." in str(sent.result.content)
    assert "tj-FAKE" not in str(sent.result.content)


def test_a_module_that_does_not_parse_is_a_failed_tool_result(
    tmp_path: Path,
) -> None:
    from pydantic_tutor.tools.workspace_tools import build_workspace_toolset

    package = tmp_path / "broken_project" / "src" / "broken"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(data="def half_written(:\n")
    sent = Sent(calls=[call(tool_name="read_docstring", module="broken")])
    run_with(
        toolset=build_workspace_toolset(root=tmp_path / "broken_project"), sent=sent
    )
    # The run goes on: the model is told the call failed, and answers.
    assert isinstance(sent.result, ToolReturnPart)
    assert sent.result.outcome == "failed"
    assert "broken cannot be read" in sent.result.model_response_str()


def test_the_time_limit_is_on_the_toolset(workspace: Path) -> None:
    from pydantic_tutor.tools.workspace_tools import build_workspace_toolset

    # An agent's `tool_timeout` covers only its own `tools=`, so the limit
    # must be on the toolset to stop a call that hangs.
    assert build_workspace_toolset(root=workspace).timeout == 10


def test_the_chat_is_offered_the_toolset_and_the_tools_it_had(
    workspace: Path,
) -> None:
    # Moved from 03-05, where the chat was offered twelve tools.
    from pydantic_tutor.composition import build_app
    from pydantic_tutor.settings import Settings

    sent = Sent()
    app = build_app(
        settings=Settings(chat_model="test", web_search=False, workspace_root=workspace)
    )
    app.agent.run_sync(user_prompt="hi", model=sent.model())
    assert len(sent.tools) == FILE_AND_SHELL_TOOLS + 2
    assert {"list_modules", "read_docstring", "read_file", "run_command"} <= set(
        sent.tools
    )


def test_the_chat_sends_the_toolsets_instructions_after_its_own(
    workspace: Path,
) -> None:
    from pydantic_tutor.chat.agent import build_chat_agent

    sent = Sent()
    agent = build_chat_agent(web_search=False, workspace_root=workspace)
    agent.run_sync(user_prompt="hi", model=sent.model())
    assert sent.instructions == (
        "You are a helpful assistant. Answer briefly.\n\n"
        "Your workspace holds the Python project tinyjobs."
    )


def test_the_chat_tools_read_the_folder_the_chat_was_built_with(
    workspace: Path, tmp_path: Path
) -> None:
    from pydantic_tutor.chat.agent import build_chat_agent

    other = tmp_path / "work" / "other"
    (other / "src" / "lonely").mkdir(parents=True)
    (other / "src" / "lonely" / "__init__.py").write_text(data='"""Alone."""\n')
    sent = Sent(calls=[call(tool_name="list_modules")])
    agent = build_chat_agent(web_search=False, workspace_root=other)
    agent.run_sync(user_prompt="hi", model=sent.model())
    assert sent.result.content == ["lonely"]


# Point 2: a wrapped toolset is a new one; the original is unchanged.


def test_prefixed_renames_the_tools_and_the_instructions_still_pass(
    workspace: Path,
) -> None:
    from pydantic_tutor.tools.workspace_tools import build_workspace_toolset

    toolset = build_workspace_toolset(root=workspace)
    sent = Sent()
    run_with(toolset=toolset.prefixed(prefix="workspace"), sent=sent)
    assert sent.tools == ["workspace_list_modules", "workspace_read_docstring"]
    assert sent.instructions == "Your workspace holds the Python project tinyjobs."

    # The original is not changed.
    again = Sent()
    run_with(toolset=toolset, sent=again)
    assert again.tools == ["list_modules", "read_docstring"]


def test_a_prefix_does_not_rewrite_the_text_that_names_a_tool(
    workspace: Path,
) -> None:
    from pydantic_tutor.tools.workspace_tools import build_workspace_toolset

    toolset = build_workspace_toolset(root=workspace).prefixed(prefix="workspace")
    sent = Sent(calls=[call(tool_name="workspace_read_docstring", module="jobs")])
    run_with(toolset=toolset, sent=sent)
    # The retry still tells the model to call a tool this agent does not have.
    assert "Call list_modules to see them." in str(sent.result.content)


def test_filtered_offers_only_what_the_function_keeps(workspace: Path) -> None:
    from pydantic_tutor.tools.workspace_tools import build_workspace_toolset

    def only_listing(ctx: RunContext[None], tool: ToolDefinition) -> bool:
        return tool.name == "list_modules"

    toolset = build_workspace_toolset(root=workspace).filtered(filter_func=only_listing)
    sent = Sent(calls=[call(tool_name="read_docstring", module="tinyjobs.jobs")])
    run_with(toolset=toolset, sent=sent)
    assert sent.tools == ["list_modules"]
    assert isinstance(sent.result, RetryPromptPart)
    assert "Unknown tool name: 'read_docstring'" in str(sent.result.content)


def test_a_tool_name_can_be_used_once(workspace: Path) -> None:
    from pydantic_ai.exceptions import UserError

    from pydantic_tutor.tools.workspace_tools import build_workspace_toolset

    def list_modules() -> list[str]:
        """A second tool with the toolset's name."""
        return []

    toolset = build_workspace_toolset(root=workspace)
    agent: Agent[None, str] = Agent(tools=[list_modules], toolsets=[toolset])
    with pytest.raises(UserError, match="list_modules"):
        agent.run_sync(user_prompt="x", model=Sent().model())


# The scripts.


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


@pytest.fixture
def project(tmp_path: Path) -> Path:
    """A stand-in project folder, with a fresh working copy made by the script."""
    shutil.copytree(src=PROJECT_ROOT / "sample_code", dst=tmp_path / "sample_code")
    shutil.copy(src=PROJECT_ROOT / "pyproject.toml", dst=tmp_path / "pyproject.toml")
    made = run_script(name="fresh_copy.py", folder=tmp_path)
    assert made.returncode == 0, made.stderr
    return tmp_path


def test_workspace_reach_now_counts_the_toolset_too(project: Path) -> None:
    # Moved from 03-05, where it printed 12.
    result = run_script(name="workspace_reach.py", folder=project)
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert "tools the model is offered: 14" in lines
    assert "README.md: # tinyjobs" in lines


def test_toolset_views_prints_the_chat_and_what_each_wrapper_sends(
    project: Path,
) -> None:
    result = run_script(name="toolset_views.py", folder=project)
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert any(line.startswith("  tools: read_file, ") for line in lines)
    assert any(line.endswith(", list_modules, read_docstring") for line in lines)
    assert (
        "  read_docstring('tinyjobs.jobs') returned: "
        "The job record and the queue that holds jobs." in lines
    )
    assert "  tools: list_modules, read_docstring" in lines
    assert "  tools: workspace_list_modules, workspace_read_docstring" in lines
    assert "  tools: list_modules" in lines
    assert (
        "  the model got back: There is no module 'jobs'. "
        "Call list_modules to see them." in lines
    )
    assert (
        "  instructions: 'Your workspace holds the Python project tinyjobs.'" in lines
    )
