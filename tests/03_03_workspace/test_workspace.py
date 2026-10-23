"""Checks that the project from the workspace lesson is in place.

Our checks for the workspace lesson: they verify the lesson, and the
student is not asked to run them. Run them with `uv run pytest`. They fail
until the lesson is done. None of them needs an API key or makes a
request: every run is on a scripted model.

Every import of the project is inside the test that needs it, so that
before the lesson each test fails on its own, instead of the whole file
failing to load.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic_ai import RunContext, models
from pydantic_ai.exceptions import UserError
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
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai.workspaces import LocalWorkspaceBackend

# This file is tests/03_03_workspace/test_workspace.py: two folders below the root.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def no_real_model(monkeypatch: pytest.MonkeyPatch) -> None:
    # Any request to a real model raises instead of reaching a provider.
    monkeypatch.setattr(target=models, name="ALLOW_MODEL_REQUESTS", value=False)


def make_codebase(*, root: Path, package: str) -> Path:
    """A codebase in the src layout, with one package and one module."""
    source = root / "src" / package
    source.mkdir(parents=True)
    (source / "__init__.py").write_text(f'"""The {package} package."""\n')
    (source / "util.py").write_text(f'"""Helpers for {package}."""\n')
    return root


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


async def working_dir(ctx: RunContext[object]) -> str:
    """A test-only tool: the folder of the run's workspace."""
    return await ctx.workspace.working_dir()


async def first_line(ctx: RunContext[object], *, path: str) -> str:
    """A test-only tool: a file's first line, read through the run's workspace."""
    return (await ctx.workspace.read_text(path=path)).splitlines()[0]


# Goal 1: the agent has a workspace, in the codebase each run is given.


def test_a_run_has_a_workspace_in_its_codebase(tmp_path: Path) -> None:
    from pydantic_tutor.agent import agent
    from pydantic_tutor.codebase_tools import CodebaseDeps

    for name in ("one", "two"):
        codebase = make_codebase(root=tmp_path / name, package=name)
        result = agent.run_sync(
            user_prompt="x",
            model=calls(tool_name="working_dir", args={}),
            deps=CodebaseDeps(root=codebase),
            toolsets=[FunctionToolset(tools=[working_dir])],
        )
        # The workspace's folder is the run's codebase, as an absolute path.
        assert result.output == str(codebase.resolve())


def test_list_modules_reads_through_the_workspace(tmp_path: Path) -> None:
    from pydantic_tutor.agent import agent
    from pydantic_tutor.codebase_tools import CodebaseDeps

    given = make_codebase(root=tmp_path / "given", package="given")
    other = make_codebase(root=tmp_path / "other", package="other")
    # A workspace passed to the run wins over the agent's own. If the tool
    # still read `deps.root` from disk, it would list `given`.
    result = agent.run_sync(
        user_prompt="x",
        model=calls(tool_name="list_modules", args={}),
        deps=CodebaseDeps(root=given),
        workspace=LocalWorkspaceBackend(other),
    )
    assert result.output == '["other","other.util"]'


def test_read_docstring_reads_through_the_workspace(tmp_path: Path) -> None:
    from pydantic_tutor.agent import agent
    from pydantic_tutor.codebase_tools import CodebaseDeps

    given = make_codebase(root=tmp_path / "given", package="given")
    other = make_codebase(root=tmp_path / "other", package="other")
    result = agent.run_sync(
        user_prompt="x",
        model=calls(tool_name="read_docstring", args={"module": "other.util"}),
        deps=CodebaseDeps(root=given),
        workspace=LocalWorkspaceBackend(other),
    )
    assert result.output == "Helpers for other."


def test_list_modules_does_not_enter_a_linked_folder(tmp_path: Path) -> None:
    from pydantic_tutor.agent import agent
    from pydantic_tutor.codebase_tools import CodebaseDeps

    codebase = make_codebase(root=tmp_path / "codebase", package="pkg")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "far.py").write_text('"""Outside the codebase."""\n')
    # One link back up (a loop, if entered) and one out of the codebase.
    (codebase / "src" / "pkg" / "up").symlink_to(target="..")
    (codebase / "src" / "pkg" / "out").symlink_to(target=outside)
    result = agent.run_sync(
        user_prompt="x",
        model=calls(tool_name="list_modules", args={}),
        deps=CodebaseDeps(root=codebase),
    )
    assert result.output == '["pkg","pkg.util"]'


def test_a_linked_file_out_of_the_codebase_is_not_read(tmp_path: Path) -> None:
    from pydantic_tutor.agent import agent
    from pydantic_tutor.codebase_tools import CodebaseDeps

    codebase = make_codebase(root=tmp_path / "codebase", package="pkg")
    (tmp_path / "far.py").write_text('"""Outside the codebase."""\n')
    (codebase / "src" / "pkg" / "leak.py").symlink_to(target=tmp_path / "far.py")
    listed = agent.run_sync(
        user_prompt="x",
        model=calls(tool_name="list_modules", args={}),
        deps=CodebaseDeps(root=codebase),
    )
    assert listed.output == '["pkg","pkg.util"]'
    read = agent.run_sync(
        user_prompt="x",
        model=calls(tool_name="read_docstring", args={"module": "pkg.leak"}),
        deps=CodebaseDeps(root=codebase),
    )
    assert "Outside the codebase" not in read.output


def test_a_codebase_without_src_has_no_modules(tmp_path: Path) -> None:
    from pydantic_tutor.agent import agent
    from pydantic_tutor.codebase_tools import CodebaseDeps

    result = agent.run_sync(
        user_prompt="x",
        model=calls(tool_name="list_modules", args={}),
        deps=CodebaseDeps(root=tmp_path),
    )
    # Read the tool's own return: an empty list reads as no text at all.
    returns = [
        part.content
        for message in result.all_messages()
        for part in message.parts
        if isinstance(part, ToolReturnPart)
    ]
    assert returns == [[]]


def test_a_folder_that_cannot_be_read_is_skipped(tmp_path: Path) -> None:
    from pydantic_tutor.agent import agent
    from pydantic_tutor.codebase_tools import CodebaseDeps

    codebase = make_codebase(root=tmp_path / "codebase", package="pkg")
    shut = codebase / "src" / "pkg" / "shut"
    shut.mkdir()
    (shut / "hidden.py").write_text('"""Cannot be listed."""\n')
    shut.chmod(mode=0)
    try:
        result = agent.run_sync(
            user_prompt="x",
            model=calls(tool_name="list_modules", args={}),
            deps=CodebaseDeps(root=codebase),
        )
    finally:
        shut.chmod(mode=0o755)  # so that pytest can remove it
    # As `Path.rglob` did: the folder is left out, and the run carries on.
    assert result.output == '["pkg","pkg.util"]'


def test_a_module_that_cannot_be_opened_fails_the_tool_not_the_run(
    tmp_path: Path,
) -> None:
    from pydantic_tutor.agent import agent
    from pydantic_tutor.codebase_tools import CodebaseDeps

    codebase = make_codebase(root=tmp_path / "codebase", package="pkg")
    # Listed, as `Path.rglob` listed it, but there is nothing to open.
    (codebase / "src" / "pkg" / "broken.py").symlink_to(target="nowhere.py")
    result = agent.run_sync(
        user_prompt="x",
        model=calls(tool_name="read_docstring", args={"module": "pkg.broken"}),
        deps=CodebaseDeps(root=codebase),
    )
    # The model is told, and answers; the run is not ended by the error.
    assert "pkg.broken cannot be opened" in result.output


# Goal 2: what the workspace does not give you. These pin the gaps as they
# are at 2.56, so a release that changes them fails here, and the lesson
# is updated.


def test_the_workspace_is_not_a_boundary(tmp_path: Path) -> None:
    from pydantic_tutor.agent import agent
    from pydantic_tutor.codebase_tools import CodebaseDeps

    codebase = make_codebase(root=tmp_path / "codebase", package="pkg")
    (tmp_path / "outside.txt").write_text("outside the codebase\n")
    result = agent.run_sync(
        user_prompt="x",
        model=calls(tool_name="first_line", args={"path": "../outside.txt"}),
        deps=CodebaseDeps(root=codebase),
        toolsets=[FunctionToolset(tools=[first_line])],
    )
    assert result.output == "outside the codebase"


def test_a_conversation_stays_in_its_workspace(tmp_path: Path) -> None:
    from pydantic_tutor.agent import agent
    from pydantic_tutor.codebase_tools import CodebaseDeps

    first = make_codebase(root=tmp_path / "first", package="first")
    second = make_codebase(root=tmp_path / "second", package="second")
    model = calls(tool_name="list_modules", args={})
    turn_one = agent.run_sync(
        user_prompt="x", model=model, deps=CodebaseDeps(root=first)
    )
    history = turn_one.all_messages()

    # The same codebase carries on.
    agent.run_sync(
        user_prompt="y",
        model=model,
        deps=CodebaseDeps(root=first),
        message_history=history,
    )
    # Another one is refused: the history names the first workspace.
    with pytest.raises(
        expected_exception=UserError, match="message history continues in workspace"
    ):
        agent.run_sync(
            user_prompt="y",
            model=model,
            deps=CodebaseDeps(root=second),
            message_history=history,
        )


def test_workspace_reach_prints_a_file_inside_and_one_outside() -> None:
    result = subprocess.run(
        args=[sys.executable, "scripts/workspace_reach.py"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
        # The banner goes to stderr; turning it off keeps a failure readable.
        env={**os.environ, "PYDANTIC_AI_NO_BANNER": "1"},
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [
        "README.md: # tinyjobs",
        "../../pyproject.toml: [project]",
    ]
