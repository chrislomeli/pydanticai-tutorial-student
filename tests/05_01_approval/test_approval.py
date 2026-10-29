"""Checks that the project from the approval lesson is in place.

Our checks for the approval lesson: they verify the lesson, and the student
is not asked to run them. Run them with `uv run pytest`. They fail until
the lesson is done. None of them needs an API key or makes a request: every
run is on a stand-in model.

Thirteen checks were moved here, because the chat now holds every write and
command for approval, and refuses it when no one answers. Each says where it
came from:

- Two from the file tools lesson (03-04) and six from the shell lesson
  (03-05) ran writes and commands through the chat, with no one to approve
  them. They prove the same points here with every call approved: approval
  is one more check, and the tools' own limits still hold after a yes.
- One from the memory lesson (04-05) tried to write into the notebook with
  a file tool; it is refused after a yes as before.
- Four, from the hooks (04-02), capability (04-03) and seam (04-06)
  lessons, counted the chat's capabilities or pinned the request log last
  of them. The guardrail added here declares itself innermost too, beside
  the log. One test here proves those points as the chat is now built.

The case that the notebook's write needs no approval moved to the deferred
lesson (06-02): the notebook must be loaded before it is called.

Each test runs in an empty folder of its own, so the project's own `.env`
and `work/` folder are never read. A test that needs the workspace folder
makes `work/tinyjobs` there, from the sample code, with a fake `.env`.

Every import of the project is inside the test that needs it, so that
before the lesson each test fails on its own, instead of the whole file
failing to load.
"""

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest
from pydantic_ai import Agent, RunContext, models
from pydantic_ai.capabilities import HandleDeferredToolCalls, LocalWorkspace
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    RetryPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import (
    DeferredToolRequests,
    DeferredToolResults,
    ToolApproved,
    ToolDenied,
)
from pydantic_ai_harness import FileSystem, ToolGuardrail

# This file is tests/05_01_approval/test_approval.py: two folders below the root.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
FAKE_TOKEN = "tj-FAKE-0000-not-a-real-secret"
PREFIX = "PYDANTIC_TUTOR_"
KEY_VARIABLES = ("OPENAI_API_KEY", "ANTHROPIC_API_KEY")
# Of the tools the chat offers, the ones that ask, by the lesson's policy.
ASKING = {
    "write_file",
    "edit_file",
    "create_directory",
    "run_command",
    "start_command",
}
NOT_ASKING = {
    "read_file",
    "list_directory",
    "search_files",
    "find_files",
    "file_info",
    "check_command",
    "stop_command",
    "list_modules",
    "read_docstring",
    "write_memory",
    "read_memory",
    "delete_memory",
    "search_memory",
    "list_memory",
}
UNATTENDED = (
    "Not run: no one was there to approve it. Tell the user what it would have done."
)


@pytest.fixture(autouse=True)
def no_real_models(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # None of this project's variables and no key, whatever the shell
    # running pytest has, and no real model call.
    for name in list(os.environ):
        if name.upper().startswith(PREFIX) or name.upper() in KEY_VARIABLES:
            monkeypatch.delenv(name)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(target=models, name="ALLOW_MODEL_REQUESTS", value=False)


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """`work/tinyjobs` in the empty folder: the sample code, and a fake `.env`."""
    root = tmp_path / "work" / "tinyjobs"
    shutil.copytree(src=PROJECT_ROOT / "sample_code" / "tinyjobs", dst=root)
    (root / ".env").write_text(f"TINYJOBS_RUNNER_TOKEN={FAKE_TOKEN}\n")
    return root


@dataclass(frozen=True)
class ToolResult:
    """What one tool call gave the model."""

    # The tool return's outcome ("success", "denied", "failed"), or "retry"
    # for a retry prompt, asking the model to try again.
    outcome: str
    content: object  # as the model received it

    @property
    def text(self) -> str:
        return str(self.content)


def one_call(*, tool_name: str, args: dict[str, object]) -> FunctionModel:
    """A model that makes one tool call, then ends with "done"."""

    def reply(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        for part in messages[-1].parts:
            if isinstance(part, ToolReturnPart | RetryPromptPart):
                return ModelResponse(parts=[TextPart(content="done")])
        return ModelResponse(parts=[ToolCallPart(tool_name=tool_name, args=args)])

    return FunctionModel(function=reply)


def outcome_of(*, messages: list[ModelMessage]) -> ToolResult:
    """The first tool return or retry prompt in a run's messages."""
    for message in messages:
        for part in message.parts:
            if isinstance(part, ToolReturnPart):
                return ToolResult(outcome=part.outcome, content=part.content)
            if isinstance(part, RetryPromptPart):
                # A tool the agent does not have is refused too; that is not
                # the refusal these tests are about.
                assert "Unknown tool name" not in str(part.content), part.content
                return ToolResult(outcome="retry", content=part.content)
    raise AssertionError("the tool was never called")


def chat_call(
    *,
    root: Path,
    tool_name: str,
    args: dict[str, object],
    approve: bool = False,
) -> ToolResult:
    """Call a tool of the project's chat, working in `root`.

    With `approve`, the run's answerer approves every held call; without it,
    the chat's own answer applies.
    """
    from pydantic_tutor.capabilities.approval import approver
    from pydantic_tutor.chat.agent import build_chat_agent
    from tutor_given.stand_ins import approve_all

    chat = build_chat_agent(web_search=False, workspace_root=root)
    result = chat.run_sync(
        user_prompt="x",
        model=one_call(tool_name=tool_name, args=args),
        capabilities=[approver(answer=approve_all)] if approve else [],
    )
    return outcome_of(messages=result.all_messages())


def command(*, root: Path, line: str, tool_name: str = "run_command") -> ToolResult:
    """Run one command line through the chat, every call approved."""
    return chat_call(
        root=root, tool_name=tool_name, args={"command": line}, approve=True
    )


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


# Goal 1: make the chat's writes and commands ask before they act; a refusal
# reaches the model as the tool's result.


def test_the_policy_holds_the_writes_and_commands_and_nothing_else() -> None:
    from pydantic_ai_harness import GuardrailResult
    from pydantic_ai_harness.guardrails import ToolCallInfo

    from pydantic_tutor.capabilities.approval import ASK_FIRST, ask_first

    # `shell` is the tool `Shell` adds only when asked: it asks too.
    assert ASKING | {"shell"} == ASK_FIRST
    assert not ASKING & NOT_ASKING

    def verdict(name: str) -> GuardrailResult:
        return ask_first(ToolCallInfo(name=name, args={}, tool_call_id="call-1"))

    for name in ASKING | {"shell"}:
        assert verdict(name) == GuardrailResult.approve(), name
    for name in NOT_ASKING:
        assert verdict(name) == GuardrailResult.allow(), name


def test_unattended_a_write_is_refused_and_nothing_is_written(workspace: Path) -> None:
    result = chat_call(
        root=workspace,
        tool_name="write_file",
        args={"path": "notes.txt", "content": "from the agent\n"},
    )
    # A tool return the model reads, not a retry prompt and not an error.
    assert result.outcome == "denied"
    assert result.content == UNATTENDED
    assert not (workspace / "notes.txt").exists()


def test_unattended_a_command_is_refused_and_does_not_start(workspace: Path) -> None:
    result = chat_call(
        root=workspace,
        tool_name="run_command",
        args={"command": "pytest -q --tb=no"},
    )
    assert result.outcome == "denied"
    assert result.content == UNATTENDED


def test_unattended_the_shells_opt_in_tool_is_refused_too(workspace: Path) -> None:
    from pydantic_ai.capabilities import LocalWorkspace as Workspace
    from pydantic_ai_harness import Shell

    from pydantic_tutor.capabilities.approval import (
        approver,
        ask_first,
        refuse_unattended,
    )

    agent: Agent[None, str] = Agent(
        capabilities=[
            Workspace(working_dir=workspace),
            Shell(allowed_commands=("pytest",), tools=["run_command", "shell"]),
            ToolGuardrail(guard=ask_first),
            approver(answer=refuse_unattended),
        ]
    )
    result = agent.run_sync(
        user_prompt="x",
        model=one_call(tool_name="shell", args={"command": "pytest --version"}),
    )
    assert outcome_of(messages=result.all_messages()).outcome == "denied"


def test_reads_do_not_ask(workspace: Path) -> None:
    read = chat_call(root=workspace, tool_name="read_file", args={"path": "README.md"})
    assert read.outcome == "success"
    assert "# tinyjobs" in read.text
    listed = chat_call(root=workspace, tool_name="list_modules", args={})
    assert listed.outcome == "success"


def test_with_no_answerer_the_run_stops_before_anything_runs(workspace: Path) -> None:
    from pydantic_tutor.capabilities.approval import ask_first

    # The guard alone: a held call has no one to answer it. A run whose
    # output is text cannot return the request, so it raises.
    agent: Agent[None, str] = Agent(
        capabilities=[
            LocalWorkspace(working_dir=workspace),
            FileSystem(),
            ToolGuardrail(guard=ask_first),
        ]
    )
    with pytest.raises(UserError):
        agent.run_sync(
            user_prompt="x",
            model=one_call(
                tool_name="write_file", args={"path": "a.txt", "content": "x\n"}
            ),
        )
    assert not (workspace / "a.txt").exists()


def test_the_chat_streams_the_refusal_too(workspace: Path) -> None:
    from pydantic_tutor.chat.agent import build_chat_agent

    # The chat reads its answers with `run_stream_sync`, as here. The test
    # model calls the one tool it is told to, and ends with what came back.
    chat = build_chat_agent(web_search=False, workspace_root=workspace)
    result = chat.run_stream_sync(
        user_prompt="x", model=TestModel(call_tools=["write_file"])
    )
    text = "".join(result.stream_text(delta=True))
    assert UNATTENDED in text


# Goal 2: the chat carries the guard and a safe answer; a run answers in its
# place, call by call.


def test_the_chat_carries_the_guardrail_and_an_answerer(workspace: Path) -> None:
    from pydantic_tutor.capabilities.approval import (
        APPROVER_ID,
        ask_first,
        refuse_unattended,
    )
    from pydantic_tutor.chat.agent import chat_capabilities

    stack = chat_capabilities(web_search=False, workspace_root=workspace)
    [guard] = [c for c in stack if isinstance(c, ToolGuardrail)]
    assert guard.guard is ask_first
    [answerer] = [c for c in stack if isinstance(c, HandleDeferredToolCalls)]
    assert answerer.handler is refuse_unattended
    assert answerer.id == APPROVER_ID


def approve_the_write_refuse_the_rest(
    ctx: RunContext[None], requests: DeferredToolRequests
) -> DeferredToolResults:
    answers: dict[str, ToolApproved | ToolDenied] = {}
    for call in requests.approvals:
        if call.tool_name == "write_file":
            answers[call.tool_call_id] = ToolApproved()
        else:
            answers[call.tool_call_id] = ToolDenied(message="Not now.")
    return requests.build_results(approvals=answers)


def two_calls() -> FunctionModel:
    """A model that asks for a write and a command in one response."""

    def reply(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        returns = [p for p in messages[-1].parts if isinstance(p, ToolReturnPart)]
        if returns:
            seen = "; ".join(f"{p.tool_name} {p.outcome}: {p.content}" for p in returns)
            return ModelResponse(parts=[TextPart(content=seen)])
        return ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name="write_file",
                    args={"path": "a.txt", "content": "from the agent\n"},
                ),
                ToolCallPart(
                    tool_name="run_command", args={"command": "pytest --version"}
                ),
            ]
        )

    return FunctionModel(function=reply)


def test_a_runs_answerer_replaces_the_chats_and_each_call_gets_its_own(
    workspace: Path,
) -> None:
    from pydantic_tutor.capabilities.approval import approver
    from pydantic_tutor.chat.agent import build_chat_agent

    chat = build_chat_agent(web_search=False, workspace_root=workspace)
    result = chat.run_sync(
        user_prompt="x",
        model=two_calls(),
        capabilities=[approver(answer=approve_the_write_refuse_the_rest)],
    )
    # One run, both calls answered: the write ran once, the command did not.
    assert (workspace / "a.txt").read_text() == "from the agent\n"
    assert "write_file success: Wrote" in result.output
    assert "run_command denied: Not now." in result.output
    # The chat is unchanged for the next run.
    again = chat_call(
        root=workspace,
        tool_name="write_file",
        args={"path": "b.txt", "content": "x\n"},
    )
    assert again.outcome == "denied"
    assert not (workspace / "b.txt").exists()


def test_an_answerer_without_the_id_is_asked_second(workspace: Path) -> None:
    from pydantic_tutor.chat.agent import build_chat_agent

    asked: list[str] = []

    def spy(
        ctx: RunContext[None], requests: DeferredToolRequests
    ) -> DeferredToolResults:
        asked.append("asked")
        return requests.build_results(approve_all=True)

    chat = build_chat_agent(web_search=False, workspace_root=workspace)
    # No `id`: both answerers stay, the chat's first, which refuses everything.
    result = chat.run_sync(
        user_prompt="x",
        model=two_calls(),
        capabilities=[HandleDeferredToolCalls(handler=spy)],
    )
    assert asked == []
    assert "write_file denied" in result.output
    assert not (workspace / "a.txt").exists()


def test_approve_changes_prints_what_was_asked_and_what_the_model_got(
    project: Path,
) -> None:
    result = run_script(name="approve_changes.py", folder=project)
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert any(
        line.startswith("  asked: write_file {'path': 'tests/") for line in lines
    )
    assert (
        "  asked: run_command {'command': 'pytest -q tests/test_verbose.py'}" in lines
    )
    assert any(
        line.startswith("  model got: write_file, success: Wrote") for line in lines
    )
    assert (
        "  model got: run_command, denied: Not now: I'll read the test before "
        "anything runs." in lines
    )
    assert "tests/test_verbose.py written: True" in lines


# Moved from earlier lessons: proved as the chat is now built, every call
# approved.


def test_the_chat_stack_ends_with_the_cap_the_log_and_the_guardrail(
    workspace: Path,
) -> None:
    from pydantic_ai.capabilities import Toolset, WebSearch
    from pydantic_ai_harness import Shell

    from pydantic_tutor.capabilities.batch_limit import BatchLimit
    from pydantic_tutor.capabilities.listable_memory import ListableMemory
    from pydantic_tutor.capabilities.prompt_file import PromptFile
    from pydantic_tutor.capabilities.request_log import request_log
    from pydantic_tutor.chat.agent import (
        PROMPT_PATH,
        build_chat_agent,
        chat_capabilities,
    )

    # Moved from 04-06: each part built anew on each call, ten parts, eleven
    # with web search.
    first = chat_capabilities(web_search=False, workspace_root=workspace)
    second = chat_capabilities(web_search=False, workspace_root=workspace)
    kinds = (
        PromptFile,
        LocalWorkspace,
        FileSystem,
        Shell,
        ToolGuardrail,
        HandleDeferredToolCalls,
        Toolset,
        ListableMemory,
        BatchLimit,
    )
    for kind in kinds:
        ours = [c for c in first if isinstance(c, kind)]
        assert len(ours) == 1, kind
        theirs = [c for c in second if isinstance(c, kind)]
        assert ours[0] is not theirs[0], kind
    assert request_log in first
    assert len(first) == 10
    assert not any(isinstance(c, WebSearch) for c in first)
    assert len(chat_capabilities(web_search=True, workspace_root=workspace)) == 11
    # Moved from 04-02, 04-03 and 04-04: as the agent sorts them, the prompt
    # is first of all, and the last three are the cap, the log and the
    # guardrail, the log and the guardrail both declared innermost.
    for web_search in (False, True):
        agent = build_chat_agent(web_search=web_search, workspace_root=workspace)
        capabilities = agent.root_capability.capabilities
        assert [type(c).__name__ for c in capabilities[:3]] == [
            "PromptFile",
            "ToolSearch",
            "PendingMessageDrainCapability",
        ]
        [prompt] = [c for c in capabilities if isinstance(c, PromptFile)]
        assert prompt.path == PROMPT_PATH
        assert isinstance(capabilities[-3], BatchLimit)
        assert request_log in capabilities[-2:]
        assert any(isinstance(c, ToolGuardrail) for c in capabilities[-2:])


def test_an_approved_write_is_written_in_the_workspace_folder(workspace: Path) -> None:
    # Moved from 03-04.
    wrote = chat_call(
        root=workspace,
        tool_name="write_file",
        args={"path": "notes.txt", "content": "from the agent\n"},
        approve=True,
    )
    assert wrote.outcome == "success", wrote.text
    assert (workspace / "notes.txt").read_text() == "from the agent\n"


def test_an_approved_write_keeps_the_default_protection(workspace: Path) -> None:
    # Moved from 03-04. Setting `read_only_patterns` yourself would replace
    # this list, and `.git/` would be writable.
    (workspace / ".git").mkdir()
    (workspace / ".git" / "config").write_text("[core]\n")
    for path in (".git/config", ".env"):
        result = chat_call(
            root=workspace,
            tool_name="write_file",
            args={"path": path, "content": "x\n"},
            approve=True,
        )
        assert result.outcome == "retry", f"{path} was written"
    assert (workspace / ".git" / "config").read_text() == "[core]\n"
    assert FAKE_TOKEN in (workspace / ".env").read_text()


def test_an_approved_write_may_not_touch_the_notebook(workspace: Path) -> None:
    # Moved from 04-05: the notebook is the memory tools' alone.
    notebook = workspace / ".agent-memory" / "main" / "logging.md"
    notebook.parent.mkdir(parents=True)
    notebook.write_text("- Logging goes to stderr.\n")
    result = chat_call(
        root=workspace,
        tool_name="write_file",
        args={"path": ".agent-memory/main/logging.md", "content": "x"},
        approve=True,
    )
    assert result.outcome == "retry"
    assert "denied by pattern" in result.text
    assert notebook.read_text() == "- Logging goes to stderr.\n"


def test_an_approved_command_runs_pytest_in_the_workspace_folder(
    workspace: Path,
) -> None:
    # Moved from 03-05. The sample's one failing test is on purpose.
    result = command(root=workspace, line="pytest -q --tb=no")
    assert result.outcome == "success"
    assert "test_cancel" in result.text
    assert "[exit code: 1]" in result.text


@pytest.mark.parametrize(
    argnames="line",
    argvalues=["rm README.md", "cat .env", "uv run pytest", 'sh -c "rm README.md"'],
)
def test_approved_a_command_that_does_not_start_with_pytest_is_refused(
    workspace: Path, line: str
) -> None:
    # Moved from 03-05: approval is one more check, in front of the shell's.
    result = command(root=workspace, line=line)
    assert result.outcome == "retry"
    assert "is not in the allowed list" in result.text
    assert (workspace / "README.md").exists()
    assert FAKE_TOKEN not in result.text


def test_approved_the_refusal_reaches_the_model_as_text_it_can_read(
    workspace: Path,
) -> None:
    result = command(root=workspace, line="rm README.md")
    assert result.content == "Command 'rm' is not in the allowed list."


def test_approved_the_file_tools_refuse_the_env_file_the_shell_prints(
    workspace: Path,
) -> None:
    # Moved from 03-05: the allow list checks a first word, so it is no boundary.
    read = chat_call(
        root=workspace, tool_name="read_file", args={"path": ".env"}, approve=True
    )
    assert read.outcome == "retry"
    assert FAKE_TOKEN not in read.text
    printed = command(root=workspace, line="pytest --version && cat .env")
    assert printed.outcome == "success"
    assert FAKE_TOKEN in printed.text


def test_approved_the_shell_reaches_outside_the_workspace_folder(
    workspace: Path,
) -> None:
    # Moved from 03-05. A person who approves a command approves all of it.
    outside = workspace.parent / "outside.txt"
    outside.write_text("outside the workspace\n")
    result = command(root=workspace, line="pytest --version; cat ../outside.txt")
    assert "outside the workspace" in result.text


def test_approved_pytest_runs_any_test_file_it_is_given(workspace: Path) -> None:
    # Moved from 03-05.
    (workspace / "test_peek.py").write_text(
        "def test_peek():\n    print(open('.env').read())\n"
    )
    result = command(root=workspace, line="pytest -q -s test_peek.py")
    assert FAKE_TOKEN in result.text
