"""Checks that the project from the approval lesson is in place.

Our checks for the approval lesson: they verify the lesson, and the
student is not asked to run them. Run them with `uv run pytest`. They fail
until the lesson is done. None of them needs an API key or makes a
request: every run is on a scripted model.

None of them touches your own `work/` folder: each makes a working copy of
its own, with your `scripts/fresh_copy.py`, in a folder of its own.

Twelve tests were moved here, because the agent now holds every write
and command for approval, and refuses it when no one answers. Each says
where it came from:

- Two from the file tools lesson (03-04) and six from the shell lesson
  (03-05) ran writes and commands through the agent, with no one to
  approve them. They prove the same points here with every call approved:
  approval is one more check, and the tools' own limits still hold after
  a yes.
- Two from the seam lesson (04-06), and one each from the hooks (04-02)
  and capability (04-03) lessons, counted the agent's capabilities or
  pinned the request log last of them. The guardrail added here declares
  itself innermost too, beside the log. Two tests here prove those four
  points as the agent is now built.

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
from pydantic_ai.tools import DeferredToolRequests, DeferredToolResults, ToolDenied

# This file is tests/05_01_approval/test_approval.py: two folders below the root.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
TINYJOBS_ROOT: Path = PROJECT_ROOT / "sample_repos" / "tinyjobs"
FAKE_TOKEN = "tj-FAKE-0000-not-a-real-secret"
PREFIX = "PYDANTIC_TUTOR_"
# Of the tools the agent offers, the ones that ask, by the lesson's policy.
ASKS = {
    "write_file",
    "edit_file",
    "create_directory",
    "run_command",
    "start_command",
}
LOGGER = "pydantic_tutor.request_log"


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    # None of this project's variables and no key, whatever the shell
    # running pytest has, and no real model call.
    for name in list(os.environ):
        if name.upper().startswith(PREFIX) or name.upper() == "OPENAI_API_KEY":
            monkeypatch.delenv(name)
    monkeypatch.setattr(target=models, name="ALLOW_MODEL_REQUESTS", value=False)
    # The shell finds `pytest` on PATH, as it does under `uv run`: put this
    # project's environment first, however pytest itself was started.
    monkeypatch.setenv(
        "PATH", f"{Path(sys.executable).parent}{os.pathsep}{os.environ['PATH']}"
    )


def run_script(*, name: str, cwd: Path) -> str:
    """Run one of the project's scripts from `cwd`; return its stdout."""
    result = subprocess.run(
        args=[sys.executable, name],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
        # The banner goes to stderr; turning it off keeps a failure readable.
        env={**os.environ, "PYDANTIC_AI_NO_BANNER": "1"},
    )
    assert result.returncode == 0, f"{name} failed:\n{result.stderr}"
    return result.stdout


@pytest.fixture
def project(tmp_path: Path) -> Path:
    """A stand-in project folder: the sample repo and the lesson's scripts, nothing else."""
    root = tmp_path / "project"
    shutil.copytree(src=TINYJOBS_ROOT, dst=root / "sample_repos" / "tinyjobs")
    (root / "scripts").mkdir()
    for script in ("fresh_copy.py", "approve_changes.py"):
        # Before the lesson there is no approve_changes.py: the test that
        # runs it fails there, not every test that makes a copy.
        if (PROJECT_ROOT / "scripts" / script).is_file():
            shutil.copy(src=PROJECT_ROOT / "scripts" / script, dst=root / "scripts")
    return root


@pytest.fixture
def working_copy(project: Path) -> Path:
    """A working copy, made by the student's `scripts/fresh_copy.py`."""
    run_script(name="scripts/fresh_copy.py", cwd=project)
    return project / "work" / "tinyjobs"


@dataclass(frozen=True)
class ToolResult:
    """What one tool call gave the model."""

    tool_name: str
    # The tool return's outcome ("success", "denied", "failed"), or "retry"
    # for a retry prompt, asking the model to try again.
    outcome: str
    text: str


@dataclass(frozen=True)
class Run:
    """What a scripted run gave the model, and how it ended."""

    results: list[ToolResult]
    output: str
    requests: int


def run_agent(
    *,
    root: Path,
    responses: list[list[ToolCallPart]],
    approved: bool = False,
) -> Run:
    """Run the student's agent on `root`, with a model that sends `responses`, one per request.

    `approved`: a run's answerer approves every call held for approval;
    otherwise the agent's own answers.
    """
    from pydantic_tutor import approval
    from pydantic_tutor.agent import agent
    from pydantic_tutor.repo_tools import RepoDeps

    queue = list(responses)

    def reply(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if queue:
            return ModelResponse(parts=list(queue.pop(0)))
        return ModelResponse(parts=[TextPart(content="done")])

    result = agent.run_sync(
        user_prompt="x",
        model=FunctionModel(function=reply),
        deps=RepoDeps(root=root),
        capabilities=[approval.approver(answer=approval.approve_all)]
        if approved
        else [],
    )
    results: list[ToolResult] = []
    for message in result.all_messages():
        if not isinstance(message, ModelRequest):
            continue
        for part in message.parts:
            if isinstance(part, ToolReturnPart):
                results.append(
                    ToolResult(
                        tool_name=part.tool_name,
                        outcome=part.outcome,
                        text=str(part.content),
                    )
                )
            if isinstance(part, RetryPromptPart):
                results.append(
                    ToolResult(
                        tool_name=str(part.tool_name),
                        outcome="retry",
                        text=str(part.content),
                    )
                )
    assert len(results) == sum(len(r) for r in responses), "a tool was never called"
    return Run(results=results, output=result.output, requests=result.usage.requests)


def one_call(
    *, root: Path, tool_name: str, args: dict[str, object], approved: bool = False
) -> ToolResult:
    """Run one tool call through the student's agent; what the model got back."""
    call = ToolCallPart(tool_name=tool_name, args=args)
    return run_agent(root=root, responses=[[call]], approved=approved).results[0]


def approved_command(*, root: Path, command: str) -> ToolResult:
    """Run one shell command through the agent, approved; what the model got back."""
    return one_call(
        root=root, tool_name="run_command", args={"command": command}, approved=True
    )


# Goal 1: make the agent's writes and commands ask before they act; a
# refusal reaches the model as the tool's result.


def test_the_policy_holds_the_writes_and_commands_and_nothing_else(
    working_copy: Path,
) -> None:
    from pydantic_ai_harness.guardrails import ToolCallInfo

    from pydantic_tutor import approval
    from pydantic_tutor.agent import agent
    from pydantic_tutor.repo_tools import RepoDeps

    offered: list[str] = []

    def reply(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        offered.extend(tool.name for tool in info.function_tools)
        return ModelResponse(parts=[TextPart(content="done")])

    agent.run_sync(
        user_prompt="x",
        model=FunctionModel(function=reply),
        deps=RepoDeps(root=working_copy),
    )
    # Of the tools the agent offers, the writes and commands ask, and the
    # rest run. `Shell`'s opt-in `shell` tool, which the agent does not
    # offer, asks too.
    asking = {*ASKS, "shell"}
    assert asking == approval.ASK_FIRST
    assert set(offered) >= ASKS
    for name in sorted({*offered, "shell"}):
        verdict = approval.ask_first(ToolCallInfo(name=name, args={}, tool_call_id="c"))
        expected = "approve" if name in asking else "allow"
        assert verdict.action == expected, name


def test_unattended_the_shells_opt_in_tool_is_refused_too(working_copy: Path) -> None:
    # The harness's `Shell` can also offer a `shell` tool, which runs a
    # command as `run_command` does. A policy that named only the tools
    # that ask would let it run unasked; this one holds it.
    from pydantic_ai_harness import Shell
    from pydantic_ai_harness.shell import SHELL_TOOL_NAMES

    from pydantic_tutor import approval
    from pydantic_tutor.agent import SHELL_COMMANDS, build_agent, repo_capabilities
    from pydantic_tutor.repo_tools import RepoDeps

    stack = [
        Shell(allowed_commands=SHELL_COMMANDS, tools=SHELL_TOOL_NAMES)
        if isinstance(c, Shell)
        else c
        for c in repo_capabilities()
    ]
    agent = build_agent(capabilities=stack)
    calls = [
        ToolCallPart(
            tool_name="shell", args={"command": "pytest --version && touch ran.txt"}
        )
    ]
    seen: list[ToolReturnPart] = []

    def reply(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        returns = [p for p in messages[-1].parts if isinstance(p, ToolReturnPart)]
        if returns:
            seen.extend(returns)
            return ModelResponse(parts=[TextPart(content="done")])
        return ModelResponse(parts=list(calls))

    agent.run_sync(
        user_prompt="x",
        model=FunctionModel(function=reply),
        deps=RepoDeps(root=working_copy),
    )
    assert [(p.tool_name, p.outcome, p.content) for p in seen] == [
        ("shell", "denied", approval.UNATTENDED)
    ]
    assert not (working_copy / "ran.txt").exists()


@pytest.mark.parametrize(
    argnames=("tool_name", "args"),
    argvalues=[
        ("write_file", {"path": "notes.txt", "content": "from the agent\n"}),
        (
            "edit_file",
            {"path": "README.md", "old_text": "# tinyjobs", "new_text": "# changed"},
        ),
        ("create_directory", {"path": "made_by_the_agent"}),
        ("run_command", {"command": "pytest --version && touch ran.txt"}),
        ("start_command", {"command": "pytest --version && touch ran.txt"}),
    ],
)
def test_unattended_a_change_is_refused_and_the_model_told(
    working_copy: Path, tool_name: str, args: dict[str, object]
) -> None:
    # The lesson's point: the call never runs, and the refusal is the tool's
    # result, a tool return marked denied, not an error and not a retry.
    from pydantic_tutor import approval

    readme = (working_copy / "README.md").read_text()
    run = run_agent(
        root=working_copy,
        responses=[[ToolCallPart(tool_name=tool_name, args=args)]],
    )
    assert run.results == [
        ToolResult(tool_name=tool_name, outcome="denied", text=approval.UNATTENDED)
    ]
    assert not (working_copy / "notes.txt").exists()
    assert (working_copy / "README.md").read_text() == readme
    assert not (working_copy / "made_by_the_agent").exists()
    assert not (working_copy / "ran.txt").exists()
    # The run went on: the model was sent the refusal, and answered.
    assert run.output == "done"
    assert run.requests == 2


@pytest.mark.parametrize(
    argnames=("tool_name", "args"),
    argvalues=[
        ("read_file", {"path": "README.md", "limit": 1}),
        ("list_modules", {}),
        ("write_memory", {"file": "MEMORY.md", "content": "- a note\n"}),
    ],
)
def test_reads_and_the_notebook_do_not_ask(
    working_copy: Path, tool_name: str, args: dict[str, object]
) -> None:
    result = one_call(root=working_copy, tool_name=tool_name, args=args)
    assert result.outcome == "success", result.text


def test_with_no_answerer_the_run_stops_before_anything_runs(
    working_copy: Path,
) -> None:
    # Why the agent carries an answerer: a call held for approval that no
    # one answers ends the run with an error, here, where the agent's
    # output is text.
    from pydantic_ai.capabilities import HandleDeferredToolCalls

    from pydantic_tutor.agent import build_agent, repo_capabilities
    from pydantic_tutor.repo_tools import RepoDeps

    stack = [
        c for c in repo_capabilities() if not isinstance(c, HandleDeferredToolCalls)
    ]
    agent = build_agent(capabilities=stack)
    calls = [
        ToolCallPart(tool_name="write_file", args={"path": "notes.txt", "content": "x"})
    ]

    def reply(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=list(calls))

    with pytest.raises(UserError, match="DeferredToolRequests"):
        agent.run_sync(
            user_prompt="x",
            model=FunctionModel(function=reply),
            deps=RepoDeps(root=working_copy),
        )
    assert not (working_copy / "notes.txt").exists()


# Goal 2: the agent carries a safe answer; a run answers in its place,
# call by call.


def test_the_agent_carries_the_guardrail_and_an_answerer() -> None:
    from pydantic_ai.capabilities import HandleDeferredToolCalls
    from pydantic_ai_harness import ToolGuardrail

    from pydantic_tutor import approval
    from pydantic_tutor.agent import agent, repo_capabilities

    stack = repo_capabilities()
    guards = [c for c in stack if isinstance(c, ToolGuardrail)]
    assert len(guards) == 1
    assert guards[0].guard is approval.ask_first
    answerers = [c for c in stack if isinstance(c, HandleDeferredToolCalls)]
    assert len(answerers) == 1
    assert answerers[0].id == approval.APPROVER_ID
    assert answerers[0].handler is approval.refuse_unattended
    assert any(
        isinstance(c, HandleDeferredToolCalls)
        for c in agent.root_capability.capabilities
    )
    run_answerer = approval.approver(answer=approval.approve_all)
    assert run_answerer.id == approval.APPROVER_ID
    assert run_answerer.handler is approval.approve_all


def test_a_runs_answerer_replaces_the_agents(working_copy: Path) -> None:
    run = run_agent(
        root=working_copy,
        responses=[
            [
                ToolCallPart(
                    tool_name="write_file",
                    args={"path": "notes.txt", "content": "from the agent\n"},
                ),
                ToolCallPart(
                    tool_name="run_command",
                    args={"command": "pytest --version && touch ran.txt"},
                ),
            ]
        ],
        approved=True,
    )
    assert [r.outcome for r in run.results] == ["success", "success"]
    assert (working_copy / "notes.txt").read_text() == "from the agent\n"
    assert (working_copy / "ran.txt").exists()


def test_each_call_gets_its_own_answer(working_copy: Path) -> None:
    from pydantic_tutor import approval
    from pydantic_tutor.agent import agent
    from pydantic_tutor.repo_tools import RepoDeps

    asked: list[list[str]] = []

    def write_only(
        ctx: RunContext[RepoDeps], requests: DeferredToolRequests
    ) -> DeferredToolResults:
        asked.append([call.tool_name for call in requests.approvals])
        return requests.build_results(
            approvals={
                call.tool_call_id: (
                    True
                    if call.tool_name == "write_file"
                    else ToolDenied(message="Not the shell.")
                )
                for call in requests.approvals
            }
        )

    calls = [
        ToolCallPart(
            tool_name="write_file", args={"path": "notes.txt", "content": "x"}
        ),
        ToolCallPart(
            tool_name="run_command",
            args={"command": "pytest --version && touch ran.txt"},
        ),
    ]
    seen: list[ToolReturnPart] = []

    def reply(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        returns = [p for p in messages[-1].parts if isinstance(p, ToolReturnPart)]
        if returns:
            seen.extend(returns)
            return ModelResponse(parts=[TextPart(content="done")])
        return ModelResponse(parts=list(calls))

    agent.run_sync(
        user_prompt="x",
        model=FunctionModel(function=reply),
        deps=RepoDeps(root=working_copy),
        capabilities=[approval.approver(answer=write_only)],
    )
    # Both calls of the one response are asked about together, once.
    assert asked == [["write_file", "run_command"]]
    assert [(p.tool_name, p.outcome) for p in seen] == [
        ("write_file", "success"),
        ("run_command", "denied"),
    ]
    assert seen[1].content == "Not the shell."
    assert (working_copy / "notes.txt").exists()
    assert not (working_copy / "ran.txt").exists()


def test_approve_changes_prints_what_was_asked_and_what_the_model_got(
    working_copy: Path, project: Path
) -> None:
    printed = run_script(name="scripts/approve_changes.py", cwd=project).splitlines()
    assert printed[0].startswith(
        "  asked: write_file {'path': 'tests/test_verbose.py', 'content': "
    )
    assert (
        printed[1]
        == "  asked: run_command {'command': 'pytest -q tests/test_verbose.py'}"
    )
    assert printed[2].startswith(
        "  model got: write_file, success: Wrote 136 chars (5 lines) to tests/test_verbose.py."
    )
    assert printed[3] == (
        "  model got: run_command, denied: Not now: I'll read the test before anything runs."
    )
    assert printed[4] == "tests/test_verbose.py written: True"
    assert len(printed) == 5
    assert "--verbose" in (working_copy / "tests" / "test_verbose.py").read_text()


# Moved from earlier lessons: proved as the agent is now built.


def test_repo_capabilities_are_new_on_each_call_and_hold_the_whole_stack() -> None:
    # Moved from 04-06, which counted eight: the guardrail and the
    # answerer make ten.
    from pydantic_ai.capabilities import HandleDeferredToolCalls, Toolset
    from pydantic_ai_harness import FileSystem, Shell, ToolGuardrail

    from pydantic_tutor.agent import repo_capabilities, repo_workspace
    from pydantic_tutor.batch_limit import BatchLimit
    from pydantic_tutor.listable_memory import ListableMemory
    from pydantic_tutor.prompt_file import PromptFile
    from pydantic_tutor.repo_tools import repo_toolset
    from pydantic_tutor.request_log import request_log

    first, second = repo_capabilities(), repo_capabilities()
    kinds = (
        PromptFile,
        Toolset,
        FileSystem,
        Shell,
        ToolGuardrail,
        HandleDeferredToolCalls,
        ListableMemory,
        BatchLimit,
    )
    for kind in kinds:
        ours = [c for c in first if isinstance(c, kind)]
        assert len(ours) == 1, kind
        theirs = [c for c in second if isinstance(c, kind)]
        assert ours[0] is not theirs[0], kind  # built anew, not shared
    assert repo_workspace in first
    assert request_log in first
    toolsets = [c for c in first if isinstance(c, Toolset)]
    assert toolsets[0].toolset is repo_toolset
    assert len(first) == 10


def test_the_agent_takes_its_prompt_first_and_logs_after_the_cap() -> None:
    # Moved from 04-02 and 04-03, which found the request log last of the
    # agent's capabilities and the cap just before it, and from 04-06 (by
    # way of 04-04), which found the prompt first and the log last. The
    # guardrail declares itself innermost too, so the log shares the last
    # place with it; what the log relies on still holds: it comes after
    # every capability that changes a request, the cap among them.
    import logging

    from pydantic_ai.capabilities import Hooks

    from pydantic_tutor import prompt_file, request_log
    from pydantic_tutor.agent import PROMPT_PATH, agent
    from pydantic_tutor.batch_limit import BatchLimit

    capabilities = list(agent.root_capability.capabilities)
    assert isinstance(capabilities[0], prompt_file.PromptFile)
    assert capabilities[0].path == PROMPT_PATH
    assert isinstance(request_log.request_log, Hooks)
    assert request_log.logger.name == LOGGER
    assert logging.getLogger(name=LOGGER) is request_log.logger
    caps = [c for c in capabilities if isinstance(c, BatchLimit)]
    assert len(caps) == 1
    assert caps[0].max_tool_calls == 3
    log_at = capabilities.index(request_log.request_log)
    assert capabilities.index(caps[0]) < log_at
    # Only declared-innermost capabilities come after the log.
    for later in capabilities[log_at + 1 :]:
        ordering = later.get_ordering()
        assert ordering is not None and ordering.position == "innermost", later


def test_an_approved_write_is_written_in_the_working_copy(working_copy: Path) -> None:
    # Moved from 03-04, where the write ran with no one to approve it.
    read = one_call(
        root=working_copy, tool_name="read_file", args={"path": "README.md"}
    )
    assert read.outcome == "success"
    assert "# tinyjobs" in read.text
    wrote = one_call(
        root=working_copy,
        tool_name="write_file",
        args={"path": "notes.txt", "content": "from the agent\n"},
        approved=True,
    )
    assert wrote.outcome == "success", wrote.text
    assert (working_copy / "notes.txt").read_text() == "from the agent\n"


def test_an_approved_write_still_keeps_the_default_protection(
    working_copy: Path,
) -> None:
    # Moved from 03-04. Approved, the file tools still refuse `.git/` and
    # the secret files: approval is one more check, not a key to the rest.
    (working_copy / ".git").mkdir()
    (working_copy / ".git" / "config").write_text("[core]\n")
    for path in (".git/config", ".env"):
        result = one_call(
            root=working_copy,
            tool_name="write_file",
            args={"path": path, "content": "x\n"},
            approved=True,
        )
        assert result.outcome == "retry", f"{path} was written"
    assert (working_copy / ".git" / "config").read_text() == "[core]\n"
    assert FAKE_TOKEN in (working_copy / ".env").read_text()


def test_an_approved_command_runs_the_working_copys_tests(working_copy: Path) -> None:
    # Moved from 03-05, as are the five tests after it.
    result = approved_command(root=working_copy, command="pytest -q --tb=no")
    assert result.outcome == "success", result.text
    # Found and run in the copy: its one planted failure, and the rest.
    assert "FAILED tests/test_jobs.py::test_cancel" in result.text
    assert "1 failed, 138 passed" in result.text
    assert "[exit code: 1]" in result.text


@pytest.mark.parametrize(
    argnames="command",
    argvalues=[
        "rm README.md",
        'sh -c "rm README.md"',
        "cat .env",
        "python -c \"print(open('.env').read())\"",
        "uv run pytest",
    ],
)
def test_approved_the_shell_still_starts_only_pytest(
    working_copy: Path, command: str
) -> None:
    result = approved_command(root=working_copy, command=command)
    assert result.outcome == "retry", f"{command!r} ran"
    assert "is not in the allowed list" in result.text
    assert (working_copy / "README.md").is_file()
    assert FAKE_TOKEN not in result.text


# The allow list is a guard against mistakes, not a boundary, and approval
# does not change that: a person who approves a command approves all of it.
# These pass because the gaps are there; if one fails, the library closed it.


def test_approved_only_the_first_word_is_checked(working_copy: Path) -> None:
    result = approved_command(root=working_copy, command="pytest --version && cat .env")
    assert result.outcome == "success", result.text
    assert FAKE_TOKEN in result.text


def test_approved_pytest_runs_code_the_agent_wrote(working_copy: Path) -> None:
    peek = 'def test_peek() -> None:\n    print(open(".env").read())\n'
    run = run_agent(
        root=working_copy,
        responses=[
            [
                ToolCallPart(
                    tool_name="write_file",
                    args={"path": "tests/test_peek.py", "content": peek},
                )
            ],
            [
                ToolCallPart(
                    tool_name="run_command",
                    args={"command": "pytest -q -s tests/test_peek.py"},
                )
            ],
        ],
        approved=True,
    )
    wrote, ran = run.results
    assert wrote.outcome == "success", wrote.text
    assert ran.outcome == "success", ran.text
    assert FAKE_TOKEN in ran.text


def test_approved_the_shell_reaches_past_the_working_copy(working_copy: Path) -> None:
    # The file tools refuse this path (03-04); the shell does not.
    (working_copy.parent / "outside.txt").write_text("outside the working copy\n")
    result = approved_command(
        root=working_copy, command="pytest --version && cat ../outside.txt"
    )
    assert result.outcome == "success", result.text
    assert "outside the working copy" in result.text


def test_approved_a_key_in_the_apps_environment_does_not_reach_commands(
    working_copy: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A guard the workspace meets: a local workspace passes on only PATH,
    # HOME and the locale variables.
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-placeholder")
    result = approved_command(root=working_copy, command="pytest --version && env")
    assert result.outcome == "success", result.text
    assert "PATH=" in result.text
    assert "sk-test-placeholder" not in result.text
