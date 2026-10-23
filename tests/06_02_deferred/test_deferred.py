"""Checks that the project from the lesson on loading tools on demand is in place.

Our checks for the deferred-loading lesson: they verify the lesson, and
the student is not asked to run them. Run them with `uv run pytest`. They
fail until the lesson is done. None of them needs an API key or makes a
request: every run is on a scripted model, and every line "typed" at the
keyboard is scripted too.

None of them touches your own `work/` folder: each uses a folder of its own.

Nineteen earlier tests no longer hold, because the notebook now loads on
demand: until the model loads it, a request carries a catalog line in
place of its guidance, and `load_capability` in place of its five tools.
Sixteen were moved here, each saying where it came from and proving its
point with the load included:

- Twelve from the memory lesson (04-05): they pinned the instructions'
  text or the request log's tool count, or had the model call a memory
  tool with nothing loaded.
- Three from the seam lesson (04-06): the stack's instructions and tools,
  in any order, `stack_order.py`'s output, and the repository toolset
  offered first.
- One from the approval lesson (05-01), its `write_memory` case of
  `test_reads_and_the_notebook_do_not_ask`, now
  `test_writing_to_the_loaded_notebook_does_not_ask`.

Three are proved by this lesson's own tests: 04-05's tools' signatures
and its "the prompt, then the memory, then the repository" by
`test_before_the_load_a_request_carries_the_catalog_not_the_notebook`, and
04-02's "the log counts the tools actually sent" by
`test_the_request_log_counts_the_tools_the_model_may_call`.

Every import of the project is inside the test that needs it, so that
before the lesson each test fails on its own, instead of the whole file
failing to load.
"""

import json
import logging
import os
import random
import re
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
from pydantic_ai import ModelRequestContext, RunContext, models
from pydantic_ai.capabilities import AbstractCapability, AgentCapability, Hooks
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
from pydantic_ai.tools import ToolDefinition
from pydantic_ai_harness import Memory

# This file is tests/06_02_deferred/test_deferred.py: two folders below the root.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
TINYJOBS_ROOT: Path = PROJECT_ROOT / "sample_repos" / "tinyjobs"
PREFIX = "PYDANTIC_TUTOR_"
LOGGER = "pydantic_tutor.request_log"
# The agent's prompt since the agent lesson, word for word.
PROMPT = "You answer questions about a Python code repository. Answer briefly."
# The library's own text, at pydantic-ai 2.56.0, before its list of
# capabilities that load on demand.
CATALOG_PREFIX = (
    "The following capabilities are deferred and can be loaded using the "
    "`load_capability` tool. A capability's tools stay hidden until it is loaded:"
)
FILE_TOOLS = [
    "create_directory",
    "edit_file",
    "file_info",
    "find_files",
    "list_directory",
    "read_file",
    "search_files",
    "write_file",
]
SHELL_TOOLS = ["check_command", "run_command", "start_command", "stop_command"]
MEMORY_TOOLS = [
    "delete_memory",
    "list_memory",
    "read_memory",
    "search_memory",
    "write_memory",
]
# What the model may call before the notebook is loaded: ours, the file
# tools, the shell's, and the library's `load_capability`.
BEFORE_LOAD = sorted(
    ["list_modules", "read_docstring", *FILE_TOOLS, *SHELL_TOOLS, "load_capability"]
)
AFTER_LOAD = sorted([*BEFORE_LOAD, *MEMORY_TOOLS])
LOAD = ToolCallPart(
    tool_name="load_capability", args={"id": "memory"}, tool_call_id="load"
)


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    # None of this project's variables and no key, whatever the shell
    # running pytest has, and no real model call.
    for name in list(os.environ):
        if name.upper().startswith(PREFIX) or name.upper() == "OPENAI_API_KEY":
            monkeypatch.delenv(name)
    monkeypatch.setattr(target=models, name="ALLOW_MODEL_REQUESTS", value=False)


# The banner goes to stderr; turning it off keeps a failure readable.
CHILD_ENV = {**os.environ, "PYDANTIC_AI_NO_BANNER": "1", "PYTHONUNBUFFERED": "1"}


def run_script(*, name: str, cwd: Path = PROJECT_ROOT, stdin: str = "") -> str:
    """Run one of the project's scripts from `cwd`, `stdin` piped in; return its stdout."""
    result = subprocess.run(
        args=[sys.executable, name],
        cwd=cwd,
        input=stdin,
        capture_output=True,
        text=True,
        check=False,
        env=CHILD_ENV,
    )
    assert result.returncode == 0, f"{name} failed:\n{result.stderr}"
    return result.stdout


@pytest.fixture
def project(tmp_path: Path) -> Path:
    """A stand-in project folder: the sample repo and the scripts these tests run."""
    root = tmp_path / "project"
    shutil.copytree(src=TINYJOBS_ROOT, dst=root / "sample_repos" / "tinyjobs")
    (root / "scripts").mkdir()
    for script in (
        "fresh_copy.py",
        "cost_session.py",
        "notebook.py",
        "memory_injection.py",
    ):
        shutil.copy(src=PROJECT_ROOT / "scripts" / script, dst=root / "scripts")
    run_script(name="scripts/fresh_copy.py", cwd=root)
    return root


@pytest.fixture
def repo_with_notes(tmp_path: Path) -> Path:
    """A copy of tinyjobs whose notebook already holds two files."""
    root = tmp_path / "tinyjobs"
    shutil.copytree(src=TINYJOBS_ROOT, dst=root)
    notes = root / ".agent-memory" / "main"
    notes.mkdir(parents=True)
    (notes / "MEMORY.md").write_text("- Prefers terse answers.\n")
    (notes / "logging.md").write_text("- Logging goes to stderr.\n")
    return root


def expected_instructions(repository: str) -> str:
    """The prompt, the catalog of what loads on demand, then the repository's name."""
    from pydantic_tutor.agent import MEMORY_DESCRIPTION

    catalog = f"{CATALOG_PREFIX}\n- memory: {MEMORY_DESCRIPTION}"
    return f"{PROMPT}\n\n{catalog}\n\nThe repository is {repository}."


class Recorder:
    """A model that records each request, making the calls it is given first."""

    def __init__(self, *, calls: list[ToolCallPart] | None = None) -> None:
        self.calls = list(calls or [])
        self.infos: list[AgentInfo] = []
        self.messages: list[list[ModelMessage]] = []

    def reply(self, messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        self.infos.append(info)
        self.messages.append(list(messages))
        if self.calls:
            return ModelResponse(parts=[self.calls.pop(0)])
        return ModelResponse(parts=[TextPart(content="ok")])

    @property
    def requests(self) -> list[ModelRequest]:
        """The newest message of each request: what that request added."""
        latest = [messages[-1] for messages in self.messages]
        return [m for m in latest if isinstance(m, ModelRequest)]

    def tools(self, *, request: int) -> list[str]:
        """The names of the tools the model was sent with request `request`, from 0."""
        return sorted(tool.name for tool in self.infos[request].function_tools)

    def returns(self) -> list[ToolReturnPart]:
        """Every tool return the model was sent, in order."""
        return [
            p for r in self.requests for p in r.parts if isinstance(p, ToolReturnPart)
        ]

    def returned(self) -> str:
        """The last tool return the model was sent."""
        return self.returns()[-1].model_response_str()

    def retries(self) -> list[RetryPromptPart]:
        """Every retry the model was sent."""
        return [
            p for r in self.requests for p in r.parts if isinstance(p, RetryPromptPart)
        ]


def run_agent(
    *,
    root: Path = TINYJOBS_ROOT,
    calls: list[ToolCallPart] | None = None,
    capabilities: list[AgentCapability[object]] | None = None,
) -> Recorder:
    """Run the student's agent once on `root`; return what its model was given."""
    from pydantic_tutor.agent import agent
    from pydantic_tutor.repo_tools import RepoDeps

    recorder = Recorder(calls=calls)
    agent.run_sync(
        user_prompt="What does this repo do?",
        model=FunctionModel(function=recorder.reply),
        deps=RepoDeps(root=root),
        capabilities=capabilities,
    )
    return recorder


def logged(*, caplog: pytest.LogCaptureFixture) -> list[str]:
    """The messages our request log wrote during the test."""
    return [record.getMessage() for record in caplog.records if record.name == LOGGER]


# Goal 1: the notebook loads on demand.


def test_the_notebook_loads_on_demand_with_a_description() -> None:
    from pydantic_tutor.agent import MEMORY_DESCRIPTION, agent

    [memory] = [c for c in agent.root_capability.capabilities if isinstance(c, Memory)]
    assert memory.defer_loading is True
    # The id the model loads it by, and that history records: Memory's own.
    assert memory.id == "memory"
    assert memory.description == MEMORY_DESCRIPTION
    # Enough for the model to know when to load it.
    assert "remember" in MEMORY_DESCRIPTION
    # Still the notebook of 04-05, injection off.
    assert memory.inject_memory is False


def test_before_the_load_a_request_carries_the_catalog_not_the_notebook() -> None:
    # The lesson's point. Also 04-05's "the prompt, then the memory, then
    # the repository", now the prompt, the catalog, then the repository,
    # and its check of the tools' signatures.
    from pydantic_tutor.listable_memory import LISTING_GUIDANCE

    recorder = run_agent()
    first = recorder.infos[0]
    assert recorder.tools(request=0) == BEFORE_LOAD
    assert first.instructions == expected_instructions("tinyjobs")
    assert LISTING_GUIDANCE not in (first.instructions or "")
    parts = first.model_request_parameters.instruction_parts or []
    assert [part.content for part in parts][0:1] == [PROMPT]
    assert [part.dynamic for part in parts] == [False, True, True]
    tools = {tool.name: tool for tool in first.function_tools}
    read = tools["read_docstring"]
    assert read.description, "read_docstring has no docstring summary"
    assert "Args" not in read.description, "the Args: section was not parsed"
    schema = read.parameters_json_schema
    assert schema["required"] == ["module"]
    assert schema["properties"]["module"]["type"] == "string"
    assert schema["properties"]["module"].get("description")
    assert tools["list_modules"].parameters_json_schema["properties"] == {}


def test_the_load_brings_the_guidance_and_five_tools_from_the_next_request() -> None:
    from pydantic_tutor.listable_memory import LISTING_GUIDANCE

    recorder = run_agent(calls=[LOAD])
    assert recorder.tools(request=0) == BEFORE_LOAD
    assert recorder.tools(request=1) == AFTER_LOAD
    # The guidance comes back once, as the load's return, not in the
    # instructions, which stay as they were.
    [loaded] = recorder.returns()
    assert loaded.tool_name == "load_capability"
    assert loaded.content == {"instructions": LISTING_GUIDANCE}
    assert [info.instructions for info in recorder.infos] == [
        expected_instructions("tinyjobs")
    ] * 2
    tools = {tool.name: tool for tool in recorder.infos[1].function_tools}
    assert tools["list_memory"].parameters_json_schema["properties"] == {}


def test_a_conversation_keeps_the_notebook_loaded_and_a_new_one_starts_without() -> (
    None
):
    from pydantic_tutor.agent import agent
    from pydantic_tutor.repo_tools import RepoDeps

    first = Recorder(calls=[LOAD])
    result = agent.run_sync(
        user_prompt="Load your notebook.",
        model=FunctionModel(function=first.reply),
        deps=RepoDeps(root=TINYJOBS_ROOT),
    )
    # The next turn of the same conversation: loaded from its first request.
    later = Recorder()
    agent.run_sync(
        user_prompt="And now?",
        model=FunctionModel(function=later.reply),
        deps=RepoDeps(root=TINYJOBS_ROOT),
        conversation=result.conversation,
    )
    assert later.tools(request=0) == AFTER_LOAD
    # A new conversation: hidden again until loaded.
    assert run_agent().tools(request=0) == BEFORE_LOAD


def test_the_request_log_counts_the_tools_the_model_may_call(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Also 04-02's point: the log counts what is sent, not what the agent
    # holds. The notebook's tools are in every request's definitions, and
    # hidden from the model until it is loaded.
    caplog.set_level(level=logging.INFO, logger=LOGGER)
    recorder = run_agent(calls=[LOAD])
    assert logged(caplog=caplog) == [
        f"request step=1 messages=1 tools={len(BEFORE_LOAD)}",
        f"request step=2 messages=3 tools={len(AFTER_LOAD)}",
    ]
    assert [len(info.function_tools) for info in recorder.infos] == [
        len(BEFORE_LOAD),
        len(AFTER_LOAD),
    ]

    # Hide every tool but ours for one run: `load_capability` is added
    # after every capability's `prepare_tools`, and is sent, and counted.
    def only_ours(
        ctx: RunContext[object], tool_defs: list[ToolDefinition]
    ) -> list[ToolDefinition]:
        return [
            tool
            for tool in tool_defs
            if tool.name in {"list_modules", "read_docstring"}
        ]

    caplog.clear()
    recorder = run_agent(capabilities=[Hooks(prepare_tools=only_ours)])
    assert recorder.tools(request=0) == [
        "list_modules",
        "load_capability",
        "read_docstring",
    ]
    assert logged(caplog=caplog) == ["request step=1 messages=1 tools=3"]


def test_cost_session_measures_the_saving_and_the_load(project: Path) -> None:
    out = run_script(
        name="scripts/cost_session.py",
        cwd=project,
        stdin=(
            "What does tinyjobs.jobs do?\n"
            "Remember that the tests run with pytest.\n"
            "And again?\n/exit\n"
        ),
    )
    carried = re.findall(
        pattern=r"request (\d+): instructions ([\d,]+) · "
        r"tool definitions ([\d,]+) \((\d+)\) · messages ([\d,]+) \((\d+)\)",
        string=out,
    )
    assert [int(found[0]) for found in carried] == list(range(1, 10))
    # The instructions never change: the guidance comes with the load, as
    # a message, not in them.
    assert len({found[1] for found in carried}) == 1
    # Turn 1, and turn 2 until the load: 16 tools (the chat adds
    # ask_user). From the request after the load: the notebook's five too.
    counts = [int(found[3]) for found in carried]
    assert counts == [16] * 4 + [21] * 5
    sizes = {int(found[3]): int(found[2].replace(",", "")) for found in carried}
    assert sizes[16] < sizes[21]
    assert out.count("ran load_capability") == 1
    assert out.count("ran write_memory") == 1
    assert "\nNoted.\n" in out
    assert len(re.findall(pattern=r"\[context [\d,]+ of 200,000", string=out)) == 3
    notebook = project / "work" / "tinyjobs" / ".agent-memory" / "main" / "MEMORY.md"
    assert notebook.read_text() == "- the tests run with pytest.\n"


def test_notebook_prints_two_sessions(project: Path) -> None:
    # Moved from 04-05: each session is a run of its own, so each loads
    # the notebook before it uses it.
    from pydantic_tutor.listable_memory import LISTING_GUIDANCE

    loaded = "  result " + json.dumps(
        obj={"instructions": LISTING_GUIDANCE}, separators=(",", ":")
    )
    load = [
        "  request ['user-prompt']",
        '  call   load_capability {"id": "memory"}',
        "  request ['tool-return', 'user-prompt']",
        loaded,
    ]
    lines = run_script(name="scripts/notebook.py", cwd=project).splitlines()
    assert lines[0] == "session: Remember that we log to stderr."
    assert lines[1:6] == [
        *load,
        '  call   write_memory {"content": "- Logging goes to stderr, so stdout '
        'stays pipeable.\\n", "file": "logging.md"}',
    ]
    assert lines[6] == "  request ['tool-return']"
    assert '"status":"created"' in lines[7]
    assert lines[8:] == [
        "  output Noted.",
        "session: What did we decide about logging?",
        *load,
        "  call   list_memory {}",
        "  request ['tool-return']",
        '  result {"files":["logging.md"],"truncated":false}',
        '  call   read_memory {"file": "logging.md"}',
        "  request ['tool-return']",
        "  result - Logging goes to stderr, so stdout stays pipeable.",
        "",
        "  output We decided: - Logging goes to stderr, so stdout stays pipeable.",
    ]
    notebook = project / "work" / "tinyjobs" / ".agent-memory" / "main" / "logging.md"
    assert (
        notebook.read_text() == "- Logging goes to stderr, so stdout stays pipeable.\n"
    )

    # Unchanged: injection would send the notebook beside the newest
    # question, and take it off the one before.
    lines = run_script(name="scripts/memory_injection.py", cwd=project).splitlines()
    block = "<memory> ### Other memory files - logging.md </memory>"
    assert lines == [
        "turn: What does tinyjobs do?",
        f"  request: ['What does tinyjobs do?', '{block}']",
        "turn: And how is it tested?",
        "  request: ['What does tinyjobs do?']",
        "  response: ['ok']",
        f"  request: ['And how is it tested?', '{block}']",
    ]


# Moved from 04-05: the notebook's own points, with the load first.


def test_no_request_carries_the_notebook(repo_with_notes: Path) -> None:
    # With injection off, the notebook's files are on disk, and no request
    # carries them, loaded or not. The model lists them when it asks.
    recorder = run_agent(
        root=repo_with_notes,
        calls=[LOAD, ToolCallPart(tool_name="list_memory", args={}, tool_call_id="c1")],
    )
    assert recorder.tools(request=0) == BEFORE_LOAD
    for messages in recorder.messages:
        for message in messages:
            for part in message.parts:
                text = str(getattr(part, "content", ""))
                assert "<memory>" not in text
                assert "Prefers terse answers" not in text
    first = recorder.requests[0]
    assert [part.part_kind for part in first.parts] == ["user-prompt"]
    assert (
        recorder.returned() == '{"files":["MEMORY.md","logging.md"],"truncated":false}'
    )


def test_a_note_written_in_one_run_is_found_in_the_next(tmp_path: Path) -> None:
    root = tmp_path / "tinyjobs"
    shutil.copytree(src=TINYJOBS_ROOT, dst=root)
    first = run_agent(
        root=root,
        calls=[
            LOAD,
            ToolCallPart(
                tool_name="write_memory",
                args={"content": "- Logging goes to stderr.\n", "file": "logging.md"},
                tool_call_id="w1",
            ),
        ],
    )
    assert first.retries() == []
    assert (root / ".agent-memory" / "main" / "logging.md").read_text() == (
        "- Logging goes to stderr.\n"
    )
    # A second run, with no history: only the file carries over, and the
    # notebook is loaded again before it is read.
    later = run_agent(
        root=root,
        calls=[
            LOAD,
            ToolCallPart(tool_name="list_memory", args={}, tool_call_id="l1"),
            ToolCallPart(
                tool_name="read_memory", args={"file": "logging.md"}, tool_call_id="r1"
            ),
        ],
    )
    assert len(later.messages[0]) == 1
    assert later.tools(request=0) == BEFORE_LOAD
    assert later.retries() == []
    assert later.returned() == "- Logging goes to stderr.\n"


def test_a_memory_tool_called_before_the_load_is_refused(tmp_path: Path) -> None:
    # What a stand-in written for the eager notebook now meets: the call
    # is not run, and the model is told to load the notebook first.
    root = tmp_path / "tinyjobs"
    shutil.copytree(src=TINYJOBS_ROOT, dst=root)
    recorder = run_agent(
        root=root,
        calls=[ToolCallPart(tool_name="write_memory", args={"content": "- x\n"})],
    )
    [retry] = recorder.retries()
    assert "load_capability" in str(retry.content)
    assert "'memory'" in str(retry.content)
    assert not (root / ".agent-memory").exists()


def test_list_memory_names_only_files_read_memory_takes(repo_with_notes: Path) -> None:
    # A file in a folder below the notebook's, and one not in Markdown:
    # `read_memory` refuses both, so the listing must not offer them.
    notes = repo_with_notes / ".agent-memory" / "main"
    (notes / "team").mkdir()
    (notes / "team" / "deep.md").write_text("x\n")
    (notes / "draft.txt").write_text("x\n")
    recorder = run_agent(
        root=repo_with_notes,
        calls=[LOAD, ToolCallPart(tool_name="list_memory", args={}, tool_call_id="c1")],
    )
    assert recorder.retries() == []
    assert (
        recorder.returned() == '{"files":["MEMORY.md","logging.md"],"truncated":false}'
    )


# Moved from 04-05: they pinned the instructions' text, which now carries
# the catalog in place of the notebook's guidance.


def test_the_instructions_name_the_repository_each_run_is_given(tmp_path: Path) -> None:
    # One agent, two runs, two renderings, each from its deps.
    other = tmp_path / "otherrepo"
    (other / "src" / "otherpkg").mkdir(parents=True)
    (other / "src" / "otherpkg" / "__init__.py").write_text('"""Another package."""\n')

    assert run_agent().infos[0].instructions == expected_instructions("tinyjobs")
    assert run_agent(root=other).infos[0].instructions == expected_instructions(
        "otherrepo"
    )


def test_the_instructions_are_rendered_for_every_request_and_sent_once() -> None:
    recorder = run_agent(
        calls=[ToolCallPart(tool_name="list_modules", args={}, tool_call_id="c1")],
    )
    expected = expected_instructions("tinyjobs")
    assert [info.instructions for info in recorder.infos] == [expected, expected]
    requests = [m for m in recorder.messages[-1] if isinstance(m, ModelRequest)]
    assert [request.instructions for request in requests] == [expected, expected]
    assert expected.count("The repository is") == 1


def test_the_agent_sends_instructions_and_no_system_prompt() -> None:
    # The instructions travel on the request, not as a part, and the agent
    # has no system prompt. Nor a memory part.
    [request] = run_agent().messages[0]
    assert isinstance(request, ModelRequest)
    assert [part.part_kind for part in request.parts] == ["user-prompt"]
    assert request.instructions == expected_instructions("tinyjobs")


def test_inspect_run_shows_the_rendered_instructions() -> None:
    printed = run_script(name="scripts/inspect_run.py")
    assert f"  instructions: {expected_instructions('tinyjobs')}\n" in printed
    assert "request ['user-prompt']" in printed


def test_scripted_run_prints_what_the_function_decided() -> None:
    # The reply is built from the prompt and the instructions the model
    # was sent.
    printed = run_script(name="scripts/scripted_run.py")
    expected = (
        "You asked 'What does this repo do?', and my instructions say: "
        f"{expected_instructions('tinyjobs')}"
    )
    assert printed.strip() == expected


def test_toolset_views_shows_the_agents_tools_and_instructions() -> None:
    # The agent's line: `load_capability` first, no memory tool; the rest
    # of that script's output is unchanged.
    lines = run_script(name="scripts/toolset_views.py").splitlines()
    assert lines[0] == "the repo assistant:"
    assert lines[1].startswith(
        "  tools: load_capability, list_modules, read_docstring, read_file,"
    )
    assert "memory" not in lines[1]
    assert lines[2] == f"  instructions: {expected_instructions('tinyjobs')!r}"
    assert lines[3:] == [
        "a second agent, given the toolset:",
        "  tools: list_modules, read_docstring",
        "  instructions: 'The repository is tinyjobs.'",
        "prefixed with 'repo':",
        "  tools: repo_list_modules, repo_read_docstring",
        "  instructions: 'The repository is tinyjobs.'",
        "prefixed, calling read_docstring with a wrong module:",
        "  asked to retry: There is no module 'jobs'. Call list_modules to see them.",
        "filtered to list_modules:",
        "  tools: list_modules",
        "  instructions: 'The repository is tinyjobs.'",
    ]


# Moved from 04-05: they pinned the request log's tool count.


def test_each_request_is_logged_with_its_step_messages_and_tools(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(level=logging.INFO, logger=LOGGER)
    recorder = run_agent(
        calls=[ToolCallPart(tool_name="list_modules", args={}, tool_call_id="c1")],
    )
    assert logged(caplog=caplog) == [
        f"request step=1 messages=1 tools={len(BEFORE_LOAD)}",
        f"request step=2 messages=3 tools={len(BEFORE_LOAD)}",
    ]
    assert [len(messages) for messages in recorder.messages] == [1, 3]


def test_watch_requests_prints_the_log() -> None:
    lines = run_script(name="scripts/watch_requests.py").splitlines()
    assert lines[:2] == [
        f"{LOGGER}: request step=1 messages=1 tools={len(BEFORE_LOAD)}",
        f"{LOGGER}: request step=2 messages=3 tools={len(BEFORE_LOAD)}",
    ]
    assert lines[2].startswith("output: ")
    assert lines[3].startswith("usage: ")


# Moved from 04-06: the stack, in any order, sends the same.


class KeepLastTwo(AbstractCapability[object]):
    """Send the model only the last two messages: a capability that changes
    the request, as the seam lesson's stand-in does."""

    async def before_model_request(
        self, ctx: RunContext[object], request_context: ModelRequestContext
    ) -> ModelRequestContext:
        return replace(request_context, messages=request_context.messages[-2:])


def list_then_answer(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """A stand-in that lists the modules once, then answers."""
    if any(isinstance(part, ToolReturnPart) for part in messages[-1].parts):
        return ModelResponse(parts=[TextPart(content="ok")])
    return ModelResponse(parts=[ToolCallPart(tool_name="list_modules", args={})])


def test_any_order_of_the_stack_sends_the_same_and_logs_what_is_sent(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # With a capability that changes the request added last, in every
    # order tried: the instructions are the prompt, the catalog, then the
    # repository's name; the same 15 tools; and the log counts the
    # messages the model is sent.
    from pydantic_tutor.agent import build_agent, repo_capabilities
    from pydantic_tutor.repo_tools import RepoDeps

    caplog.set_level(level=logging.INFO, logger=LOGGER)
    catalog = expected_instructions("tinyjobs").split("\n\n")[1]
    shuffle = random.Random(4).shuffle  # fixed: the same orders every run
    for index in range(2 + 12):
        stack: list[AgentCapability[object]] = [*repo_capabilities(), KeepLastTwo()]  # type: ignore[list-item]
        if index == 1:
            stack.reverse()
        elif index > 1:
            shuffle(stack)
        caplog.clear()
        infos: list[AgentInfo] = []
        sizes: list[int] = []

        def reply(
            messages: list[ModelMessage],
            info: AgentInfo,
            infos: list[AgentInfo] = infos,
            sizes: list[int] = sizes,
        ) -> ModelResponse:
            infos.append(info)
            sizes.append(len(messages))
            return list_then_answer(messages, info)

        build_agent(capabilities=stack).run_sync(
            user_prompt="What does this repo do?",
            model=FunctionModel(function=reply),
            deps=RepoDeps(root=TINYJOBS_ROOT),
        )
        parts = infos[0].model_request_parameters.instruction_parts or []
        assert [part.content for part in parts] == [
            PROMPT,
            catalog,
            "The repository is tinyjobs.",
        ]
        names = [sorted(tool.name for tool in info.function_tools) for info in infos]
        assert names == [BEFORE_LOAD, BEFORE_LOAD]
        assert sizes == [1, 2]
        counted = [
            int(message.split("messages=")[1].split()[0])
            for message in logged(caplog=caplog)
        ]
        assert counted == sizes


def test_stack_order_prints_two_orders_alike() -> None:
    out = run_script(name="scripts/stack_order.py").splitlines()
    request_lines = [
        f"{LOGGER}: request step=1 messages=1 tools={len(BEFORE_LOAD)}",
        "  model: sent 1 message(s)",
        "    instructions: You answer questions about ...",
        "    instructions: The following capabilities are ...",
        "    instructions: The repository is tinyjobs. ...",
        f"{LOGGER}: request step=2 messages=2 tools={len(BEFORE_LOAD)}",
        "  model: sent 2 message(s)",
    ]
    assert out == ["forwards", *request_lines, "reversed", *request_lines]


def test_the_agent_takes_the_repo_toolset_as_a_capability() -> None:
    # Moved from 04-06 (from 04-01, by way of 04-05): the toolset reaches
    # the agent through `Toolset(...)` in the stack, and the model is
    # offered our two tools first of the stack's, after the library's
    # `load_capability`.
    from pydantic_ai.capabilities import Toolset
    from pydantic_ai.toolsets import FunctionToolset

    from pydantic_tutor.agent import agent
    from pydantic_tutor.repo_tools import repo_toolset

    assert not any(toolset is repo_toolset for toolset in agent.toolsets)
    own = agent.toolsets[0]
    assert isinstance(own, FunctionToolset)
    assert own.tools == {}  # no tools of its own: every one is from the stack
    capabilities = agent.root_capability.capabilities
    assert any(
        isinstance(c, Toolset) and c.toolset is repo_toolset for c in capabilities
    )
    names = [tool.name for tool in run_agent().infos[0].function_tools]
    assert names[:3] == ["load_capability", "list_modules", "read_docstring"]
    assert len(names) == len(BEFORE_LOAD)


# Moved from 05-01: writing to the notebook does not ask, once it is loaded.


def test_writing_to_the_loaded_notebook_does_not_ask(tmp_path: Path) -> None:
    from pydantic_tutor.listable_memory import LISTING_GUIDANCE

    root = tmp_path / "tinyjobs"
    shutil.copytree(src=TINYJOBS_ROOT, dst=root)
    recorder = run_agent(
        root=root,
        calls=[
            LOAD,
            ToolCallPart(
                tool_name="write_memory",
                args={"file": "MEMORY.md", "content": "- a note\n"},
            ),
        ],
    )
    loaded, written = recorder.returns()
    assert loaded.content == {"instructions": LISTING_GUIDANCE}
    assert written.tool_name == "write_memory"
    assert '"status":"created"' in written.model_response_str()
    assert recorder.retries() == []
    assert (root / ".agent-memory" / "main" / "MEMORY.md").read_text() == "- a note\n"
