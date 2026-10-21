"""Checks that the project from the memory lesson is in place.

Our checks for the memory lesson: they verify the lesson, and the student
is not asked to run them. Run them with `uv run pytest`. They fail until
the lesson is done. None of them needs an API key or makes a request:
every run is on a scripted model.

Twelve tests were moved here: the memory capability adds its guidance to
the instructions and five tools to the agent, and they pinned the
instructions' text or the number of tools. Each says where it came from,
and proves its point with the memory's part included.

Every import of the project is inside the test that needs it, so that
before the lesson each test fails on its own, instead of the whole file
failing to load.
"""

import asyncio
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic_ai import Agent, models
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai_harness import Memory
from pydantic_ai_harness.memory import FileStore, InMemoryStore

# This file is tests/04_05_memory/test_memory.py: two folders below the root.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
TINYJOBS_ROOT: Path = PROJECT_ROOT / "sample_repos" / "tinyjobs"
# The agent's prompt since the agent lesson, word for word.
PROMPT = "You answer questions about a Python code repository. Answer briefly."
LOGGER = "pydantic_tutor.request_log"
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
ALL_TOOLS = 2 + len(FILE_TOOLS) + len(SHELL_TOOLS) + len(MEMORY_TOOLS)  # 19


@pytest.fixture(autouse=True)
def no_real_model(monkeypatch: pytest.MonkeyPatch) -> None:
    # Any request to a real model raises instead of reaching a provider.
    monkeypatch.setattr(target=models, name="ALLOW_MODEL_REQUESTS", value=False)


def run_script(name: str, *, cwd: Path = PROJECT_ROOT) -> str:
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


def expected_instructions(repository: str) -> str:
    """The prompt, the memory's guidance, then the repository's name."""
    from pydantic_tutor.listable_memory import LISTING_GUIDANCE

    return f"{PROMPT}\n\n{LISTING_GUIDANCE}\n\nThe repository is {repository}."


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

    def returned(self) -> str:
        """The last tool result the model was sent."""
        parts = [
            p for r in self.requests for p in r.parts if isinstance(p, ToolReturnPart)
        ]
        return parts[-1].model_response_str()


def run_agent(*, root: Path, calls: list[ToolCallPart] | None = None) -> Recorder:
    """Run the student's agent once on `root`; return what its model was given."""
    from pydantic_tutor.agent import agent
    from pydantic_tutor.repo_tools import RepoDeps

    recorder = Recorder(calls=calls)
    agent.run_sync(
        user_prompt="What does this repo do?",
        model=FunctionModel(function=recorder.reply),
        deps=RepoDeps(root=root),
    )
    return recorder


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


# Goal 1: the agent has a notebook it reads and writes, and a tool to list it.


def test_listable_memory_is_memory_with_its_own_guidance() -> None:
    from pydantic_tutor.listable_memory import LISTING_GUIDANCE, ListableMemory

    memory: ListableMemory[None] = ListableMemory()
    assert isinstance(memory, Memory)
    assert memory.guidance == LISTING_GUIDANCE
    assert "list_memory" in LISTING_GUIDANCE
    assert memory.max_list_files == 100
    # `Memory`'s own id, kept: an agent has one capability of that id.
    assert memory.id == "memory"


def test_the_agent_has_one_notebook_in_its_workspace_with_injection_off() -> None:
    # One import: before the lesson, Ruff would sort a second one, from a
    # module not yet written, as a third-party import.
    from pydantic_tutor.agent import agent

    memories = [c for c in agent.root_capability.capabilities if isinstance(c, Memory)]
    assert len(memories) == 1
    [memory] = memories
    assert type(memory).__module__ == "pydantic_tutor.listable_memory"
    assert type(memory).__name__ == "ListableMemory"
    assert memory.inject_memory is False
    assert isinstance(memory.store, FileStore)
    assert memory.store.directory == ".agent-memory"
    assert memory.store.workspace is None  # the run's workspace: the repository


def test_the_model_is_sent_our_signatures_beside_the_file_shell_and_memory_tools() -> (
    None
):
    # Moved from 03-05 (and before it 03-04 and 01-05), which pinned the
    # agent's tools before the memory's five were added.
    tools = {
        tool.name: tool
        for tool in run_agent(root=TINYJOBS_ROOT).infos[0].function_tools
    }
    assert sorted(tools) == sorted(
        ["list_modules", "read_docstring", *FILE_TOOLS, *SHELL_TOOLS, *MEMORY_TOOLS]
    )
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
    assert tools["list_memory"].parameters_json_schema["properties"] == {}


def test_no_request_carries_the_notebook(repo_with_notes: Path) -> None:
    # The lesson's point: with injection off, the notebook's files are on
    # disk, and no request carries them. The model lists them when it asks.
    recorder = run_agent(
        root=repo_with_notes,
        calls=[ToolCallPart(tool_name="list_memory", args={}, tool_call_id="c1")],
    )
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
    run_agent(
        root=root,
        calls=[
            ToolCallPart(
                tool_name="write_memory",
                args={"content": "- Logging goes to stderr.\n", "file": "logging.md"},
                tool_call_id="w1",
            )
        ],
    )
    assert (root / ".agent-memory" / "main" / "logging.md").read_text() == (
        "- Logging goes to stderr.\n"
    )
    # A second run, with no history: only the file carries over.
    later = run_agent(
        root=root,
        calls=[
            ToolCallPart(tool_name="list_memory", args={}, tool_call_id="l1"),
            ToolCallPart(
                tool_name="read_memory", args={"file": "logging.md"}, tool_call_id="r1"
            ),
        ],
    )
    assert len(later.messages[0]) == 1
    assert later.returned() == "- Logging goes to stderr.\n"


def test_list_memory_lists_one_notebook_and_says_when_it_is_cut() -> None:
    from pydantic_tutor.listable_memory import ListableMemory, MemoryListing

    async def fill() -> InMemoryStore:
        store = InMemoryStore()
        for name in ["a.md", "b.md", "c.md"]:
            await store.write(f"main/{name}", "x\n", expected_version=None)
        # Another notebook in the same store: not this one's to list.
        await store.write("other/secret.md", "x\n", expected_version=None)
        return store

    store = asyncio.run(fill())
    agent = Agent(capabilities=[ListableMemory(store=store, max_list_files=2)])
    recorder = Recorder(
        calls=[ToolCallPart(tool_name="list_memory", args={}, tool_call_id="c1")]
    )
    agent.run_sync(user_prompt="x", model=FunctionModel(function=recorder.reply))
    listing = MemoryListing.model_validate_json(recorder.returned())
    assert listing == MemoryListing(files=["a.md", "b.md"], truncated=True)


def test_list_memory_names_only_files_read_memory_takes(repo_with_notes: Path) -> None:
    # A file in a folder below the notebook's, and one not in Markdown:
    # `read_memory` refuses both, so the listing must not offer them.
    notes = repo_with_notes / ".agent-memory" / "main"
    (notes / "team").mkdir()
    (notes / "team" / "deep.md").write_text("x\n")
    (notes / "draft.txt").write_text("x\n")
    recorder = run_agent(
        root=repo_with_notes,
        calls=[ToolCallPart(tool_name="list_memory", args={}, tool_call_id="c1")],
    )
    assert (
        recorder.returned() == '{"files":["MEMORY.md","logging.md"],"truncated":false}'
    )


def test_list_memory_joins_memorys_own_toolset() -> None:
    # `Memory`'s tools find the run's copy of the capability by comparing
    # its toolset with themselves, and durable engines know it by its id:
    # `get_toolset()` must return Memory's one object, every time.
    from pydantic_tutor.agent import agent

    [memory] = [c for c in agent.root_capability.capabilities if isinstance(c, Memory)]
    toolset = memory.get_toolset()
    assert toolset is memory.get_toolset()
    assert toolset is Memory.get_toolset(memory)
    assert getattr(toolset, "id", None) == "memory"


def test_notebook_prints_two_sessions(tmp_path: Path) -> None:
    # In a stand-in project, so the student's own work/ copy is left alone.
    project = tmp_path / "project"
    shutil.copytree(src=TINYJOBS_ROOT, dst=project / "sample_repos" / "tinyjobs")
    (project / "scripts").mkdir()
    for script in ("fresh_copy.py", "notebook.py", "memory_injection.py"):
        shutil.copy(src=PROJECT_ROOT / "scripts" / script, dst=project / "scripts")
    run_script("scripts/fresh_copy.py", cwd=project)

    lines = run_script("scripts/notebook.py", cwd=project).splitlines()
    assert lines[0] == "session: Remember that we log to stderr."
    assert lines[1:3] == [
        "  request ['user-prompt']",
        '  call   write_memory {"content": "- Logging goes to stderr, so stdout '
        'stays pipeable.\\n", "file": "logging.md"}',
    ]
    assert '"status":"created"' in lines[4]
    assert lines[5:] == [
        "  output Noted.",
        "session: What did we decide about logging?",
        "  request ['user-prompt']",
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

    # Goal 2: injection would send the notebook beside the newest question,
    # and take it off the one before.
    lines = run_script("scripts/memory_injection.py", cwd=project).splitlines()
    block = "<memory> ### Other memory files - logging.md </memory>"
    assert lines == [
        "turn: What does tinyjobs do?",
        f"  request: ['What does tinyjobs do?', '{block}']",
        "turn: And how is it tested?",
        "  request: ['What does tinyjobs do?']",
        "  response: ['ok']",
        f"  request: ['And how is it tested?', '{block}']",
    ]


# Moved: they pinned the instructions' text before the memory's guidance.


def test_the_instructions_name_the_repository_each_run_is_given(tmp_path: Path) -> None:
    # Moved from 01-05: one agent, two runs, two renderings, each from its deps.
    other = tmp_path / "otherrepo"
    (other / "src" / "otherpkg").mkdir(parents=True)
    (other / "src" / "otherpkg" / "__init__.py").write_text('"""Another package."""\n')

    assert run_agent(root=TINYJOBS_ROOT).infos[0].instructions == expected_instructions(
        "tinyjobs"
    )
    assert run_agent(root=other).infos[0].instructions == expected_instructions(
        "otherrepo"
    )


def test_the_instructions_are_rendered_for_every_request_and_sent_once() -> None:
    # Moved from 01-05 (every request) and 04-01 (the toolset's text, once).
    recorder = run_agent(
        root=TINYJOBS_ROOT,
        calls=[ToolCallPart(tool_name="list_modules", args={}, tool_call_id="c1")],
    )
    expected = expected_instructions("tinyjobs")
    assert [info.instructions for info in recorder.infos] == [expected, expected]
    requests = [m for m in recorder.messages[-1] if isinstance(m, ModelRequest)]
    assert [request.instructions for request in requests] == [expected, expected]
    assert expected.count("The repository is") == 1


def test_the_agent_sends_instructions_and_no_system_prompt() -> None:
    # Moved from 01-05: the instructions travel on the request, not as a
    # part, and the agent has no system prompt. Nor a memory part.
    [request] = run_agent(root=TINYJOBS_ROOT).messages[0]
    assert isinstance(request, ModelRequest)
    assert [part.part_kind for part in request.parts] == ["user-prompt"]
    assert request.instructions == expected_instructions("tinyjobs")


def test_the_model_is_sent_the_prompt_then_the_memory_then_the_repository() -> None:
    # Moved from 04-04's guard, which pinned the prompt then the toolset's
    # text. The memory's guidance is text, like the prompt: it goes after
    # the prompt (list order) and before the toolset's function.
    infos = run_agent(root=TINYJOBS_ROOT).infos
    parts = infos[0].model_request_parameters.instruction_parts or []
    assert [part.content for part in parts][0:1] == [PROMPT]
    assert [part.dynamic for part in parts] == [False, False, True]


def test_inspect_run_shows_the_rendered_instructions() -> None:
    # Moved from 01-05.
    printed = run_script("scripts/inspect_run.py")
    assert f"  instructions: {expected_instructions('tinyjobs')}\n" in printed
    assert "request ['user-prompt']" in printed


def test_scripted_run_prints_what_the_function_decided() -> None:
    # Moved from 01-05: the reply is built from the prompt and the
    # instructions the model was sent.
    printed = run_script("scripts/scripted_run.py")
    expected = (
        "You asked 'What does this repo do?', and my instructions say: "
        f"{expected_instructions('tinyjobs')}"
    )
    assert printed.strip() == expected


def test_toolset_views_shows_the_agents_tools_and_instructions() -> None:
    # Moved from 04-01, which pinned the agent's instructions on line 3;
    # the rest of that script's output is unchanged.
    lines = run_script("scripts/toolset_views.py").splitlines()
    assert lines[0] == "the repo assistant:"
    assert lines[1].startswith("  tools: list_modules, read_docstring, read_file,")
    assert "list_memory" in lines[1]
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


# Moved: they pinned the number of tools before the memory's five.


def test_the_agent_takes_the_toolset_and_has_no_tools_of_its_own() -> None:
    # Moved from 04-01.
    from pydantic_ai.toolsets import FunctionToolset

    from pydantic_tutor.agent import agent
    from pydantic_tutor.repo_tools import repo_toolset

    assert any(toolset is repo_toolset for toolset in agent.toolsets)
    own = agent.toolsets[0]
    assert isinstance(own, FunctionToolset)
    assert own.tools == {}
    names = [
        tool.name for tool in run_agent(root=TINYJOBS_ROOT).infos[0].function_tools
    ]
    assert names[:2] == ["list_modules", "read_docstring"]
    assert len(names) == ALL_TOOLS


def test_each_request_is_logged_with_its_step_messages_and_tools(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Moved from 04-02.
    caplog.set_level(level=logging.INFO, logger=LOGGER)
    recorder = run_agent(
        root=TINYJOBS_ROOT,
        calls=[ToolCallPart(tool_name="list_modules", args={}, tool_call_id="c1")],
    )
    logged = [record.getMessage() for record in caplog.records if record.name == LOGGER]
    assert logged == [
        f"request step=1 messages=1 tools={ALL_TOOLS}",
        f"request step=2 messages=3 tools={ALL_TOOLS}",
    ]
    assert [len(messages) for messages in recorder.messages] == [1, 3]
    assert [len(info.function_tools) for info in recorder.infos] == [ALL_TOOLS] * 2


def test_watch_requests_prints_the_log() -> None:
    # Moved from 04-02.
    lines = run_script("scripts/watch_requests.py").splitlines()
    assert lines[:2] == [
        f"{LOGGER}: request step=1 messages=1 tools={ALL_TOOLS}",
        f"{LOGGER}: request step=2 messages=3 tools={ALL_TOOLS}",
    ]
    assert lines[2].startswith("output: ")
    assert lines[3].startswith("usage: ")


def test_no_script_leaves_a_notebook_in_the_sample_repo() -> None:
    # The scripts that run on the sample repo never write memory; the
    # series keeps the sample repo as shipped.
    assert not (TINYJOBS_ROOT / ".agent-memory").exists()
