"""Checks that the project from the memory lesson is in place.

Our checks for the memory lesson: they verify the lesson, and the student
is not asked to run them. Run them with `uv run pytest`. They fail until
the lesson is done. None of them needs an API key or makes a request:
every run is on a stand-in model.

Eight checks were moved here from earlier lessons, because this lesson
adds five tools and a paragraph of guidance to the chat, and they pinned
the number of tools or the instructions' text: from the toolsets lesson
(04-01) four, from the hooks lesson (04-02) two, and from the prompt
lesson (04-04) two. Two of them said the same, so seven remain here. Each
says where it came from.

Each test runs in an empty folder of its own, so the project's own `.env`
and `work/` folder are never read. A test that needs the workspace folder
makes `work/tinyjobs` there, from the sample code.

Every import of the project is inside the test that needs it, so that
before the lesson each test fails on its own, instead of the whole file
failing to load.

The check that the file tools cannot write into the notebook moved to the
approval lesson (05-01): a write is now held, and the check proves the same
point with it approved.
"""

import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic_ai import Agent, models
from pydantic_ai.capabilities import LocalWorkspace
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai_harness import Memory
from pydantic_ai_harness.memory import FileStore, InMemoryStore

# This file is tests/04_05_memory/test_memory.py: two folders below the root.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
PREFIX = "PYDANTIC_TUTOR_"
KEY_VARIABLES = ("OPENAI_API_KEY", "ANTHROPIC_API_KEY")
LOGGER = "pydantic_tutor.capabilities.request_log"
# The chat's prompt since the console-chat lesson, word for word.
PROMPT = "You are a helpful assistant. Answer briefly."
WORKSPACE_NAME = "Your workspace holds the Python project tinyjobs."
FILE_TOOLS = [
    "read_file",
    "write_file",
    "edit_file",
    "list_directory",
    "search_files",
    "find_files",
    "create_directory",
    "file_info",
]
SHELL_TOOLS = ["run_command", "start_command", "check_command", "stop_command"]
OUR_TOOLS = ["list_modules", "read_docstring"]
MEMORY_TOOLS = [
    "write_memory",
    "read_memory",
    "delete_memory",
    "search_memory",
    "list_memory",
]
ALL_TOOLS = len(FILE_TOOLS) + len(SHELL_TOOLS) + len(OUR_TOOLS) + len(MEMORY_TOOLS)


@pytest.fixture(autouse=True)
def no_real_models(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for name in list(os.environ):
        if name.upper().startswith(PREFIX) or name.upper() in KEY_VARIABLES:
            monkeypatch.delenv(name)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(target=models, name="ALLOW_MODEL_REQUESTS", value=False)


@pytest.fixture
def project(tmp_path: Path) -> Path:
    """A stand-in project folder with a fresh `work/tinyjobs`, made by the script."""
    shutil.copytree(src=PROJECT_ROOT / "sample_code", dst=tmp_path / "sample_code")
    shutil.copy(src=PROJECT_ROOT / "pyproject.toml", dst=tmp_path / "pyproject.toml")
    made = run_script(name="fresh_copy.py", folder=tmp_path)
    assert made.returncode == 0, made.stderr
    return tmp_path


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


class Recorder:
    """A model that records each request, making the calls it is given first."""

    def __init__(self, *, calls: list[ToolCallPart] | None = None) -> None:
        self.calls = list(calls or [])
        self.infos: list[AgentInfo] = []
        self.messages: list[list[ModelMessage]] = []

    def model(self) -> FunctionModel:
        return FunctionModel(function=self.reply)

    def reply(self, messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        self.infos.append(info)
        self.messages.append(list(messages))
        if self.calls:
            return ModelResponse(parts=[self.calls.pop(0)])
        return ModelResponse(parts=[TextPart(content="ok")])

    @property
    def tools(self) -> list[str]:
        return [tool.name for tool in self.infos[0].function_tools]

    @property
    def instructions(self) -> str | None:
        return self.infos[0].instructions

    @property
    def returned(self) -> str:
        """The last tool result, or refusal, the model was sent."""
        parts = [
            part
            for messages in self.messages
            for part in messages[-1].parts
            if isinstance(part, ToolReturnPart | RetryPromptPart)
        ]
        last = parts[-1]
        if isinstance(last, RetryPromptPart):
            return last.model_response()
        return last.model_response_str()


def call(*, tool_name: str, **args: str) -> ToolCallPart:
    return ToolCallPart(tool_name=tool_name, args=args)


def run_chat(*, root: Path, recorder: Recorder, prompt: str = "hi") -> None:
    """Run the chat's agent once on `root`, on the recorder's model."""
    from pydantic_tutor.chat.agent import build_chat_agent

    chat = build_chat_agent(web_search=False, workspace_root=root)
    chat.run_sync(user_prompt=prompt, model=recorder.model())


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """`work/tinyjobs`, with a notebook that already holds two files."""
    root = tmp_path / "work" / "tinyjobs"
    shutil.copytree(src=PROJECT_ROOT / "sample_code" / "tinyjobs", dst=root)
    notes = root / ".agent-memory" / "main"
    notes.mkdir(parents=True)
    (notes / "MEMORY.md").write_text("- Prefers terse answers.\n")
    (notes / "logging.md").write_text("- Logging goes to stderr.\n")
    return root


# Point 1: the chat has a notebook it reads and writes, and a tool to list it.


def test_listable_memory_is_memory_with_its_own_guidance() -> None:
    from pydantic_tutor.capabilities.listable_memory import ListableMemory
    from tutor_given.memory_text import LISTING_GUIDANCE

    memory: ListableMemory[None] = ListableMemory()
    assert isinstance(memory, Memory)
    assert memory.guidance == LISTING_GUIDANCE
    assert "list_memory" in LISTING_GUIDANCE
    assert memory.max_list_files == 100
    # `Memory`'s own id, kept: an agent has one capability of that id.
    assert memory.id == "memory"


def test_the_chat_has_one_notebook_in_its_workspace_with_injection_off(
    workspace: Path,
) -> None:
    from pydantic_tutor.chat.agent import build_chat_agent

    chat = build_chat_agent(web_search=False, workspace_root=workspace)
    memories = [c for c in chat.root_capability.capabilities if isinstance(c, Memory)]
    assert len(memories) == 1
    [memory] = memories
    assert type(memory).__module__ == "pydantic_tutor.capabilities.listable_memory"
    assert memory.inject_memory is False
    assert isinstance(memory.store, FileStore)
    assert memory.store.directory == ".agent-memory"
    assert memory.store.workspace is None  # the run's workspace: the chat's folder


def test_no_request_carries_the_notebook_and_the_model_lists_it(
    workspace: Path,
) -> None:
    # The lesson's point: the notebook's files are on disk, and no request
    # carries them. The model gets the names when it asks.
    recorder = Recorder(calls=[call(tool_name="list_memory")])
    run_chat(root=workspace, recorder=recorder)
    for messages in recorder.messages:
        for message in messages:
            for part in message.parts:
                text = str(getattr(part, "content", ""))
                assert "<memory>" not in text
                assert "Prefers terse answers" not in text
    [first, *_] = recorder.messages
    assert [part.part_kind for part in first[0].parts] == ["user-prompt"]
    assert recorder.returned == '{"files":["MEMORY.md","logging.md"],"truncated":false}'


def test_injection_on_would_send_the_block_beside_the_question(
    workspace: Path,
) -> None:
    # The same notebook, the same stand-in, one setting different.
    recorder = Recorder()
    injecting: Agent[None, str] = Agent(
        capabilities=[
            LocalWorkspace(working_dir=workspace),
            Memory(store=FileStore(directory=".agent-memory"), inject_memory=True),
        ]
    )
    injecting.run_sync(user_prompt="hi", model=recorder.model())
    [request] = recorder.messages[0]
    assert isinstance(request, ModelRequest)
    assert [part.part_kind for part in request.parts] == ["user-prompt", "user-prompt"]
    block = request.parts[1]
    assert isinstance(block, UserPromptPart)
    assert "logging.md" in str(block.content)


def test_a_note_written_in_one_run_is_found_in_the_next(workspace: Path) -> None:
    shutil.rmtree(workspace / ".agent-memory")
    note = "- Logging goes to stderr.\n"
    run_chat(
        root=workspace,
        recorder=Recorder(
            calls=[call(tool_name="write_memory", content=note, file="logging.md")]
        ),
    )
    assert (workspace / ".agent-memory" / "main" / "logging.md").read_text() == note
    # A second run, with no history: only the file carries over.
    later = Recorder(
        calls=[
            call(tool_name="list_memory"),
            call(tool_name="read_memory", file="logging.md"),
        ]
    )
    run_chat(root=workspace, recorder=later)
    assert len(later.messages[0]) == 1
    assert later.returned == note


def test_the_file_tools_still_read_and_write_the_rest_of_the_workspace(
    workspace: Path,
) -> None:
    # The deny list gained one pattern; it did not close the workspace.
    recorder = Recorder(calls=[call(tool_name="read_file", path="README.md")])
    run_chat(root=workspace, recorder=recorder)
    assert "# tinyjobs" in recorder.returned


def test_list_memory_lists_one_notebook_and_says_when_it_is_cut() -> None:
    import asyncio

    from pydantic_tutor.capabilities.listable_memory import ListableMemory
    from tutor_given.memory_text import MemoryListing

    async def fill() -> InMemoryStore:
        store = InMemoryStore()
        for name in ["a.md", "b.md", "c.md"]:
            await store.write(f"main/{name}", "x\n", expected_version=None)
        # Another notebook in the same store: not this one's to list.
        await store.write("other/secret.md", "x\n", expected_version=None)
        return store

    store = asyncio.run(fill())
    second: Agent[None, str] = Agent(
        capabilities=[ListableMemory(store=store, max_list_files=2)]
    )
    recorder = Recorder(calls=[call(tool_name="list_memory")])
    second.run_sync(user_prompt="x", model=recorder.model())
    listing = MemoryListing.model_validate_json(recorder.returned)
    assert listing == MemoryListing(files=["a.md", "b.md"], truncated=True)


def test_list_memory_names_only_files_read_memory_takes(workspace: Path) -> None:
    # A file in a folder below the notebook's, and one not in Markdown:
    # `read_memory` refuses both, so the listing must not offer them.
    notes = workspace / ".agent-memory" / "main"
    (notes / "team").mkdir()
    (notes / "team" / "deep.md").write_text("x\n")
    (notes / "draft.txt").write_text("x\n")
    recorder = Recorder(calls=[call(tool_name="list_memory")])
    run_chat(root=workspace, recorder=recorder)
    assert recorder.returned == '{"files":["MEMORY.md","logging.md"],"truncated":false}'


def test_list_memory_joins_memorys_own_toolset() -> None:
    # `Memory`'s tools find the run's copy of the capability by comparing
    # its toolset with themselves, and durable engines know it by its id:
    # `get_toolset()` must return Memory's one object, every time.
    from pydantic_tutor.capabilities.listable_memory import ListableMemory

    memory: ListableMemory[None] = ListableMemory()
    toolset = memory.get_toolset()
    assert toolset is memory.get_toolset()
    assert toolset is Memory.get_toolset(memory)
    assert getattr(toolset, "id", None) == "memory"


def test_notebook_prints_two_sessions(project: Path) -> None:
    result = run_script(name="notebook.py", folder=project)
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
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


def test_a_fresh_copy_starts_with_an_empty_notebook(project: Path) -> None:
    assert run_script(name="notebook.py", folder=project).returncode == 0
    run_script(name="fresh_copy.py", folder=project)
    assert not (project / "work" / "tinyjobs" / ".agent-memory").exists()


# Point 2: what injection does to every request.


def test_memory_injection_moves_the_block_to_the_newest_question(
    project: Path,
) -> None:
    # Goal 2: injection would send the notebook beside the newest question,
    # and take it off the one before.
    assert run_script(name="notebook.py", folder=project).returncode == 0
    result = run_script(name="memory_injection.py", folder=project)
    assert result.returncode == 0, result.stderr
    block = "<memory> ### Other memory files - logging.md </memory>"
    assert result.stdout.splitlines() == [
        "turn: What does tinyjobs do?",
        f"  request: ['What does tinyjobs do?', '{block}']",
        "turn: And how is it tested?",
        "  request: ['What does tinyjobs do?']",
        "  response: ['ok']",
        f"  request: ['And how is it tested?', '{block}']",
    ]


def test_the_guidance_is_sent_after_the_prompt_and_before_the_workspace(
    workspace: Path,
) -> None:
    # Off is not "no memory": the guidance is text in the instructions, so
    # it follows the prompt (list order), and the toolset's sentence, also
    # static text, comes last.
    from tutor_given.memory_text import LISTING_GUIDANCE

    recorder = Recorder()
    run_chat(root=workspace, recorder=recorder)
    parts = recorder.infos[0].model_request_parameters.instruction_parts or []
    assert [part.content for part in parts] == [
        PROMPT,
        LISTING_GUIDANCE,
        WORKSPACE_NAME,
    ]
    assert [part.dynamic for part in parts] == [False, False, False]


# Moved: they pinned the chat's tools or instructions before the memory.


def test_the_chat_is_offered_the_toolset_and_the_tools_it_had(
    workspace: Path,
) -> None:
    # Moved from 04-01 (and before it 03-05): the chat's tools, now with
    # the notebook's five.
    from pydantic_tutor.composition import build_app
    from pydantic_tutor.settings import Settings

    recorder = Recorder()
    app = build_app(
        settings=Settings(chat_model="test", web_search=False, workspace_root=workspace)
    )
    app.agent.run_sync(user_prompt="hi", model=recorder.model())
    assert sorted(recorder.tools) == sorted(
        [*FILE_TOOLS, *SHELL_TOOLS, *OUR_TOOLS, *MEMORY_TOOLS]
    )
    assert len(recorder.tools) == ALL_TOOLS == 19


def test_the_chat_sends_the_prompt_the_notebooks_guidance_then_the_toolsets(
    workspace: Path,
) -> None:
    # Moved from 04-01 and 04-04, which pinned the instructions as the
    # prompt then the workspace sentence.
    from tutor_given.memory_text import LISTING_GUIDANCE

    recorder = Recorder()
    run_chat(root=workspace, recorder=recorder)
    assert (
        recorder.instructions == f"{PROMPT}\n\n{LISTING_GUIDANCE}\n\n{WORKSPACE_NAME}"
    )


def test_the_prompt_file_is_still_what_the_chat_is_sent_first(
    workspace: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Moved from 04-04: change the file, and the chat built next is sent
    # the change, ahead of the guidance.
    from pydantic_tutor.chat import agent as chat_agent
    from tutor_given.memory_text import LISTING_GUIDANCE

    edited = tmp_path / "chat.md"
    edited.write_text("You are a pirate.\n", encoding="utf-8")
    monkeypatch.setattr(target=chat_agent, name="PROMPT_PATH", value=edited)
    recorder = Recorder()
    run_chat(root=workspace, recorder=recorder)
    assert recorder.instructions == (
        f"You are a pirate.\n\n{LISTING_GUIDANCE}\n\n{WORKSPACE_NAME}"
    )


def test_workspace_reach_counts_the_notebooks_tools_too(project: Path) -> None:
    # Moved from 04-01, where it printed 14.
    result = run_script(name="workspace_reach.py", folder=project)
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert f"tools the model is offered: {ALL_TOOLS}" in lines
    assert "README.md: # tinyjobs" in lines


def test_toolset_views_prints_the_chat_and_what_each_wrapper_sends(
    project: Path,
) -> None:
    # Moved from 04-01: the chat's line now ends with the notebook's tools.
    result = run_script(name="toolset_views.py", folder=project)
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    [chat_tools] = [line for line in lines if line.startswith("  tools: read_file, ")]
    assert ", list_modules, read_docstring, " in chat_tools
    assert chat_tools.endswith(", ".join(MEMORY_TOOLS))
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


def test_each_request_is_logged_with_its_step_messages_and_tools(
    workspace: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # Moved from 04-02, where the log said tools=14.
    caplog.set_level(level=logging.INFO, logger=LOGGER)
    recorder = Recorder(calls=[call(tool_name="list_modules")])
    run_chat(root=workspace, recorder=recorder)
    logged = [r.getMessage() for r in caplog.records if r.name == LOGGER]
    assert logged == [
        f"request step=1 messages=1 tools={ALL_TOOLS}",
        f"request step=2 messages=3 tools={ALL_TOOLS}",
    ]
    assert [len(info.function_tools) for info in recorder.infos] == [ALL_TOOLS] * 2


def test_watch_requests_prints_the_log(project: Path) -> None:
    # Moved from 04-02.
    result = run_script(name="watch_requests.py", folder=project)
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert lines[:2] == [
        f"{LOGGER}: request step=1 messages=1 tools={ALL_TOOLS}",
        f"{LOGGER}: request step=2 messages=3 tools={ALL_TOOLS}",
    ]
    assert lines[2].startswith("output: ")
    assert lines[3].startswith("usage: ")
