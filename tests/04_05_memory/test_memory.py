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

Twelve checks moved to the deferred lesson (06-02), which loads the
notebook on demand: they pinned its five tools and its guidance in every
request, or a call to one of its tools made with no load before it.

The check that the file tools cannot write into the notebook moved to the
approval lesson (05-01): a write is now held, and the check proves the same
point with it approved.
"""

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


# Moved: they pinned the chat's tools or instructions before the memory.
