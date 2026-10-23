"""Checks that the project from the memory lesson is in place.

Our checks for the memory lesson: they verify the lesson, and the student
is not asked to run them. Run them with `uv run pytest`. They fail until
the lesson is done. None of them needs an API key or makes a request:
every run is on a scripted model.

Twelve tests were moved here: the memory capability adds its guidance to
the instructions and five tools to the agent, and they pinned the
instructions' text or the number of tools. Each says where it came from,
and proves its point with the memory's part included.

One of them was moved again by the seam lesson (04-06), which gives the
agent the codebase toolset as a capability, not on `toolsets=`: it is
now `test_the_agent_takes_the_codebase_toolset_as_a_capability` in
`tests/04_06_seam/test_seam.py`.

Fourteen were moved on by the lesson on loading tools on demand (06-02),
which loads the notebook only when the model asks for it: every test that
pinned the instructions' text, the tools the model is sent or the request
log's count, or had the model call a memory tool with nothing loaded. They
are in `tests/06_02_deferred/test_deferred.py`, twelve by the same names;
the tools' signatures and the order of the instructions are proved there
by `test_before_the_load_a_request_carries_the_catalog_not_the_notebook`.

Every import of the project is inside the test that needs it, so that
before the lesson each test fails on its own, instead of the whole file
failing to load.
"""

import asyncio
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
TINYJOBS_ROOT: Path = PROJECT_ROOT / "sample_code" / "tinyjobs"


@pytest.fixture(autouse=True)
def no_real_model(monkeypatch: pytest.MonkeyPatch) -> None:
    # Any request to a real model raises instead of reaching a provider.
    monkeypatch.setattr(target=models, name="ALLOW_MODEL_REQUESTS", value=False)


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
    assert memory.store.workspace is None  # the run's workspace: the codebase


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


def test_no_script_leaves_a_notebook_in_the_sample_codebase() -> None:
    # The scripts that run on the sample codebase never write memory; the
    # series keeps the sample codebase as shipped.
    assert not (TINYJOBS_ROOT / ".agent-memory").exists()
