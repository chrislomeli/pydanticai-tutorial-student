"""The library's `Memory`, plus a tool that lists the notebook's files.

From the memory lesson (04-05). With injection off nothing names the
files to the model, and none of `Memory`'s tools lists them:
`ListableMemory` adds `list_memory`, and guidance that names it.
"""

from dataclasses import dataclass

from pydantic_ai import RunContext
from pydantic_ai.toolsets import AgentToolset, FunctionToolset
from pydantic_ai_harness import Memory

from tutor_given.memory_text import LISTING_GUIDANCE, MemoryListing


@dataclass(kw_only=True)
class ListableMemory[DepsT](Memory[DepsT]):
    """`Memory`, with a `list_memory` tool beside its own four."""

    guidance: str | None = LISTING_GUIDANCE
    max_list_files: int = 100  # the most names one call returns

    def get_toolset(self) -> AgentToolset[DepsT] | None:
        """Return `Memory`'s own toolset, with `list_memory` added once."""
        memory_tools = super().get_toolset()
        # `Memory` returns its one FunctionToolset: tells mypy, fails if not.
        if not isinstance(memory_tools, FunctionToolset):
            raise TypeError("Memory.get_toolset() did not return a FunctionToolset")
        # The same object every time, never a new one: `Memory`'s tools find
        # the run's copy of the capability by it. The agent asks on every
        # memory tool call, hence "once".
        if "list_memory" not in memory_tools.tools:
            memory_tools.add_function(func=self.list_memory, name="list_memory")
        return memory_tools

    async def list_memory(self, ctx: RunContext[DepsT]) -> MemoryListing:
        """List the files in your memory notebook, MEMORY.md included.

        Call it before `read_memory` when you do not know a file's name.
        """
        # The run's store and this notebook's folder: the other tools' own.
        store, scope = self.resolve_scope(ctx=ctx)
        prefix = f"{scope}/"
        # One over the limit, so exactly that many files is not "cut".
        paths = await store.list_paths(prefix=prefix, limit=self.max_list_files + 1)
        # Only names `read_memory` takes: Markdown, not in a folder below.
        names = sorted(
            name
            for name in (path.removeprefix(prefix) for path in paths)
            if "/" not in name and name.endswith(".md")
        )
        return MemoryListing(
            files=names[: self.max_list_files],
            truncated=len(paths) > self.max_list_files,
        )
