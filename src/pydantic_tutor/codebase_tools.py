"""Tools that read a Python codebase, for the agent to call.

Each tool takes the run's context first: the agent passes it, and the model
never sees it. The codebase a tool reads is in the context's `deps`,
chosen by the code that runs the agent; the tool reads it with the codebase API.
"""

from dataclasses import dataclass
from pathlib import Path

from pydantic_ai import ModelRetry, RunContext, ToolFailed

from pydantic_tutor import codebase


@dataclass(frozen=True)
class CodebaseDeps:
    """What a run of the agent needs: the codebase its tools read."""

    # The codebase's top folder. The codebase API reads its modules under
    # `src/`, the layout every codebase in this series uses.
    root: Path


def list_modules(ctx: RunContext[CodebaseDeps]) -> list[str]:
    """List the Python modules in the codebase, by dotted name."""
    return codebase.list_modules(root=ctx.deps.root)


def read_docstring(ctx: RunContext[CodebaseDeps], *, module: str) -> str:
    """Read a module's docstring, without running the module.

    Args:
        module: The module's dotted name, as `list_modules` gives it.
    """
    try:
        return codebase.read_docstring(root=ctx.deps.root, module=module)
    except codebase.UnknownModuleError as error:
        # The model chose `module`: tell it what went wrong, and what to do.
        raise ModelRetry(message=f"{error} Call list_modules to see them.") from error
    except codebase.UnreadableModuleError as error:
        # Asking again will not help: the file itself is broken. Say so, and
        # the model can tell the user instead of retrying.
        raise ToolFailed(message=str(error)) from error
