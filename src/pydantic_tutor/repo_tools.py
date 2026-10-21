"""Tools that read a Python repository, for the agent to call.

Each tool takes the run's context first: the agent passes it, and the model
never sees it. The repository a tool reads is in the context's `deps`,
chosen by the code that runs the agent; the tool reads it with the repo API.
"""

from dataclasses import dataclass
from pathlib import Path

from pydantic_ai import ModelRetry, RunContext, ToolFailed

from pydantic_tutor import repo


@dataclass(frozen=True)
class RepoDeps:
    """What a run of the agent needs: the repository its tools read."""

    # The repository's top folder. The repo API reads its modules under
    # `src/`, the layout every repository in this series uses.
    root: Path


def list_modules(ctx: RunContext[RepoDeps]) -> list[str]:
    """List the Python modules in the repository, by dotted name."""
    return repo.list_modules(root=ctx.deps.root)


def read_docstring(ctx: RunContext[RepoDeps], *, module: str) -> str:
    """Read a module's docstring, without running the module.

    Args:
        module: The module's dotted name, as `list_modules` gives it.
    """
    try:
        return repo.read_docstring(root=ctx.deps.root, module=module)
    except repo.UnknownModuleError as error:
        # The model chose `module`: tell it what went wrong, and what to do.
        raise ModelRetry(message=f"{error} Call list_modules to see them.") from error
    except repo.UnreadableModuleError as error:
        # Asking again will not help: the file itself is broken. Say so, and
        # the model can tell the user instead of retrying.
        raise ToolFailed(message=str(error)) from error
