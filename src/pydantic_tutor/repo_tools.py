"""Tools that read a Python repository, for the agent to call.

Each tool takes the run's context first: the agent passes it, and the model
never sees it. Since the workspace lesson (03-03), a tool reads the
repository through the run's workspace, `ctx.workspace`, whose folder is
the repository's root: the tool reads wherever the workspace is.
"""

import ast
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from pydantic_ai import ModelRetry, RunContext, ToolFailed
from pydantic_ai.workspaces import Workspace


@dataclass(frozen=True)
class RepoDeps:
    """What a run of the agent needs: the repository its tools read."""

    # The repository's top folder, and the folder of the run's workspace
    # (agent.py). Its modules are under `src/`, the layout every repository
    # in this series uses.
    root: Path


async def list_modules(ctx: RunContext[RepoDeps]) -> list[str]:
    """List the Python modules in the repository, by dotted name."""
    return sorted(await _modules(workspace=ctx.workspace))


async def read_docstring(ctx: RunContext[RepoDeps], *, module: str) -> str:
    """Read a module's docstring, without running the module.

    Args:
        module: The module's dotted name, as `list_modules` gives it.
    """
    # The model chose `module`: look it up, never open it as a path.
    path = (await _modules(workspace=ctx.workspace)).get(module)
    if path is None:
        raise ModelRetry(
            message=f"There is no module {module!r}. Call list_modules to see them."
        )
    try:
        # Bytes, so that `ast` reads the file's own encoding, not your system's.
        tree = ast.parse(source=await ctx.workspace.read_bytes(path=path))
    except OSError as error:
        # Listed, but it cannot be opened: a broken link, or no permission.
        # Asking again will not help either.
        raise ToolFailed(
            message=f"{module} cannot be opened ({type(error).__name__})."
        ) from error
    except SyntaxError as error:
        # Asking again will not help: the file itself is broken. Say so, and
        # the model can tell the user instead of retrying.
        raise ToolFailed(
            message=f"{module} cannot be read: {error.msg} on line {error.lineno}."
        ) from error
    docstring = ast.get_docstring(node=tree)
    return docstring or "(This module has no docstring.)"


async def _modules(*, workspace: Workspace) -> dict[str, str]:
    """Every module under `src/`, by dotted name: `tinyjobs/jobs.py` is `tinyjobs.jobs`.

    Each path is absolute, in the workspace. A linked folder is not
    entered, as `Path.rglob` does not enter one: it could lead back up
    and round forever.
    """
    modules: dict[str, str] = {}
    if not await workspace.exists(path="src"):
        return modules  # a repository with no `src/` has no modules
    source = await workspace.realpath(path="src")
    folders = [source]
    while folders:
        try:
            entries = await workspace.list_dir(path=folders.pop())
        except OSError:
            continue  # a folder you may not read is skipped, as `Path.rglob` skips it
        for entry in entries:
            if entry.is_dir:
                # A folder's real path is its own path unless it is a link.
                if await workspace.realpath(path=entry.path) == entry.path:
                    folders.append(entry.path)
            elif entry.name.endswith(".py"):
                relative = PurePosixPath(entry.path).relative_to(other=source)
                parts = relative.with_suffix(suffix="").parts
                if parts[-1] == "__init__":  # a package's own module is the package
                    parts = parts[:-1]
                modules[".".join(parts)] = entry.path
    return modules
