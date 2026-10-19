"""Tools that read a Python repository, for the agent to call.

Each tool takes the run's context first: the agent passes it, and the model
never sees it. The repository a tool reads is in the context's `deps`,
chosen by the code that runs the agent.
"""

import ast
from dataclasses import dataclass
from pathlib import Path

from pydantic_ai import ModelRetry, RunContext


@dataclass(frozen=True)
class RepoDeps:
    """What a run of the agent needs: the repository its tools read."""

    # The repository's top folder. Its modules are under `src/`, the layout
    # every repository in this series uses.
    root: Path


def list_modules(ctx: RunContext[RepoDeps]) -> list[str]:
    """List the Python modules in the repository, by dotted name."""
    return sorted(_modules(source=ctx.deps.root / "src"))


def read_docstring(ctx: RunContext[RepoDeps], *, module: str) -> str:
    """Read a module's docstring, without running the module.

    Args:
        module: The module's dotted name, as `list_modules` gives it.
    """
    # The model chose `module`: look it up, never open it as a path.
    path = _modules(source=ctx.deps.root / "src").get(module)
    if path is None:
        raise ModelRetry(
            message=f"There is no module {module!r}. Call list_modules to see them."
        )
    # Bytes, so that `ast` reads the file's own encoding, not your system's.
    docstring = ast.get_docstring(node=ast.parse(source=path.read_bytes()))
    return docstring or "(This module has no docstring.)"


def _modules(*, source: Path) -> dict[str, Path]:
    """Every module under `source`, by dotted name: `tinyjobs/jobs.py` is `tinyjobs.jobs`."""
    modules: dict[str, Path] = {}
    for path in source.rglob(pattern="*.py"):
        parts = path.relative_to(other=source).with_suffix(suffix="").parts
        if parts[-1] == "__init__":  # a package's own module is the package
            parts = parts[:-1]
        modules[".".join(parts)] = path
    return modules
