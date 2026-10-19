"""Tools that read a Python repository, for the agent to call.

Each one is a plain function: the agent sends the model its name, its
docstring and its parameters, and runs it when the model asks.
"""

import ast
from pathlib import Path

from pydantic_ai import ModelRetry

# The source folder of the repository the tools read. Relative, so run from
# the project root. The next lesson passes the repository in instead.
SOURCE = Path("sample_repos/tinyjobs/src")


def list_modules() -> list[str]:
    """List the Python modules in the repository, by dotted name."""
    return sorted(_modules())


def read_docstring(*, module: str) -> str:
    """Read a module's docstring, without running the module.

    Args:
        module: The module's dotted name, as `list_modules` gives it.
    """
    # The model chose `module`: look it up, never open it as a path.
    path = _modules().get(module)
    if path is None:
        raise ModelRetry(
            f"There is no module {module!r}. Call list_modules to see them."
        )
    # Bytes, so that `ast` reads the file's own encoding, not your system's.
    docstring = ast.get_docstring(node=ast.parse(source=path.read_bytes()))
    return docstring or "(This module has no docstring.)"


def _modules() -> dict[str, Path]:
    """Every module under SOURCE, by dotted name: `tinyjobs/jobs.py` is `tinyjobs.jobs`."""
    modules: dict[str, Path] = {}
    for path in SOURCE.rglob(pattern="*.py"):
        parts = path.relative_to(other=SOURCE).with_suffix(suffix="").parts
        if parts[-1] == "__init__":  # a package's own module is the package
            parts = parts[:-1]
        modules[".".join(parts)] = path
    return modules
