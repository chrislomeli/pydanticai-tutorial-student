"""Read a Python repository: the example's API, given to you.

You call it; you do not need to read it. The agent's tools use it to answer
questions about a repository's modules. A repository's modules are under
its `src/` folder, the layout every repository in this series uses.

It reads a repository in one of two ways, and both behave the same:

- from a folder on this machine: `list_modules(root=...)` and
  `read_docstring(root=..., module=...)`;
- through a workspace (from the workspace lesson, 03-03):
  `await list_modules_in(workspace=...)` and
  `await read_docstring_in(workspace=..., module=...)`.

A module is always named, never given as a path: a name such as
`"../../.env"` is not one of `list_modules`' names, so it reads nothing.
Nor does a link: a linked folder under `src/` is not entered, and a linked
file that leads out of `src/` is not listed.
"""

import ast
from pathlib import Path, PurePosixPath

from pydantic_ai.workspaces import Workspace


class UnknownModuleError(Exception):
    """Raised for a module name the repository does not have."""


class UnreadableModuleError(Exception):
    """Raised for a module that is listed but cannot be opened or parsed."""


def list_modules(*, root: Path) -> list[str]:
    """Every module in the repository at `root`, by dotted name, sorted."""
    return sorted(_modules(source=root / "src"))


def read_docstring(*, root: Path, module: str) -> str:
    """A module's docstring, read without running the module.

    Raises:
        UnknownModuleError: `module` is not one of `list_modules`' names.
        UnreadableModuleError: the module cannot be opened or parsed.
    """
    path = _modules(source=root / "src").get(module)
    if path is None:
        raise UnknownModuleError(f"There is no module {module!r}.")
    try:
        source = path.read_bytes()
    except OSError as error:
        raise _cannot_open(module=module, error=error) from error
    return _docstring(module=module, source=source)


async def list_modules_in(*, workspace: Workspace) -> list[str]:
    """Every module in the repository at the workspace's folder, by dotted name, sorted."""
    return sorted(await _workspace_modules(workspace=workspace))


async def read_docstring_in(*, workspace: Workspace, module: str) -> str:
    """A module's docstring, read through the workspace without running the module.

    Raises:
        UnknownModuleError: `module` is not one of `list_modules_in`' names.
        UnreadableModuleError: the module cannot be opened or parsed.
    """
    path = (await _workspace_modules(workspace=workspace)).get(module)
    if path is None:
        raise UnknownModuleError(f"There is no module {module!r}.")
    try:
        source = await workspace.read_bytes(path=path)
    except OSError as error:
        raise _cannot_open(module=module, error=error) from error
    return _docstring(module=module, source=source)


def _cannot_open(*, module: str, error: OSError) -> UnreadableModuleError:
    # Listed, but it cannot be opened: a broken link, or no permission.
    return UnreadableModuleError(f"{module} cannot be opened ({type(error).__name__}).")


def _docstring(*, module: str, source: bytes) -> str:
    """The docstring of a module's source, or a line saying it has none."""
    try:
        # Bytes, so that `ast` reads the file's own encoding, not your system's.
        tree = ast.parse(source=source)
    except SyntaxError as error:
        where = f" on line {error.lineno}" if error.lineno is not None else ""
        raise UnreadableModuleError(
            f"{module} cannot be read: {error.msg}{where}."
        ) from error
    return ast.get_docstring(node=tree) or "(This module has no docstring.)"


def _dotted(*, relative: PurePosixPath) -> str:
    """A module's dotted name from its path under `src/`: `tinyjobs/jobs.py` is `tinyjobs.jobs`."""
    parts = relative.with_suffix(suffix="").parts
    if parts[-1] == "__init__":  # a package's own module is the package
        parts = parts[:-1]
    return ".".join(parts)


def _modules(*, source: Path) -> dict[str, Path]:
    """Every module under `source`, by dotted name.

    A linked folder is not entered (`Path.rglob` does not enter one), and a
    linked file that leads out of `source` is left out.
    """
    inside = source.resolve()
    return {
        _dotted(relative=PurePosixPath(path.relative_to(other=source).as_posix())): path
        for path in source.rglob(pattern="*.py")
        if path.resolve().is_relative_to(other=inside)
    }


async def _workspace_modules(*, workspace: Workspace) -> dict[str, str]:
    """Every module under the workspace's `src/`, by dotted name, each with its absolute path.

    As in the folder form, a linked folder is not entered (it could lead back
    up and round forever), and a linked file that leads out of `src/` is
    left out.
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
            # An entry's real path is its own path unless it is a link.
            real = await workspace.realpath(path=entry.path)
            if entry.is_dir:
                if real == entry.path:
                    folders.append(entry.path)
            elif entry.name.endswith(".py"):
                if not PurePosixPath(real).is_relative_to(other=source):
                    continue  # a linked file that leads out of `src/`
                relative = PurePosixPath(entry.path).relative_to(other=source)
                modules[_dotted(relative=relative)] = entry.path
    return modules
