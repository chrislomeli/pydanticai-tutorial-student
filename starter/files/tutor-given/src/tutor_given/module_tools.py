"""Two read-only tools for looking at the modules of a Python project.

Given, from the toolsets lesson (04-01): you run it, you don't write it. Hand
`list_modules` and `read_docstring` to a toolset, and the model calls them. The
project's folder is set once, when you build `ModuleTools`, so the tools take
no deps and the model never sees the path.
"""

import ast
from dataclasses import dataclass
from pathlib import Path

from pydantic_ai import ModelRetry, ToolFailed


@dataclass(frozen=True, kw_only=True)
class ModuleTools:
    """The tools, bound to the project at `root`, whose code is under `root/src`."""

    root: Path

    def _modules(self) -> dict[str, Path]:
        """Every module under `root/src`, by dotted name."""
        source = self.root / "src"
        found: dict[str, Path] = {}
        for path in sorted(source.rglob(pattern="*.py")):
            parts = path.relative_to(source).with_suffix(suffix="").parts
            if parts[-1] == "__init__":  # a package's own module is the package
                parts = parts[:-1]
            found[".".join(parts)] = path
        return found

    def list_modules(self) -> list[str]:
        """List the Python modules in the workspace, by dotted name."""
        return list(self._modules())

    def read_docstring(self, *, module: str) -> str:
        """Read a module's docstring, without running the module.

        Args:
            module: The module's dotted name, as `list_modules` gives it.
        """
        # Looked up among the listed names, never turned into a path.
        path = self._modules().get(module)
        if path is None:
            raise ModelRetry(
                message=f"There is no module {module!r}. Call list_modules to see them."
            )
        try:
            tree = ast.parse(source=path.read_bytes())
        except SyntaxError as error:
            raise ToolFailed(
                message=f"{module} cannot be read: {error.msg}."
            ) from error
        return ast.get_docstring(node=tree) or "(This module has no docstring.)"
