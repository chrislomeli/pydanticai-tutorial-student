"""Two read-only tools over a Python project's modules, as one toolset.

From the toolsets lesson (04-01). Built once, where the app is built, with
the folder the tools read: they take no deps.
"""

import ast
from pathlib import Path

from pydantic_ai import ModelRetry, ToolFailed
from pydantic_ai.toolsets import FunctionToolset


def build_workspace_toolset(*, root: Path) -> FunctionToolset[None]:
    """The tools that read the modules under `root/src`, with their instructions."""

    def modules() -> dict[str, Path]:
        """Every module under `root/src`, by dotted name."""
        found: dict[str, Path] = {}
        for path in sorted((root / "src").rglob(pattern="*.py")):
            parts = path.relative_to(root / "src").with_suffix(suffix="").parts
            if parts[-1] == "__init__":  # a package's own module is the package
                parts = parts[:-1]
            found[".".join(parts)] = path
        return found

    def list_modules() -> list[str]:
        """List the Python modules in the workspace, by dotted name."""
        return list(modules())

    def read_docstring(*, module: str) -> str:
        """Read a module's docstring, without running the module.

        Args:
            module: The module's dotted name, as `list_modules` gives it.
        """
        # Looked up among the listed names, never turned into a path.
        path = modules().get(module)
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

    return FunctionToolset(
        tools=[list_modules, read_docstring],
        # Sent with every request, beside the agent's own instructions.
        instructions=f"Your workspace holds the Python project {root.resolve().name}.",
        # Set here: an agent's `tool_timeout` covers only its own `tools=`.
        timeout=10,
    )
