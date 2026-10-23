"""The agent's tools: each one calls the codebase API for the model.

Each is a plain function: the agent sends the model its name, its
description and its parameters, and runs it when the model asks.
"""

from pathlib import Path

from pydantic_ai import ModelRetry

from pydantic_tutor import codebase

# The codebase the tools read. Relative, so run from the project root.
# The next lesson passes the codebase in instead.
CODEBASE = Path("sample_code/tinyjobs")


def list_modules() -> list[str]:
    """List the Python modules in the codebase, by dotted name."""
    return codebase.list_modules(root=CODEBASE)


def read_docstring(*, module: str) -> str:
    """Read a module's docstring, without running the module.

    Args:
        module: The module's dotted name, as `list_modules` gives it.
    """
    try:
        return codebase.read_docstring(root=CODEBASE, module=module)
    except codebase.UnknownModuleError as error:
        # The model chose `module`: tell it what went wrong, and what to do.
        raise ModelRetry(message=f"{error} Call list_modules to see them.") from error
