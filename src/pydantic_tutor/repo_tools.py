"""The agent's tools: each one calls the repo API for the model.

Each is a plain function: the agent sends the model its name, its
description and its parameters, and runs it when the model asks.
"""

from pathlib import Path

from pydantic_ai import ModelRetry

from pydantic_tutor import repo

# The repository the tools read. Relative, so run from the project root.
# The next lesson passes the repository in instead.
REPO = Path("sample_repos/tinyjobs")


def list_modules() -> list[str]:
    """List the Python modules in the repository, by dotted name."""
    return repo.list_modules(root=REPO)


def read_docstring(*, module: str) -> str:
    """Read a module's docstring, without running the module.

    Args:
        module: The module's dotted name, as `list_modules` gives it.
    """
    try:
        return repo.read_docstring(root=REPO, module=module)
    except repo.UnknownModuleError as error:
        # The model chose `module`: tell it what went wrong, and what to do.
        raise ModelRetry(message=f"{error} Call list_modules to see them.") from error
