"""Tools that read a Python repository, and the toolset that carries them.

Each tool takes the run's context first: the agent passes it, and the model
never sees it. A tool reads the repository through the run's workspace,
`ctx.workspace`, whose folder is the repository's root, with the repo API's
workspace functions: the tool reads wherever the workspace is. An agent
gets the tools as one object, `repo_toolset`.
"""

from dataclasses import dataclass
from pathlib import Path

from pydantic_ai import ModelRetry, RunContext, ToolFailed
from pydantic_ai.toolsets import FunctionToolset

from pydantic_tutor import repo


@dataclass(frozen=True)
class RepoDeps:
    """What a run of the agent needs: the repository its tools read."""

    # The repository's top folder, and the folder of the run's workspace
    # (agent.py). The repo API reads its modules under `src/`, the layout
    # every repository in this series uses.
    root: Path


async def list_modules(ctx: RunContext[RepoDeps]) -> list[str]:
    """List the Python modules in the repository, by dotted name."""
    return await repo.list_modules_in(workspace=ctx.workspace)


async def read_docstring(ctx: RunContext[RepoDeps], *, module: str) -> str:
    """Read a module's docstring, without running the module.

    Args:
        module: The module's dotted name, as `list_modules` gives it.
    """
    try:
        return await repo.read_docstring_in(workspace=ctx.workspace, module=module)
    except repo.UnknownModuleError as error:
        # The model chose `module`: tell it what went wrong, and what to do.
        raise ModelRetry(message=f"{error} Call list_modules to see them.") from error
    except repo.UnreadableModuleError as error:
        # Asking again will not help: the file is broken, or cannot be
        # opened. Say so, and the model can tell the user instead of retrying.
        raise ToolFailed(message=str(error)) from error


def name_the_repository(ctx: RunContext[RepoDeps]) -> str:
    """Tell the model which repository this run reads."""
    # Called for every request of a run, with that run's deps.
    return f"The repository is {ctx.deps.root.resolve().name}."


# The repository tools as one object: an agent whose deps are `RepoDeps`
# takes them with `toolsets=[repo_toolset]`, and gets their instructions
# and their time limit with them. Not the workspace the tools read
# through: the agent supplies that.
repo_toolset: FunctionToolset[RepoDeps] = FunctionToolset(
    tools=[list_modules, read_docstring],
    # Sent with every request, beside the agent's own instructions.
    instructions=name_the_repository,
    # A call that takes longer is abandoned, and the model told so. Set
    # here: an agent's `tool_timeout` covers only the tools in its `tools=`.
    timeout=10,
)
