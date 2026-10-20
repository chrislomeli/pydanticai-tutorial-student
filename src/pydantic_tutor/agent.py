from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import LocalWorkspace
from pydantic_ai.settings import ModelSettings

from pydantic_tutor.repo_tools import RepoDeps, list_modules, read_docstring


def repo_workspace(ctx: RunContext[RepoDeps]) -> LocalWorkspace[RepoDeps]:
    """Give each run a workspace on this machine, in the repository it reads."""
    # Called once per run, with that run's deps: the workspace follows the
    # repository the run is given. Its folder is where relative paths
    # start, not a boundary: `..` and absolute paths reach past it.
    # Keep it to building the capability from deps: no I/O, no state.
    return LocalWorkspace(working_dir=ctx.deps.root)


# No model here: each run says which model to use, so the same agent runs on
# a test model now and a real one later. No repository either: each run
# passes one, as `deps`, and gets a workspace there.
agent = Agent(
    name="repo_assistant",
    deps_type=RepoDeps,
    instructions="You answer questions about a Python code repository. Answer briefly.",
    tools=[list_modules, read_docstring],
    capabilities=[repo_workspace],
    # A tool call that takes longer is abandoned, and the model told so.
    tool_timeout=10,
    # Each attempt at a request that takes longer fails; most hosted
    # models honour it, the scripted models in these lessons do not.
    model_settings=ModelSettings(timeout=60),
)


@agent.instructions
def name_the_repository(ctx: RunContext[RepoDeps]) -> str:
    """Tell the model which repository this run reads."""
    # Called for every request of a run, with that run's deps.
    return f"The repository is {ctx.deps.root.resolve().name}."
