from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import LocalWorkspace
from pydantic_ai.settings import ModelSettings
from pydantic_ai_harness import FileSystem

from pydantic_tutor.codebase_tools import CodebaseDeps, list_modules, read_docstring

# Files the file tools may neither read nor write, at any depth: the
# library's own secret patterns, which by default it only keeps from being
# written. A leading `**/` means "in any folder, the top one included".
SECRET_PATTERNS = ("**/.env", "**/.env.*", "*.pem", "*.key", "**/secrets*")


def codebase_workspace(ctx: RunContext[CodebaseDeps]) -> LocalWorkspace[CodebaseDeps]:
    """Give each run a workspace on this machine, in the codebase it reads."""
    # Called once per run, with that run's deps: the workspace follows the
    # codebase the run is given. Its folder is where relative paths
    # start, not a boundary: `..` and absolute paths reach past it.
    # Keep it to building the capability from deps: no I/O, no state.
    return LocalWorkspace(working_dir=ctx.deps.root)


# No model here: each run says which model to use, so the same agent runs on
# a test model now and a real one later. No codebase either: each run
# passes one, as `deps`, and gets a workspace there.
agent = Agent(
    name="assistant",
    deps_type=CodebaseDeps,
    instructions="You answer questions about a Python codebase. Answer briefly.",
    tools=[list_modules, read_docstring],
    capabilities=[
        codebase_workspace,
        # File tools in the run's workspace. They keep to its folder, and
        # refuse the secret files; the default `read_only_patterns` still
        # keeps `.git/` and the rest from being written.
        FileSystem(denied_patterns=SECRET_PATTERNS),
    ],
    # A tool call that takes longer is abandoned, and the model told so.
    tool_timeout=10,
    # Each attempt at a request that takes longer fails; most hosted
    # models honour it, the scripted models in these lessons do not.
    model_settings=ModelSettings(timeout=60),
)


@agent.instructions
def name_the_codebase(ctx: RunContext[CodebaseDeps]) -> str:
    """Tell the model which codebase this run reads."""
    # Called for every request of a run, with that run's deps.
    return f"The codebase is {ctx.deps.root.resolve().name}."
