from pydantic_ai import Agent, RunContext
from pydantic_ai.settings import ModelSettings

from pydantic_tutor.codebase_tools import CodebaseDeps, list_modules, read_docstring

# No model here: each run says which model to use, so the same agent runs on
# a test model now and a real one later. No codebase either: each run
# passes one, as `deps`.
agent = Agent(
    name="assistant",
    deps_type=CodebaseDeps,
    instructions="You answer questions about a Python codebase. Answer briefly.",
    tools=[list_modules, read_docstring],
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
