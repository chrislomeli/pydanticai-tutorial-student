from pydantic_ai import Agent, RunContext
from pydantic_ai.settings import ModelSettings

from pydantic_tutor.repo_tools import RepoDeps, list_modules, read_docstring

# No model here: each run says which model to use, so the same agent runs on
# a test model now and a real one later. No repository either: each run
# passes one, as `deps`.
agent = Agent(
    name="repo_assistant",
    deps_type=RepoDeps,
    instructions="You answer questions about a Python code repository. Answer briefly.",
    tools=[list_modules, read_docstring],
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
