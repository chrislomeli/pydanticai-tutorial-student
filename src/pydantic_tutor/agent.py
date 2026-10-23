from pydantic_ai import Agent

from pydantic_tutor.codebase_tools import CodebaseDeps, list_modules, read_docstring

# No model here: each run says which model to use, so the same agent runs on
# a test model now and a real one later. No codebase either: each run
# passes one, as `deps`.
agent = Agent(
    name="assistant",
    deps_type=CodebaseDeps,
    instructions="You answer questions about a Python codebase. Answer briefly.",
    tools=[list_modules, read_docstring],
)
