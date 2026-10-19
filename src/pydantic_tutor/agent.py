from pydantic_ai import Agent

from pydantic_tutor.repo_tools import list_modules, read_docstring

# No model here: each run says which model to use, so the same agent runs on
# a test model now and a real one later.
agent = Agent(
    name="repo_assistant",
    instructions="You answer questions about a Python code repository. Answer briefly.",
    tools=[list_modules, read_docstring],
)
