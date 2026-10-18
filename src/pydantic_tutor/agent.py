from pydantic_ai import Agent

# No model here: each run says which model to use, so the same agent runs on
# a test model now and a real one later.
agent = Agent(
    name="repo_assistant",
    instructions="You answer questions about a Python code repository. Answer briefly.",
)
