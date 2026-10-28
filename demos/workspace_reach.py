"""Show what the chat's workspace is, and what it is not.

From the workspace lesson (03-03). Needs no API key. Run `demos/fresh_copy.py`
first. Kept working: a later lesson that changes how the chat's workspace
is built updates this script, or removes it and says why.
"""

from pydantic_ai import RunContext
from pydantic_ai.models.test import TestModel
from pydantic_ai.toolsets import FunctionToolset

from pydantic_tutor.chat.agent import build_chat_agent
from pydantic_tutor.settings import Settings
from tutor_given.stand_ins import calls_tool_with_the_prompt

root = Settings().workspace_root
agent = build_chat_agent(web_search=False, workspace_root=root)

# Offer the tools without calling them: TestModel calls every tool it is offered.
test_model = TestModel(call_tools=[])
agent.run_sync(user_prompt="hello", model=test_model)
parameters = test_model.last_model_request_parameters
assert parameters is not None  # the run made a request
print(f"tools the model is offered: {len(parameters.function_tools)}")


async def first_line(ctx: RunContext[None], *, path: str) -> str:
    """Read the first line of a file, by a path the model gives."""
    text = await ctx.workspace.read_text(path=path)
    return text.splitlines()[0]


# The folder is `work/tinyjobs`: `..` is `work/`, and `../..` is the project.
outside = "../../pyproject.toml"
paths = {
    "README.md": "README.md",
    outside: outside,
    "absolute path": str((root / outside).resolve()),
}
for label, path in paths.items():
    result = agent.run_sync(
        user_prompt=path,
        model=calls_tool_with_the_prompt(tool_name="first_line", argument="path"),
        # A tool for this run only, that reads through the workspace as the file
        # tools do.
        toolsets=[FunctionToolset(tools=[first_line])],
    )
    print(f"{label}: {result.output}")
