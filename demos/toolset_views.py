"""Show what the chat's toolset sends the model, then give it to a second agent.

From the toolsets lesson (04-01). Needs no API key; run `demos/fresh_copy.py`
first.
"""

from pydantic_ai import Agent, RunContext
from pydantic_ai.models import Model
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.toolsets import AbstractToolset

from pydantic_tutor.chat.agent import build_chat_agent
from pydantic_tutor.settings import Settings
from pydantic_tutor.tools.workspace_tools import build_workspace_toolset
from tutor_given.stand_ins import (
    calls_tool_with_the_prompt,
    reports_what_it_is_sent,
)

root = Settings().workspace_root
toolset = build_workspace_toolset(root=root)


def only_listing(ctx: RunContext[None], tool: ToolDefinition) -> bool:
    """Keep `list_modules`; hide every other tool, in every run."""
    return tool.name == "list_modules"


def show(*, title: str, toolset: AbstractToolset[None], model: Model) -> None:
    """Run a second agent that has only `toolset`, on `model`."""
    reader: Agent[None, str] = Agent(name="reader", toolsets=[toolset])
    print(title)
    result = reader.run_sync(user_prompt="jobs", model=model)
    if result.output != "Done.":
        print("  the model got back:", result.output)


print("the chat agent:")
chat = build_chat_agent(web_search=False, workspace_root=root)
chat.run_sync(user_prompt="hi", model=reports_what_it_is_sent())
result = chat.run_sync(
    user_prompt="tinyjobs.jobs",
    model=calls_tool_with_the_prompt(tool_name="read_docstring", argument="module"),
)
print("  read_docstring('tinyjobs.jobs') returned:", result.output.splitlines()[0])
show(
    title="a second agent, given the toolset:",
    toolset=toolset,
    model=reports_what_it_is_sent(),
)
show(
    title="prefixed with 'workspace':",
    toolset=toolset.prefixed(prefix="workspace"),
    model=reports_what_it_is_sent(),
)
show(
    title="prefixed, calling read_docstring with a wrong module:",
    toolset=toolset.prefixed(prefix="workspace"),
    model=calls_tool_with_the_prompt(
        tool_name="workspace_read_docstring", argument="module"
    ),
)
show(
    title="filtered to list_modules:",
    toolset=toolset.filtered(filter_func=only_listing),
    model=reports_what_it_is_sent(),
)
