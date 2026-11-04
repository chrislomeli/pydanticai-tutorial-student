"""Show a second agent two views of the workspace toolset.

Needs no API key; run `demos/fresh_copy.py`
first.
"""

from pydantic_ai import Agent, RunContext
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.toolsets import AbstractToolset

from pydantic_tutor.settings import Settings
from pydantic_tutor.tools.workspace_toolset import build_workspace_toolset
from tutor_given.stand_ins import reports_what_it_is_sent

toolset = build_workspace_toolset(root=Settings().workspace_root)


def only_listing(ctx: RunContext[None], tool: ToolDefinition) -> bool:
    """Keep `list_modules`; hide every other tool."""
    return tool.name == "list_modules"


views: dict[str, AbstractToolset[None]] = {
    "prefixed": toolset.prefixed(prefix="workspace"),
    "filtered": toolset.filtered(filter_func=only_listing),
}
for title, view in views.items():
    print(title)
    reader = Agent[None, str](name="reader", toolsets=[view])
    reader.run_sync(user_prompt="hi", model=reports_what_it_is_sent())
