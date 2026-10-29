"""The agent the chat talks to.

From the console-chat lesson (02-02); web search from the web-search
lesson (02-05); built from a setting in the settings lesson (03-01); a
workspace from the workspace lesson (03-03). No model here, as for every
agent in this project: the chat passes one on each run.
"""

from pathlib import Path

from pydantic_ai import Agent
from pydantic_ai.capabilities import AbstractCapability, LocalWorkspace, WebSearch


def build_chat_agent(*, web_search: bool, workspace_root: Path) -> Agent[None, str]:
    """Build the chat's agent, its runs working in the `workspace_root` folder."""
    capabilities: list[AbstractCapability[None]] = [
        # A place for tools to work in. It adds no tool: they use `ctx.workspace`.
        LocalWorkspace(working_dir=workspace_root)
    ]
    if web_search:
        # The model may search when it needs to. Each search is billed by
        # the provider, on top of the tokens.
        capabilities.append(WebSearch())
    return Agent(
        name="chat",
        instructions="You are a helpful assistant. Answer briefly.",
        capabilities=capabilities,
    )


# The agent with web search on and the default folder, as before: earlier
# scripts and the tests import it by name. The chat itself gets its agent
# from `build_app`.
chat_agent = build_chat_agent(web_search=True, workspace_root=Path("work/tinyjobs"))
