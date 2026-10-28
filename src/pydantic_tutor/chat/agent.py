"""The agent the chat talks to.

From the console-chat lesson (02-02); web search from the web-search
lesson (02-05); built from a setting in the settings lesson (03-01). No
model here, as for every agent in this project: the chat passes one on
each run.
"""

from pydantic_ai import Agent
from pydantic_ai.capabilities import AbstractCapability, WebSearch


def build_chat_agent(*, web_search: bool) -> Agent[None, str]:
    """Build the chat's agent; `web_search` says whether it may search the web."""
    capabilities: list[AbstractCapability[None]] = []
    if web_search:
        # The model may search when it needs to. Each search is billed by
        # the provider, on top of the tokens.
        capabilities.append(WebSearch())
    return Agent(
        name="chat",
        instructions="You are a helpful assistant. Answer briefly.",
        capabilities=capabilities,
    )


# The agent with web search on, as before: earlier scripts and the tests
# import it by name. The chat itself gets its agent from `build_app`.
chat_agent = build_chat_agent(web_search=True)
