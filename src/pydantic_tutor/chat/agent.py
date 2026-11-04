"""The agent the chat talks to.

No model here, as for every agent in this project: the chat passes
`CHAT_MODEL` on each run.
"""

from pydantic_ai import Agent
from pydantic_ai.capabilities import WebSearch

chat_agent = Agent(
    name="chat",
    instructions="You are a helpful assistant. Answer briefly.",
    # The model may search the web when it needs to. Each search is billed
    # by the provider, on top of the tokens.
    capabilities=[WebSearch()],
)
