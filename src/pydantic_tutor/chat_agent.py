"""The agent the chat talks to.

From the console-chat lesson (02-02). No model here, as for every agent in
this project: the chat passes `CHAT_MODEL` on each run.
"""

from pydantic_ai import Agent

chat_agent = Agent(
    name="chat",
    instructions="You are a helpful assistant. Answer briefly.",
)
