"""Compare a system prompt with instructions: which one a later run replays.

From the instructions lesson (01-05). It builds two small agents of its
own, not the project's, so a change to the project's agent leaves it
working. Kept as it is: a later lesson that changes it says why.
"""

from pydantic_ai import Agent
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel


def reply(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """Stand in for a model: print what this request carries, then answer."""
    # The system prompt is a part inside the messages.
    in_messages = [
        part.content
        for message in messages
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, SystemPromptPart)
    ]
    # The instructions are attached beside the messages.
    print("  in the messages:", in_messages)
    print("  beside them:    ", info.instructions)
    return ModelResponse(parts=[TextPart(content="ok")])


model = FunctionModel(function=reply)
# Each agent has both kinds of text, and each string names its owner.
agent_a = Agent(system_prompt="A's system prompt", instructions="A's instructions")
agent_b = Agent(system_prompt="B's system prompt", instructions="B's instructions")

print("A, a new run:")
a_run = agent_a.run_sync(user_prompt="What does this codebase do?", model=model)

print("B, continuing A's messages:")
agent_b.run_sync(
    user_prompt="And what do its tests cover?",
    model=model,
    message_history=a_run.all_messages(),
)
