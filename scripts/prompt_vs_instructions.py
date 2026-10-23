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
    system_prompts = [
        part.content
        for message in messages
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, SystemPromptPart)
    ]
    print("  system prompt parts:", system_prompts)
    print("  instructions:", info.instructions)
    return ModelResponse(parts=[TextPart(content="ok")])


model = FunctionModel(function=reply)
# The first agent has both: a system prompt, and instructions.
first = Agent(system_prompt="You review Python code.", instructions="Answer briefly.")
# The second has instructions only.
second = Agent(instructions="Answer in detail.")

print("The first agent, a new run:")
first_run = first.run_sync(user_prompt="What does this codebase do?", model=model)

print("The second agent, continuing the first run's messages:")
second.run_sync(
    user_prompt="And what do its tests cover?",
    model=model,
    message_history=first_run.all_messages(),
)
