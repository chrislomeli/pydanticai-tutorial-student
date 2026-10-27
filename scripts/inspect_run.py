"""Run the agent once on TestModel and print every message it sent and received.

From the agent lesson (01-02).
"""

from pydantic_ai.messages import ModelRequest
from pydantic_ai.models.test import TestModel

from pydantic_tutor.agent import agent

# TestModel answers at once, inside this program: no key, no cost.
result = agent.run_sync(user_prompt="What does this codebase do?", model=TestModel())

# Every message of the run, in order: requests to the model, responses back.
for message in result.all_messages():
    # The message's kind, and the kind of each of its parts.
    print(message.kind, [part.part_kind for part in message.parts])
    if isinstance(message, ModelRequest):
        # The instructions travel on the request itself, not as a part.
        print("  instructions:", message.instructions)

print("output:", result.output)
