"""Run the price agent on a model that calls a tool, gets it wrong, then answers.

From the tools lesson (01-03).
"""

from pydantic_ai.messages import RetryPromptPart

from pydantic_tutor.given.tool_models import asks_with_a_wrong_name
from pydantic_tutor.prices import price_agent

result = price_agent.run_sync(
    user_prompt="How much is an apple?", model=asks_with_a_wrong_name()
)

for message in result.all_messages():
    print(message.kind, [part.part_kind for part in message.parts])
    for part in message.parts:
        if isinstance(part, RetryPromptPart):
            print("  told the model:", part.content)
print("output:", result.output)
print("requests:", result.usage.requests)
