"""Run the price agent from the tools lesson on a real model, unchanged.

From the real-model lesson (02-01). Run it with `OPENAI_API_KEY` set:
`uv run scripts/first_real_run.py`.
"""

from pydantic_tutor.model import CHAT_MODEL
from pydantic_tutor.prices import price_agent

result = price_agent.run_sync(user_prompt="How much is a coffee?", model=CHAT_MODEL)

for message in result.all_messages():
    print(message.kind, [part.part_kind for part in message.parts])
print("output:", result.output)
print("requests:", result.usage.requests)
print("cost in dollars:", result.usage.cost)
