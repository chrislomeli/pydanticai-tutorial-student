"""Give up on a tool that takes too long, with a tool timeout.

From the limits lesson (01-08).
"""

import asyncio

from pydantic_ai import Agent

from pydantic_tutor.given.limit_models import calls_once_then_answers


async def wait_for_delivery() -> str:
    """Wait for the supplier to confirm a delivery."""
    await asyncio.sleep(delay=2)  # the supplier is slow to answer
    return "Delivery confirmed."


# Half a second, so the timeout shows quickly. A real tool gets seconds,
# set from what it waits on.
slow = Agent(name="deliveries", tools=[wait_for_delivery], tool_timeout=0.5)
result = slow.run_sync(
    user_prompt="Is the delivery confirmed?", model=calls_once_then_answers()
)
print("output:", result.output)
