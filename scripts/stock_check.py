"""Ask about two products the tool can't answer for: one misspelt, one whose feed is down."""

from pydantic_tutor.given.limit_models import misspells_then_asks_for_bread
from pydantic_tutor.stock import stock_agent

result = stock_agent.run_sync(
    user_prompt="How much bread is there?", model=misspells_then_asks_for_bread()
)
print("output:", result.output)
