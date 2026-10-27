"""Print what the model is told about the price agent's tools.

From the tools lesson (01-03).
"""

from pydantic_tutor.given.tool_models import prints_tool_definitions
from pydantic_tutor.prices import price_agent

price_agent.run_sync(
    user_prompt="How much is an apple?", model=prints_tool_definitions()
)
