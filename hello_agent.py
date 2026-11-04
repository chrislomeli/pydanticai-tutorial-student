from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

agent = Agent(model=TestModel())
result = agent.run_sync(user_prompt="Hello!")
print(result.output)
