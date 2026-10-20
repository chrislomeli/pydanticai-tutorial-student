"""Give up on a tool that takes too long, with a tool timeout.

From the limits lesson (01-08). It builds a small agent of its own, with a
tool that sleeps, so the timeout shows in a second, not ten. Kept as it
is: a later lesson that changes it says why.
"""

import asyncio

from pydantic_ai import Agent
from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    RetryPromptPart,
    TextPart,
    ToolCallPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel


async def wait_for_runner() -> str:
    """Wait for the test runner to report."""
    await asyncio.sleep(delay=2)  # the runner is stuck
    return "all tests passed"


def reply(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """Stand in for a model: call the tool once, then answer with what came back."""
    for part in messages[-1].parts:
        if isinstance(part, RetryPromptPart):
            print("asked to retry:", part.model_response())
            return ModelResponse(
                parts=[TextPart(content="The test runner did not answer.")]
            )
    return ModelResponse(parts=[ToolCallPart(tool_name="wait_for_runner", args={})])


slow = Agent(name="slow", tools=[wait_for_runner], tool_timeout=0.5)
result = slow.run_sync(
    user_prompt="Did the tests pass?", model=FunctionModel(function=reply)
)
print("output:", result.output)
