"""Stop a run whose model never stops calling a tool, with a request limit.

From the limits lesson (01-08). Kept working: a later lesson that changes
the agent updates this script, or removes it and says why.
"""

from pathlib import Path

from pydantic_ai.exceptions import UsageLimitExceeded
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ToolCallPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.usage import UsageLimits

from pydantic_tutor.agent import agent
from pydantic_tutor.codebase_tools import CodebaseDeps


def reply(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """Stand in for a model stuck in a loop: ask for the module list, every time."""
    request = sum(isinstance(message, ModelRequest) for message in messages)
    print(f"request {request}: the model calls list_modules")
    return ModelResponse(parts=[ToolCallPart(tool_name="list_modules", args={})])


try:
    agent.run_sync(
        user_prompt="What does this codebase do?",
        model=FunctionModel(function=reply),
        deps=CodebaseDeps(root=Path("sample_code/tinyjobs")),
        # This job needs two or three requests; five is room to spare.
        usage_limits=UsageLimits(request_limit=5),
    )
except UsageLimitExceeded as error:
    print("stopped:", error)
