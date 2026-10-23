"""Run one agent on two codebases: only the run's dependencies change.

From the dependencies lesson (01-04). Kept working: a later lesson that
changes the agent updates this script, or removes it and says why.
"""

from pathlib import Path

from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

from pydantic_tutor.agent import agent
from pydantic_tutor.codebase_tools import CodebaseDeps


def reply(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """Stand in for a model: ask for list_modules, then answer with its result."""
    for part in messages[-1].parts:
        if isinstance(part, ToolReturnPart):
            return ModelResponse(parts=[TextPart(content=f"Modules: {part.content}")])
    return ModelResponse(parts=[ToolCallPart(tool_name="list_modules", args={})])


model = FunctionModel(function=reply)
# Two codebases, relative to the project root, where you run this from:
# the sample codebase, and this project itself.
for root in [Path("sample_code/tinyjobs"), Path(".")]:
    result = agent.run_sync(
        user_prompt="Which modules does this codebase have?",
        model=model,
        deps=CodebaseDeps(root=root),
    )
    print(f"{root}: {result.output}")
