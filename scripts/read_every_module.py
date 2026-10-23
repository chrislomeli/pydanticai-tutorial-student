"""Ask the agent to read every module, on a model that asks for them all at once.

From the capability lesson (04-03). The stand-in model lists the modules,
then asks for the docstring of every module it has not read, in one
response. Kept working: a later lesson that changes the agent updates
this script, or removes it and says why.
"""

from pathlib import Path

from pydantic import TypeAdapter
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

# What list_modules returns, checked as the stand-in model reads it.
MODULE_NAMES = TypeAdapter(list[str])


def read_all_at_once(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """Stand in for a model: list the modules, then ask to read every unread one at once."""
    listed: list[str] = []
    read = 0
    for message in messages:
        for part in message.parts:
            if isinstance(part, ToolReturnPart) and part.tool_name == "list_modules":
                listed = MODULE_NAMES.validate_python(part.content)
            elif (
                isinstance(part, ToolReturnPart) and part.tool_name == "read_docstring"
            ):
                read += 1
    if not listed:
        print("the model asks for the module list")
        return ModelResponse(parts=[ToolCallPart(tool_name="list_modules", args={})])
    # Each docstring returned is one module read, in the order asked for.
    unread = listed[read:]
    if not unread:
        print("the model answers")
        return ModelResponse(parts=[TextPart(content=f"I read {read} modules.")])
    print(f"the model asks to read {len(unread)} modules: {', '.join(unread)}")
    calls = [
        ToolCallPart(tool_name="read_docstring", args={"module": module})
        for module in unread
    ]
    return ModelResponse(parts=calls)


result = agent.run_sync(
    user_prompt="Read every module and summarise each.",
    model=FunctionModel(function=read_all_at_once),
    deps=CodebaseDeps(root=Path("sample_code/tinyjobs")),
)
print("the run's history, response by response:")
for message in result.all_messages():
    if isinstance(message, ModelResponse):
        names = [call.tool_name for call in message.tool_calls]
        print("  tool calls:", ", ".join(names) or "none, the answer")
print("output:", result.output)
print("usage:", result.usage)
