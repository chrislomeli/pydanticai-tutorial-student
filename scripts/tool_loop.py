"""Run the agent on a model that calls a tool, then answers from its result.

From the function tools lesson (01-03). Kept working: a later lesson that
changes the agent updates this script, or removes it and says why.
"""

from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

from pydantic_tutor.agent import agent


def reply(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """Stand in for a model that uses a tool: ask for one, then answer from it.

    Called once per request. The first carries the user's prompt; the
    second carries the tool's result.
    """
    request = messages[-1]
    for part in request.parts:
        if isinstance(part, ToolReturnPart):
            # The tool has run: answer from what it returned.
            return ModelResponse(
                parts=[TextPart(content=f"From its docstring: {part.content}")]
            )
    # No result yet: ask the agent to run a tool. The model never runs it.
    call = ToolCallPart(tool_name="read_docstring", args={"module": "tinyjobs.jobs"})
    return ModelResponse(parts=[call])


result = agent.run_sync(
    user_prompt="What does the jobs module do?", model=FunctionModel(function=reply)
)

for message in result.all_messages():
    print(message.kind, [part.part_kind for part in message.parts])
    for part in message.parts:
        if isinstance(part, ToolCallPart | ToolReturnPart):
            # A tool's return carries the id of the call it answers.
            print("  tool_call_id:", part.tool_call_id)
print("output:", result.output)
print("requests:", result.usage.requests)
