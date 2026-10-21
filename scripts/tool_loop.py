"""Run the agent on a model that calls a tool, gets it wrong, then answers.

From the function tools lesson (01-03). Kept working: a later lesson that
changes the agent updates this script, or removes it and says why.
"""

from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    RetryPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

from pydantic_tutor.agent import agent


def reply(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """Stand in for a model: ask with a wrong name, retry, then answer.

    Called once per request. It reads the latest request to see where the
    run is: the user's prompt, a retry, or the tool's result.
    """
    for part in messages[-1].parts:
        if isinstance(part, ToolReturnPart):
            # The tool has run: answer from what it returned.
            return ModelResponse(
                parts=[TextPart(content=f"From its docstring: {part.content}")]
            )
        if isinstance(part, RetryPromptPart):
            # The tool refused the name: ask again, with the right one.
            call = ToolCallPart(
                tool_name="read_docstring", args={"module": "tinyjobs.jobs"}
            )
            return ModelResponse(parts=[call])
    # The first request: ask for a tool, with a name that does not exist.
    call = ToolCallPart(tool_name="read_docstring", args={"module": "jobs"})
    return ModelResponse(parts=[call])


result = agent.run_sync(
    user_prompt="What does the jobs module do?", model=FunctionModel(function=reply)
)

for message in result.all_messages():
    print(message.kind, [part.part_kind for part in message.parts])
    for part in message.parts:
        if isinstance(part, RetryPromptPart):
            print("  told the model:", part.content)
print("output:", result.output)
print("requests:", result.usage.requests)
