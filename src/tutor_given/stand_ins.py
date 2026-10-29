"""Stand-in models for the lessons' keyless scripts.

Given, from the workspace lesson (03-03). Each is a function with a
readable name that returns a model; the lesson says what it does.
"""

from collections.abc import Mapping

from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    RetryPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel


def calls_tool_with_the_prompt(
    *,
    tool_name: str,
    argument: str,
    other_arguments: Mapping[str, str | int] | None = None,
) -> FunctionModel:
    """A model that calls `tool_name` with the user's prompt as `argument`.

    `other_arguments`, if given, are passed too (from the file tools lesson,
    03-04). The model then answers with the text the tool returned or the
    reason it was refused, and nothing else.
    """

    def reply(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        for part in messages[-1].parts:
            # A refusal comes back as a RetryPromptPart, asking the model to
            # try something else; this model passes the reason on.
            if isinstance(part, ToolReturnPart | RetryPromptPart):
                return ModelResponse(parts=[TextPart(content=str(part.content))])
            if isinstance(part, UserPromptPart):
                args = {**(other_arguments or {}), argument: part.content}
                call = ToolCallPart(tool_name=tool_name, args=args)
                return ModelResponse(parts=[call])
        raise ValueError("the request has neither a prompt nor a tool result")

    return FunctionModel(function=reply)
