"""Stand-in models for the tools lesson (01-03).

Given code: you run it; you don't need to read it.
"""

import json

from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    RetryPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel


def _print_tool_definitions(
    messages: list[ModelMessage], info: AgentInfo
) -> ModelResponse:
    for tool in info.function_tools:
        print(tool.name)
        print("  description:", tool.description)
        print("  parameters:", json.dumps(obj=tool.parameters_json_schema, indent=2))
    return ModelResponse(parts=[TextPart(content="Done.")])


def prints_tool_definitions() -> FunctionModel:
    """A model that prints the tool definitions it is sent, then answers "Done."."""
    return FunctionModel(function=_print_tool_definitions)


def _ask_with_a_wrong_name(
    messages: list[ModelMessage], info: AgentInfo
) -> ModelResponse:
    for part in messages[-1].parts:
        if isinstance(part, ToolReturnPart):
            return ModelResponse(
                parts=[TextPart(content=f"An apple costs {part.content} cents.")]
            )
        if isinstance(part, RetryPromptPart):
            call = ToolCallPart(tool_name="price_of", args={"product": "apple"})
            return ModelResponse(parts=[call])
    call = ToolCallPart(tool_name="price_of", args={"product": "apples"})
    return ModelResponse(parts=[call])


def asks_with_a_wrong_name() -> FunctionModel:
    """A model that asks for price_of("apples"), then "apple" when told, then answers."""
    return FunctionModel(function=_ask_with_a_wrong_name)
