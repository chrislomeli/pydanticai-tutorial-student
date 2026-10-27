"""A stand-in model for the dependencies lesson (01-04).

Given code: you run it; you don't need to read it.
"""

from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel


def _ask_for_coffee(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    for part in messages[-1].parts:
        if isinstance(part, ToolReturnPart):
            return ModelResponse(
                parts=[TextPart(content=f"Coffee is {part.content} cents.")]
            )
    # The first request: print everything the model is told, then call the tool.
    print("  instructions:", repr(info.instructions))
    for tool in info.function_tools:
        print(
            f"  {tool.name}'s arguments:",
            list(tool.parameters_json_schema["properties"]),
        )
    return ModelResponse(
        parts=[ToolCallPart(tool_name="price_of", args={"product": "coffee"})]
    )


def asks_for_coffee() -> FunctionModel:
    """A model that prints what it is sent, asks price_of for coffee, then answers."""
    return FunctionModel(function=_ask_for_coffee)
