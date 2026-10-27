"""Stand-in models for the limits lesson (01-08).

Given code: you run it; you don't need to read it.
"""

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

# The products the first model asks about, in order: a misspelling, then a
# product whose stock can't be checked.
_ASKS = ("bred", "bread")


def _print_what_came_back(messages: list[ModelMessage]) -> None:
    for part in messages[-1].parts:
        if isinstance(part, RetryPromptPart):
            print("asked to retry:", part.model_response())
        if isinstance(part, ToolReturnPart):
            print(f"tool return ({part.outcome}):", part.model_response_str())


def _misspell_then_ask_for_bread(
    messages: list[ModelMessage], info: AgentInfo
) -> ModelResponse:
    _print_what_came_back(messages)
    asked = sum(isinstance(message, ModelResponse) for message in messages)
    if asked < len(_ASKS):
        call = ToolCallPart(tool_name="stock_of", args={"product": _ASKS[asked]})
        return ModelResponse(parts=[call])
    return ModelResponse(parts=[TextPart(content="I can't check bread right now.")])


def misspells_then_asks_for_bread() -> FunctionModel:
    """A model that asks stock_of for "bred", then "bread", then answers.

    It prints what each call gets back.
    """
    return FunctionModel(function=_misspell_then_ask_for_bread)


def _keep_asking_for_bread(
    messages: list[ModelMessage], info: AgentInfo
) -> ModelResponse:
    request = sum(isinstance(message, ModelRequest) for message in messages)
    print(f"request {request}: the model asks for bread")
    call = ToolCallPart(tool_name="stock_of", args={"product": "bread"})
    return ModelResponse(parts=[call])


def keeps_asking_for_bread() -> FunctionModel:
    """A model stuck in a loop: it asks stock_of for bread on every request."""
    return FunctionModel(function=_keep_asking_for_bread)


def _call_once_then_answer(
    messages: list[ModelMessage], info: AgentInfo
) -> ModelResponse:
    for part in messages[-1].parts:
        if isinstance(part, RetryPromptPart):
            print("asked to retry:", part.model_response())
            return ModelResponse(
                parts=[TextPart(content="The supplier didn't answer.")]
            )
        if isinstance(part, ToolReturnPart):
            return ModelResponse(parts=[TextPart(content=str(part.content))])
    call = ToolCallPart(tool_name=info.function_tools[0].name, args={})
    return ModelResponse(parts=[call])


def calls_once_then_answers() -> FunctionModel:
    """A model that calls the agent's one tool, then answers with what came back.

    It prints anything it is asked to retry.
    """
    return FunctionModel(function=_call_once_then_answer)
