"""A stand-in model for the output-trap lesson (01-07).

Given code: you run it; you don't need to read it.
"""

from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    RetryPromptPart,
    TextPart,
    ToolCallPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

GREETING = "Hello! Send me an invoice and I will read it."
# INV-0042, as a model reading it sends back: a value for each field.
INV_0042 = {
    "number": "INV-0042",
    "issued": "2026-09-30",
    "supplier": "Acme Paper Co.",
    "currency": "GBP",
    "lines": [
        {"description": "A4 paper, box of 5", "quantity": 3, "unit_price": "24.50"},
        {"description": "Stapler", "quantity": 1, "unit_price": "12.00"},
    ],
    "total": "85.50",
}


def _greet_on_hello(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    print("  a text answer offered:", info.allow_text_output)
    for part in messages[-1].parts:
        if isinstance(part, RetryPromptPart):
            print("  sent back to the model:")
            print(part.model_response())
    prompt = messages[0].parts[0]  # the run's first request starts with the prompt
    if isinstance(prompt, UserPromptPart) and prompt.content == "Hello":
        return ModelResponse(parts=[TextPart(content=GREETING)])
    return ModelResponse(
        parts=[ToolCallPart(tool_name=info.output_tools[0].name, args=INV_0042)]
    )


def greets_on_hello() -> FunctionModel:
    """A model that answers "Hello" in words, and reads anything else as INV-0042.

    It prints whether it was offered a text answer, and anything it is sent back.
    """
    return FunctionModel(function=_greet_on_hello)
