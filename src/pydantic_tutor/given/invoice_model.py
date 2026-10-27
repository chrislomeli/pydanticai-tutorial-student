"""A stand-in model, and the document it reads, for the typed-answer lesson (01-06).

Given code: you run it; you don't need to read it.
"""

from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    RetryPromptPart,
    ToolCallPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

# The document the agent reads: an invoice, as plain text.
DOCUMENT = """\
ACME PAPER CO.
12 Mill Lane, Leeds LS1 4AB

INVOICE                                  No. INV-0042
Date: 30 September 2026
Bill to: Harbour Books Ltd, 4 Harbour Street, Hull HU1 2AA

Description                     Qty    Unit price      Amount
A4 paper, box of 5                3         24.50       73.50
Stapler                           1         12.00       12.00

Total due (GBP)                                         85.50

Payment within 30 days. Thank you for your business.
"""

# What a model reading INV-0042 sends back: a value for each field of
# Invoice. The first time, it takes the first line's amount for the total.
FIRST_READING = {
    "number": "INV-0042",
    "issued": "2026-09-30",
    "supplier": "Acme Paper Co.",
    "currency": "GBP",
    "lines": [
        {"description": "A4 paper, box of 5", "quantity": 3, "unit_price": "24.50"},
        {"description": "Stapler", "quantity": 1, "unit_price": "12.00"},
    ],
    "total": "73.50",
}
SECOND_READING = {**FIRST_READING, "total": "85.50"}


def _misread_the_total(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    # A model sends structured values as a call to a tool the agent offers
    # for the purpose.
    answer_with = info.output_tools[0].name
    for part in messages[-1].parts:
        if isinstance(part, RetryPromptPart):
            print("sent back to the model:")
            print(part.model_response())
            return ModelResponse(
                parts=[ToolCallPart(tool_name=answer_with, args=SECOND_READING)]
            )
    return ModelResponse(
        parts=[ToolCallPart(tool_name=answer_with, args=FIRST_READING)]
    )


def misreads_the_total() -> FunctionModel:
    """A model that reads the invoice's total wrong once, then right when sent back.

    It prints what it is sent back.
    """
    return FunctionModel(function=_misread_the_total)
