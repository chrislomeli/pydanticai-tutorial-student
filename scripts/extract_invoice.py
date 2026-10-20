"""Extract an invoice from a text document, on a model that misreads it once.

From the typed-answer lesson (01-06). Kept working: a later lesson that
changes the extractor updates this script, or removes it and says why.
"""

from pathlib import Path

from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    RetryPromptPart,
    ToolCallPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

from pydantic_tutor.invoices import extractor

# What a model reading INV-0042 answers: the arguments of its call to the
# output tool, as JSON. The first time, it takes the first line's amount for
# the total.
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


def reply(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """Stand in for a model: answer with the first reading, or the second if sent back."""
    output_tool = info.output_tools[0]  # the one tool the agent gave it to answer with
    for part in messages[-1].parts:
        if isinstance(part, RetryPromptPart):
            print("sent back to the model:")
            print(part.model_response())
            return ModelResponse(
                parts=[ToolCallPart(tool_name=output_tool.name, args=SECOND_READING)]
            )
    print("the model answers by calling:", output_tool.name)
    return ModelResponse(
        parts=[ToolCallPart(tool_name=output_tool.name, args=FIRST_READING)]
    )


# Relative to the project root, where you run this from.
document = Path("sample_docs/invoices/INV-0042.txt").read_text()
result = extractor.run_sync(user_prompt=document, model=FunctionModel(function=reply))
invoice = result.output  # an Invoice, not text
print("number:", invoice.number)
print("issued:", invoice.issued)
print("total:", invoice.total, invoice.currency)
print("requests:", result.usage.requests)
