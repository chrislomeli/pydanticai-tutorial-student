"""Extract an invoice from a scanned image, on a scripted model.

From the multimodal lesson (02-05). Reuses the extractor from the
typed-answer lesson (01-06) unchanged: nothing about it assumed its
document would arrive as text.
"""

from pathlib import Path

from pydantic_ai.messages import (
    BinaryContent,
    ModelMessage,
    ModelResponse,
    ToolCallPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

from pydantic_tutor.invoices import extractor

# What a model reading the scan answers: the arguments of its call to the
# output tool, as JSON. This scan reads cleanly the first time, unlike
# 01-06's text invoice -- that lesson's retry path is not this one's point.
READING = {
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


def reply(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """Stand in for a model: answer the scan correctly in one call."""
    output_tool = info.output_tools[0]
    return ModelResponse(parts=[ToolCallPart(tool_name=output_tool.name, args=READING)])


# Relative to the project root, where you run this from.
scan = BinaryContent.from_path(Path("sample_docs/invoices/INV-0042-scan.png"))
result = extractor.run_sync(
    user_prompt=["Pull the total out of this scan.", scan],
    model=FunctionModel(function=reply),
)
invoice = result.output  # an Invoice, not text
print("number:", invoice.number)
print("total:", invoice.total, invoice.currency)
print("requests:", result.usage.requests)
