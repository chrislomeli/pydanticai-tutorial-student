"""Say "Hello" to an agent whose answer must be an Invoice, then to one that may answer in words.

From the output-trap lesson (01-07). Kept working: a later lesson that
changes the extractor updates this script, or removes it and says why.
"""

from pathlib import Path

from pydantic_ai import Agent
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    RetryPromptPart,
    TextPart,
    ToolCallPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

from pydantic_tutor.invoices import Invoice, extractor

GREETING = "Hello! Send me an invoice and I will read it."
# INV-0042, as a model reading it answers: the output tool's arguments.
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


def reply(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """Stand in for a model: answer "Hello" in words; read anything else as INV-0042."""
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


model = FunctionModel(function=reply)

print('The extractor, told "Hello":')
try:
    extractor.run_sync(user_prompt="Hello", model=model)
except UnexpectedModelBehavior as error:
    print("  the run failed:", error)

# Words or an Invoice. mypy cannot read that from the list, so the type
# parameters say it: the deps type (none), then the output type.
desk = Agent[None, str | Invoice](
    name="invoice_desk",
    output_type=[str, Invoice],
    instructions="Greet people briefly. When you are given an invoice, extract it.",
)
# Relative to the project root, where you run this from.
document = Path("sample_docs/invoices/INV-0042.txt").read_text()
for label, prompt in [('told "Hello"', "Hello"), ("sent INV-0042", document)]:
    print(f"The desk, {label}:")
    answer = desk.run_sync(user_prompt=prompt, model=model).output
    if isinstance(answer, Invoice):
        print("  an Invoice:", answer.number, answer.total)
    else:
        print("  words:", answer)
