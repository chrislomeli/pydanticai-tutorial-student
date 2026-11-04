"""Say "Hello" to an agent whose answer must be an Invoice, then to one that may answer in words."""

from pydantic_ai import Agent
from pydantic_ai.exceptions import UnexpectedModelBehavior

from pydantic_tutor.given.greeting_model import greets_on_hello
from pydantic_tutor.given.invoice_model import DOCUMENT
from pydantic_tutor.invoices import Invoice, extractor

print('The extractor, told "Hello":')
try:
    extractor.run_sync(user_prompt="Hello", model=greets_on_hello())
except UnexpectedModelBehavior as error:
    print("  the run failed:", error)

# Words or an Invoice: either ends a run. With a list, the type parameters
# say so to the type checker: the deps type (none), then the output type.
desk = Agent[None, str | Invoice](
    name="invoice_desk",
    output_type=[str, Invoice],
    instructions="Greet people briefly. When you are given an invoice, extract it.",
)
for prompt in ["Hello", DOCUMENT]:
    answer = desk.run_sync(user_prompt=prompt, model=greets_on_hello()).output
    if isinstance(answer, Invoice):
        print("an Invoice:", answer.number, answer.total)
    else:
        print("words:", answer)
