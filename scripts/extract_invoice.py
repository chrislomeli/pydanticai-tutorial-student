"""Extract an invoice from a text document, on a model that misreads it once.

From the typed-answer lesson (01-06).
"""

from pydantic_tutor.given.invoice_model import DOCUMENT, misreads_the_total
from pydantic_tutor.invoices import extractor

result = extractor.run_sync(user_prompt=DOCUMENT, model=misreads_the_total())
invoice = result.output  # an Invoice, not text
print("number:", invoice.number)
print("issued:", invoice.issued)
print("total:", invoice.total, invoice.currency)
print("requests:", result.usage.requests)
