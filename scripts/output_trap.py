"""Say "Hello" to an agent whose answer must be an Invoice, then to one that may answer in words.

From the output-trap lesson (01-07).
"""

from pydantic_ai.exceptions import UnexpectedModelBehavior

from pydantic_tutor.given.greeting_model import greets_on_hello
from pydantic_tutor.invoices import extractor

print('The extractor, told "Hello":')
try:
    extractor.run_sync(user_prompt="Hello", model=greets_on_hello())
except UnexpectedModelBehavior as error:
    print("  the run failed:", error)
