"""An invoice, as a type, and an agent that reads one out of a document."""

from datetime import date
from decimal import Decimal
from typing import Self

from pydantic import BaseModel, Field, model_validator
from pydantic_ai import Agent


class LineItem(BaseModel):
    """One line of an invoice."""

    description: str
    quantity: int
    unit_price: Decimal = Field(description="The price of one, as printed.")


class Invoice(BaseModel):
    """An invoice, as printed on the document."""

    number: str = Field(description="The invoice number, as printed.")
    issued: date = Field(description="The date the invoice was issued.")
    supplier: str = Field(description="Who sent the invoice.")
    currency: str = Field(description="The three-letter currency code, such as GBP.")
    lines: list[LineItem]
    total: Decimal = Field(description="The total due, as printed.")

    @model_validator(mode="after")
    def check_total(self) -> Self:
        """Refuse a total that is not the sum of the lines."""
        expected = sum(
            (line.quantity * line.unit_price for line in self.lines),
            start=Decimal(value=0),
        )
        if self.total != expected:
            raise ValueError(
                f"The total {self.total} is not the sum of the lines, {expected}."
            )
        return self


# No model here, as for the repository agent: each run says which to use.
extractor = Agent(
    name="invoice_extractor",
    output_type=Invoice,
    instructions="Extract the invoice from the document you are given. Copy every value as printed.",
)
