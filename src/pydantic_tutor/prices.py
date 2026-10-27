"""A small agent with two tools: a price list it can read for the model.

From the tools lesson (01-03). The prices live in this file because the
lesson is about the tools, not about where data comes from.
"""

from pydantic_ai import Agent, ModelRetry

# Whole cents, so no price is ever rounded.
PRICES_IN_CENTS = {"apple": 50, "bread": 225, "coffee": 700}


def list_products() -> list[str]:
    """List the products that have a price."""
    return sorted(PRICES_IN_CENTS)


def price_of(*, product: str) -> int:
    """Look up the price of one product, in cents.

    Args:
        product: The product's name, as `list_products` gives it.
    """
    # The model wrote `product`: check it before using it.
    if product not in PRICES_IN_CENTS:
        raise ModelRetry(
            message=f"There is no product {product!r}. Call list_products to see them."
        )
    return PRICES_IN_CENTS[product]


# No model here: each run says which model to use.
price_agent = Agent(
    name="prices",
    instructions="You answer questions about prices. Answer briefly.",
    tools=[list_products, price_of],
)
