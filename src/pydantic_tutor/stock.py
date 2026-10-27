"""An agent with one tool that can fail in two ways.

From the limits lesson (01-08).
"""

from pydantic_ai import Agent, ModelRetry, ToolFailed

IN_STOCK = {"apple": 12, "bread": 8, "coffee": 40}
# Products whose stock feed is down, so their stock can't be read.
FEED_DOWN = frozenset({"bread"})


def stock_of(*, product: str) -> int:
    """Look up how many of one product are in stock.

    Args:
        product: The product's name: apple, bread or coffee.
    """
    if product not in IN_STOCK:
        # The model wrote `product`: tell it what to fix.
        raise ModelRetry(message=f"There is no product {product!r}.")
    if product in FEED_DOWN:
        # Asking again won't help: the feed is down. Say so, and the model
        # can tell the user instead of retrying.
        raise ToolFailed(message=f"The stock feed for {product} is down.")
    return IN_STOCK[product]


# No model here: each run says which model to use.
stock_agent = Agent(
    name="stock",
    instructions="You answer questions about stock. Answer briefly.",
    tools=[stock_of],
)
