"""An agent that answers for whichever shop a run is about.

From the dependencies lesson (01-04). The shop is the run's dependency:
each run passes one, and the tool and the instructions read it.
"""

from collections.abc import Mapping
from dataclasses import dataclass

from pydantic_ai import Agent, ModelRetry, RunContext


@dataclass(frozen=True)
class Shop:
    """What one run needs: the shop it answers for."""

    name: str
    prices_in_cents: Mapping[str, int]


def price_of(ctx: RunContext[Shop], *, product: str) -> int:
    """Look up the price of one product, in cents.

    Args:
        product: The product's name.
    """
    # ctx.deps is the Shop this run was given; the model never sees it.
    prices = ctx.deps.prices_in_cents
    if product not in prices:
        raise ModelRetry(message=f"{ctx.deps.name} doesn't sell {product!r}.")
    return prices[product]


# No model and no shop here: each run says which model to use, and which
# shop it's about.
shop_agent = Agent(
    name="shop",
    deps_type=Shop,
    instructions="You answer questions about prices. Answer briefly.",
    tools=[price_of],
)


@shop_agent.instructions
def name_the_shop(ctx: RunContext[Shop]) -> str:
    """Tell the model which shop this run is for."""
    return f"You answer for {ctx.deps.name}."
