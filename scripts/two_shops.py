"""Run one agent for two shops: only the run's dependencies change.

From the dependencies lesson (01-04).
"""

from pydantic_tutor.given.shop_model import asks_for_coffee
from pydantic_tutor.shop import Shop, shop_agent

shops = [
    Shop(name="Corner Cafe", prices_in_cents={"coffee": 350, "tea": 300}),
    Shop(name="Station Kiosk", prices_in_cents={"coffee": 500}),
]
for shop in shops:
    print(shop.name)
    result = shop_agent.run_sync(
        user_prompt="How much is a coffee?", model=asks_for_coffee(), deps=shop
    )
    print("  output:", result.output)
