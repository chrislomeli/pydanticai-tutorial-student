"""Stop a run whose model never stops calling a tool, with a request limit.

From the limits lesson (01-08).
"""

from pydantic_ai.exceptions import UsageLimitExceeded
from pydantic_ai.usage import UsageLimits

from pydantic_tutor.given.limit_models import keeps_asking_for_bread
from pydantic_tutor.stock import stock_agent

try:
    stock_agent.run_sync(
        user_prompt="How much bread is there?",
        model=keeps_asking_for_bread(),
        # This job needs two or three requests; five is room to spare.
        usage_limits=UsageLimits(request_limit=5),
    )
except UsageLimitExceeded as error:
    print("stopped:", error)
