"""A hook that logs each request the agent is about to send to a model.

One line per request: the run's step, and how many messages and tools the
request carries. An agent takes it as one of its capabilities,
`request_log`. It logs with Python's `logging`, at INFO, to the logger
`pydantic_tutor.request_log`; nothing is shown until the program that runs
the agent turns that logger on.
"""

import logging

from pydantic_ai import ModelRequestContext, RunContext
from pydantic_ai.capabilities import Hooks

from pydantic_tutor.codebase_tools import CodebaseDeps

logger = logging.getLogger(name=__name__)


async def log_request(
    ctx: RunContext[CodebaseDeps], request_context: ModelRequestContext
) -> ModelRequestContext:
    """Log the request about to be sent, and pass it on unchanged."""
    # Called before each request of a run. The request is as the
    # capabilities before this one in the agent's list have left it.
    logger.info(
        "request step=%d messages=%d tools=%d",
        ctx.run_step,
        len(request_context.messages),
        len(request_context.model_request_parameters.function_tools),
    )
    # The hook returns the request to send. Return it as it came: this
    # hook only watches.
    return request_context


# The hook as a capability: an agent takes it in `capabilities=[...]`.
request_log: Hooks[CodebaseDeps] = Hooks(before_model_request=log_request)
