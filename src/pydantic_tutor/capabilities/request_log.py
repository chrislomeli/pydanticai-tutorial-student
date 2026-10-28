"""Log each request the agent is about to send to a model.

From the hooks lesson (04-02). One line per request, to the logger
`pydantic_tutor.capabilities.request_log` at INFO. Nothing is shown until the program
that runs the agent turns that logger on.
"""

import logging

from pydantic_ai import ModelRequestContext, RunContext
from pydantic_ai.capabilities import Hooks

logger = logging.getLogger(name=__name__)


async def log_request(
    ctx: RunContext[None], request_context: ModelRequestContext
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
request_log: Hooks[None] = Hooks(before_model_request=log_request)
