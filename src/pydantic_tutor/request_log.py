"""A hook that logs each request the agent is about to send to a model.

One line per request: the run's step, how many messages the request
carries, and how many tools the model may call. An agent takes it as one
of its capabilities, `request_log`. It logs with Python's `logging`, at
INFO, to the logger `pydantic_tutor.request_log`; nothing is shown until
the program that runs the agent turns that logger on.
"""

import logging

from pydantic_ai import ModelRequestContext, RunContext
from pydantic_ai.capabilities import CapabilityOrdering, Hooks

from pydantic_tutor.codebase_tools import CodebaseDeps

logger = logging.getLogger(name=__name__)


async def log_request(
    ctx: RunContext[CodebaseDeps], request_context: ModelRequestContext
) -> ModelRequestContext:
    """Log the request about to be sent, and pass it on unchanged."""
    # Called before each request of a run, after every other capability's
    # `before_model_request`: the request is as they have left it.
    # The tools the model may call: the request's definitions also hold
    # the tools of a capability that loads on demand, hidden until it is
    # loaded.
    tools = request_context.model_request_parameters.function_tools
    logger.info(
        "request step=%d messages=%d tools=%d",
        ctx.run_step,
        len(request_context.messages),
        sum(ctx.is_tool_available(tool=tool) for tool in tools),
    )
    # The hook returns the request to send. Return it as it came: this
    # hook only watches.
    return request_context


# The hook as a capability: an agent takes it in `capabilities=[...]`.
# Innermost, wherever it is listed: last before the model, so it logs the
# request every other capability has finished changing.
request_log: Hooks[CodebaseDeps] = Hooks(
    before_model_request=log_request,
    ordering=CapabilityOrdering(position="innermost"),
)
