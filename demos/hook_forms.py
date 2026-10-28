"""The request log as a class, and why its place in the list matters.

From the hooks lesson (04-02). Needs no API key; run `demos/fresh_copy.py`
first. Kept working: a later lesson that changes the request log updates
it, or removes it and says why.
"""

import logging
import sys

from pydantic_ai import Agent, ModelRequestContext, RunContext
from pydantic_ai.capabilities import AbstractCapability, Hooks

from pydantic_tutor.capabilities.request_log import log_request
from pydantic_tutor.settings import Settings
from pydantic_tutor.tools.workspace_tools import build_workspace_toolset
from tutor_given.stand_ins import (
    keeps_the_latest_exchange,
    lists_reads_then_answers,
)

# As in watch_requests.py: the request log on.
logging.basicConfig(stream=sys.stdout, format="%(name)s: %(message)s")
logging.getLogger(name="pydantic_tutor.capabilities.request_log").setLevel(
    level=logging.INFO
)


class RequestLog(AbstractCapability[None]):
    """The request log as a class: the hook is a method of the same name."""

    async def before_model_request(
        self, ctx: RunContext[None], request_context: ModelRequestContext
    ) -> ModelRequestContext:
        """Log the request about to be sent, as `request_log` does."""
        return await log_request(ctx=ctx, request_context=request_context)


def show(*, title: str, capabilities: list[AbstractCapability[None]]) -> None:
    """Run a second agent with these capabilities, and print what happens."""
    reader: Agent[None, str] = Agent(
        name="reader",
        toolsets=[build_workspace_toolset(root=Settings().workspace_root)],
        capabilities=capabilities,
    )
    print(title)
    reader.run_sync(
        user_prompt="What does this project do?", model=lists_reads_then_answers()
    )


show(
    title="the log, as a class, before a hook that trims the history:",
    capabilities=[
        RequestLog(),
        Hooks(before_model_request=keeps_the_latest_exchange),
    ],
)
show(
    title="the log, as a class, after it:",
    capabilities=[
        Hooks(before_model_request=keeps_the_latest_exchange),
        RequestLog(),
    ],
)
