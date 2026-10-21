"""The request log written as a class, and why its place in the list matters.

From the hooks lesson (04-02). Each run is a second agent, given the
repository toolset, on a stand-in model that prints how many messages and
tools it receives. Kept working: a later lesson that changes the request
log updates this script, or removes it and says why.
"""

import logging
import sys
from dataclasses import replace
from pathlib import Path

from pydantic_ai import Agent, ModelRequestContext, RunContext
from pydantic_ai.capabilities import AbstractCapability, Hooks
from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

from pydantic_tutor.agent import repo_workspace
from pydantic_tutor.repo_tools import RepoDeps, repo_toolset
from pydantic_tutor.request_log import log_request

# As in watch_requests.py: the request log on, on stdout.
logging.basicConfig(stream=sys.stdout, format="%(name)s: %(message)s")
logging.getLogger(name="pydantic_tutor.request_log").setLevel(level=logging.INFO)


class RequestLog(AbstractCapability[RepoDeps]):
    """The request log as a class: the hook is a method of the same name."""

    async def before_model_request(
        self, ctx: RunContext[RepoDeps], request_context: ModelRequestContext
    ) -> ModelRequestContext:
        """Log the request about to be sent, as `request_log` does."""
        return await log_request(ctx=ctx, request_context=request_context)


def trim_history(
    ctx: RunContext[RepoDeps], request_context: ModelRequestContext
) -> ModelRequestContext:
    """Send the question and the latest exchange only: a stand-in for a history trimmer."""
    messages = request_context.messages
    if len(messages) <= 3:
        return request_context
    # A new request with a shorter list: the run's history keeps every
    # message. A tool's call and its return stay together.
    return replace(request_context, messages=[messages[0], *messages[-2:]])


def two_tools_then_answer(
    messages: list[ModelMessage], info: AgentInfo
) -> ModelResponse:
    """Stand in for a model: print what it receives; list, read, then answer."""
    print(
        f"  the model receives: messages={len(messages)} tools={len(info.function_tools)}"
    )
    returned = [
        part.tool_name
        for part in messages[-1].parts
        if isinstance(part, ToolReturnPart)
    ]
    if not returned:
        return ModelResponse(parts=[ToolCallPart(tool_name="list_modules", args={})])
    if returned == ["list_modules"]:
        call = ToolCallPart(
            tool_name="read_docstring", args={"module": "tinyjobs.jobs"}
        )
        return ModelResponse(parts=[call])
    return ModelResponse(parts=[TextPart(content="Done.")])


def show(*, title: str, capabilities: list[AbstractCapability[RepoDeps]]) -> None:
    """Run a second agent with these capabilities after its workspace, and print what happens."""
    reader = Agent(
        name="reader",
        deps_type=RepoDeps,
        toolsets=[repo_toolset],
        capabilities=[repo_workspace, *capabilities],
    )
    print(title)
    reader.run_sync(
        user_prompt="What does this repo do?",
        model=FunctionModel(function=two_tools_then_answer),
        deps=RepoDeps(root=Path("sample_repos/tinyjobs")),
    )


show(title="the request log as a class:", capabilities=[RequestLog()])
show(
    title="the log before a hook that trims the history:",
    capabilities=[RequestLog(), Hooks(before_model_request=trim_history)],
)
show(
    title="the log after it:",
    capabilities=[Hooks(before_model_request=trim_history), RequestLog()],
)
