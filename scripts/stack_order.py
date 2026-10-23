"""The agent with its capabilities listed forwards, then reversed: what changes?

From the seam lesson (04-06). Builds the project's agent twice with
`build_agent()`, its stack in each order, plus a stand-in capability that
sends the model only the last two messages of each request, as a later
lesson's compaction will shorten it. A stand-in model prints what it is
sent; the request log prints what it saw. Kept working: a later lesson
that changes the agent's stack updates this script, or removes it and says
why.
"""

import logging
import sys
from dataclasses import replace
from pathlib import Path

from pydantic_ai import ModelRequestContext, RunContext
from pydantic_ai.capabilities import AbstractCapability, AgentCapability
from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

from pydantic_tutor.agent import build_agent, codebase_capabilities
from pydantic_tutor.codebase_tools import CodebaseDeps

# The request log on, printing on stdout in order with this script's lines.
logging.basicConfig(stream=sys.stdout, format="%(name)s: %(message)s")
logging.getLogger(name="pydantic_tutor.request_log").setLevel(level=logging.INFO)


class KeepLastTwo(AbstractCapability[CodebaseDeps]):
    """Send the model only the last two messages of each request."""

    async def before_model_request(
        self, ctx: RunContext[CodebaseDeps], request_context: ModelRequestContext
    ) -> ModelRequestContext:
        """Return the request with its earlier messages left out."""
        # Two, so a tool call is sent with its return. A copy: the run's
        # history keeps every message.
        return replace(request_context, messages=request_context.messages[-2:])


def reply(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """Stand in for the model: say what it is sent, list the modules, then answer."""
    print(f"  model: sent {len(messages)} message(s)")
    if any(isinstance(part, ToolReturnPart) for part in messages[-1].parts):
        return ModelResponse(parts=[TextPart(content="Done.")])
    # The first request: show how each piece of the instructions starts.
    for part in info.model_request_parameters.instruction_parts or []:
        print("    instructions:", " ".join(part.content.split()[:4]), "...")
    return ModelResponse(parts=[ToolCallPart(tool_name="list_modules", args={})])


def stack() -> list[AgentCapability[CodebaseDeps]]:
    """The agent's capabilities, each one new, and the stand-in last."""
    return [*codebase_capabilities(), KeepLastTwo()]


for label, capabilities in [("forwards", stack()), ("reversed", stack()[::-1])]:
    print(label)
    agent = build_agent(capabilities=capabilities)
    agent.run_sync(
        user_prompt="What does this codebase do?",
        model=FunctionModel(function=reply),
        deps=CodebaseDeps(root=Path("sample_code/tinyjobs")),
    )
