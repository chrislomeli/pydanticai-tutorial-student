"""Where a capability's hooks run: before the request in list order, after it in reverse.

From the capability lesson (04-03). A second agent, given the codebase
toolset, has `BatchLimit` between two capabilities that only print what
they see; a stand-in model asks for five tool calls at once. Kept
working: a later lesson that changes `BatchLimit` updates this script,
or removes it and says why.
"""

from dataclasses import dataclass
from pathlib import Path

from pydantic_ai import Agent, ModelRequestContext, RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

from pydantic_tutor.agent import codebase_workspace
from pydantic_tutor.batch_limit import BatchLimit
from pydantic_tutor.codebase_tools import CodebaseDeps, codebase_toolset

# The five modules of sample_code/tinyjobs, as list_modules gives them.
MODULES = (
    "tinyjobs",
    "tinyjobs.cli",
    "tinyjobs.jobs",
    "tinyjobs.log",
    "tinyjobs.storage",
)


@dataclass(kw_only=True)
class Watch(AbstractCapability[CodebaseDeps]):
    """Print each request going out and each response coming back, unchanged."""

    # Printed at the start of each line, to tell two of them apart.
    name: str

    async def before_model_request(
        self, ctx: RunContext[CodebaseDeps], request_context: ModelRequestContext
    ) -> ModelRequestContext:
        """Say that a request is about to go out, and pass it on."""
        print(f"{self.name}: before the request")
        return request_context

    async def after_model_request(
        self,
        ctx: RunContext[CodebaseDeps],
        *,
        request_context: ModelRequestContext,
        response: ModelResponse,
    ) -> ModelResponse:
        """Say how many tool calls the response holds, and pass it on."""
        print(
            f"{self.name}: after, the response has {len(response.tool_calls)} tool calls"
        )
        return response


def five_reads_then_answer(
    messages: list[ModelMessage], info: AgentInfo
) -> ModelResponse:
    """Stand in for a model: ask for five docstrings at once, then answer."""
    if any(isinstance(part, ToolReturnPart) for part in messages[-1].parts):
        print("  the model answers")
        return ModelResponse(parts=[TextPart(content="Done.")])
    print("  the model asks for 5 tool calls")
    calls = [
        ToolCallPart(tool_name="read_docstring", args={"module": module})
        for module in MODULES
    ]
    return ModelResponse(parts=calls)


reader = Agent(
    name="reader",
    deps_type=CodebaseDeps,
    toolsets=[codebase_toolset],
    capabilities=[
        codebase_workspace,
        Watch(name="first"),
        BatchLimit(max_tool_calls=3),
        Watch(name="last"),
    ],
)
reader.run_sync(
    user_prompt="Read every module and summarise each.",
    model=FunctionModel(function=five_reads_then_answer),
    deps=CodebaseDeps(root=Path("sample_code/tinyjobs")),
)
