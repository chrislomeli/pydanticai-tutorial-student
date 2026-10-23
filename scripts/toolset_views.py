"""Give the codebase toolset to a second agent, then prefix and filter it.

From the toolsets lesson (04-01). Each run is on a stand-in model that
prints what it is sent: the tools' names, and the instructions. Kept
working: a later lesson that changes the toolset updates this script, or
removes it and says why.
"""

from pathlib import Path

from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    RetryPromptPart,
    TextPart,
    ToolCallPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionDef, FunctionModel
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.toolsets import AbstractToolset

from pydantic_tutor.agent import agent, codebase_workspace
from pydantic_tutor.codebase_tools import CodebaseDeps, codebase_toolset

TINYJOBS = CodebaseDeps(root=Path("sample_code/tinyjobs"))


def report(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """Stand in for a model: print what it is sent, then answer."""
    print("  tools:", ", ".join(tool.name for tool in info.function_tools))
    print("  instructions:", repr(info.instructions))
    return ModelResponse(parts=[TextPart(content="Done.")])


def ask_for_a_wrong_module(
    messages: list[ModelMessage], info: AgentInfo
) -> ModelResponse:
    """Stand in for a model: call the prefixed read_docstring with a wrong name."""
    for part in messages[-1].parts:
        if isinstance(part, RetryPromptPart):
            print("  asked to retry:", part.content)
            return ModelResponse(parts=[TextPart(content="Done.")])
    call = ToolCallPart(tool_name="codebase_read_docstring", args={"module": "jobs"})
    return ModelResponse(parts=[call])


def only_listing(ctx: RunContext[CodebaseDeps], tool: ToolDefinition) -> bool:
    """Keep `list_modules`; hide every other tool, in every run."""
    return tool.name == "list_modules"


def show(
    *, title: str, toolset: AbstractToolset[CodebaseDeps], reply: FunctionDef
) -> None:
    """Run a second agent, given `toolset`, on a model that answers with `reply`."""
    # It must have a workspace: the codebase tools read through it, and
    # the toolset does not bring one.
    reader = Agent(
        name="reader",
        deps_type=CodebaseDeps,
        toolsets=[toolset],
        capabilities=[codebase_workspace],
    )
    print(title)
    reader.run_sync(
        user_prompt="What does this codebase do?",
        model=FunctionModel(function=reply),
        deps=TINYJOBS,
    )


print("the codebase assistant:")
agent.run_sync(
    user_prompt="What does this codebase do?",
    model=FunctionModel(function=report),
    deps=TINYJOBS,
)
show(title="a second agent, given the toolset:", toolset=codebase_toolset, reply=report)
show(
    title="prefixed with 'codebase':",
    toolset=codebase_toolset.prefixed(prefix="codebase"),
    reply=report,
)
show(
    title="prefixed, calling read_docstring with a wrong module:",
    toolset=codebase_toolset.prefixed(prefix="codebase"),
    reply=ask_for_a_wrong_module,
)
show(
    title="filtered to list_modules:",
    toolset=codebase_toolset.filtered(filter_func=only_listing),
    reply=report,
)
