"""Show what the agent's workspace does not do: keep a tool inside its folder.

From the workspace lesson (03-03). Kept working: a later lesson that
changes the agent's workspace updates this script, or removes it and says
why.
"""

from pathlib import Path

from pydantic_ai import RunContext
from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.toolsets import FunctionToolset

from pydantic_tutor.agent import agent
from pydantic_tutor.codebase_tools import CodebaseDeps


async def first_line(ctx: RunContext[CodebaseDeps], *, path: str) -> str:
    """Read the first line of a file, by a path relative to the workspace's folder."""
    text = await ctx.workspace.read_text(path=path)
    return text.splitlines()[0]


def reply(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """Stand in for a model: read the file the prompt names, then answer with its line."""
    for part in messages[-1].parts:
        if isinstance(part, ToolReturnPart):
            return ModelResponse(parts=[TextPart(content=str(part.content))])
        if isinstance(part, UserPromptPart):
            call = ToolCallPart(tool_name="first_line", args={"path": part.content})
            return ModelResponse(parts=[call])
    raise ValueError("the request has neither a prompt nor a tool result")


# The workspace's folder is the sample codebase. The first path is inside it;
# the second climbs two folders, to this project's own pyproject.toml.
for path in ["README.md", "../../pyproject.toml"]:
    result = agent.run_sync(
        user_prompt=path,
        model=FunctionModel(function=reply),
        deps=CodebaseDeps(root=Path("sample_code/tinyjobs")),
        # A tool for this script only, beside the agent's own two.
        toolsets=[FunctionToolset(tools=[first_line])],
    )
    print(f"{path}: {result.output}")
