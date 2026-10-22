"""Show what the agent's file tools read, write and refuse, in the working copy.

From the file tools lesson (03-04). Run `scripts/fresh_copy.py` first: the
tools work in `work/tinyjobs`. Kept working: a later lesson that changes the
agent's file tools updates this script, or removes it and says why.
"""

from pathlib import Path

from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    RetryPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

from pydantic_tutor.agent import agent
from pydantic_tutor.approval import approve_all, approver
from pydantic_tutor.repo_tools import RepoDeps

# Each prompt names the one call the stand-in model makes for it.
CALLS = {
    "read README.md": ToolCallPart(
        tool_name="read_file", args={"path": "README.md", "limit": 1}
    ),
    "read .env": ToolCallPart(tool_name="read_file", args={"path": ".env"}),
    "write .env": ToolCallPart(
        tool_name="write_file", args={"path": ".env", "content": "TOKEN=mine\n"}
    ),
    "read ../../pyproject.toml": ToolCallPart(
        tool_name="read_file", args={"path": "../../pyproject.toml"}
    ),
}


def reply(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """Stand in for a model: make the prompt's call, then answer with what came back."""
    for part in messages[-1].parts:
        # A tool's result comes back as a ToolReturnPart; a refusal, as a
        # RetryPromptPart asking the model to try something else.
        if isinstance(part, ToolReturnPart | RetryPromptPart):
            return ModelResponse(parts=[TextPart(content=str(part.content))])
        if isinstance(part, UserPromptPart):
            return ModelResponse(parts=[CALLS[str(part.content)]])
    raise ValueError("the request has neither a prompt nor a tool result")


for prompt in CALLS:
    result = agent.run_sync(
        user_prompt=prompt,
        model=FunctionModel(function=reply),
        # The working copy, which scripts/fresh_copy.py makes.
        deps=RepoDeps(root=Path("work/tinyjobs")),
        # Every call approved: this script shows what the tools themselves
        # allow and refuse, once a person has said yes.
        capabilities=[approver(answer=approve_all)],
    )
    print(f"{prompt}:")
    for line in result.output.splitlines():
        print(f"  {line}")
