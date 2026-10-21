"""Show what the agent's shell runs and refuses, in the working copy.

From the shell lesson (03-05). Run `scripts/fresh_copy.py` first: the
commands run in `work/tinyjobs`, and one of them may delete its README.
Kept working: a later lesson that changes the agent's shell updates this
script, or removes it and says why.
"""

import textwrap
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
from pydantic_tutor.repo_tools import RepoDeps

# Each one is a prompt, and the command the stand-in model runs for it.
COMMANDS = [
    "pytest -q --tb=no",
    "rm README.md",
    'sh -c "rm README.md && echo removed"',
    "cat .env",
    "head -n 1 ../../pyproject.toml",
    "pytest --version && cat .env",
]


def reply(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """Stand in for a model: run the prompt as a command, then answer with what came back."""
    for part in messages[-1].parts:
        # A command's output comes back as a ToolReturnPart; a refusal, as a
        # RetryPromptPart asking the model to try something else.
        if isinstance(part, ToolReturnPart | RetryPromptPart):
            return ModelResponse(parts=[TextPart(content=str(part.content))])
        if isinstance(part, UserPromptPart):
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="run_command", args={"command": str(part.content)}
                    )
                ]
            )
    raise ValueError("the request has neither a prompt nor a tool result")


for command in COMMANDS:
    result = agent.run_sync(
        user_prompt=command,
        model=FunctionModel(function=reply),
        # The working copy, which scripts/fresh_copy.py makes.
        deps=RepoDeps(root=Path("work/tinyjobs")),
    )
    print(f"$ {command}")
    print(textwrap.indent(text=result.output.rstrip(), prefix="  "))
