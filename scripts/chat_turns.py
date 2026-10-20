"""Thread two turns of a conversation through the agent, with message_history.

From the history lesson (02-01). Kept working: a later lesson that changes
the agent updates this script, or removes it and says why.
"""

from pathlib import Path

from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

from pydantic_tutor.agent import agent
from pydantic_tutor.repo_tools import RepoDeps


def reply(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """Stand in for the model: look up the module once, then answer from history."""
    turn = sum(isinstance(part, UserPromptPart) for m in messages for part in m.parts)
    returns = [
        part for m in messages for part in m.parts if isinstance(part, ToolReturnPart)
    ]
    if turn == 1 and not returns:
        # First turn, nothing read yet: ask for the module's docstring.
        return ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name="read_docstring", args={"module": "tinyjobs.jobs"}
                )
            ]
        )
    if turn == 1:
        # The tool has answered; read its content into the reply.
        docstring = str(returns[-1].content).splitlines()[0]
        return ModelResponse(parts=[TextPart(content=f"jobs.py: {docstring}")])
    # Second turn: "it" means jobs.py, only because the first turn is in the
    # history this run was given.
    return ModelResponse(
        parts=[TextPart(content="run_next calls it, from the CLI's loop.")]
    )


deps = RepoDeps(root=Path("sample_repos/tinyjobs"))
model = FunctionModel(function=reply)

print("Turn 1:")
first = agent.run_sync(user_prompt="What does jobs.py do?", model=model, deps=deps)
print(" ", first.output)

print("Turn 2, with the first turn's history:")
second = agent.run_sync(
    user_prompt="And who calls it?",
    model=model,
    deps=deps,
    message_history=first.all_messages(),
)
print(" ", second.output)

print("Turn 2 again, without the history:")
alone = agent.run_sync(user_prompt="And who calls it?", model=model, deps=deps)
print(" ", alone.output)

print("\nThe conversation so far, as messages:")
for message in second.all_messages():
    kinds = [type(part).__name__ for part in message.parts]
    print(f"  {type(message).__name__}: {kinds}")
