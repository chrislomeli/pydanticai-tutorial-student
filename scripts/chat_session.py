"""Open the chat on a stand-in model, and answer its held calls yourself.

From the chat loop lesson (05-02). The chat is `pydantic_tutor.chat`, the
one the app opens; only the model is a stand-in, on `work/tinyjobs`. Run
`scripts/fresh_copy.py` first. Kept working: a later lesson that changes
the chat updates this script, or removes it and says why.
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
from pydantic_tutor.chat import chat
from pydantic_tutor.composition import App
from pydantic_tutor.repo_tools import RepoDeps

TEST_PATH = "tests/test_verbose.py"
# The test the stand-in model writes. It fails until the flag exists.
TEST = (
    "from tinyjobs.cli import build_parser\n\n\n"
    "def test_verbose() -> None:\n"
    '    assert build_parser().parse_args(["--verbose", "list"]).verbose\n'
)


def reply(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """Stand in for the model, whatever was typed.

    The first turn reads a docstring, then asks in one response to write
    `TEST` and to run it, then reports what each call returned. A later
    turn says what the conversation it was sent holds.
    """
    parts = [part for message in messages for part in message.parts]
    turn = sum(isinstance(part, UserPromptPart) for part in parts)
    returns = [part for part in parts if isinstance(part, ToolReturnPart)]
    if turn > 1:
        # A later turn: what the conversation it was sent holds.
        held = ", ".join(f"{part.tool_name} {part.outcome}" for part in returns)
        return ModelResponse(parts=[TextPart(content=f"Turn {turn}. So far: {held}.")])
    if not returns:
        call = ToolCallPart(tool_name="read_docstring", args={"module": "tinyjobs.cli"})
        return ModelResponse(parts=[call])
    if len(returns) == 1:
        return ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name="write_file", args={"path": TEST_PATH, "content": TEST}
                ),
                ToolCallPart(
                    tool_name="run_command", args={"command": f"pytest -q {TEST_PATH}"}
                ),
            ]
        )
    report = [
        f"{part.tool_name}, {part.outcome}: {part.content}" for part in returns[1:]
    ]
    return ModelResponse(parts=[TextPart(content="\n".join(report))])


chat(
    app=App(
        agent=agent,
        model=FunctionModel(function=reply, model_name="a stand-in model"),
        deps=RepoDeps(root=Path("work/tinyjobs")),
    )
)
