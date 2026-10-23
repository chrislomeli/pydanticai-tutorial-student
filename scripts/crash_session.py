"""Open the chat on a stand-in model whose turn ends in a slow build, so you can stop it mid-tool.

The chat is `pydantic_tutor.chat`, the one the app opens, on
`work/tinyjobs`; it saves its conversations and records its runs in
`work/crash.db`, not the app's own file. Asked anything, the stand-in
model lists the modules, reads one docstring, writes `NOTES.md` (you
approve it), then calls `wait_for_build`, a stand-in tool that waits ten
seconds. Run `scripts/fresh_copy.py` first. From 05-05-crash; a later
lesson that changes the chat keeps this script working, or removes it
and says why.
"""

import asyncio
from pathlib import Path

from pydantic_ai import FunctionToolset
from pydantic_ai.capabilities import Toolset
from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai_harness.step_persistence import SqliteStepStore

from pydantic_tutor.agent import build_agent, codebase_capabilities
from pydantic_tutor.chat import chat
from pydantic_tutor.codebase_tools import CodebaseDeps
from pydantic_tutor.composition import App
from pydantic_tutor.conversations import ConversationStore

DATABASE = Path("work/crash.db")
BUILD_SECONDS = 10


async def wait_for_build() -> str:
    """Wait for the build to finish, and report it."""
    print(f"  building... ({BUILD_SECONDS} seconds)")
    await asyncio.sleep(delay=BUILD_SECONDS)
    return "build passed"


def reply(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """Stand in for the model: four tools in turn, then say what came back.

    It reads the turn so far, from its last prompt on, and asks for the
    first of its four tools that has not returned in it yet.
    """
    start = max(
        index
        for index, message in enumerate(messages)
        if any(isinstance(part, UserPromptPart) for part in message.parts)
    )
    returned = {
        part.tool_name: part.content
        for message in messages[start:]
        for part in message.parts
        if isinstance(part, ToolReturnPart)
    }
    if "list_modules" not in returned:
        return ModelResponse(parts=[ToolCallPart(tool_name="list_modules", args={})])
    if "read_docstring" not in returned:
        return ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name="read_docstring", args={"module": "tinyjobs.jobs"}
                )
            ]
        )
    if "write_file" not in returned:
        summary = str(returned["read_docstring"]).splitlines()[0]
        notes = f"# Notes\n\ntinyjobs.jobs: {summary}\n"
        return ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name="write_file", args={"path": "NOTES.md", "content": notes}
                )
            ]
        )
    if "wait_for_build" not in returned:
        return ModelResponse(parts=[ToolCallPart(tool_name="wait_for_build", args={})])
    return ModelResponse(
        parts=[TextPart(content=f"Wrote NOTES.md; {returned['wait_for_build']}.")]
    )


chat(
    app=App(
        agent=build_agent(
            capabilities=[
                *codebase_capabilities(),
                Toolset(toolset=FunctionToolset(tools=[wait_for_build])),
            ]
        ),
        model=FunctionModel(function=reply, model_name="a stand-in model"),
        deps=CodebaseDeps(root=Path("work/tinyjobs")),
        store=ConversationStore(path=DATABASE),
        steps=SqliteStepStore(database=DATABASE),
    )
)
