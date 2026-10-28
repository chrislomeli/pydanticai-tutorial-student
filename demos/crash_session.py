"""Open the chat on a stand-in model whose turn ends in a slow build, so you can stop it mid-tool.

Given, from the crash lesson (05-05): you run it, you don't write it. It opens
the chat of `pydantic_tutor/chat/loop.py` on `work/tinyjobs`, with the given stand-in model
`writes_notes_then_builds()` and a stand-in tool, `wait_for_build`, that waits
ten seconds. The chat saves its conversations and records its runs in
`work/crash.db`, not the app's own file. Needs no API key. Run
`demos/fresh_copy.py` first.
"""

import asyncio
from pathlib import Path

from pydantic_ai import FunctionToolset
from pydantic_ai.capabilities import Toolset
from pydantic_ai_harness.step_persistence import SqliteStepStore

from pydantic_tutor.chat.agent import build_chat_agent, chat_capabilities
from pydantic_tutor.chat.loop import main
from pydantic_tutor.composition import App
from pydantic_tutor.settings import Settings
from pydantic_tutor.stores.sqlite_conversations import SyncSqliteConversationStore
from tutor_given.stand_ins import writes_notes_then_builds

DATABASE = Path("work/crash.db")
BUILD_SECONDS = 10


async def wait_for_build() -> str:
    """Wait for the build to finish, and report it."""
    print(f"  building... ({BUILD_SECONDS} seconds)", flush=True)
    await asyncio.sleep(delay=BUILD_SECONDS)
    return "build passed"


settings = Settings()
main(
    app=App(
        agent=build_chat_agent(
            web_search=False,
            workspace_root=settings.workspace_root,
            capabilities=[
                *chat_capabilities(
                    web_search=False, workspace_root=settings.workspace_root
                ),
                Toolset(toolset=FunctionToolset(tools=[wait_for_build])),
            ],
        ),
        model=writes_notes_then_builds(),
        store=SyncSqliteConversationStore(path=DATABASE),
        steps=SqliteStepStore(database=DATABASE),
    )
)
