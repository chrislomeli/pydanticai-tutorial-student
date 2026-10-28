"""Open the chat on a stand-in model, saving its conversations in a file.

Given, from the restart lesson (05-04): you run it, you don't write it. It opens
the chat of `pydantic_tutor/chat/loop.py` with the agent the app builds, the given stand-in
model `says_which_turn()` and a store on `work/conversations.db`, not the app's
own file. Needs no API key. Run `demos/fresh_copy.py` first.
"""

from pathlib import Path

from pydantic_tutor.stores.sqlite_conversations import SyncSqliteConversationStore

from pydantic_tutor.chat.agent import build_chat_agent
from pydantic_tutor.chat.loop import main
from pydantic_tutor.composition import App
from pydantic_tutor.settings import Settings
from tutor_given.stand_ins import says_which_turn

settings = Settings()
main(
    app=App(
        agent=build_chat_agent(
            web_search=False, workspace_root=settings.workspace_root
        ),
        model=says_which_turn(),
        store=SyncSqliteConversationStore(path=Path("work/conversations.db")),
    )
)
