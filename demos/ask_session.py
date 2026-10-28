"""Open the chat on a stand-in model that asks you a question each turn.

Given, from the ask-user lesson (05-03): you run it, you don't write it. It
opens the chat of `pydantic_tutor/chat/loop.py`, with the agent the app builds and a given
stand-in model, `asks_which_flag()`, in place of a real one. Whatever you
type, the model asks which flag the CLI should gain, then says what it was
told. Needs no API key. Run `demos/fresh_copy.py` first.
"""

from pydantic_tutor.chat.agent import build_chat_agent
from pydantic_tutor.chat.loop import main
from pydantic_tutor.composition import App
from pydantic_tutor.settings import Settings
from tutor_given.stand_ins import asks_which_flag

settings = Settings()
main(
    app=App(
        agent=build_chat_agent(
            web_search=False, workspace_root=settings.workspace_root
        ),
        model=asks_which_flag(),
    )
)
