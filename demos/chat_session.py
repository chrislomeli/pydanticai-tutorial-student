"""Open the chat on a stand-in model, so you can answer its held calls.

Given, from the chat-loop lesson (05-02): you run it, you don't write it. It
opens the chat of `pydantic_tutor/chat/loop.py`, with the agent the app builds and a given
stand-in model, `asks_for_a_test_and_its_run()`, in place of a real one. Whatever
you type, the model asks to write a test and to run it. Needs no API key. Run
`demos/fresh_copy.py` first.
"""

from pydantic_tutor.chat.agent import build_chat_agent
from pydantic_tutor.chat.loop import main
from pydantic_tutor.composition import App
from pydantic_tutor.settings import Settings
from tutor_given.stand_ins import asks_for_a_test_and_its_run

settings = Settings()
main(
    app=App(
        agent=build_chat_agent(
            web_search=False, workspace_root=settings.workspace_root
        ),
        model=asks_for_a_test_and_its_run(),
    )
)
