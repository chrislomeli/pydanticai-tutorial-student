"""Open the chat on a stand-in model that says what each request carried.

Given, from the cost lesson (06-01): you run it, you don't write it. It opens
the chat of `pydantic_tutor/chat/loop.py` on `work/tinyjobs`, nothing saved, with the
given stand-in model `counts_like_a_provider()`. Whatever you type, the model
lists the modules, reads one docstring and answers: three requests a turn.
Before each it prints the tokens of what the request carries (instructions,
tool definitions, messages) and reports their sum as the request's input
tokens, as a provider does. Needs no API key. Run `demos/fresh_copy.py`
first.
"""

from pydantic_tutor.chat.agent import build_chat_agent
from pydantic_tutor.chat.loop import main
from pydantic_tutor.composition import App
from pydantic_tutor.settings import Settings
from tutor_given.provider_counts import counts_like_a_provider

settings = Settings()
main(
    app=App(
        agent=build_chat_agent(
            web_search=False, workspace_root=settings.workspace_root
        ),
        model=counts_like_a_provider(),
    )
)
