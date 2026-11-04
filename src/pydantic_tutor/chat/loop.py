"""The terminal chat, with streaming and conversation history.

The application is assembled from Settings before the chat opens.
"""

from pydantic import ValidationError
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import ModelMessage

from pydantic_tutor.chat.console import blue, dim
from pydantic_tutor.composition import build_app
from pydantic_tutor.settings import Settings


def main() -> None:
    """Build the app, then run the chat with streaming and history."""
    try:
        app = build_app(settings=Settings())
    except (ValidationError, UserError) as error:
        # A bad setting or a missing key: say which, and do not open the chat.
        raise SystemExit(f"pydantic-tutor cannot start:\n{error}") from error
    print("Chat with the agent. An empty line ends it.")
    history: list[ModelMessage] = []  # every message of the chat so far
    while True:
        try:
            line = input("you> ").strip()
        except EOFError:
            break
        if not line:
            break
        result = app.agent.run_stream_sync(
            user_prompt=line, model=app.model, message_history=history
        )
        print(blue(text="agent> "), end="")
        for piece in result.stream_text(delta=True):
            print(blue(text=piece), end="", flush=True)
        print()
        print(dim(text=f"  (the model was sent {result.usage.input_tokens} tokens)"))
        history = result.all_messages()
