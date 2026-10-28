"""A chat in the terminal that remembers: type a message, read the answer.

From the console-chat lesson (02-02); history from the history lesson
(02-03); streaming from the streaming lesson (02-04); its agent and model
from the settings lesson (03-01). Run it with `OPENAI_API_KEY` set:
`uv run python -m pydantic_tutor`. An empty line, or Ctrl-D, ends
it. Each run is handed the messages of the runs before it, so the model is
sent the whole chat so far, and its answer is printed as it is written.
"""

from pydantic import ValidationError
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import ModelMessage

from pydantic_tutor.chat.console import blue, dim
from pydantic_tutor.composition import build_app
from pydantic_tutor.settings import Settings


def main() -> None:
    """Read a line, run the agent on it with the chat so far, print the answer."""
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
        except EOFError:  # Ctrl-D, or the end of piped input
            break
        if not line:
            break
        result = app.agent.run_stream_sync(
            user_prompt=line, model=app.model, message_history=history
        )
        print(blue(text="agent> "), end="")
        # Each piece of the answer as it arrives; flush, so it shows at once.
        for piece in result.stream_text(delta=True):
            print(blue(text=piece), end="", flush=True)
        print()
        print(dim(text=f"  (the model was sent {result.usage.input_tokens} tokens)"))
        history = result.all_messages()
