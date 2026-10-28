"""A chat in the terminal that remembers: type a message, read the answer.

From the console-chat lesson (02-02); history from the history lesson
(02-03). Run it with `OPENAI_API_KEY` set:
`uv run scripts/chat.py`. An empty line, or Ctrl-D, ends
it. Each run is handed the messages of the runs before it, so the model is
sent the whole chat so far.
"""

from pydantic_ai.messages import ModelMessage

from pydantic_tutor.chat_agent import chat_agent
from pydantic_tutor.color import blue, dim
from pydantic_tutor.model import CHAT_MODEL


def main() -> None:
    """Read a line, run the agent on it with the chat so far, print the answer."""
    print("Chat with the agent. An empty line ends it.")
    history: list[ModelMessage] = []  # every message of the chat so far
    while True:
        try:
            line = input("you> ").strip()
        except EOFError:  # Ctrl-D, or the end of piped input
            break
        if not line:
            break
        result = chat_agent.run_sync(
            user_prompt=line, model=CHAT_MODEL, message_history=history
        )
        print(blue(text=f"agent> {result.output}"))
        print(dim(text=f"  (the model was sent {result.usage.input_tokens} tokens)"))
        history = result.all_messages()


if __name__ == "__main__":
    main()
