"""A chat in the terminal: type a message, read the answer.

From the console-chat lesson (02-02). Run it with `OPENAI_API_KEY` set:
`uv run scripts/chat.py`. An empty line, or Ctrl-D, ends
it. Each turn is a run of its own, so the agent remembers nothing between
turns; the next lesson changes that.
"""

from pydantic_tutor.chat_agent import chat_agent
from pydantic_tutor.color import blue
from pydantic_tutor.model import CHAT_MODEL


def main() -> None:
    """Read a line, run the agent on it, print the answer; until an empty line."""
    print("Chat with the agent. An empty line ends it.")
    while True:
        try:
            line = input("you> ").strip()
        except EOFError:  # Ctrl-D, or the end of piped input
            break
        if not line:
            break
        result = chat_agent.run_sync(user_prompt=line, model=CHAT_MODEL)
        print(blue(text=f"agent> {result.output}"))


if __name__ == "__main__":
    main()
