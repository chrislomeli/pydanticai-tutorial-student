"""A chat in the terminal: type a message, read the answer.

Run it with `OPENAI_API_KEY` set:
`uv run python -m pydantic_tutor`. An empty line, or Ctrl-D, ends
it. Each line you type starts a run of its own, so the agent remembers
nothing from one run to the next.
"""

from pydantic_tutor.chat.agent import chat_agent
from pydantic_tutor.chat.console import blue
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
