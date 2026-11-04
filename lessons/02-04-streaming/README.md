# Streaming the answer

## Welcome

Welcome back. In the last lesson your chat learned to remember, by
sending the chat so far with each run.

You'll keep improving the chat program, this time changing the user's
experience rather than its memory. **Streaming** lets the terminal show
pieces of a reply as they arrive instead of displaying the entire reply
at the end.

By the end, you'll be able to:

1. **Stream the chat's answers** as they're written.
2. **Say what streaming changes, and what it doesn't.**

Tested on `gpt-4o-mini`. About 10 minutes.

## Concepts

### The answer arrives in pieces

The model generates its output incrementally. Your existing `run_sync`
call waits until the run finishes, then gives you the answer. For a long answer, that
means staring at nothing, then a wall of text.

Streaming hands you the pieces as they arrive. You start the run with
`run_stream_sync` instead of `run_sync`, and read the text from
`result.stream_text(delta=True)`: each piece is the new text since the
last one. Print each as it comes, and the answer appears as it's
written.

### The run is the same

Streaming changes how the final text reaches your program. You still
supply `message_history`, and after consuming the stream, you still
save the messages for the next turn. The underlying run may involve
other events, including tool calls; this example focuses on streaming
the final answer.

## What you'll build

You'll revise `chat/loop.py` again. The chat still accepts a line
and keeps its history; the difference is that it prints response text
incrementally.

```text
pydantic-tutor/
└── src/
    └── pydantic_tutor/
        └── chat/
            └── loop.py  changed: answers are streamed
```

There's one step: **stream the answers.**

## Before you start

Carry on in your `pydantic-tutor` project from the last lesson. If you're in a new terminal, set `OPENAI_API_KEY` and `PYDANTIC_AI_NO_BANNER` again: an
environment variable lasts only as long as the terminal you set it in.

> [!TIP]
> **Check:** from the project folder, start the chat, send one line, and
> end it with an empty line:
>
> ```bash
> uv run python -m pydantic_tutor
> ```
>
> The agent should answer your line. The answer appears all at once: that is this lesson.

> [!WARNING]
> **Joining here, or has your project gone wrong?** If your project works,
> skip this. To refresh a broken project, delete its `pydantic-tutor` folder
> first: you lose your work and anything else in it, including local
> credentials. Then, from the folder that should hold the project:
>
> ```bash
> git clone --branch 02-03-history/end --single-branch https://github.com/chrislomeli/pydanticai-tutorial-student.git pydantic-tutor
> cd pydantic-tutor
> uv sync
> ```
>
> The copy also has a `tests/` folder, our own checks for each lesson. You
> don't need it. Then run the check above.

## Step 1: Stream the answers

> **Goal:** a chat whose answers appear as they're written.

So far the program waits until a reply is complete before showing any
of it. Feel that once before we change it: start the chat as it is,
and ask for something with a little length:

```bash
uv run python -m pydantic_tutor
```

```text
Tell me a short story about a lighthouse keeper, in four sentences.
```

A pause, then the whole story lands in one block. For four sentences
that's a moment; a long answer is a wait spent staring at nothing. End
the chat with an empty line, and open `src/pydantic_tutor/chat/loop.py`.

**Let's make the chat stream its answers!**

You'll edit `loop.py`. Three small edits, all inside the loop; the
question-collecting and history-keeping don't move. Try the edits
yourself, and compare with the full file at the end.

**1. Use `run_stream_sync`.** `run_sync` becomes `run_stream_sync`,
with the same arguments:

```python
        result = chat_agent.run_stream_sync(
            user_prompt=line, model=CHAT_MODEL, message_history=history
        )
```

The result can now hand over the answer in pieces as they arrive, not
only as one string at the end.

**2. Print each piece as it arrives.** `result.stream_text(delta=True)`
yields just the new text since the last piece — a few words at a time:

```python
        for piece in result.stream_text(delta=True):
            print(piece, end="", flush=True)
```

`flush=True` makes each piece show at once; without it, Python may hold
the text back until the line ends.

**3. Split the `agent>` print around the stream.** The single print
becomes an opening print with no newline — so the first piece lands on
the same line — and a closing `print()` that ends the line when the
stream ends. The answer stays blue (`blue` is the helper from the
console-chat lesson), so each piece is colored on its own:

```python
        print(blue(text="agent> "), end="")
        for piece in result.stream_text(delta=True):
            print(blue(text=piece), end="", flush=True)
        print()
```

Because every piece carries its own color and reset, stopping the chat
with Ctrl-C in the middle of an answer can't leave your terminal blue.

With all three edits, the middle of the loop reads like this — and here
is the whole file, with history updated after the answer as before:

```python
"""A chat in the terminal that remembers: type a message, read the answer.

Run it with `OPENAI_API_KEY` set:
`uv run python -m pydantic_tutor`. An empty line, or Ctrl-D, ends
it. Each run is handed the messages of the runs before it, so the model is
sent the whole chat so far, and its answer is printed as it is written.
"""

from pydantic_ai.messages import ModelMessage

from pydantic_tutor.chat.agent import chat_agent
from pydantic_tutor.chat.console import blue, dim
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
        result = chat_agent.run_stream_sync(
            user_prompt=line, model=CHAT_MODEL, message_history=history
        )
        print(blue(text="agent> "), end="")
        # Each piece of the answer as it arrives; flush, so it shows at once.
        for piece in result.stream_text(delta=True):
            print(blue(text=piece), end="", flush=True)
        print()
        print(dim(text=f"  (the model was sent {result.usage.input_tokens} tokens)"))
        history = result.all_messages()
```

Run the chat again, and ask it the same question:

```bash
uv run python -m pydantic_tutor
```

> [!TIP]
> **Check:** the story appears a few words at a time, not all at once.
> When it's done, you see the tokens line, as before. Ours:
>
> ```text
> you> Tell me a short story about a lighthouse keeper, in four sentences.
> agent> In a stormy coastal village, an old lighthouse keeper named Elias tended to his beacon with unwavering dedication. One fateful night, a fierce tempest threatened to engulf a passing ship, casting it into turmoil among the waves. As the light flickered, Elias steadfastly kept the flame alive, guiding the vessel to safety with his unwavering resolve. When dawn broke, the villagers gathered to thank the keeper, who simply smiled, saying, "The light shines brightest in the darkest of times."
>   (the model was sent 34 tokens)
> ```
>
> Your story will differ. Then ask it something about the story, such
> as "What was his name?": it still remembers.

**What it means**

The reply appeared gradually, but the follow-up still worked. That
combination is the point: a change to how the terminal displays an
answer didn't require changing how the application remembers the
conversation. The same history is supplied before the run and saved
once the response is complete.

One thing to know if you stream in a script of your own: keep the
streamed run inside a function, as `main()` does here. At this version,
a streamed result left at the top level of a script can keep the program
from exiting when it's done.

## What you can now do

- **Streaming shows the answer as it's written:** `run_stream_sync`,
  then print each piece from `stream_text(delta=True)`.
- **The application retains history.** Streaming changes how this
  example receives the final text, not how it supplies previous messages.

## Next

The chat now remembers and streams. Next you'll give that same agent a
new source of information: the ability to request a web search.

---

*Tested with `pydantic-ai` 2.56.0 and `pydantic-ai-harness` 0.56.0.*
