# A chat that remembers

## Welcome

Welcome back. In the last lesson your chat forgot your name one line
after you told it.

This lesson is about **history**: handing each run the messages of the
chat so far, so the model can see what was said.

By the end, you'll be able to:

1. **Make the chat remember**, by passing each run the messages before it.
2. **Say what the model is sent** each time, and why it grows.

Tested on `gpt-4o-mini`. About 10 minutes.

## Concepts

### The chat sends what came before

Your terminal loop already works. What's missing is the conversation.
The model doesn't retain the earlier requests on its own, so your
program must keep the messages and supply them on later runs.

A run's result has them all, `result.all_messages()`: every request and
response of this run, and of any history it was given. You pass them to
the next run as `message_history=`, and that run sends them to the model
before your new question.

### Every request carries the whole chat

So the model is sent more each time: the first question, its answer, the
second question, its answer, and so on, then the new one. A long chat
means long requests, and on a real model you pay for every token sent.

This is the central idea behind conversation history: continuity comes
from context the application supplies, not from a model quietly
remembering a previous request. Production applications can manage,
summarize or retrieve history in other ways; this chat starts by
resending the messages.

## What you'll build

You'll revise the **same `chat/loop.py`** from the previous lesson.
It will keep messages between runs and pass them to the agent. You'll
also print input-token usage to see a consequence of including history.

```text
pydantic-tutor/
└── src/
    └── pydantic_tutor/
        └── chat/
            └── loop.py  changed: history between runs, and the tokens sent
```

There's one step: **give the chat a history**, and watch it remember.

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
> The agent should answer your line. It still forgets: that is this lesson.

> [!WARNING]
> **Joining here, or has your project gone wrong?** If your project works,
> skip this. To refresh a broken project, delete its `pydantic-tutor` folder
> first: you lose your work and anything else in it, including local
> credentials. Then, from the folder that should hold the project:
>
> ```bash
> git clone --branch 02-02-console-chat/end --single-branch https://github.com/chrislomeli/pydanticai-tutorial-student.git pydantic-tutor
> cd pydantic-tutor
> uv sync
> ```
>
> The copy also has a `tests/` folder, our own checks for each lesson. You
> don't need it. Then run the check above.

## Step 1: Give the chat a history

> **Goal:** a chat that remembers what you told it, and shows what the
> model is sent.

As it stands, your loop works fine: you can chat with the model as long
as you like. But the model forgets each line the moment it answers it,
because every run is sent only the one new line. To answer a follow-up,
the model needs the conversation that came before — and it can't get
that on its own. Your program has to keep a record of the chat and pass
that history along with every question.

Here's the part that makes this easy: pydantic-ai already keeps the
record. Each run's result carries the log of every message that run
saw — the history it was handed, plus its own request and answer — and
`result.all_messages()` gives you that whole log. Feeding it forward
takes one new argument: `message_history=`.

So the terminal loop stays as it is, and only a few lines around the
run change. Open `src/pydantic_tutor/chat/loop.py` and make these edits.

Add the import for `ModelMessage` — the type of the history list — at
the top of the file, and extend the color import with `dim`:

```python
from pydantic_ai.messages import ModelMessage
from pydantic_tutor.chat.console import blue, dim
```

Before the `while` loop, start an empty history:

```python
history: list[ModelMessage] = []  # every message of the chat so far
```

Then, inside the loop, hand that history to the run, print what the
model was sent, and save the log the run returns for next time:

```python
result = chat_agent.run_sync(
    user_prompt=line, model=CHAT_MODEL, message_history=history
)
print(blue(text=f"agent> {result.output}"))
print(dim(text=f"  (the model was sent {result.usage.input_tokens} tokens)"))
history = result.all_messages()
```

With those edits in, `src/pydantic_tutor/chat/loop.py` should look like this (the
docstrings are updated to match):

```python
"""A chat in the terminal that remembers: type a message, read the answer.

Run it with `OPENAI_API_KEY` set:
`uv run python -m pydantic_tutor`. An empty line, or Ctrl-D, ends
it. Each run is handed the messages of the runs before it, so the model is
sent the whole chat so far.
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
        result = chat_agent.run_sync(
            user_prompt=line, model=CHAT_MODEL, message_history=history
        )
        print(blue(text=f"agent> {result.output}"))
        print(dim(text=f"  (the model was sent {result.usage.input_tokens} tokens)"))
        history = result.all_messages()
```

Start the chat, and send these three lines, one at a time:

```bash
uv run python -m pydantic_tutor
```

```text
My name is Sam.
```

```text
What is my name?
```

```text
Thanks!
```

Before starting, notice the order: the program first sends `history`,
then replaces it with the messages returned by this run. Watch whether
the input-token count grows as you add turns.

> [!TIP]
> **Check:** something like this:
>
> ```text
> Chat with the agent. An empty line ends it.
> you> My name is Sam.
> agent> Nice to meet you, Sam! How can I help you today?
>   (the model was sent 25 tokens)
> you> What is my name?
> agent> Your name is Sam.
>   (the model was sent 52 tokens)
> you> Thanks!
> agent> You’re welcome, Sam! If you need anything else, just let me know!
>   (the model was sent 67 tokens)
> ```
>
> The words and numbers will differ. The important observations are that
> the second answer can use your name and that sending longer history
> generally increases input-token usage.

**What it means**

The second answer could use the name from the first exchange because
your program carried that exchange forward. The model didn't gain
persistent memory; the next run received earlier messages as input.
The next model request included "My name is Sam." along with
"What is my name?".

In this sample, input usage grew from 25 to 52 to 67 tokens as the
conversation accumulated. Exact counts vary and aren't guaranteed to
rise on every individual turn, but carrying more history generally
requires more input tokens. Nothing about the model changed; your
program supplied more context.

## What you can now do

- **The chat remembers because it sends what came before:** each run
  gets `message_history=` the messages of the runs before it, from
  `result.all_messages()`.
- **This version passes the accumulated chat** on every run. Longer
  context generally uses more tokens and can increase cost.

## Next

The application now remembers within a session. Next you'll keep that
history behavior while changing how replies reach the terminal: instead
of waiting for a whole answer, you'll stream it as it arrives.

---

*Tested with `pydantic-ai` 2.56.0 and `pydantic-ai-harness` 0.56.0.*
