# A chat in the terminal

## Welcome

Welcome back. In the last lesson you ran an agent on a real model, once,
and the script ended.

Now you'll create the first version of the **chat program** that you'll
keep extending in later lessons and Chapters. It will be simple at first:
type a line, receive an answer, and repeat. Building the loop yourself
makes it clear where the application ends and an agent run begins.

By the end, you'll be able to:

1. **Write a chat in the terminal**, as a loop around the agent's runs.
2. **Say why it forgets** what you told it a moment ago.

Tested on `gpt-4o-mini`. About 10 minutes.

## Concepts

### A chat is a loop around runs

You've run an agent many times now: one prompt in, one answer out. A
chat is that, again and again. Read what the person types, run the agent
on it, print the answer, and go back for the next line. The loop is
yours, in plain Python; the agent does what it always did.

A **run** is one question and its answer: one call to `run_sync`.
Everything the agent does before it answers, every tool call and every
request to the model, is part of the same run. In this chat, each line
you type starts a run, and its answer ends it.

### Each run starts fresh

In this particular chat, a run includes the agent's instructions and
the current prompt, but **no earlier chat messages**. Other agents can
supply tools or additional context; we're deliberately leaving out
conversation history for now.

So when the loop starts a new run for each line you type, the model
doesn't receive the earlier lines. It never sees the questions before it. Tell the chat
your name, then ask it what your name is, and it can't say: the second
run didn't carry the first.

That's not the model being forgetful. It remembers nothing between
requests anyway. It's the chat not sending what came before. The next
lesson does.

## What you'll build

The first version of your continuing application: a chat agent and a
terminal program that calls it repeatedly. They live in a `chat/`
package, which you'll keep as the lessons add memory, streaming and web
search.

```text
pydantic-tutor/
└── src/
    └── pydantic_tutor/
        ├── __main__.py      new: runs the chat with `python -m pydantic_tutor`
        └── chat/
            ├── __init__.py  new: makes `chat/` a package
            ├── agent.py     new: the agent the chat talks to
            ├── console.py   new: blue and dim, only on a terminal
            └── loop.py      new: the chat, a loop around the agent
```

And the steps, in order:

1. **Create the chat's agent.**
2. **Write the loop**, and chat.
3. **Watch it forget.**

Each step ends with a **Check**. If it doesn't show what it says, stop
there: the next step builds on it.

## Before you start

Carry on in your `pydantic-tutor` project from the last lesson. If you're in a new terminal, set `OPENAI_API_KEY` and `PYDANTIC_AI_NO_BANNER` again: an
environment variable lasts only as long as the terminal you set it in.

> [!TIP]
> **Check:** the key is set in this terminal. Print its first three
> characters, and no more:
>
> ```bash
> echo $OPENAI_API_KEY | cut -c1-3
> ```
>
> You should see:
>
> ```text
> sk-
> ```
>
> If you see nothing, set the key again (Step 1 of the last lesson): an
> environment variable lasts only as long as the terminal you set it in.
>
> Then run the last lesson's script and wait for the response:
>
> ```bash
> uv run scripts/first_real_run.py
> ```
>
> It should end with an answer saying coffee is 700 cents, and a cost.

> [!WARNING]
> **Joining here, or has your project gone wrong?** If your project works,
> skip this. To refresh a broken project, delete its `pydantic-tutor` folder
> first: you lose your work and anything else in it, including local
> credentials. Then, from the folder that should hold the project:
>
> ```bash
> git clone --branch 02-01-real-model/end --single-branch https://github.com/chrislomeli/pydanticai-tutorial-student.git pydantic-tutor
> cd pydantic-tutor
> uv sync
> ```
>
> The copy also has a `tests/` folder, our own checks for each lesson. You
> don't need it. Then run the check above.

## Step 1: Create the chat's agent

> **Goal:** an agent for the chat, with instructions and no model.

The chat will need an agent to answer each message. We'll keep its
definition separate from the terminal loop, so later lessons can add
capabilities without changing how we read and print messages. For now,
the agent only needs a name and brief instructions; the loop will select
the model when it runs.

First, make `chat/` a package. Create `src/pydantic_tutor/chat/__init__.py`:

```python
"""The chat application: its agent, console and loop."""
```

Then create `src/pydantic_tutor/chat/agent.py`:

```python
"""The agent the chat talks to.

No model here, as for every agent in
this project: the chat passes `CHAT_MODEL` on each run.
"""

from pydantic_ai import Agent

chat_agent = Agent(
    name="chat",
    instructions="You are a helpful assistant. Answer briefly.",
)
```

> [!TIP]
> **Check:** print its name and its model:
>
> ```bash
> uv run python -c "from pydantic_tutor.chat.agent import chat_agent; print(chat_agent.name, chat_agent.model)"
> ```
>
> You should see:
>
> ```text
> chat None
> ```

**What it means**

An agent like the first one you built: a name, instructions, and no
model. The chat gives it `CHAT_MODEL` on each run.

## Step 2: Write the loop

> **Goal:** a chat in your terminal: you type, the agent answers.

Now we can build the part that makes this feel like a chat. The program
will wait for a line of text, hand it to our agent, show the reply, and
wait again until you enter an empty line. That's the job of the Python
loop, not the model. The first version won't pass earlier messages to a
new run; we'll see the consequence shortly.

First, a small helper so the chat can color its lines. Colors are
cosmetic and aren't what this lesson teaches. The helper colors text only
when the output is a terminal and `NO_COLOR` is not set, so piped output
and files stay plain. Create `src/pydantic_tutor/chat/console.py`:

```python
"""Terminal colors for the chat, used only where someone will see them.

Text is colored only when the output
is a terminal and `NO_COLOR` is not set, so piped output, files and tests
get the plain text.
"""

import os
import sys

_BLUE = "\033[38;5;75m"
_DIM = "\033[2m"
_RESET = "\033[0m"


def _wanted() -> bool:
    """Whether to color: output goes to a terminal, and `NO_COLOR` is unset."""
    return sys.stdout.isatty() and not os.environ.get("NO_COLOR")


def blue(*, text: str) -> str:
    """The text in blue, or as it is when colors are not wanted."""
    return f"{_BLUE}{text}{_RESET}" if _wanted() else text


def dim(*, text: str) -> str:
    """The text dimmed, or as it is when colors are not wanted."""
    return f"{_DIM}{text}{_RESET}" if _wanted() else text
```

Now create `src/pydantic_tutor/chat/loop.py`. As you read it, find the line
that calls the agent, and the condition that ends the loop:

```python
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
```

`loop.py` only defines `main`. To run the package, create
`src/pydantic_tutor/__main__.py`, which Python runs for
`python -m pydantic_tutor`:

```python
"""Run the chat: `uv run python -m pydantic_tutor`."""

from pydantic_tutor.chat.loop import main

main()
```

Now chat:

```bash
uv run python -m pydantic_tutor
```

Type a message after `you>`, press Enter, and read the answer. Press
Enter on an empty line to stop.

> [!TIP]
> **Check:** you should see something like this, with your own
> messages:
>
> ```text
> Chat with the agent. An empty line ends it.
> you> Hello!
> agent> Hello! How can I assist you today?
> you>
> ```
>
> The agent's words will differ; what matters is that it answers each
> line you type. In a terminal the
> `agent>` line is blue; the colors are not part of the check.

**What it means**

You can now stay in one terminal program and exchange several messages
with the agent. Each reply still comes from a separate agent run. The
Python loop gives the application its repeated interaction; it doesn't
yet give those runs a shared conversation.

## Step 3: Watch it forget

> **Goal:** see that each line you type is a run of its own.

Now test whether a chat interface is the same thing as conversation
memory. You'll tell it a name, then ask for that name in the very next
message. If the program hasn't passed earlier messages along, the
second run has no reliable way to know the answer.

Start the chat again, and type these two lines, one at a time:

```text
My name is Sam.
```

```text
What is my name?
```

Before you send the second, guess: will it know?

> [!TIP]
> **Check:** something like this:
>
> ```text
> you> My name is Sam.
> agent> Hi, Sam! How can I assist you today?
> you> What is my name?
> agent> I’m not sure what your name is. Can you tell me?
> ```
>
> In another of our runs it said "I don’t know your name. You can tell
> me if you’d like!" The wording varies; it doesn't know.

**What it means**

It greeted you by name, then couldn't reliably recall it. This isn't a
failure of the loop: you wrote the loop to pass only the current prompt.
The next improvement to this **same program** will be to preserve and
resend previous messages.

## What you can now do

- **A chat is a loop around runs:** read a line, run the agent on it,
  print the answer. A run is one question and its answer, tool calls
  included. The loop is plain Python.
- **Each run starts fresh,** so a chat that runs the agent once per line
  forgets everything before the current question.

## Next

The chat now works, but its conversations don't continue across runs.
Next you'll change `chat/loop.py` to carry earlier messages into each new
run. The loop stays; the application gets memory of the current chat.

---

*Tested with `pydantic-ai` 2.56.0 and `pydantic-ai-harness` 0.56.0.*
