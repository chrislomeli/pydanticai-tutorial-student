---
title: "Settings and a composition root"
description: "Give the chat's choices one typed home, Settings, and build the objects they call for in one place, so a bad setting or a missing key stops the app before the chat opens."
---

# Settings and a composition root

## Welcome

Welcome back. Your chat now remembers conversations, streams answers, and
searches the web. We're beginning to turn it into an application you can
configure without rewriting it.

Right now two decisions are made in Python code: the model's name lives
in `model.py`, and web search is switched on inside `chat/agent.py`.
That's a reasonable way to start. But suppose you want to switch off
web search for one run, or choose a different model on another machine.
Why should either choice require an edit?

Applications usually have a way to collect these choices, often called **configuration**.
Some values have sensible defaults in code; others come from a file or
from the environment where the program runs. We'll bring our two choices
into a typed `Settings` class, try changing their values without
editing Python, and then build the chat from those settings at startup.

There's a second payoff: a missing API key currently makes the chat fail
*after* you type your first message. We'll move that failure to startup,
before the chat invites you to type.

By the end, you'll be able to:

1. **Separate values from objects:** read every choice into one typed
   `Settings`, and build the agent and model from it in one place.
2. **Make a bad setting or a missing key stop the app at startup**, by
   name, before the chat opens.

About 15 minutes. From Step 2 on, no API key is needed.

## Where the configuration lives

Our chat has two choices built into Python: which model it uses and
whether web search is available. We'll collect them in a `Settings`
class. Each setting has a name, a Python type, and a default value.

Pydantic Settings can read those choices from several places, including
environment variables and files. It interprets incoming text according
to the field's type: for example, `false` from the environment becomes
the Python boolean `False`. That gives our application one consistent
way to obtain its configuration, even when the values come from
different places.

Later in the lesson, we'll use those settings to assemble the agent and
model. For now, we'll concentrate on gathering the choices and seeing
how overrides work.

## What you'll build

```text
pydantic-tutor/
├── pyproject.toml                    changed: pydantic-settings, pinned
├── uv.lock                           changed: the same, resolved
└── src/
    └── pydantic_tutor/
        ├── chat/
        │   ├── agent.py              changed: build_chat_agent(web_search=...)
        │   └── loop.py               changed: reads the settings, runs on the built app
        ├── composition.py            new: App and build_app, the one place objects are built
        └── settings.py               new: Settings, every choice, typed
```

And the steps, in order:

1. **Create settings:** collect the chat's two defaults in one class.
2. **Try overrides:** read settings from the environment and a file.
3. **Assemble the application:** build an agent and model from settings.
4. **Update the chat loop:** use the assembled application.

Each step ends with a **Check**. If it doesn't show what it says, stop
there: the next step builds on it.

## Before you start

Continue with the `pydantic-tutor` project you've been building. From
its root folder, check that the Chapter 2 chat still works:

```bash
uv run python -m pydantic_tutor
```

**Check:** send one short message, confirm that the agent replies, then
end the chat with an empty line. Use the API key you set up in Chapter 2.

> [!WARNING]
> **Only start fresh if you need to.**
>
> If you've been following the lessons, **keep using your existing
> project**. Do not run these commands.
>
> If you're joining at this lesson or starting again without an existing
> project, run these commands from a parent directory where **no folder
> named `pydantic-tutor` exists**. Every lesson uses that same project
> folder name. If you already have `pydantic-tutor`, keep using it; do
> not run the clone commands or overwrite that folder. Starting again
> would lose the work and settings in your existing project.
>
> ```bash
> git clone --branch 02-05-web-search/end --single-branch https://github.com/chrislomeli/pydanticai-tutorial-student.git pydantic-tutor
> cd pydantic-tutor
> uv sync
> ```

## Step 1: Create the chat's settings

> **Goal:** put the two existing choices in one typed class, with
> defaults that keep the chat behaving as before.

### Add `pydantic-settings` as a dependency

We're about to import `pydantic-settings` directly, so add it to our
project's dependencies even if another package already installed it:

```bash
uv add pydantic-settings==2.16.0
```

**Check:** open `pyproject.toml` and confirm that `pydantic-settings`
appears in its dependencies. If the command reported an error, stop
before continuing.

### Create `Settings` with two defaults

Now create `src/pydantic_tutor/settings.py`. This is a new class with
two fields: `chat_model` holds the model name, and `web_search` holds
a true-or-false choice. **Both start with the same values our chat
already used in Python code.** We're moving them into one place, not
changing the behavior yet.

```python
"""Every value the app is configured with, typed and checked at startup.

Values only: strings and flags read from the environment. The objects built
from them are in `composition.py`.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict

from pydantic_tutor.model import CHAT_MODEL


class Settings(BaseSettings):
    """Read from variables named `PYDANTIC_TUTOR_<FIELD>`; frozen once created."""

    model_config = SettingsConfigDict(env_prefix="PYDANTIC_TUTOR_", frozen=True)

    # A `<provider>:<model>` string, the name of the model: not the model.
    chat_model: str = CHAT_MODEL
    # Whether the chat may search the web. Each search is billed.
    web_search: bool = True
```

A `Settings` instance can read environment variables as well as those
defaults. `frozen=True` prevents assigning a new value to a field
after it has been created.

Let's first see the defaults:

```bash
uv run python -c "from pydantic_tutor.settings import Settings; print(Settings())"
```

> [!TIP]
> **Check:** if you have no `PYDANTIC_TUTOR_` overrides in your shell:
>
> ```text
> chat_model='openai:gpt-4o-mini' web_search=True
> ```

The chat's two choices now have one home. We haven't yet made the
running chat use this class.

## Step 2: Change a setting without editing Python

> **Goal:** see how the same typed setting can get a value from an
> environment variable or a file without changing the class.

### Override a setting from the environment

Try changing web search for just this one command:

```bash
PYDANTIC_TUTOR_WEB_SEARCH=false uv run python -c "from pydantic_tutor.settings import Settings; print(Settings().web_search)"
```

> [!TIP]
> **Check:** it prints `False`.

The environment supplies text, but Pydantic Settings interprets
`false` as a Python boolean. We didn't change the class or its default.

### Read a setting from a file

So far, the values came from Python defaults or variables on a command
line. If you regularly run the chat with web search off, you might not
want to type that override every time. A `.env` file is a plain text
file of `NAME=value` lines. Create a practice file in the project root with an ordinary setting.
We use a separate name so we don't overwrite your real `.env` file,
which might contain an API key:

```bash
printf 'PYDANTIC_TUTOR_WEB_SEARCH=false\n' > lesson-settings.env
```

**Check:** open `lesson-settings.env` and confirm it contains
the line `PYDANTIC_TUTOR_WEB_SEARCH=false`.

Now ask `Settings` to read it:

```bash
uv run python -c "from pydantic_tutor.settings import Settings; print(Settings(_env_file='lesson-settings.env').web_search)"
```

> [!TIP]
> **Check:** it prints `False`. The file overrides the Python default.

Finally, override the file with an environment variable:

```bash
PYDANTIC_TUTOR_WEB_SEARCH=true uv run python -c "from pydantic_tutor.settings import Settings; print(Settings(_env_file='lesson-settings.env').web_search)"
```

> [!TIP]
> **Check:** this time it prints `True`.

**What it means**

The class hasn't changed. Its Python default is `True`; the file
overrides it with `false`; and a variable supplied for that command
takes precedence over the file. That's the useful idea: **one setting,
several possible sources, a predictable result**. We're opting into the
file for this experiment with `_env_file`. In the next lesson we'll
configure `Settings` to read it routinely, including the credentials
needed by model providers.

Your real `.env` is excluded by the project's `.gitignore`; keep it
that way. You can delete `lesson-settings.env` after this experiment.
Don't put real credentials in the practice file. A file on your
machine is convenient configuration, not a secure secret store.

## Step 3: Assemble the agent and model from settings

> **Goal:** `build_app(settings=...)` returns the agent and the model,
> built from the settings.

We've seen where configuration values can come from. Now it's time to
make them change the running application. The agent needs the web-search
flag, while the chat loop needs a model object rather than a name.
We'll assemble those two pieces in `build_app`, so the loop doesn't
need to know how either was chosen.

First we'll **define** a new `build_chat_agent` function. Instead of
constructing the agent immediately with web search hard-coded, the
function takes a `web_search` argument and builds the agent accordingly.
This is a *factory function*—one straightforward way to configure an
agent before using it. In the next file we'll **call** that function
from `build_app`. We're separating those two actions deliberately.
Replace `src/pydantic_tutor/chat/agent.py`:

```python
"""The agent the chat talks to.

No model here, as for every agent in this project: the chat passes one on
each run.

Built from these pieces:
- Web search: switched on by a setting.
"""

from pydantic_ai import Agent
from pydantic_ai.capabilities import AbstractCapability, WebSearch


def build_chat_agent(*, web_search: bool) -> Agent[None, str]:
    """Build the chat's agent."""
    capabilities: list[AbstractCapability[None]] = []
    if web_search:
        # Web search: the model may search when it needs to. Each search is
        # billed by the provider, on top of the tokens.
        capabilities.append(WebSearch())
    return Agent[None, str](
        name="chat",
        instructions="You are a helpful assistant. Answer briefly.",
        capabilities=capabilities,
    )


# Earlier demos import this name; the console chat builds its agent in build_app.
chat_agent = build_chat_agent(web_search=True)
```

The header lists the pieces the agent is built from, and each is marked in the code by name. Each lesson that changes this file adds one.

We have now defined the factory, but our running chat doesn't use it
yet. Next we'll call it from `build_app`, together with `infer_model`,
which creates the model from its name. That gives us a single place to
assemble the application's main pieces before the chat opens. This
assembly point is often called the **composition root**.

Create `src/pydantic_tutor/composition.py`:

```python
"""The composition root: the one place the app's objects are built.

`build_app` turns `Settings` (values) into the agent and the model (objects).
The model is built here, at startup, so an unknown provider or a missing API key
stops the app before the chat opens, not when the first message is sent.
"""

from dataclasses import dataclass

from pydantic_ai import Agent
from pydantic_ai.models import Model, infer_model

from pydantic_tutor.chat.agent import build_chat_agent
from pydantic_tutor.settings import Settings


@dataclass(frozen=True)
class App:
    """What the chat needs to run, built once from the settings."""

    agent: Agent[None, str]
    model: Model


def build_app(*, settings: Settings) -> App:
    """Build the agent and the model from `settings`.

    Raises `pydantic_ai.exceptions.UserError` if the model name's provider is
    unknown or its API key is not set.
    """
    return App(
        agent=build_chat_agent(web_search=settings.web_search),
        model=infer_model(model=settings.chat_model),
    )
```

`App` is a small frozen container for the agent and model. Notice
the two different jobs in `build_app`: `build_chat_agent` uses the
web-search choice to configure the agent; `infer_model` turns the model
name into an object. The latter also builds the provider and checks for
its key. That is why this is an application *assembly* step, rather than
another way to read settings.

Try assembling an app with Pydantic AI's keyless test model. No model
request will be sent:

```bash
uv run python -c "from pydantic_tutor.composition import build_app; from pydantic_tutor.settings import Settings; app = build_app(settings=Settings(chat_model='test')); print(type(app.model).__name__, app.agent.name)"
```

> [!TIP]
> **Check:** the output is `TestModel chat`.

The string `test` was a configuration value; `build_app` constructed
a `TestModel` and an agent. A real provider's missing key can now be
noticed at assembly time, before the conversation starts.

## Step 4: Update the chat loop to use the configured application

> **Goal:** run the same chat using the agent and model assembled from
> `Settings`.

Our chat already handles keyboard input, streams replies, and remembers
message history. None of that needs to change. What *does* change is
where the chat gets its agent and model.

Previously the loop imported a ready-made `chat_agent` and passed a
hard-coded model name on each run. Now we'll build both at the beginning of `main`, before the first
prompt. In the loop itself,
`app.agent` and `app.model` replace those two old values.

We'll also handle startup configuration errors here. If settings are
invalid or the model can't be built, the program should explain why
**before** showing the `you>` prompt. We aren't going to trigger an
error deliberately.

Replace `src/pydantic_tutor/chat/loop.py` with this complete version;
compare the configuration block at the beginning of `main`:

```python
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
```

Notice the division of work: the beginning of `main` assembles the
application once; the loop that follows only runs the conversation.
The project's existing `__main__.py` can still call `main()` as before.

Run the chat with the keyless test model. Since that model doesn't
support provider-hosted web search, disable web search for this run:

```bash
env -u OPENAI_API_KEY PYDANTIC_AI_NO_BANNER=1 PYDANTIC_TUTOR_CHAT_MODEL=test PYDANTIC_TUTOR_WEB_SEARCH=false uv run python -m pydantic_tutor
```

> [!TIP]
> **Check:** type `hello`, then an empty line. You should see:
>
> ```text
> Chat with the agent. An empty line ends it.
> you> hello
> agent> success (no tool calls)
>   (the model was sent 51 tokens)
> you>
> ```

**What it means**

We've changed the agent's capabilities and the model with configuration,
not by editing the conversation loop for each choice. The same chat
still streams responses and carries its message history from one turn
to the next.

Because assembly now happens before the chat opens, a missing key or
an unknown provider is also reported earlier. (A misspelled model with a
known provider, such as `openai:gpt-4o-minii`, isn't caught here: it
builds, and the mistake shows only when the first message is sent.) That's a useful benefit of
this design, although configuration—not failure handling—is the main
idea of the lesson.

## What you can now do

- **Separate values from objects:** `Settings` holds the chat's choices as
  typed values read from the environment, and `build_app` is the one place
  that builds the agent and model from them.
- **Make a bad setting or a missing key stop the app at startup:** a
  value that doesn't fit its type, an unknown provider and a missing key
  each end the app with a message before the chat opens.

## Next

The chat now reads configuration before it opens. In the next lesson you
give that place more to decide: which provider the name resolves to, and
how the provider gets its key.

---

*Tested with `pydantic-ai` 2.56.0 and `pydantic-ai-harness` 0.56.0.*
