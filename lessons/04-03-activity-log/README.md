---
title: "Put the activity log in the chat"
description: "Connect the activity log to the chat through the composition root, choose what it records from Settings, and read the log of a real conversation."
---

# Put the activity log in the chat

## Welcome

This lesson adds the activity log we created in the previous lesson to our chat. We'll connect the hooks from the previous lesson to the application and choose which events to record.

Last time, the log ran only in a demo with a stand-in model. Now we will integrate these hooks into our chat, and every conversation you have with the chat leaves a record, and you'll read what a real model did, including something it was refused.

By the end, you'll be able to:

- Say why a hook is the right place for a rule only your application knows.
- Choose which hooks an agent gets from `Settings`.
- Read an activity log, and tell a call that finished from one that didn't.

About 15 minutes. Steps 1-6 need no API key. Step 7 uses your OpenAI key.

## Concepts

### Your application decides what to log

Our chat can read files and run commands. We want to record what it attempted and what finished. These hooks write that record to a local file.

The activity log helps us understand a run. It isn't a security audit: settings can disable it, and other programs can change the file.

### Settings control which hooks run

`Hooks(...)` accepts one function for each event. We can omit events we don't want to record. We'll make that choice through `Settings`.

### The log records attempted calls

`before_tool_execute` records an attempt before a tool runs. `after_tool_execute` records a completed call. If a call doesn't finish, the log may contain only the first event.

## What you'll build

- `src/pydantic_tutor/settings.py`, changed: where the log goes, and what it records
- `src/pydantic_tutor/chat/agent.py`, changed: the chat's agent can take the log
- `src/pydantic_tutor/composition.py`, changed: `build_app` builds the log and hands it over
- `demos/watch_run.py`, changed: it uses the same route as the chat

The steps:

1. Add log settings.
2. Add the log to the chat agent.
3. Pass the log through `build_app`.
4. Update the demo.
5. Check your work.
6. Try it.
7. Read a real chat's activity log.

## Before you start

Carry on in your `pydantic-tutor` project from the last lesson.

> [!TIP]
> **Check:** run `uv run demos/fresh_copy.py`, then `uv run demos/watch_run.py`. Its last line is `{"hook":"before_model_request","step":2,"messages":3,"tools":14}`.

> [!WARNING]
> **Joining here, or has your project gone wrong?** If your project works, skip this. To refresh a broken project, delete its folder first: you lose your work and anything else in it, including your `.env` and its API keys.
>
> ```bash
> git clone --branch 04-02-hooks/end --single-branch https://github.com/chrislomeli/pydanticai-tutorial-student.git pydantic-tutor
> cd pydantic-tutor
> uv sync
> ```
>
> Then run the check above.

So far `watch_run.py` sets up the log for one run. The chat gets everything from `build_app`, so that's where the log belongs. We'll also make the log configurable: a setting chooses which events it records.

## Step 1: Add log settings

In `src/pydantic_tutor/settings.py`, import `HookName` after the `pydantic_settings` import:

```python
from pydantic_tutor.capabilities.activity_log import HookName
```

Add these fields after `workspace_root`. `activity_log` holds `HookName` values, so a typo in the setting stops the app at startup instead of quietly recording nothing:

```python
    # The activity log's file. Beside the workspace and not in it, so the file
    # tools can't reach it. `work/` is already ignored by git.
    activity_log_path: Path = project_root() / "work" / "activity.jsonl"
    # Which hooks the activity log records. An empty list turns it off.
    activity_log: frozenset[HookName] = frozenset(HookName)
```

> **Check your work:** compare your `src/pydantic_tutor/settings.py` with the finished version in [Step 5](#step-5-check-your-work).

## Step 2: Add the log to the chat agent

Four small changes in `src/pydantic_tutor/chat/agent.py`. First, add `Hooks` to the `pydantic_ai.capabilities` import:

```python
from pydantic_ai.capabilities import (
    AbstractCapability,
    Hooks,
    LocalWorkspace,
    Toolset,
    WebSearch,
)
```

Add `activity_log` to the signature. The parameter is optional, so earlier scripts still work without a log:

```python
def build_chat_agent(
    *, web_search: bool, workspace_root: Path, activity_log: Hooks[None] | None = None
) -> Agent[None, str]:
```

In the same function, add this after the `if web_search:` block. The log goes last, so it records what the other capabilities have left:

```python
    if activity_log is not None:
        # Activity log: last, so it records what the other capabilities have
        # left.
        capabilities.append(activity_log)
```

Optionally, add it to the docstring's piece list, after "Web search":

```python
- Web search: switched on by a setting.
- Activity log: a record of the run, if one is given.
```

> **Check your work:** compare your `src/pydantic_tutor/chat/agent.py` with the finished version in [Step 5](#step-5-check-your-work).

## Step 3: Pass the log through `build_app`

In `src/pydantic_tutor/composition.py`, add this import:

```python
from pydantic_tutor.capabilities.activity_log import build_activity_hooks
```

Then pass the log to the agent, inside `build_app`:

```python
        agent=build_chat_agent(
            web_search=settings.web_search,
            workspace_root=settings.workspace_root,
            activity_log=build_activity_hooks(
                path=settings.activity_log_path, hooks=settings.activity_log
            ),
        ),
```

> **Check your work:** compare your `src/pydantic_tutor/composition.py` with the finished version in [Step 5](#step-5-check-your-work).

## Step 4: Update the demo

Make `demos/watch_run.py` take the same route. Remove the `project_root` import and the two lines that set up `LOG_PATH`, along with their comment. Replace everything from `chat = build_chat_agent` to the end of the file with:

```python
settings = Settings()
settings.activity_log_path.unlink(missing_ok=True)

chat = build_chat_agent(
    web_search=False,
    workspace_root=settings.workspace_root,
    activity_log=build_activity_hooks(
        path=settings.activity_log_path, hooks=settings.activity_log
    ),
)
chat.run_sync(
    user_prompt="What does this project do?",
    model=TestModel(call_tools=["list_modules", "write_file"]),
)
print(
    settings.activity_log_path.read_text()
    if settings.activity_log_path.exists()
    else "(nothing recorded)"
)
```

> **Check your work:** compare your `demos/watch_run.py` with the finished version in [Step 5](#step-5-check-your-work).

## Step 5: Check your work

<details>
<summary>Show the finished files</summary>

`src/pydantic_tutor/settings.py`:

```python
"""Every value the app is configured with, typed and checked at startup.

Values only: strings and flags read from the environment. The objects built
from them are in `composition.py`.
"""

from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from pydantic_tutor.capabilities.activity_log import HookName
from pydantic_tutor.model import CHAT_MODEL
from tutor_given.paths import project_root


class Settings(BaseSettings):
    """Read from variables named `PYDANTIC_TUTOR_<FIELD>`; frozen once created."""

    model_config = SettingsConfigDict(
        env_prefix="PYDANTIC_TUTOR_",
        frozen=True,
        # Read here, into these fields only: a key in `.env` never enters
        # the process environment. A shell variable wins over `.env`.
        env_file=project_root() / ".env",
        # `OPENAI_API_KEY=` with nothing after it counts as unset.
        env_ignore_empty=True,
        # `.env` may hold variables for other tools; skip them.
        extra="ignore",
    )

    # A `<provider>:<model>` string, the name of the model: not the model.
    chat_model: str = CHAT_MODEL
    # Whether the chat may search the web. Each search is billed.
    web_search: bool = True
    # The folder the chat works in. `demos/fresh_copy.py` makes the default.
    workspace_root: Path = project_root() / "work" / "tinyjobs"
    # The activity log's file. Beside the workspace and not in it, so the file
    # tools can't reach it. `work/` is already ignored by git.
    activity_log_path: Path = project_root() / "work" / "activity.jsonl"
    # Which hooks the activity log records. An empty list turns it off.
    activity_log: frozenset[HookName] = frozenset(HookName)
    # One key per provider the composition root can build. SecretStr:
    # printing the settings shows `**********`, never the key.
    openai_api_key: SecretStr | None = Field(
        default=None, validation_alias="OPENAI_API_KEY"
    )
    anthropic_api_key: SecretStr | None = Field(
        default=None, validation_alias="ANTHROPIC_API_KEY"
    )
```

`src/pydantic_tutor/chat/agent.py`:

```python
"""The agent the chat talks to.

No model here, as for every agent in this project: the chat passes one on
each run.

Built from these pieces:
- Workspace: the folder the tools work in.
- File tools: read and write files there.
- Shell tools: run a few commands there.
- Toolset: the workspace's own tools.
- Web search: switched on by a setting.
- Activity log: a record of the run, if one is given.
"""

from pathlib import Path

from pydantic_ai import Agent
from pydantic_ai.capabilities import (
    AbstractCapability,
    Hooks,
    LocalWorkspace,
    Toolset,
    WebSearch,
)
from pydantic_ai_harness import FileSystem, Shell

from pydantic_tutor.tools.workspace_toolset import build_workspace_toolset
from tutor_given.paths import project_root

# File names the file tools may not touch. A precaution, not a complete
# data-security policy.
SECRET_PATTERNS = ("**/.env", "**/.env.*", "*.pem", "*.key", "**/secrets*")

# The commands the shell may start. A precaution, not a boundary: it checks
# only a command's first word, so `ls && cat .env` still runs.
SHELL_COMMANDS = ("pytest", "pwd", "ls", "cat")


def build_chat_agent(
    *, web_search: bool, workspace_root: Path, activity_log: Hooks[None] | None = None
) -> Agent[None, str]:
    """Build the chat's agent."""
    capabilities: list[AbstractCapability[None]] = [
        # Workspace: a place for tools to work in. It adds no tool: they use
        # `ctx.workspace`.
        LocalWorkspace(working_dir=workspace_root),
        # File tools: read and write files in that workspace, minus the secret
        # patterns.
        FileSystem(denied_patterns=SECRET_PATTERNS),
        # Shell tools: run the allowed commands in that workspace.
        Shell(allowed_commands=SHELL_COMMANDS),
        # Toolset: the workspace's own tools, built for this folder.
        Toolset(toolset=build_workspace_toolset(root=workspace_root)),
    ]
    if web_search:
        # Web search: the model may search when it needs to. Each search is
        # billed by the provider, on top of the tokens.
        capabilities.append(WebSearch())
    if activity_log is not None:
        # Activity log: last, so it records what the other capabilities have
        # left.
        capabilities.append(activity_log)
    return Agent[None, str](
        name="chat",
        instructions="You are a helpful assistant. Answer briefly.",
        capabilities=capabilities,
    )


# Earlier demos import this name; the console chat builds its agent in build_app.
chat_agent = build_chat_agent(
    web_search=True, workspace_root=project_root() / "work" / "tinyjobs"
)
```

`src/pydantic_tutor/composition.py`:

```python
"""The composition root: the one place the app's objects are built.

`build_app` turns `Settings` (values) into the agent and the model (objects).
The model is built here, at startup, so an unknown provider or a missing API key
stops the app before the chat opens, not when the first message is sent.
"""

from dataclasses import dataclass

from pydantic_ai import Agent
from pydantic_ai.exceptions import UserError
from pydantic_ai.models import Model, infer_model

from pydantic_tutor.capabilities.activity_log import build_activity_hooks
from pydantic_tutor.chat.agent import build_chat_agent
from pydantic_tutor.providers import ProviderFactory, build_provider
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
    if not settings.workspace_root.is_dir():
        raise UserError(
            f"PYDANTIC_TUTOR_WORKSPACE_ROOT is {str(settings.workspace_root)!r}, "
            "which is not a folder. Make the default with "
            "`uv run demos/fresh_copy.py`, or set the variable to another folder."
        )
    return App(
        agent=build_chat_agent(
            web_search=settings.web_search,
            workspace_root=settings.workspace_root,
            activity_log=build_activity_hooks(
                path=settings.activity_log_path, hooks=settings.activity_log
            ),
        ),
        model=build_model(settings=settings),
    )


def build_model(
    *, settings: Settings, provider_factory: ProviderFactory | None = None
) -> Model:
    """Build the model `settings.chat_model` names, its provider holding its key.

    `infer_model` picks the model class from the string, and asks
    `provider_factory` for the provider, by the part before the colon. The
    default is `build_provider`; pass another function to replace it.
    """
    return infer_model(
        model=settings.chat_model,
        # `infer_model` calls this with one argument, the provider's name.
        provider_factory=provider_factory
        or (lambda name: build_provider(name=name, settings=settings)),
    )
```

`demos/watch_run.py`:

```python
"""Run the chat's agent on a stand-in model, and print its activity log.

Needs no API key; run `demos/fresh_copy.py` first. It starts the log again,
so the file holds this run only.
"""

from pydantic_ai.models.test import TestModel

from pydantic_tutor.capabilities.activity_log import build_activity_hooks
from pydantic_tutor.chat.agent import build_chat_agent
from pydantic_tutor.settings import Settings

settings = Settings()
settings.activity_log_path.unlink(missing_ok=True)

chat = build_chat_agent(
    web_search=False,
    workspace_root=settings.workspace_root,
    activity_log=build_activity_hooks(
        path=settings.activity_log_path, hooks=settings.activity_log
    ),
)
chat.run_sync(
    user_prompt="What does this project do?",
    model=TestModel(call_tools=["list_modules", "write_file"]),
)
print(
    settings.activity_log_path.read_text()
    if settings.activity_log_path.exists()
    else "(nothing recorded)"
)
```

</details>

## Step 6: Try it

```bash
uv run demos/fresh_copy.py
uv run demos/watch_run.py
```

> [!TIP]
> **Check:** ignoring any startup banner, you should see the same lines as at the end of the last lesson:
>
> ```text
> {"hook":"before_model_request","step":1,"messages":1,"tools":14}
> {"hook":"before_tool_execute","step":1,"tool":"list_modules"}
> {"hook":"before_tool_execute","step":1,"tool":"write_file","target":"a"}
> {"hook":"after_tool_execute","step":1,"tool":"list_modules","result_chars":137}
> {"hook":"after_tool_execute","step":1,"tool":"write_file","result_chars":49}
> {"hook":"before_model_request","step":2,"messages":3,"tools":14}
> ```

Now try filtering. The setting is a JSON list, so it needs the quotes:

```bash
uv run demos/fresh_copy.py
PYDANTIC_TUTOR_ACTIVITY_LOG='["before_tool_execute"]' uv run demos/watch_run.py
```

> [!TIP]
> **Check:** you should see only the two tool lines:
>
> ```text
> {"hook":"before_tool_execute","step":1,"tool":"list_modules"}
> {"hook":"before_tool_execute","step":1,"tool":"write_file","target":"a"}
> ```
>
> Then turn it off:
>
> ```bash
> uv run demos/fresh_copy.py
> PYDANTIC_TUTOR_ACTIVITY_LOG='[]' uv run demos/watch_run.py
> ```
>
> You should see `(nothing recorded)`.

**What it means**

The hooks arrive the way the app's other capabilities do. They're last in the list, and `build_app` now gives them to the chat too. The setting decides which hooks fire: with an empty list, the agent has nothing to call.

## Step 7: Read a real chat's activity log

This step uses your API key, and the model decides what it does, so your lines will differ from anyone else's. Start the chat:

```bash
uv run demos/fresh_copy.py
uv run python -m pydantic_tutor
```

Ask it something that takes a few tools, then ask for something the chat's shell doesn't allow:

```text
Which module in this project handles retries? Read it and summarize it.
```

```text
Please run the command rm -rf build, then tell me what happened.
```

End the chat with an empty line, then open `work/activity.jsonl` in your editor.

> [!TIP]
> **Check:** you should see a `before_model_request` line for each trip to the model, and tool lines between them. `step` starts again at 1 for your second question, because each question is a new run. `messages` keeps growing across questions, because each run is sent the chat so far.
>
> If the model tried the command, you'll find a line like this, with no `after_tool_execute` for that call:
>
> ```text
> {"hook":"before_tool_execute","step":1,"tool":"run_command","target":"rm -rf build"}
> ```
>
> The model may decline without trying. Then there's no such line, and the log is still telling the truth.

**What it means**

A `before_tool_execute` with no matching `after_tool_execute` means the call didn't finish. The shell refused this one, because `rm` isn't an allowed command. The refusal went back to the model, and no `after_tool_execute` fired.

The log can't tell you why a call didn't finish. A crash inside a tool could leave the same gap. It does tell you that the agent tried, and that's something only a hook sitting in the run could record.

There's one more thing a hook being part of the run means. If the log can't be written, the error stops the run before the tool starts. For a log you'd rather lose than stop a chat for, you'd catch that error in `write`.

## What you can now do

- **Hooks hold what only your application knows.** The chat now keeps a record of what it did on your machine.
- **A setting can decide which hooks an agent gets.** `Settings` chooses how much the log records, or turns it off.
- **The log shows what was tried.** You can tell a call that finished from one that didn't.

## Next

Our log only watches. Next we'll write hooks that change a run, and see why the order of the `capabilities` list starts to matter.

---

*Tested with `pydantic-ai` 2.56.0 and `pydantic-ai-harness` 0.56.0.*
