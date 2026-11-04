---
title: "Hooks that change a run"
description: "Write hooks that refuse a chained shell command and remove this project's tokens from tool results, and see why their place in the list decides what the activity log records."
---

# Change tool calls with hooks

## Welcome

This lesson uses **hooks to change a run**. Our earlier hooks observed events and returned their inputs unchanged. Now we'll use hooks to refuse some tool calls and change some tool returns.

In Chapter 3, we allowed selected shell commands, but the check looked only at the first word. A command could therefore chain another operation onto an allowed command. We'll demonstrate that gap in a disposable workspace, then write one hook to refuse chained commands and another to remove our practice tokens from tool returns. Finally, we'll adjust the activity log so it records refused calls too.

By the end, you'll be able to:

- Change what the model is sent by returning something else from a hook.
- Refuse a tool call from a hook, and say what the model sees.
- Choose a hook's place in the list by what it needs to see.

About 20 minutes. Steps 1 to 3 need no API key. Step 4 uses your OpenAI key.

## Concepts

### Hooks can change or refuse tool calls

The run carries on with whatever a hook returns. So a `before_tool_execute` hook can return different arguments, and an `after_tool_execute` hook can return a different result. The model then sees the changed version and nothing else.

A hook can also refuse. If `before_tool_execute` raises `ModelRetry`, the tool doesn't run, and the model gets your message and can try something else. `ModelRetry` is the same exception you raised from a tool earlier in the series. Any other exception stops the whole run.

These two hooks hold rules only this application knows: which shell commands we consider safe, and what our tokens look like. The library ships general versions you'd reach for first. `ToolOutputLimits` shortens oversized results, and `ToolGuardrail` runs checks you supply on a tool's arguments and results.

### Hook order changes what gets recorded

For each tool call, `before_` hooks run in list order. The tool runs next. Then `after_` hooks run in reverse order.

So a hook sees what the hooks before it have done. A `before_` hook first in the list sees every attempt, including ones a later hook will refuse. An `after_` hook first in the list runs last, so it sees the result in its final form.

## What you'll build

- `src/pydantic_tutor/capabilities/guards.py`, new: the two hooks, as one capability
- `src/pydantic_tutor/chat/agent.py`, changed: the chat takes them, and the activity log moves
- `demos/guard_check.py`, new: runs the chat's agent on a given model that reads a token and asks for a chained command

Given, in `tutor_given/stand_ins.py`: `reads_a_token_then_chains_a_command()`. It's a model that asks for those two things, and prints what comes back.

The steps:

1. See a secret reach the model.
2. Write the two hooks.
3. Move the activity log to see every attempt.
4. Try it in a real chat.

## Before you start

Carry on in your `pydantic-tutor` project from the last lesson.

> [!TIP]
> **Check:** run `uv run demos/fresh_copy.py`, then `uv run demos/watch_run.py`. Its last line is `{"hook":"before_model_request","step":2,"messages":3,"tools":14}`.

> [!WARNING]
> **Joining here, or has your project gone wrong?** If your project works, skip this. To refresh a broken project, delete its folder first: you lose your work and anything else in it, including your `.env` and its API keys.
>
> ```bash
> git clone --branch 04-03-activity-log/end --single-branch https://github.com/chrislomeli/pydanticai-tutorial-student.git pydantic-tutor
> cd pydantic-tutor
> uv sync
> ```
>
> Then run the check above.

## Step 1: Demonstrate the unsafe command

> **Goal:** watch two ways a token reaches the model today.

We'll reproduce the gap in a disposable workspace before adding the guards. The script adds a `deploy.toml` holding a fake token to your workspace copy. That's a file nobody thought to add to `SECRET_PATTERNS`. The given model reads it, then asks to run `pytest --version && cat .env`.

Everything secret here is fake, including the `.env` that `fresh_copy.py` puts in the workspace copy. Your real `.env` isn't involved.

Create `demos/guard_check.py`:

```python
"""Run the chat's agent on a model that reads a token and chains a command.

Needs no API key. It resets the log, and adds a `deploy.toml` with a fake token
to the workspace copy; run `demos/fresh_copy.py` first.
"""

from pydantic_tutor.capabilities.activity_log import HookName, build_activity_hooks
from pydantic_tutor.chat.agent import build_chat_agent
from pydantic_tutor.settings import Settings
from tutor_given.stand_ins import reads_a_token_then_chains_a_command

settings = Settings()
(settings.workspace_root / "deploy.toml").write_text(
    'runner_token = "tj-FAKE-0000-not-a-real-secret"\n'
)
log_path = settings.activity_log_path
log_path.unlink(missing_ok=True)

chat = build_chat_agent(
    web_search=False,
    workspace_root=settings.workspace_root,
    activity_log=build_activity_hooks(
        path=log_path,
        hooks=frozenset({HookName.BEFORE_TOOL_EXECUTE, HookName.AFTER_TOOL_EXECUTE}),
    ),
)
chat.run_sync(
    user_prompt="Check the deployment settings.",
    model=reads_a_token_then_chains_a_command(),
)
print("the log:")
print(log_path.read_text(), end="")
```

```bash
uv run demos/fresh_copy.py
uv run demos/guard_check.py
```

> [!TIP]
> **Check:** ignoring any startup banner, you should see:
>
> ```text
>   model got, from read_file:
>     [deploy.toml | 1 lines | hash:e3b623df92da]
>     1	runner_token = "tj-FAKE-0000-not-a-real-secret"
>   model got, from run_command:
>     [stdout]
>     pytest 9.1.1
>     # FAKE VALUES. A fixture: the agent must not be able to read this file.
>     TINYJOBS_RUNNER_TOKEN=tj-FAKE-0000-not-a-real-secret
>     TINYJOBS_RUNNER_URL=https://runner.example.invalid
> the log:
> {"hook":"before_tool_execute","step":1,"tool":"read_file","target":"deploy.toml"}
> {"hook":"after_tool_execute","step":1,"tool":"read_file","result_chars":99}
> {"hook":"before_tool_execute","step":2,"tool":"run_command","target":"pytest --version && cat .env"}
> {"hook":"after_tool_execute","step":2,"tool":"run_command","result_chars":198}
> ```

**What it means**

Both fake secrets reached the model. `read_file` returned the token, because `deploy.toml` isn't on the deny list. `run_command` returned the fake `.env`, because `Shell` checks only the first word, `pytest`, and the rest of the line ran too. The file tools refuse `.env`, but the shell doesn't use them.

## Step 2: Add guards for commands and tokens

> **Goal:** the model no longer gets either secret.

Create `src/pydantic_tutor/capabilities/guards.py`:

- `refuse_chained_commands` raises `ModelRetry` when a command holds more than one command.
- `remove_tokens` returns the result with anything shaped like our tokens replaced. Results that aren't text pass through.

```python
"""Two rules only this application knows, as hooks on the chat's tool calls.

A safety net for mistakes, not a security boundary: `pytest` still runs any code
it is given, and `TOKEN` matches one known format, not every secret.
"""

import re
from typing import Any

from pydantic_ai import RunContext
from pydantic_ai.capabilities import Hooks
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.tools import ToolDefinition

# What joins one shell command to another, or runs one inside another.
CHAINING = ("&&", "||", ";", "|", "`", "$(", "\n")
# The shape of this project's tokens.
TOKEN = re.compile(r"tj-[A-Za-z0-9-]+")


async def refuse_chained_commands(
    ctx: RunContext[None],
    *,
    call: ToolCallPart,
    tool_def: ToolDefinition,
    args: dict[str, Any],
) -> dict[str, Any]:
    """Refuse a shell command that holds more than one command."""
    command = str(args.get("command", ""))
    if any(mark in command for mark in CHAINING):
        raise ModelRetry("Run one command at a time, with no chaining.")
    return args


async def remove_tokens(
    ctx: RunContext[None],
    *,
    call: ToolCallPart,
    tool_def: ToolDefinition,
    args: dict[str, Any],
    result: object,
) -> object:
    """Return the tool's result with this project's tokens removed."""
    if isinstance(result, str):
        return TOKEN.sub("[token removed]", result)
    return result


guards: Hooks[None] = Hooks(
    before_tool_execute=refuse_chained_commands,
    after_tool_execute=remove_tokens,
)
```

In `src/pydantic_tutor/chat/agent.py`, import it after the other `pydantic_tutor.capabilities` import:

```python
from pydantic_tutor.capabilities.guards import guards
```

Then add it to the end of the `capabilities` list in `build_chat_agent`, after the `Toolset(...)` entry:

```python
        Toolset(toolset=build_workspace_toolset(root=workspace_root)),
        # Our rules for tool calls: see the module's limits.
        guards,
    ]
```

And name it in the docstring's piece list, after "Activity log":

```python
- Activity log: a record of the run, if one is given.
- Guards: our rules for tool calls.
```

```bash
uv run demos/fresh_copy.py
uv run demos/guard_check.py
```

> [!TIP]
> **Check:** you should see:
>
> ```text
>   model got, from read_file:
>     [deploy.toml | 1 lines | hash:e3b623df92da]
>     1	runner_token = "[token removed]"
>   model got, from run_command:
>     Run one command at a time, with no chaining.
> the log:
> {"hook":"before_tool_execute","step":1,"tool":"read_file","target":"deploy.toml"}
> {"hook":"after_tool_execute","step":1,"tool":"read_file","result_chars":99}
> ```

**What it means**

The token is gone from the first result, and the second command never ran. The model got your refusal message instead, and could try again with one command. Both hooks changed the run without touching `FileSystem`, `Shell` or the tools themselves.

Now look at the log. It doesn't mention `run_command`, and it says `read_file` returned 99 characters, though the model got fewer. Our log has stopped telling the truth about this run, which is the next step's subject.

## Step 3: Move the activity log to see every attempt

> **Goal:** a log that records refused attempts, and the result the model got.

The log is last in the list, so its `before_` hook runs after the guard's. A refused call never reaches it. Its `after_` hook runs first on the way out, before `remove_tokens` has done anything, so it counts the original result.

We want the opposite on both counts, and one move does it. In `build_chat_agent`, find the lines that add the log, and replace them:

```python
    if activity_log is not None:
        # First: its `before_` hooks see every attempt, and its `after_` hooks
        # run last, so they see the result the model gets.
        capabilities.insert(0, activity_log)
```

```bash
uv run demos/fresh_copy.py
uv run demos/guard_check.py
```

> [!TIP]
> **Check:** what the model got is unchanged, and the log now reads:
>
> ```text
> {"hook":"before_tool_execute","step":1,"tool":"read_file","target":"deploy.toml"}
> {"hook":"after_tool_execute","step":1,"tool":"read_file","result_chars":84}
> {"hook":"before_tool_execute","step":2,"tool":"run_command","target":"pytest --version && cat .env"}
> ```

**What it means**

The refused command is now in the log, as a `before_tool_execute` with no `after_tool_execute`. The size is now 84, the result with the token removed. Nothing about the hooks changed, only their place.

The same hook at either end of the list sees different things, and nothing warns you. Whenever two hooks share the same point, it's worth asking which should see the other's work.

<details>
<summary>Optional: compare your finished <code>chat/agent.py</code></summary>

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
- Guards: our rules for tool calls.
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

from pydantic_tutor.capabilities.guards import guards
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
        # Our rules for tool calls: see the module's limits.
        guards,
    ]
    if web_search:
        # Web search: the model may search when it needs to. Each search is
        # billed by the provider, on top of the tokens.
        capabilities.append(WebSearch())
    if activity_log is not None:
        # First: its `before_` hooks see every attempt, and its `after_` hooks
        # run last, so they see the result the model gets.
        capabilities.insert(0, activity_log)
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

</details>

## Step 4: Try it in a real chat

> **Goal:** see a real model meet the guard.

This step uses your API key, and the model decides what it does.

```bash
uv run demos/fresh_copy.py
uv run python -m pydantic_tutor
```

```text
Please run pytest --version && cat .env and show me the output.
```

End the chat with an empty line, then open `work/activity.jsonl`.

> [!TIP]
> **Check:** if the model tried the command, the log has a `before_tool_execute` line with that command as its `target`, and no `after_tool_execute` for it. The model may then try `pytest --version` alone, or decline to try at all. Either way, `.env` shouldn't appear in its answer.

**What it means**

The model asks, and your program decides. A real model reads your refusal message and chooses what to do next, which is why the message says what to do instead.

## What you can now do

- **A hook can change a run or refuse.** Return something else to change what happens next, or raise `ModelRetry` to refuse a tool call.
- **A hook's place decides what it sees.** `before_` hooks run in list order, and `after_` hooks in reverse.

## Next

`remove_tokens` has a flaw: the model sees `[token removed]` and nobody has told it what that means. Fixing that takes something a hook can't do alone, an instruction to the model. Next we'll see what a capability really is, and turn this hook into one.

---

*Tested with `pydantic-ai` 2.56.0 and `pydantic-ai-harness` 0.56.0.*
