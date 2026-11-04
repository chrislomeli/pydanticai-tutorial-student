---
title: "What a capability is"
description: "See what each capability in the chat's list brings, then turn the token-removing hook into a capability of your own that also tells the model what it did."
---

# Build a custom capability

## Welcome

This lesson is about **custom capabilities**: classes we write to give an agent new behavior. You've been adding them to a list since Chapter 2, and `Hooks` was one too. Now we'll build one of our own.

Last time we left a flaw. `remove_tokens` puts `[token removed]` into a tool's result, and nothing has told the model what that means. Fixing it takes a hook and an instruction that travel together, and that's what a capability you write yourself is for. We'll look at what's already in the chat's list, then turn the hook into a class.

By the end, you'll be able to:

- Say what a capability is, and name what the chat's capabilities bring.
- Write a capability with a setting, an instruction and a hook.
- Choose between `Hooks(...)` and a class of your own.

About 15 minutes. Steps 1 and 2 need no API key. Step 3 uses your OpenAI key.

## Concepts

### Capabilities add behavior to an agent

The library's base class, `AbstractCapability`, has a method for each thing a capability can bring. Each one does nothing by default, and a capability overrides the ones it needs:

| A capability can bring | By overriding |
|---|---|
| Instructions | `get_instructions()` |
| Tools | `get_toolset()` |
| A workspace | `get_workspace()` |
| Hooks | `before_tool_execute()`, `after_tool_execute()` and the rest |

`Shell`, `FileSystem`, `WebSearch` and `Hooks` are all subclasses of it. To the agent, they're the same kind of thing, so the `capabilities` list can hold them side by side. Capabilities can also supply model settings; we'll return to that later.

### A class keeps related behavior together

Use `Hooks(...)` when a function can stand on its own. We'll keep `refuse_chained_commands` that way.

`remove_tokens` also needs to explain the replacement text to the model. We'll put the hook and that instruction in one class. The class will hold the token pattern as a setting.

## What you'll build

- `src/pydantic_tutor/capabilities/redact_tokens.py`, new: `RedactTokens` (our name, not the library's)
- `src/pydantic_tutor/capabilities/guards.py`, changed: `remove_tokens` moves out
- `src/pydantic_tutor/chat/agent.py`, changed: the chat takes `RedactTokens`

Given: `demos/show_capabilities.py`, which lists what each of the chat's capabilities brings. You don't need to read it.

The steps:

1. See what's in the chat's list.
2. Turn the hook into a capability.
3. Try it in a real chat.

## Before you start

Carry on in your `pydantic-tutor` project from the last lesson.

> [!TIP]
> **Check:** run `uv run demos/fresh_copy.py`, then `uv run demos/guard_check.py`. Its output includes `runner_token = "[token removed]"`.

> [!WARNING]
> **Joining here, or has your project gone wrong?** If your project works, skip this. To refresh a broken project, delete its folder first: you lose your work and anything else in it, including your `.env` and its API keys.
>
> ```bash
> git clone --branch 04-04-guards/end --single-branch https://github.com/chrislomeli/pydanticai-tutorial-student.git pydantic-tutor
> cd pydantic-tutor
> uv sync
> ```
>
> Then run the check above.

## Step 1: Inspect the chat's capabilities

> **Goal:** see that the things you've been adding are all one kind of thing.

```bash
uv run demos/show_capabilities.py
```

> [!TIP]
> **Check:** ignoring any startup banner, you should see:
>
> ```text
> LocalWorkspace: a workspace
> FileSystem: tools, hooks
> Shell: tools, hooks
> Toolset: tools
> Hooks: hooks
> ```

**What it means**

Each line is one entry in the chat's list, and each brings something different. `LocalWorkspace` brings only a workspace. `Toolset` brings only tools. `FileSystem` and `Shell` bring tools and hooks of their own. `Hooks` is the `guards` you wrote last time, and brings hooks only.

None of them brings instructions yet. That's what we'll add.

## Step 2: Build the redaction capability

> **Goal:** one object that removes tokens and tells the model it does.

We'll create `RedactTokens` to keep the redaction hook and its explanation together. The pattern becomes a setting on the class.

Create `src/pydantic_tutor/capabilities/redact_tokens.py`. Three Python features matter here:

- `@dataclass(kw_only=True)` makes each field a keyword argument: `RedactTokens(pattern=...)`.
- `__post_init__` runs at the end of the constructor, so a bad pattern stops the app at startup, not in the middle of a run.
- `[DepsT]` means "for whatever deps type the agent has". This capability reads none, so any agent can take it.
- `after_tool_execute` is the same hook as before, now a method. It has to be `async`. A plain `def` fails the run with a `TypeError` at the first tool call, and a method with any other name is never called, with no error at all.

```python
"""Remove tokens from tool results, and tell the model that it happens.

A safety net for mistakes, not a security boundary: it matches one known
format, not every secret.
"""

import re
from dataclasses import dataclass, field
from typing import Any

from pydantic_ai import RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.tools import ToolDefinition

REPLACEMENT = "[token removed]"


@dataclass(kw_only=True)
class RedactTokens[DepsT](AbstractCapability[DepsT]):
    """Replace anything matching `pattern` in tool results, and say so."""

    # A regular expression for the tokens to remove.
    pattern: str
    compiled: re.Pattern[str] = field(init=False)

    def __post_init__(self) -> None:
        """Compile the pattern now, so a bad one fails when the app starts."""
        self.compiled = re.compile(self.pattern)

    def get_instructions(self) -> str:
        """Tell the model what the replacement text means."""
        return (
            f"{REPLACEMENT} in a tool result marks a secret removed on purpose. "
            "The file isn't broken. Don't try to recover the secret."
        )

    async def after_tool_execute(
        self,
        ctx: RunContext[DepsT],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
        result: object,
    ) -> object:
        """Return the tool's result with the tokens removed."""
        if isinstance(result, str):
            return self.compiled.sub(REPLACEMENT, result)
        return result
```

Now `guards.py` only needs the refusal. In `src/pydantic_tutor/capabilities/guards.py`, delete the `remove_tokens` function, the `TOKEN` constant and its comment, and the `import re` line. Then replace the last lines of the file, which make `guards`:

```python
guards: Hooks[None] = Hooks(before_tool_execute=refuse_chained_commands)
```

And replace the module's docstring, which no longer describes it:

```python
"""A rule only this application knows, as a hook on the chat's shell commands.

A safety net for mistakes, not a security boundary: `pytest` still runs any code
it is given.
"""
```

In `src/pydantic_tutor/chat/agent.py`, add this import after the `guards` import:

```python
from pydantic_tutor.capabilities.redact_tokens import RedactTokens
```

Add the token format after `SHELL_COMMANDS`:

```python
# The shape of this project's tokens.
TOKEN_PATTERN = r"tj-[A-Za-z0-9-]+"
```

And add the capability after `guards` in the list:

```python
        guards,
        RedactTokens(pattern=TOKEN_PATTERN),
    ]
```

And name it in the docstring's piece list, after "Guards":

```python
- Guards: our rules for tool calls.
- Token redaction: secrets removed before the model sees them.
```

```bash
uv run demos/show_capabilities.py
uv run demos/fresh_copy.py
uv run demos/guard_check.py
uv run demos/show_chat_request.py
```

> [!TIP]
> **Check:** `show_capabilities.py` now ends with these two lines:
>
> ```text
> Hooks: hooks
> RedactTokens: instructions, hooks
> ```
>
> `guard_check.py` prints exactly what it did before, including `runner_token = "[token removed]"`.
>
> `show_chat_request.py` prints the tools, then these instructions:
>
> ```text
>   instructions: "You are a helpful assistant. Answer briefly.\n\n[token removed] in a tool result marks a secret removed on purpose. The file isn't broken. Don't try to recover the secret.\n\nYour workspace holds the Python project tinyjobs."
> ```

**What it means**

The tokens are removed just as before, and the model is now told what the replacement means. Both came from one line in the list. `RedactTokens` is the first entry that brings instructions and hooks together, and a second agent could take the whole thing with that one line.

`guards` stayed a `Hooks`, because it's one function with nothing to say to the model. That's the choice from Concepts: reach for a class when pieces have to travel together.

<details>
<summary>Optional: compare your finished files</summary>

`src/pydantic_tutor/capabilities/guards.py`:

```python
"""A rule only this application knows, as a hook on the chat's shell commands.

A safety net for mistakes, not a security boundary: `pytest` still runs any code
it is given.
"""

from typing import Any

from pydantic_ai import RunContext
from pydantic_ai.capabilities import Hooks
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.tools import ToolDefinition

# What joins one shell command to another, or runs one inside another.
CHAINING = ("&&", "||", ";", "|", "`", "$(", "\n")


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


guards: Hooks[None] = Hooks(before_tool_execute=refuse_chained_commands)
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
- Guards: our rules for tool calls.
- Token redaction: secrets removed before the model sees them.
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
from pydantic_tutor.capabilities.redact_tokens import RedactTokens
from pydantic_tutor.tools.workspace_toolset import build_workspace_toolset
from tutor_given.paths import project_root

# File names the file tools may not touch. A precaution, not a complete
# data-security policy.
SECRET_PATTERNS = ("**/.env", "**/.env.*", "*.pem", "*.key", "**/secrets*")

# The commands the shell may start. A precaution, not a boundary: it checks
# only a command's first word, so `ls && cat .env` still runs.
SHELL_COMMANDS = ("pytest", "pwd", "ls", "cat")

# The shape of this project's tokens.
TOKEN_PATTERN = r"tj-[A-Za-z0-9-]+"


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
        RedactTokens(pattern=TOKEN_PATTERN),
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

## Step 3: Test redaction in the chat

> **Goal:** see whether a real model understands the replacement.

This step uses your API key, and the model decides what it says. `guard_check.py` adds `deploy.toml` to the workspace copy, so run it first and don't reset the copy afterwards:

```bash
uv run demos/fresh_copy.py
uv run demos/guard_check.py
uv run python -m pydantic_tutor
```

```text
Read deploy.toml and tell me what's in it.
```

> [!TIP]
> **Check:** the answer doesn't contain `tj-FAKE`. The model should describe the value as removed, rather than call the file broken. It's the model's wording, so yours will differ.

**What it means**

The hook guarantees what the model is sent, and that part isn't up to the model. The instruction only asks, and a real model's response to it can vary. That's the difference between code and a prompt, and a capability can carry both.

## What you can now do

- **A capability is one object that gives an agent part of what it does.** You can say what the chat's capabilities bring.
- **`Hooks(...)` is enough for a function on its own.** A class is for when a hook needs an instruction or a setting to go with it.

## Next

`RedactTokens` brings an instruction and a hook. Next we'll write a capability with no hooks at all, one that moves the chat's prompt out of Python and into a file.

---

*Tested with `pydantic-ai` 2.56.0 and `pydantic-ai-harness` 0.56.0.*
