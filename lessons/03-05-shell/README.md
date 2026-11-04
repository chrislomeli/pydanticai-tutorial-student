---
title: "Let the chat run a command"
description: "Give the continuing chat a limited shell, then use a real model to run the tinyjobs tests."
---

# Let the chat run a command

## Welcome

Last lesson you gave our chat a way to **look inside** the `tinyjobs`
project. It could list directories and read code, but it couldn't run
the project's tests.

Now we'll give the same agent a shell. You'll add one capability to
the chat, ask a real model to run `pytest`, and see it report what
happened. We'll also try a command the shell isn't allowed to run,
then discover why the agent can sometimes accomplish the same task
using its existing file tools.

By the end, you'll have a chat that can both **inspect code and run
selected commands**. More importantly, you'll know which part of your
application actually executes those commands.

> This lesson uses the real model and API key you configured earlier.
> Provider requests may incur charges. Model replies may differ from
> the recorded example.

## Concepts

### The shell is another capability

In 03-04, `LocalWorkspace` identified the folder and `FileSystem`
gave the model ways to read files there. `Shell`, from
`pydantic-ai-harness`, adds tools for running commands.

The distinction from Chapter 1 still holds. The **model requests** a
command. The **application executes** it through the shell capability
and returns the output. The model then interprets that output for you.
The model is not itself a Python process running `pytest`.

We can supply `allowed_commands` to limit which commands the shell
will start. For now we'll use `pytest`, `pwd`, `ls`, and `cat`.

### Two capabilities, two sets of rules

The shell's allowed-command list applies to **shell commands**, not
to `FileSystem` operations. An agent may be unable to run the shell
command `grep` but still search text through `FileSystem`'s search
tool.

Neither an allowlist nor the workspace folder creates a sandbox.
Shell commands execute locally with your account's permissions; even
an allowed program such as `pytest` can execute arbitrary project
code. Use only this disposable sample project, and don't let the
agent run commands against personal or sensitive files.

> [!WARNING]
> **This chat can read your API key.** The list checks only a command's
> first word, so `ls && cat ../../.env` runs. `FileSystem`'s secret
> patterns don't apply to the shell, so `cat` can read your project's
> `.env`, and whatever a tool reads is sent to the model's provider. For
> this tutorial, on your own computer, that's an accepted risk. Don't
> use this setup as it stands anywhere that matters.

## What you'll change

Just the agent file you edited in 03-04:

```text
src/pydantic_tutor/chat/agent.py    add Shell alongside FileSystem
```

Then you'll run the **existing chat**. No new demo program or
stand-in model is needed.

## Before you start

Continue in your existing `pydantic-tutor` project. From the
project root, refresh the dependencies and disposable workspace:

```bash
uv sync
uv run demos/fresh_copy.py
```

> [!TIP]
> **Check:** `work/tinyjobs` exists and your agent still has the
> `FileSystem` capability from 03-04. Keep your real API
> key and model configuration from the previous lessons.

> [!WARNING]
> **Already have `pydantic-tutor`? Do not run these clone commands.**
>
> Only if you're joining at this lesson or recovering without an existing
> project, run these commands from a parent directory where **no folder
> named `pydantic-tutor` exists**. Do not delete or overwrite your existing project: starting fresh
> would lose the work and settings you have added.
>
> ```bash
> git clone --branch 03-04-files/end --single-branch https://github.com/chrislomeli/pydanticai-tutorial-student.git pydantic-tutor
> cd pydantic-tutor
> uv sync
> ```
>
> This is a recovery option, not a step to repeat between lessons.

## Step 1: Give the chat a shell

> **Goal:** let the existing agent run a small selection of commands.

This adds `Shell` to the agent, and the commands it may start. Replace `src/pydantic_tutor/chat/agent.py` with:

```python
"""The agent the chat talks to.

No model here, as for every agent in this project: the chat passes one on
each run.

Built from these pieces:
- Workspace: the folder the tools work in.
- File tools: read and write files there.
- Shell tools: run a few commands there.
- Web search: switched on by a setting.
"""

from pathlib import Path

from pydantic_ai import Agent
from pydantic_ai.capabilities import AbstractCapability, LocalWorkspace, WebSearch
from pydantic_ai_harness import FileSystem, Shell

from tutor_given.paths import project_root

# File names the file tools may not touch. A precaution, not a complete
# data-security policy.
SECRET_PATTERNS = ("**/.env", "**/.env.*", "*.pem", "*.key", "**/secrets*")

# The commands the shell may start. A precaution, not a boundary: it checks
# only a command's first word, so `ls && cat .env` still runs.
SHELL_COMMANDS = ("pytest", "pwd", "ls", "cat")


def build_chat_agent(*, web_search: bool, workspace_root: Path) -> Agent[None, str]:
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
    ]
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
chat_agent = build_chat_agent(
    web_search=True, workspace_root=project_root() / "work" / "tinyjobs"
)
```

Notice that we haven't replaced `FileSystem`. The agent now has
both sets of tools available. We also haven't changed the chat loop
or model-building code.

## Step 2: Run the project's tests

> **Goal:** watch the agent execute a command and explain its result.

Run the same chat as before, with web search turned off:

```bash
PYDANTIC_TUTOR_WEB_SEARCH=false uv run python -m pydantic_tutor
```

Ask:

```text
Can you run pytest?
```

Here is a real conversation from the `tinyjobs` project:

![A conversation with the agent using its shell tools to investigate the tinyjobs project and run tests.](Pydantic%20AI%20agent%20session%402x.png)

*An actual agent session. The commands and responses shown are examples;
your model's wording and token usage may vary.*

In this session, `pytest` collected **151 tests: 150 passed and one
failed**. The agent reported the failing `test_cancel` test and
described its assertion. A failing test is still a successful
demonstration of the shell: the command ran, and the model interpreted
its output. We aren't fixing that test in this lesson.

> [!TIP]
> **Check:** your agent should be able to run `pytest` and tell
> you what happened. The exact test count and failure details can
> change as the sample project evolves. If it cannot run tests,
> check the `Shell` capability, its allowed commands, and that
> you're in the refreshed `tinyjobs` workspace.

You can experiment further:

- `Use the shell to run pwd. Where are you working?`
- `Use the shell to run ls. What files are here?`
- `Use the shell to run pytest and summarize the failing test.`

Each request is useful on its own. You don't need to reproduce the
transcript word for word.

### A boundary worth noticing

Now ask:

```text
Can you run the shell command grep tinyjobs README.md?
```

Because `grep` isn't in our shell's allowed-command list, the agent
should be unable to execute **that shell command**. But try a
different request:

```text
Can you find occurrences of "tinyjobs" in README.md using your file tools?
```

The agent may still succeed: `FileSystem` offers a separate way to
search files. The shell hasn't secretly acquired permission to run
`grep`; the model has another tool available for the task.

> [!TIP]
> **Check:** the explicit shell command `grep` should be refused,
> while file searching may still work. If the model reports something
> surprising, remember that its explanation is not a tool log. We
> would inspect the actual tool calls to verify exactly what ran.

**What happened?**

Your agent now has two capabilities with different responsibilities.
`FileSystem` lets the model request information about files.
`Shell` lets it request execution of allowed commands. **Your
application**, not the model, carries out both operations.

The shell's list helps avoid accidental commands, but it doesn't
protect your computer from everything an allowed command can do.
We'll return to stronger controls when we introduce human approval
and isolation later in the tutorial.

## What you can now do

You extended the same chat you started in Chapter 2: it can inspect
project files, run selected commands, and explain real test results.
You also saw why a restriction on one capability is not necessarily
a restriction on the entire agent.

## Next

We now have a chat that can read a project and run its tests. In
Chapter 4, we'll build our own tools for the agent, giving it access
to information and actions specific to our application.

---

*Tested with `pydantic-ai` 2.56.0 and `pydantic-ai-harness` 0.56.0.*
