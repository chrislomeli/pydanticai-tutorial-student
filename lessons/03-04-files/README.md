---
title: "Let the chat read your project"
description: "Add file tools to the existing chat and use a real model to explore the tinyjobs project."
---

# Let the chat read your project

## Welcome

Welcome back. Last lesson you gave our chat a workspace: a disposable
copy of the small `tinyjobs` Python project. The agent knows *where*
to work, but it still has no way to look at a file there.

In this lesson, you'll add Pydantic AI's `FileSystem` capability to
**the chat you've been building**. We'll start the
chat with a real model, and ask it to read `tinyjobs` files. We'll explore the project directories and ask the agent to read and
explain a source file.

The goal is simple: **turn our chat into an assistant that can answer
questions about the project in front of it**, rather than relying only
on what the model already knows.

By the end, you'll be able to:

1. Add file access to the existing chat agent.
2. Explain what happens when the model requests a file tool and uses
   its result to answer you.

> This lesson uses a real model and a valid API key, as in Chapter 2. It
> makes provider requests, which may incur charges. Follow the checks below in your own project; exact model replies
> may differ.

## Concepts

### A tool the model can actually use

Remember the distinction from Chapter 1: the **model** can request a
tool, but it does not execute Python functions or open files itself.
Our **agent application** receives that request, runs the tool, and
passes the result back to the model. The model can then use the result
to answer the student.

Previously, you wrote your own tools. `FileSystem`, from
`pydantic-ai-harness`, supplies ready-made tools for reading, listing,
finding and searching files. We attach it as a *capability* alongside
`LocalWorkspace`. The workspace says where files live; `FileSystem`
makes file operations available to the model.

`FileSystem` also offers tools that change files. This lesson only asks
the model to read, and nothing yet asks you before it writes: a later
chapter adds that. We'll also refuse access to common secret-file names, because
**a file tool that only reads is still not private**.

### A useful boundary, not a sandbox

The file tools default to the workspace's working directory as their
root. In our app that's `work/tinyjobs`. They check paths, including
resolved links, and refuse attempts to read outside that root.

That's useful, but it is **not** the same as isolating an agent from
your computer. `LocalWorkspace` runs with your account's access,
and other capabilities—especially a shell—can have different rules.
We'll explore that distinction later. For this lesson, work only with
the disposable `tinyjobs` copy, only ask it to read, and do not
ask the agent to inspect personal or sensitive files.

## What you'll change

Just one existing file:

```text
src/pydantic_tutor/chat/agent.py    add FileSystem to our chat
```

Then we'll run the **existing chat**, not a new demonstration agent.
The plan is: add the capability, explore the workspace, and read real
source files with the agent.

## Before you start

Continue in the `pydantic-tutor` project you've been building through
Chapter 3. From its root (the folder containing `pyproject.toml`), make
sure your dependencies are in place:

```bash
uv sync
```

You already have the workspace setup from 03-03, and this lesson needs
no new starter files. Refresh the disposable `tinyjobs` copy:

```bash
uv run demos/fresh_copy.py
```

> [!TIP]
> **Check:** the command reports a fresh copy under
> `work/tinyjobs`. It includes a fake `.env` file used in earlier
> demonstrations. We'll exclude secret-like files from the new tools.

You'll also need the **real API key** you configured in Chapter 3, in
the project's own `.env` or your shell, and a real model selected
through `PYDANTIC_TUTOR_CHAT_MODEL`. Do not use the `test` model for
this lesson: we want an actual model choosing file operations.

> [!WARNING]
> **Already have `pydantic-tutor`? Do not run these clone commands.**
>
> Only if you're joining at this lesson or recovering without an existing
> project, run these commands from a parent directory where **no folder
> named `pydantic-tutor` exists**. Keep using your current project if you
> have one; do not delete, refresh, or replace it: doing so would lose
> the work and settings you have already added.
>
> ```bash
> git clone --branch 03-03-workspace/end --single-branch https://github.com/chrislomeli/pydanticai-tutorial-student.git pydantic-tutor
> cd pydantic-tutor
> uv sync
> ```
>
> This is a recovery option, not a step to repeat between lessons.

## Step 1: Give the chat file tools

> **Goal:** attach ready-made file operations to our
> existing chat agent.

This adds `FileSystem` to the agent, and the names of the files its tools must not touch. Replace `src/pydantic_tutor/chat/agent.py` with:

```python
"""The agent the chat talks to.

No model here, as for every agent in this project: the chat passes one on
each run.

Built from these pieces:
- Workspace: the folder the tools work in.
- File tools: read and write files there.
- Web search: switched on by a setting.
"""

from pathlib import Path

from pydantic_ai import Agent
from pydantic_ai.capabilities import AbstractCapability, LocalWorkspace, WebSearch
from pydantic_ai_harness import FileSystem

from tutor_given.paths import project_root

# File names the file tools may not touch. A precaution, not a complete
# data-security policy.
SECRET_PATTERNS = ("**/.env", "**/.env.*", "*.pem", "*.key", "**/secrets*")


def build_chat_agent(*, web_search: bool, workspace_root: Path) -> Agent[None, str]:
    """Build the chat's agent."""
    capabilities: list[AbstractCapability[None]] = [
        # Workspace: a place for tools to work in. It adds no tool: they use
        # `ctx.workspace`.
        LocalWorkspace(working_dir=workspace_root),
        # File tools: read and write files in that workspace, minus the secret
        # patterns.
        FileSystem(denied_patterns=SECRET_PATTERNS),
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

`FileSystem` comes right after `LocalWorkspace`, which still supplies the folder. It makes the file tools available, the tools that change files included. The deny patterns add a separate restriction: they keep common secret files unreadable through these tools. They are a precaution, **not** a complete data-security policy.

Notice that we did **not** change `chat/loop.py` or the code that
builds the model. The same chat now has an additional ability.

## Step 2: Explore the project with your agent

> **Goal:** see a real model use file tools to answer questions about
> the project, beginning with simple exploration.

Run the chat using your configured real model. Turn off web search so
the answers come from the working copy:

```bash
PYDANTIC_TUTOR_WEB_SEARCH=false uv run python -m pydantic_tutor
```

You now have a chat that can do something new: look around a real
Python project. Start with simple questions, just as you might when
getting to know a new teammate. Ask what the agent can do, then invite
it to show you what's in the workspace.

The conversation below is taken from an **actual session** with the
file-reading agent. Notice how the questions become more specific:
first the available tools, then the project directory, and finally the
`src/` folder. Nothing about those files was included in the user's
messages.

![A real eight-turn conversation with the file-reading agent. The user asks about available tools, then explores the project directories and the tinyjobs Python package.](Agent%20chat%20session%402x.png)

*An actual conversation with the file-reading agent, recorded with its write tools turned off: yours has eight tools, not five. Your model's replies may differ; here, the agent lists the project and source files without the user supplying their contents.*

That last directory listing is the important moment. The model didn't
need a copy of the project in its prompt: it asked your application to
list the directory and then explained the result. The answer is modest,
but it's grounded in something the agent actually *observed*.

The small token counts beneath the replies are not the point of this
exercise. Focus on what changed: the agent can now look inside the
workspace and answer from what it finds.

Next, ask it to examine code instead of merely listing names:

```text
Read src/tinyjobs/jobs.py. What does the JobQueue class do?
Explain the submit() and cancel() methods.
```

In our trial, asking it to read this file exposed the `JobQueue`
class and its methods. The source shows that `submit()` creates a
named job and adds it to the queue; `cancel()` changes the status of
a job to `CANCELLED` (but refuses a running job). Ask the agent to
show the methods if you want to compare its explanation with the code.

> [!TIP]
> **Check:** your agent should be able to read
> `src/tinyjobs/jobs.py` and explain the methods using details from
> the file. If it says it cannot see the file, check your `FileSystem`
> setup and that `work/tinyjobs` exists. Exact model replies will vary.

There are other things you can try without needing to know anything
about the file tools' arguments:

- `Read README.md. What is tinyjobs for?`
- `List the files in src/tinyjobs. Which file looks most relevant to storing jobs?`
- `Read src/tinyjobs/storage.py. What does this module do?`
- `Read src/tinyjobs/jobs.py. What happens if run_next() encounters an error?`

These are *suggested explorations*, not a set of scripted results.
Feel free to follow your curiosity. We aren't relying on directory-wide
text search in this first exercise: in our initial tests, searches
sometimes missed code that a direct file read found.

**What happened?**

You asked about project-specific information without including any
source files in your question. The model requested tools, **your
application** executed them in the workspace, and the model used the
returned information to answer. That's the tool loop from Chapter 1,
now doing useful work in the chat you've built.

## What you can now do

You added `FileSystem` to the chat you already had. With a real
model, you can ask questions about files in the workspace and watch
it use file tools to inspect them. Its root and deny
rules restrict what its tools can reach. Neither turns `LocalWorkspace`
into a sandbox, and nothing asks before a file tool writes: that comes
in a later chapter.

## Next

Our chat can now inspect a project. In the next lesson we'll let it
run a few simple commands through a shell. That is a bigger step:
even commands that look harmless have different risks from reading
files, so we'll be deliberate about what we allow.

---

*Tested with `pydantic-ai` 2.56.0 and `pydantic-ai-harness` 0.56.0.*
