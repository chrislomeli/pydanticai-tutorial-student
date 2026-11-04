---
title: "Toolsets: tools that travel with their instructions"
description: "Package supplied workspace tools in a FunctionToolset, attach it to the chat, and try prefixed and filtered views."
---

# Toolsets

## Welcome

This lesson is about **toolsets**: groups of tools that carry their own instructions and time limits. Our chat can already inspect files and run selected commands. We'll take two tools supplied with the workspace starter, package them in a `FunctionToolset`, and attach it to the chat. Then we'll see how another agent can use different views of that toolset and ask the chat about the `tinyjobs` project.

By the end, you'll be able to:

- Build a toolset for useful questions about the `tinyjobs` project.
- Attach it to the chat, distinguishing the tools it exposes from the instructions it supplies.
- Give an agent a prefixed or filtered view of a toolset, without changing the original.

About 15 minutes. Steps 1–3 need no API key; Step 4 uses a configured real model.

## Concepts

### A toolset keeps tool settings together

An agent can take a list of functions as tools, and that list can be reused.
A **toolset** groups tools into one object that also carries instructions
for the model and a per-tool-call time limit. Those settings travel with the
tools rather than being configured separately for each agent.

We've used `FileSystem` and `Shell`, which already provide collections of
tools. Now we'll assemble a `FunctionToolset` from two functions of our own.
The chat will use their names, docstrings, and parameters to offer them
to the model.

### Agents can use different toolset views

An agent can use the complete toolset or take a **view** of it: `prefixed` renames its tools, and `filtered`
hides some of them. This lets one agent see only the tools it should have,
and the original toolset doesn't change.

## What you'll build

- `src/pydantic_tutor/tools/workspace_toolset.py`, new: the `FunctionToolset` that packages the two supplied tools
- `src/pydantic_tutor/chat/agent.py`, changed: the chat takes the toolset
- `demos/toolset_views.py`, new: shows two views of the toolset

## Before you start

Keep working in your existing `pydantic-tutor` project. The workspace
starter you fetched in 03-03 already supplied
`src/tutor_given/module_tools.py`; you do **not** need to create or
fetch the two tools again.

> [!TIP]
> **Check:** from your project directory, run
> `uv run demos/fresh_copy.py` and then
> `uv run demos/workspace_reach.py`. Before adding your toolset, the
> output starts with `tools the model is offered: 12`.
>
> You can also check that `src/tutor_given/module_tools.py` exists.

> [!WARNING]
> **Starting at this lesson instead of continuing?** Only use this
> recovery route if you do not already have a `pydantic-tutor` directory.
> Starting over in a fresh project will not carry over work, settings,
> or credentials from your existing project. Do not delete or overwrite
> that project to follow these commands.
>
> From a parent directory with no `pydantic-tutor` folder:
>
> ```bash
> git clone --branch 03-05-shell/end --single-branch https://github.com/chrislomeli/pydanticai-tutorial-student.git pydantic-tutor
> cd pydantic-tutor
> uv sync
> ```
>
> If you take this route, make sure you also obtain the `workspace`
> starter files as described in lesson 03-03 before continuing.

## Step 1: Build a toolset

> **Goal:** create a `FunctionToolset` that offers two read-only tools,
> together with instructions and a ten-second tool-call limit.

### Look at the two workspace tools

Our chat will use `list_modules()` to discover modules in `tinyjobs`
and `read_docstring(module=...)` to read a module's description without
executing its code.

These tools are **supplied code**, not what we're learning to implement.
You already have them: `ModuleTools` came with the starter files you
fetched for the workspace lesson. You need to know what each tool takes
and returns, not how the file discovery or AST parsing works.

`ModuleTools(root=...)` gives both tools the workspace path when the
application starts. The model doesn't choose or receive that path.

Here are their signatures and docstrings, from
`src/tutor_given/module_tools.py`. There's nothing to type here:

```python
class ModuleTools:
    root: Path

    def list_modules(self) -> list[str]:
        """List the Python modules in the workspace, by dotted name."""

    def read_docstring(self, *, module: str) -> str:
        """Read a module's docstring, without running the module.

        Args:
            module: The module's dotted name, as `list_modules` gives it.
        """
```

### Create the FunctionToolset

Now create `src/pydantic_tutor/tools/workspace_toolset.py`. This is
the file you write: it imports the supplied tools, then groups them
into a `FunctionToolset`.

The `tools=` list selects what the model can call. The `instructions=`
text tells the model which workspace it is inspecting, and `timeout=`
sets a ten-second limit on each tool call. Unlike an ordinary list of
functions, the toolset carries those settings along with its tools.

```python
"""The workspace's tools, with their instructions and time limit, as one toolset.

Built once, where the app is built, with the
folder the tools read: they take no deps.
"""

from pathlib import Path

from pydantic_ai.toolsets import FunctionToolset

from tutor_given.module_tools import ModuleTools


def build_workspace_toolset(*, root: Path) -> FunctionToolset[None]:
    """The tools that read the modules under `root/src`, with their instructions."""
    tools = ModuleTools(root=root)
    return FunctionToolset(
        tools=[tools.list_modules, tools.read_docstring],
        # Sent with every request, beside the agent's own instructions.
        instructions=f"Your workspace holds the Python project {root.resolve().name}.",
        # Set here: an agent's `tool_timeout` covers only its own `tools=`.
        timeout=10,
    )
```

> [!TIP]
> **Check:** inspect the new toolset:
>
> ```bash
> uv run python -c "from pathlib import Path; from pydantic_tutor.tools.workspace_toolset import build_workspace_toolset as build; t = build(root=Path('work/tinyjobs')); print(list(t.tools), t.timeout)"
> ```
>
> ```text
> ['list_modules', 'read_docstring'] 10
> ```

**What it means**

The toolset offers two tools and gives each call a ten-second limit. We have not run either tool yet.

## Step 2: Attach it to the chat

> **Goal:** the chat is offered your two tools.

Open `src/pydantic_tutor/chat/agent.py`. This adds `Toolset` to the agent's capabilities, built for the same folder as the others. Replace the file with:

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
"""

from pathlib import Path

from pydantic_ai import Agent
from pydantic_ai.capabilities import (
    AbstractCapability,
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
        # Toolset: the workspace's own tools, built for this folder.
        Toolset(toolset=build_workspace_toolset(root=workspace_root)),
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

This uses the same workspace root as the existing file and shell tools;
it doesn't replace either capability.

Run `workspace_reach.py` again:

```bash
uv run demos/workspace_reach.py
```

> [!TIP]
> **Check:** the first line is now `tools the model is offered: 14`. The
> other three are as before.

**What it means**

The model now sees fourteen tools, including the two you added. On the keyless test
model (`PYDANTIC_TUTOR_CHAT_MODEL=test`) the chat still ends on its first
message, as it has since the file tools: that model calls every tool it is
offered, yours now included. `demos/fresh_copy.py` resets the copy.

## Step 3: Try two toolset views

> **Goal:** see the same toolset offered under new names, then with one tool
> hidden.

We'll use a separate demonstration agent so we don't change our chat.
A stand-in model will report the tool definitions and instructions it
receives; it won't answer a question or call the tools. This lets us
compare the two views without an API key.

`prefixed(prefix="workspace")` adds a prefix to both tool names;
`filtered(filter_func=only_listing)` exposes only `list_modules`.
The original toolset stays unchanged.

Create `demos/toolset_views.py`:

```python
"""Show a second agent two views of the workspace toolset.

Needs no API key; run `demos/fresh_copy.py`
first.
"""

from pydantic_ai import Agent, RunContext
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.toolsets import AbstractToolset

from pydantic_tutor.settings import Settings
from pydantic_tutor.tools.workspace_toolset import build_workspace_toolset
from tutor_given.stand_ins import reports_what_it_is_sent

toolset = build_workspace_toolset(root=Settings().workspace_root)


def only_listing(ctx: RunContext[None], tool: ToolDefinition) -> bool:
    """Keep `list_modules`; hide every other tool."""
    return tool.name == "list_modules"


views: dict[str, AbstractToolset[None]] = {
    "prefixed": toolset.prefixed(prefix="workspace"),
    "filtered": toolset.filtered(filter_func=only_listing),
}
for title, view in views.items():
    print(title)
    reader = Agent[None, str](name="reader", toolsets=[view])
    reader.run_sync(user_prompt="hi", model=reports_what_it_is_sent())
```

Run it:

```bash
uv run demos/toolset_views.py
```

> [!TIP]
> **Check:** ignoring any startup banner, you should see:
>
> ```text
> prefixed
>   tools: workspace_list_modules, workspace_read_docstring
>   instructions: 'Your workspace holds the Python project tinyjobs.'
> filtered
>   tools: list_modules
>   instructions: 'Your workspace holds the Python project tinyjobs.'
> ```

**What it means**

Both agents received the toolset's instructions, but each saw different tools. Both views came from one `toolset`, which still has its two tools.

## Step 4: Ask about tinyjobs

> **Goal:** see your continuing chat use the toolset to answer a
> question about the project it can inspect.

The previous checks confirmed that the tools were offered to the model.
Now we'll give them a useful job. With a real model configured, run the
same console chat as in the previous lesson, with web search disabled:

```bash
PYDANTIC_TUTOR_WEB_SEARCH=false uv run python -m pydantic_tutor
```

Ask:

```text
Where does tinyjobs record a job's status? Please inspect the project's modules.
```

> [!TIP]
> **Check:** look for an answer grounded in the `tinyjobs` project,
> rather than a general guess about job queues. The model can use
> `list_modules` and `read_docstring` to discover relevant modules.
> It also has the earlier file tools, so its exact choice of tools
> and wording may vary.

Here's an example of a real conversation with the workspace toolset:

![A conversation with the chat agent using the workspace toolset's list_modules and read_docstring tools to answer a question about the tinyjobs project.](agent_session%402x.png)

*An actual agent session; your model's tool choices and wording may vary.*

**What it means**

You've moved beyond checking a list of available tools: the chat now
has project-specific information it can consult while answering. If you
haven't configured a real model yet, the keyless checks in Steps 1–3
still confirm that the toolset is wired into the application.

## What you can now do

You took two tools already supplied with the workspace, packaged them with
instructions and a time limit, and made them available to the chat. You
also saw how a second agent can see renamed or filtered tools without
changing the original toolset. In Step 4, you can use that same chat to ask a project-specific
question and see how the available tools help it answer.

## Next

The chat now has tools of its own beside the library's. Next it gets a
hook: code that records what the agent does.

---

*Tested with `pydantic-ai` 2.56.0 and `pydantic-ai-harness` 0.56.0.*
