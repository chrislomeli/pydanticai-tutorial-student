---
title: "Memory, with injection off"
description: "Give the chat a notebook of files it reads and writes across sessions, with a list tool of your own, and see what the library's memory injection would add to every request and why the chat turns it off."
---

# Save notes between chats

## Welcome

This lesson is about **memory**: notes the chat keeps between sessions. Suppose you explain a project convention today and open a new chat tomorrow. A new session doesn't automatically receive yesterday's conversation history. We'll give the chat a notebook to save project decisions and conventions. Then we'll see how it can retrieve those notes without sending every note with every request.

By the end, you'll be able to:

- Store a useful project note, start a fresh session, and retrieve it.
- Explain the trade-off between letting the model request notes and injecting notes into every model request.

About 20 minutes. Steps 1 to 4 need no API key: the model is a stand-in that
follows a script. Step 5 uses your OpenAI key.

## Concepts

### Saved notes survive a chat session

The `Memory` capability stores notes as Markdown files that outlast a run.
`MEMORY.md` is the main file, and the model can create others. It uses `write_memory` to save notes, `read_memory` to retrieve them, `search_memory` to find matching files, and `delete_memory` to remove files other than `MEMORY.md`. `Memory` also brings
**guidance**, text in the instructions that says when to use them. The model reads these files through tools. The notes are not a second conversation history. A **store** keeps the files; `FileStore`
keeps them in the run's workspace, in a folder you name.

### Injection adds notes to requests

By default, `Memory` uses **memory injection**. Before each request, it adds the contents of `MEMORY.md` and the names of other note files alongside the latest question.
The block isn't kept in the history: `Memory` takes the old one off the
earlier question and puts a new one beside the latest, so it moves.

| | Injection on | Injection off |
|---|---|---|
| Sent with each request | The guidance, and the notebook, up to 2,000 tokens together | The guidance only |
| The start of each request | Changes each turn, from the previous question on | The same as the request before |
| The notebook reaches the model | Beside your question, every time | In a tool result, when it asks |

The second row costs the most. A provider that caches prompts reuses the
unchanged start of a request; with injection on, each request changes
from the previous question onward, even when the notebook hasn't. The
library's [Memory page](https://pydantic.dev/docs/ai/harness/memory/)
recommends `inject_memory=False` for prompts that
should stay the same. Turning injection off keeps the memory tools and guidance available.

### The model can request saved notes

With injection off, the model no longer receives the file names automatically. It needs a way to discover them before calling `read_memory`. We'll add a `list_memory` tool and update the guidance to explain how to use it.

A capability gives the model tools, as `PromptFile` gave it instructions,
by overriding a method: `get_toolset()`, which returns a toolset.
`ListableMemory` (ours, not the library's) is a subclass of `Memory` whose
`get_toolset()` adds one tool. A subclass, so one capability keeps all its
tools together: `prefix_tools` renames the lot.

## What you'll build

You write `capabilities/listable_memory.py`, give it to the chat in `chat/agent.py`,
and add two scripts, `notebook.py` and `memory_injection.py`. Given, and
not written by you: the replacement guidance and the shape `list_memory`
returns (`tutor_given/memory_text.py`), and the scripts' stand-in models
(`tutor_given/stand_ins.py`).

## Before you start

Carry on in your `pydantic-tutor` project from the last lesson.

> [!TIP]
> **Check:** run `uv run demos/fresh_copy.py`, then `uv run demos/show_chat_request.py`. Its `tools:` line ends `list_modules, read_docstring`.

> [!WARNING]
> **Joining here, or has your project gone wrong?** If your project works, skip this. To refresh a broken project, delete its folder first: you lose your work and anything else in it, including your `.env` and its API keys.
>
> ```bash
> git clone --branch 04-06-prompt/end --single-branch https://github.com/chrislomeli/pydanticai-tutorial-student.git pydantic-tutor
> cd pydantic-tutor
> uv sync
> ```
>
> Then run the check above.

## Step 1: Add a notebook to the chat

> **Goal:** write `ListableMemory`, give it to the chat with injection
> off, and see what the model is now sent.

We'll use the harness's `Memory` capability to save notes, but we'll leave injection off. That means the model needs a way to discover note names before reading them. Our subclass adds a `list_memory` tool.

Create `src/pydantic_tutor/capabilities/listable_memory.py`. `LISTING_GUIDANCE` is
the guidance text, which names `list_memory`, and `MemoryListing` is what
the tool returns: its file names, and `truncated`, true when there were
more than `max_list_files`, so a cut list isn't taken for the whole.

```python
"""The library's `Memory`, plus a tool that lists the notebook's files.

With injection off nothing names the files to the model, and none of `Memory`'s
tools lists them: `ListableMemory` adds `list_memory`, and guidance that names it.
"""

from dataclasses import dataclass

from pydantic_ai import RunContext
from pydantic_ai.toolsets import AgentToolset, FunctionToolset
from pydantic_ai_harness import Memory

from tutor_given.memory_text import LISTING_GUIDANCE, MemoryListing


@dataclass(kw_only=True)
class ListableMemory[DepsT](Memory[DepsT]):
    """`Memory`, with a `list_memory` tool beside its own four."""

    guidance: str | None = LISTING_GUIDANCE
    max_list_files: int = 100  # the most names one call returns

    def get_toolset(self) -> AgentToolset[DepsT] | None:
        """Return `Memory`'s own toolset, with `list_memory` added once."""
        memory_tools = super().get_toolset()
        # `Memory` returns its one FunctionToolset: tells mypy, fails if not.
        if not isinstance(memory_tools, FunctionToolset):
            raise TypeError("Memory.get_toolset() did not return a FunctionToolset")
        # The same object every time, never a new one: `Memory`'s tools find
        # the run's copy of the capability by it. The agent asks on every
        # memory tool call, hence "once".
        if "list_memory" not in memory_tools.tools:
            memory_tools.add_function(func=self.list_memory, name="list_memory")
        return memory_tools

    async def list_memory(self, ctx: RunContext[DepsT]) -> MemoryListing:
        """List the files in your memory notebook, MEMORY.md included.

        Call it before `read_memory` when you do not know a file's name.
        """
        # The run's store and this notebook's folder: the other tools' own.
        store, scope = self.resolve_scope(ctx=ctx)
        prefix = f"{scope}/"
        # One over the limit, so exactly that many files is not "cut".
        paths = await store.list_paths(prefix=prefix, limit=self.max_list_files + 1)
        # Only names `read_memory` takes: Markdown, not in a folder below.
        names = sorted(
            name
            for name in (path.removeprefix(prefix) for path in paths)
            if "/" not in name and name.endswith(".md")
        )
        return MemoryListing(
            files=names[: self.max_list_files],
            truncated=len(paths) > self.max_list_files,
        )
```

In `src/pydantic_tutor/chat/agent.py`, import `FileStore` from
`pydantic_ai_harness.memory` and `ListableMemory` from
`pydantic_tutor.capabilities.listable_memory`, add the notebook's folder after
`SECRET_PATTERNS`, and add the capability after `Toolset(...)`:

```python
# The notebook's folder, in the workspace.
NOTEBOOK = ".agent-memory"
...
        Toolset(toolset=build_workspace_toolset(root=workspace_root)),
        # Injection off: the model lists and reads the notebook with tools.
        ListableMemory(store=FileStore(directory=NOTEBOOK), inject_memory=False),
```

```bash
uv run demos/show_chat_request.py
```

> [!TIP]
> **Check:** the `tools:` line now ends `..., read_docstring, write_memory,
> read_memory, delete_memory, search_memory, list_memory`, and its
> `instructions:` line holds the prompt, then the guidance
> (`You have a persistent memory: ...`), then what `RedactTokens` says, then the
> workspace sentence.

**What it means**

The guidance is a `str`, so it is static, in the capabilities' list order. `get_toolset()` returns `Memory`'s own toolset with
`list_memory` added, never a new one: `Memory`'s tools find the run's copy
of the capability through it, and the agent asks for it on every memory
tool call, hence the `if`.

## Step 2: Retrieve a note in a new session

> **Goal:** run two sessions with no history between them, and see a note
> written in the first found in the second.

We'll prove that the notes survive without carrying conversation history forward. Create `demos/notebook.py` to run two separate sessions against the same notebook:

```python
"""Two sessions with the chat's agent: one stores a note, the next finds it.

The notebook lives in the workspace, so run `demos/fresh_copy.py` first.
"""

from pydantic_tutor.chat.agent import build_chat_agent
from pydantic_tutor.settings import Settings
from tutor_given.stand_ins import remembers_then_recalls

chat = build_chat_agent(web_search=False, workspace_root=Settings().workspace_root)
model = remembers_then_recalls()
for prompt in ["Remember that we log to stderr.", "What did we decide about logging?"]:
    # No message_history: a new session, as far as the model can tell.
    print(f"session: {prompt}")
    result = chat.run_sync(user_prompt=prompt, model=model)
    print("  output", result.output)
```

The given stand-in writes a note when a prompt starts with "Remember",
and otherwise looks one up:

```bash
uv run demos/fresh_copy.py
uv run demos/notebook.py
```

> [!TIP]
> **Check:** you should see the output below (the `version` is a hash of
> the note, so it matches too), and
> `cat work/tinyjobs/.agent-memory/main/logging.md` prints the note.
>
> ```text
> session: Remember that we log to stderr.
>   request ['user-prompt']
>   call   write_memory {"content": "- Logging goes to stderr, so stdout stays pipeable.\n", "file": "logging.md"}
>   request ['tool-return']
>   result {"file":"logging.md","version":"9eb4a459b5f2bb5a0a1554df7230342c45d02c6c53231a4115be9b8d8a22a291","replayed":false,"status":"created"}
>   output Noted.
> session: What did we decide about logging?
>   request ['user-prompt']
>   call   list_memory {}
>   request ['tool-return']
>   result {"files":["logging.md"],"truncated":false}
>   call   read_memory {"file": "logging.md"}
>   request ['tool-return']
>   result - Logging goes to stderr, so stdout stays pipeable.
>
>   output We decided: - Logging goes to stderr, so stdout stays pipeable.
> ```

**What it means**

The second session's first request is `['user-prompt']`: the question and
nothing else. With no history and no memory part, the note can reach the
model only through a tool call, and it does: `list_memory` for the name,
then `read_memory`. The note is `.agent-memory/main/logging.md` (`main` is
`Memory`'s `agent_name`; the model never sees it), and `fresh_copy.py`
deletes it with the rest of the copy. A stand-in does what its script
says: whether a real model writes a note, or looks, isn't shown.

## Step 3: Restrict file access to notes

> **Goal:** deny the file tools the notebook's folder, and see two calls
> refused.

The notebook is in the workspace, so the chat's file tools reach it too.
In a test run for this lesson, a `write_file` replaced the note outright, and the next
`write_memory` appended to the file tool's text. The memory tools append
and keep a record; a file tool does neither. So the notebook gets one
owner. In `chat/agent.py`, put its folder in the file tools' deny list, next to
the secrets:

```python
        # The notebook is denied too, for reads and writes alike.
        FileSystem(denied_patterns=(*SECRET_PATTERNS, f"**/{NOTEBOOK}/**")),
```

and say so in `NOTEBOOK`'s comment: "The file tools may not touch it: the
memory tools are its only way in." Add two calls to `CALLS` in
`demos/file_tools.py`, before the last: `("read_file",
".agent-memory/main/logging.md", {})` and `("write_file",
".agent-memory/main/logging.md", {"content": "mine\n"})`.

```bash
uv run demos/file_tools.py
```

> [!TIP]
> **Check:** the earlier calls print as before, then:
>
> ```text
> read_file .agent-memory/main/logging.md:
>   Path '.agent-memory/main/logging.md' is denied by pattern '**/.agent-memory/**'.
> write_file .agent-memory/main/logging.md:
>   Path '.agent-memory/main/logging.md' is denied by pattern '**/.agent-memory/**'.
> ```
>
> and `cat` of the note still prints it.

**What it means**

`denied_patterns` stops reads and writes alike; `FileStore` doesn't go
through the file tools, so the memory tools are unaffected. It is a guard
against mistakes, not a boundary: `pytest` runs any code it is given. In
a real project the notebook belongs in `.gitignore` as `.agent-memory/`;
here all of `work/` is already there.

## Step 4: Compare memory injection

> **Goal:** run a second agent on the same notebook with injection on, and
> see what the chat doesn't send.

Create `demos/memory_injection.py`:

```python
"""What memory injection would send: the notebook beside every question.

A second agent reads the notebook `demos/notebook.py` left in the workspace, with
injection on.
"""

from pydantic_ai import Agent
from pydantic_ai.capabilities import LocalWorkspace
from pydantic_ai.messages import ModelMessage
from pydantic_ai_harness import Memory
from pydantic_ai_harness.memory import FileStore

from pydantic_tutor.chat.agent import NOTEBOOK
from pydantic_tutor.settings import Settings
from tutor_given.stand_ins import prints_every_message

# The chat's notebook; only `inject_memory` differs.
injecting: Agent[None, str] = Agent(
    capabilities=[
        LocalWorkspace(working_dir=Settings().workspace_root),
        Memory(store=FileStore(directory=NOTEBOOK), inject_memory=True),
    ],
)
model = prints_every_message()
history: list[ModelMessage] = []
for prompt in ["What does tinyjobs do?", "And how is it tested?"]:
    print(f"turn: {prompt}")
    result = injecting.run_sync(
        user_prompt=prompt, model=model, message_history=history
    )
    history = result.all_messages()
```

Run it on the same workspace, so `logging.md` is there:

```bash
uv run demos/memory_injection.py
```

> [!TIP]
> **Check:** each line is one message of a request:
>
> ```text
> turn: What does tinyjobs do?
>   request: ['What does tinyjobs do?', '<memory> ### Other memory files - logging.md </memory>']
> turn: And how is it tested?
>   request: ['What does tinyjobs do?']
>   response: ['ok']
>   request: ['And how is it tested?', '<memory> ### Other memory files - logging.md </memory>']
> ```

**What it means**

Every request carries the block, beside the newest question and only
there: on turn two the first question has lost it. It lists "Other memory
files": that is how this agent learns `logging.md`, and why it needs no
`list_memory`. The moving is the cost: this request's start differs from
the previous one's, though the notebook didn't change, so a provider's
cache has less to reuse. The block is also cut to a budget, 2,000 tokens
at four characters a token, shared with the guidance.

## Step 5: Try memory in real chats

> **Goal:** see whether a real model saves a note and finds it again.

This step uses your API key, and the model decides what it does.

```bash
uv run demos/fresh_copy.py
uv run python -m pydantic_tutor
```

```text
Please remember that this project logs to stderr, so stdout stays pipeable.
```

End the chat with an empty line, then start it again and ask:

```text
What did we decide about logging?
```

> [!TIP]
> **Check:** the second chat starts with no history. If the model saved and looked up the note, `work/activity.jsonl` has a `write_memory` call from the first chat, and `list_memory` or `read_memory` from the second. A model may also choose not to use the notebook, and the log shows that too.

**What it means**

The note survived because it's in a file, not in the conversation. With injection off, using it is the model's choice, which is the trade-off from Concepts. Your activity log shows you which choice it made.

## What you can now do

- **`Memory` is a notebook the model reads and writes with tools.** Kept in the workspace, it outlasts a run.
- **Injection sends the notebook beside every question.** With it off, the model asks, and needs a tool to list the files.

## Next

The chat now has a prompt file, a capability of its own and a notebook.
The next lesson brings `build_chat_agent` together in one place.

---

*Tested with `pydantic-ai` 2.56.0 and `pydantic-ai-harness` 0.56.0.*
