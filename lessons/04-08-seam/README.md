---
title: "What the agent sets and what a capability supplies"
description: "Sort what the chat agent is from what it does: the constructor keeps its name, output type and request defaults, a list of capabilities holds the rest, and the capabilities whose place matters declare it, so the list can be rearranged and added to safely."
---

# Separate agent settings from capabilities

## Welcome

This lesson is about **agent settings and capability order**. We'll separate the settings that define our agent from the capabilities that add behavior. Our chat now has tools, guards, an activity log, a prompt file, and a notebook. We'll separate constructor settings from capability behavior. Then we'll make the ordering requirements explicit.

By the end, you'll be able to:

- Explain why some settings belong on `Agent(...)` and others in capabilities.
- Make important ordering constraints explicit so adding another capability doesn't silently change behavior.

About 15 minutes. Steps 1 to 3 need no API key. Step 4 uses your OpenAI key.

## Concepts

### The constructor sets agent defaults

Capabilities can supply instructions, tools, a model, model settings, a workspace, and hooks. The `Agent(...)` constructor sets the agent's name, output type, and loop policy. Some of those choices can be overridden for a run.

We'll keep the agent's identity and defaults on the constructor, and put its tools and behavior in capabilities:

| | On the constructor | In a capability |
|---|---|---|
| Name, types, loop policy | `name`, `description`, `deps_type`, `output_type`, `retries`, `end_strategy`, `tool_timeout`, `max_concurrency` | No way to set them |
| Instructions | `instructions` | `get_instructions()` |
| Tools | `tools`, `toolsets` | `get_toolset()` |
| Model settings | `model_settings` | `get_model_settings()` |
| Model | `model` | `get_model()` |
| A workspace, hooks at each stage | No way to set them | `get_workspace()`, `before_model_request()` and the rest |

Some settings can appear in either place. In this project, capabilities supply what the model receives and what acts during a run. That lets us move related behavior between agents as one object. We'll keep ordinary request defaults, such as a timeout, on the constructor. The library's [Capabilities](https://pydantic.dev/docs/ai/capabilities/overview/) page also shows model settings there. If both sides set the same setting, they
merge key by key: a capability's value beats the constructor's, and a
run's beats both.

### Capabilities can declare their order

Capability order matters. `PromptFile` supplies the chat's initial instructions, and the activity log needs to see every tool attempt. Rearranging the list could change either behavior.

A capability can say where it belongs instead, with **`get_ordering()`**.
It returns a `CapabilityOrdering`, whose `position` is `"outermost"`,
first of all, or `"innermost"`, last before the model. When the agent is
made, it sorts its capabilities so that every declared place holds, and
keeps your list's order for the rest; two that declare the same position
share it, and list order decides between them. A `Hooks` object takes the
same thing as `ordering=`. (`CapabilityOrdering` can also say that one
capability wraps another: see "Ordering" on the [Custom
capabilities](https://pydantic.dev/docs/ai/capabilities/custom/) page.)
The harness does this itself: its input guardrail declares `innermost`, so
it checks the prompt as every other capability has left it.

## What you'll build

You split `build_chat_agent` into `chat_capabilities()`, the list of everything the chat does, and `build_chat_agent()`, whose constructor holds only what the agent is. Then `demos/stack_order.py` builds the agent with its list forwards and reversed, to show what the order changes, and `PromptFile` and the activity log declare their places.

## Before you start

Carry on in your `pydantic-tutor` project from the last lesson.

> [!TIP]
> **Check:** run `uv run demos/fresh_copy.py`, then `uv run demos/guard_check.py`. Its log's last line is the refused `pytest --version && cat .env`.

> [!WARNING]
> **Joining here, or has your project gone wrong?** If your project works, skip this. To refresh a broken project, delete its folder first: you lose your work and anything else in it, including your `.env` and its API keys.
>
> ```bash
> git clone --branch 04-07-memory/end --single-branch https://github.com/chrislomeli/pydanticai-tutorial-student.git pydantic-tutor
> cd pydantic-tutor
> uv sync
> ```
>
> Then run the check above.

## Step 1: Separate settings from capabilities

> **Goal:** turn `build_chat_agent` into two functions: one returns the
> chat's capabilities, one builds the agent from them.

We'll separate building the capability list from constructing the agent. The list stays the same, so this refactor should not change what the model receives.

In `src/pydantic_tutor/chat/agent.py`:

- Add `from collections.abc import Sequence` and `from
  pydantic_ai.settings import ModelSettings`, and import `AgentCapability`
  from `pydantic_ai.capabilities` in place of `AbstractCapability`.
- Rename `build_chat_agent` to `chat_capabilities`. It returns
  `list[AgentCapability[None]]`, as does its `capabilities` variable, and
  its docstring is `"""Everything the chat does, as a list of
  capabilities, each one new."""`. Its list stays as it is.
- At the end of that function, replace `return Agent(...)` with `return
  capabilities`, and write `build_chat_agent` after it:

```python
        capabilities.insert(0, activity_log)
    return capabilities


def build_chat_agent(
    *,
    web_search: bool,
    workspace_root: Path,
    activity_log: Hooks[None] | None = None,
    capabilities: Sequence[AgentCapability[None]] | None = None,
) -> Agent[None, str]:
    """Build the chat's agent: what it is here, what it does in `capabilities`.

    The constructor sets what no capability can. `capabilities` defaults to
    `chat_capabilities(...)`; pass another list to rearrange or extend the
    stack, and the other arguments are not used.
    """
    if capabilities is None:
        capabilities = chat_capabilities(
            web_search=web_search,
            workspace_root=workspace_root,
            activity_log=activity_log,
        )
    return Agent[None, str](
        name="chat",
        # The type the type checker holds every run to.
        output_type=str,
        # Each attempt at a request that takes longer fails; most hosted
        # models honour it, the given stand-in models do not.
        model_settings=ModelSettings(timeout=120),
        capabilities=capabilities,
    )
```

The `chat_agent = build_chat_agent(...)` line below doesn't change. Nothing the
model is sent should change either:

```bash
uv run demos/fresh_copy.py
uv run demos/guard_check.py
uv run python -c "from pydantic_tutor.chat.agent import chat_agent as a; print(a.name, a.output_type, a.model_settings)"
```

> [!TIP]
> **Check:** `guard_check.py` prints what it printed in Before you start. The
> second command prints:
>
> ```text
> chat <class 'str'> {'timeout': 120}
> ```

**What it means**

Nothing the model is sent changed: the capabilities are the same, in the same
order. The constructor holds the agent's name and output type
(`output_type=str` was the default; written out, it says what the agent
is) and a request default, the one middle row this project's rule keeps
there. The timeout is new, and is the library's: a model that honours it
fails an attempt that takes longer than 120 seconds. The stand-in models
here don't, so nothing in this lesson shows it working.

`chat_capabilities()` makes new objects on each call, so two agents built
from it share no capability that holds state of its own, such as the
notebook's. `build_app` and `chat_agent` call `build_chat_agent` as
before, so neither changed.

## Step 2: Reverse the list

> **Goal:** build the agent with its capabilities forwards and reversed, and see what the order changes.

Now we'll compare the normal list with its reverse. This shows which behaviors depend on a capability's position. It reuses the given model from the guards lesson, which also prints how each piece of its instructions starts.

Create `demos/stack_order.py`:

```python
"""The chat's agent with its capabilities listed forwards, then reversed.

Needs no API key. It adds a `deploy.toml` with a fake token to the workspace copy;
run `demos/fresh_copy.py` first.
"""

from pydantic_ai.capabilities import AgentCapability

from pydantic_tutor.capabilities.activity_log import HookName, build_activity_hooks
from pydantic_tutor.chat.agent import build_chat_agent, chat_capabilities
from pydantic_tutor.settings import Settings
from tutor_given.stand_ins import reads_a_token_then_chains_a_command

settings = Settings()
ROOT = settings.workspace_root
LOG = settings.activity_log_path
(ROOT / "deploy.toml").write_text('runner_token = "tj-FAKE-0000-not-a-real-secret"\n')


def stack() -> list[AgentCapability[None]]:
    """The chat's capabilities, each one new."""
    return chat_capabilities(
        web_search=False,
        workspace_root=ROOT,
        activity_log=build_activity_hooks(
            path=LOG, hooks=frozenset({HookName.BEFORE_TOOL_EXECUTE})
        ),
    )


for label, capabilities in [("forwards", stack()), ("reversed", stack()[::-1])]:
    print(label)
    LOG.unlink(missing_ok=True)
    chat = build_chat_agent(
        web_search=False, workspace_root=ROOT, capabilities=capabilities
    )
    chat.run_sync(
        user_prompt="Check the deployment settings.",
        model=reads_a_token_then_chains_a_command(show_instructions=True),
    )
    print("  logged:", len(LOG.read_text().splitlines()), "tool calls")
```

```bash
uv run demos/fresh_copy.py
uv run demos/stack_order.py
```

> [!TIP]
> **Check:** ignoring any startup banner, you should see:
>
> ```text
> forwards
>   instructions: You are a helpful ...
>   instructions: You have a persistent ...
>   instructions: [token removed] in a ...
>   instructions: Your workspace holds the ...
>   model got, from read_file:
>     [deploy.toml | 1 lines | hash:e3b623df92da]
>     1	runner_token = "[token removed]"
>   model got, from run_command:
>     Run one command at a time, with no chaining.
>   logged: 2 tool calls
> reversed
>   instructions: [token removed] in a ...
>   instructions: You have a persistent ...
>   instructions: You are a helpful ...
>   instructions: Your workspace holds the ...
>   model got, from read_file:
>     [deploy.toml | 1 lines | hash:e3b623df92da]
>     1	runner_token = "[token removed]"
>   model got, from run_command:
>     Run one command at a time, with no chaining.
>   logged: 1 tool calls
> ```

**What it means**

Two things differ, and they're the two facts the list's order was holding.

Reversed, the prompt isn't first: the notebook's guidance and `RedactTokens`'s sentence reach the model ahead of it, because text is sent in list order. And the log records one tool call, not two, because it now sits after the guard, which refused the second before the log saw it. The workspace sentence stays last in both, because a toolset's instructions are sent after a capability's.

## Step 3: Set capability order

> **Goal:** make `PromptFile` and the activity log declare their positions, so reversing the list no longer changes the behavior.

We'll give the prompt and activity log explicit positions so future edits don't depend on their list order.

In `src/pydantic_tutor/capabilities/prompt_file.py`, import `CapabilityOrdering` beside
`AbstractCapability`, and add a method to `PromptFile`, after
`get_instructions`:

```python
    def get_ordering(self) -> CapabilityOrdering:
        """Put this capability first, wherever it is listed."""
        # The prompt says what the agent is for, so it is the first text the
        # model is sent: of the instructions given as text, the agent sends
        # those of the outermost capability first.
        return CapabilityOrdering(position="outermost")
```

In `src/pydantic_tutor/capabilities/activity_log.py`, import `CapabilityOrdering` beside `Hooks`, and add one argument to the `Hooks[None](...)` that `build_activity_hooks` returns, after `after_tool_execute`:

```python
        # Outermost, wherever it is listed: its `before_` hooks see every
        # attempt, and its `after_` hooks see the result the model gets.
        ordering=CapabilityOrdering(position="outermost"),
```

```bash
uv run demos/fresh_copy.py
uv run demos/stack_order.py
```

> [!TIP]
> **Check:** you should see:
>
> ```text
> forwards
>   instructions: You are a helpful ...
>   instructions: You have a persistent ...
>   instructions: [token removed] in a ...
>   instructions: Your workspace holds the ...
>   model got, from read_file:
>     [deploy.toml | 1 lines | hash:e3b623df92da]
>     1	runner_token = "[token removed]"
>   model got, from run_command:
>     Run one command at a time, with no chaining.
>   logged: 2 tool calls
> reversed
>   instructions: You are a helpful ...
>   instructions: [token removed] in a ...
>   instructions: You have a persistent ...
>   instructions: Your workspace holds the ...
>   model got, from read_file:
>     [deploy.toml | 1 lines | hash:e3b623df92da]
>     1	runner_token = "[token removed]"
>   model got, from run_command:
>     Run one command at a time, with no chaining.
>   logged: 2 tool calls
> ```

**What it means**

The agent sorts its capabilities when it is made. `PromptFile` and the log declared `outermost`, so they go first whichever way the list runs: the prompt is the first text sent, and the log records both tool calls in both orders. Every other capability keeps its place from your list, which is why the guidance and `RedactTokens`'s sentence still swap. Nothing relies on their order, so they don't declare one.

When something relies on a place, it belongs in a declaration, not a comment.

## Step 4: Inspect the finished chat

> **Goal:** connect each thing the chat does to the part that brings it.

This step uses your API key, and the model decides what it does.

```bash
uv run demos/fresh_copy.py
uv run demos/show_capabilities.py
uv run python -m pydantic_tutor
```

```text
Which module handles retries in this project? Save a note about it, then run the tests.
```

End the chat with an empty line, then open `work/activity.jsonl`.

> [!TIP]
> **Check:** the log shows the tools the model chose, with a request between each round. Each tool came from a line that `show_capabilities.py` printed. Your calls will differ from anyone else's.

**What it means**

Everything the chat does beyond the model is in one list, and you can name what each entry brings. `build_chat_agent` itself says only what the agent is.

## What you can now do

- **The constructor says what the agent is, and the list says what it does.** You can put a choice on the right side.
- **A capability whose place matters declares it.** The list can be rearranged and added to without breaking what relies on it.

## Next

Chapter 4 is done: the agent is built from parts, and the list can grow. Next, the chat learns to ask you before it writes a file or runs a command.

---

*Tested with `pydantic-ai` 2.56.0 and `pydantic-ai-harness` 0.56.0.*
