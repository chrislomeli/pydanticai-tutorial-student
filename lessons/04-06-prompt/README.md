---
title: "The prompt becomes a capability"
description: "Move the chat's prompt out of Python into a file that a capability of your own reads, without changing what the model is sent, and predict the order in which instructions from several places reach the model."
---

# Move the chat prompt into a file

## Welcome

This lesson moves the chat's main instructions into a Markdown file. Last time, you wrote a capability with a hook and an instruction. Now we'll write one that supplies instructions without a hook. We'll compare what the model receives before and after the change, then look at instruction order.

By the end, you'll be able to:

- Put the chat's behavioral instructions in a file and keep the application working.
- Explain how instructions supplied from different places reach the model.

About 15 minutes. Steps 1 and 2 need no API key. Step 3 uses your OpenAI key.

## Concepts

### Capabilities can supply instructions

A capability can supply instructions without a hook. It implements `get_instructions()`, which returns text or a function that produces text. The agent keeps returned text for later requests. If it receives a function, it calls that function for each request.

Our `PromptFile` capability will read its instructions from a file. `PromptFile` is our name, not the library's. Keeping the prompt in its own file lets us edit the instructions without changing Python. `PromptFile` reads the file when we construct the capability. The agent reuses that text for each request, and edits take effect when we restart the application. We could instead return a function that reads the file on every request. We won't need that behavior in this lesson.

### Static instructions come first

Pydantic AI treats instructions as **static** text or **dynamic** functions. Static text stays the same across requests. Dynamic instructions are computed for each request. What a piece is
given as decides its kind, not where it comes from. Pydantic AI records whether each piece is dynamic.

An agent takes instructions in four places: its constructor, its
capabilities, the run (`run_sync(..., instructions=...)`) and its
toolsets. A function added with `@agent.instructions` counts as the
constructor's. The model is sent every static piece first, then every
dynamic one, and within each kind the places go in this order:

| Order | Static: text | Dynamic: functions |
|---|---|---|
| 1 | the constructor's `instructions=` | the constructor's `instructions=`, and `@agent.instructions` functions |
| 2 | each capability's `get_instructions()`, in list order | each capability's `get_instructions()`, in list order |
| 3 | the run's `instructions=` | the run's `instructions=` |
| 4 | each toolset's `instructions=` | each toolset's `instructions=` |

All static instructions come before dynamic ones. Within each kind, their source determines the order. The pieces are
joined with a blank line between them and sent as one block. Static goes
first for prompt caching: a provider that caches prompts can reuse a
request's unchanged beginning, so text that changes goes after it.

## What you'll build

- `prompts/chat.md` and `capabilities/prompt_file.py` (new); `chat/agent.py` takes
  its prompt from them, first in its capabilities
- `demos/instruction_order.py` (new): a second agent with instructions
  in all four places, printed in the order the model is sent them

## Before you start

Carry on in your `pydantic-tutor` project from the last lesson.

> [!TIP]
> **Check:** run `uv run demos/fresh_copy.py`, then `uv run demos/show_chat_request.py`. Its `instructions:` line, which this lesson must not change, reads:
>
> ```text
>   instructions: "You are a helpful assistant. Answer briefly.\n\n[token removed] in a tool result marks a secret removed on purpose. The file isn't broken. Don't try to recover the secret.\n\nYour workspace holds the Python project tinyjobs."
> ```

> [!WARNING]
> **Joining here, or has your project gone wrong?** If your project works, skip this. To refresh a broken project, delete its folder first: you lose your work and anything else in it, including your `.env` and its API keys.
>
> ```bash
> git clone --branch 04-05-capability/end --single-branch https://github.com/chrislomeli/pydanticai-tutorial-student.git pydantic-tutor
> cd pydantic-tutor
> uv sync
> ```
>
> Then run the check above.

## Step 1: Move the prompt into a file

> **Goal:** put the chat's prompt in a file, give the agent a capability
> that reads it, and see that the model is sent the same instruction text.

We'll keep the chat's current wording so moving it won't change its behavior. Create `src/pydantic_tutor/prompts/chat.md` with this line:

```markdown
You are a helpful assistant. Answer briefly.
```

Next, create `src/pydantic_tutor/capabilities/prompt_file.py`. This capability reads the Markdown file when the agent is built and supplies its text as instructions:

```python
"""A capability that gives an agent its instructions from a text file.

An agent's prompt is text, not code: in a file of its own, it can be read,
reviewed and changed without touching Python. `PromptFile` reads the file once, when it is made, and supplies the
text as the agent's instructions. It reads no deps, so any agent can take it.
"""

from dataclasses import dataclass, field
from pathlib import Path

from pydantic_ai.capabilities import AbstractCapability


@dataclass(kw_only=True)
class PromptFile[DepsT](AbstractCapability[DepsT]):
    """Supply the text of a file as the agent's instructions."""

    # The file the prompt is read from, as UTF-8 text.
    path: Path
    # The file's text, without the blank lines and spaces around it.
    text: str = field(init=False)

    def __post_init__(self) -> None:
        """Read the prompt now, so a missing or empty file fails at once."""
        self.text = self.path.read_text(encoding="utf-8").strip()
        if not self.text:
            raise ValueError(f"the prompt file {self.path} is empty")

    def get_instructions(self) -> str:
        """Return the prompt, as text."""
        # Text, not a function: the agent sends text it was given before
        # any it computes for a request, and does not ask for it again.
        # An edit to the file is read when the program next starts.
        return self.text
```

In `src/pydantic_tutor/chat/agent.py`, import `PromptFile` after the
`guards` import, and add the prompt file's path after `TOKEN_PATTERN`:

```python
from pydantic_tutor.capabilities.prompt_file import PromptFile
```

```python
# The chat's prompt, in the package's `prompts` folder, so it is found whichever
# folder the program runs from.
PROMPT_PATH = Path(__file__).parents[1] / "prompts" / "chat.md"
```

Then, in `build_chat_agent`, put `PromptFile(path=PROMPT_PATH)` first in
the capabilities, and delete the `instructions=` line from the `Agent(...)`
call:

```python
    capabilities: list[AbstractCapability[None]] = [
        # The chat's instructions, read from PROMPT_PATH when the agent is
        # built. Text, so it is sent ahead of any computed for a request.
        PromptFile(path=PROMPT_PATH),
```

```python
    return Agent[None, str](
        name="chat",
        capabilities=capabilities,
    )
```

Run the script from Before you start again:

```bash
uv run demos/show_chat_request.py
```

> [!TIP]
> **Check:** the `instructions:` line is exactly what it was:
>
> ```text
>   instructions: "You are a helpful assistant. Answer briefly.\n\n[token removed] in a tool result marks a secret removed on purpose. The file isn't broken. Don't try to recover the secret.\n\nYour workspace holds the Python project tinyjobs."
> ```

**What it means**

The instructions the model receives haven't changed. The prompt's words are in no `.py`
file now, yet the model is sent the same pieces. `PromptFile` returns
a `str`, so its text is static, like the constructor's was, and still
goes ahead of the others.

Now change the prompt without touching Python. Edit `chat.md` to read
`You are a helpful assistant. Answer in one short sentence.` and run the
script again:

```text
  instructions: "You are a helpful assistant. Answer in one short sentence.\n\n[token removed] in a tool result marks a secret removed on purpose. The file isn't broken. Don't try to recover the secret.\n\nYour workspace holds the Python project tinyjobs."
```

Put the file back as it was. `__post_init__` is what a dataclass calls
at the end of its constructor, so the file is read when `PromptFile(...)`
is made, which is when the agent is built. A missing file raises
`FileNotFoundError` then; an empty one raises `ValueError: the prompt
file .../chat.md is empty`. Either way the program fails at startup, not
in a run that goes out without its prompt. `text` is `field(init=False)`:
the constructor takes only `path`, and `__post_init__` sets `text`.

`PROMPT_PATH` starts from `Path(__file__)`, the file `chat/agent.py` itself,
and `parents[1]` is the package folder above it. Because `PromptFile` reads once, an edit made
while the chat is running is not seen until you start it again: the
choice from Concepts.

## Step 2: Observe instruction order

> **Goal:** see in which order the model is sent instructions given in
> all four places.

This step builds a second agent, `reader`, with instructions in all four
places, and runs it on a stand-in model that prints each piece it is
sent. We'll deliberately list the instructions out of order so the output shows how Pydantic AI arranges them.

Create `demos/instruction_order.py`:

```python
"""In which order instructions from several places reach the model.

A second agent is given instructions in every place an agent takes them: its
constructor, a capability, a toolset and the run, some as text and some as a
function. A stand-in model prints each part of the instructions it is sent, in
order. Needs no API key.
"""

from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from pydantic_tutor.capabilities.prompt_file import PromptFile
from pydantic_tutor.chat.agent import PROMPT_PATH
from pydantic_tutor.settings import Settings
from pydantic_tutor.tools.workspace_toolset import build_workspace_toolset


def from_the_constructor(ctx: RunContext[None]) -> str:
    """Instructions given to the constructor as a function."""
    return "From the constructor, a function."


def show_instructions(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """Stand in for a model: print each part of its instructions, then answer."""
    for part in info.model_request_parameters.instruction_parts or []:
        kind = "dynamic" if part.dynamic else "static"
        print(f"{kind:>7}: {part.content}")
    return ModelResponse(parts=[TextPart(content="Done.")])


# The function is listed before the text, so that the order you write would
# be wrong.
reader: Agent[None, str] = Agent(
    name="reader",
    instructions=[from_the_constructor, "From the constructor, text."],
    toolsets=[build_workspace_toolset(root=Settings().workspace_root)],
    capabilities=[PromptFile(path=PROMPT_PATH)],
)
reader.run_sync(
    user_prompt="Hi.",
    model=FunctionModel(function=show_instructions),
    instructions="From the run, text.",
)
```

```bash
uv run demos/instruction_order.py
```

> [!TIP]
> **Check:** ignoring any startup banner, you should see exactly:
>
> ```text
>  static: From the constructor, text.
>  static: You are a helpful assistant. Answer briefly.
>  static: From the run, text.
>  static: Your workspace holds the Python project tinyjobs.
> dynamic: From the constructor, a function.
> ```

**What it means**

Four static pieces come first, in
the order of places: the constructor's text, `PromptFile`'s prompt, the
run's text, the workspace toolset's sentence. Then the one dynamic piece,
the constructor's function.

The constructor listed its function first, and the text still went
first: kind decides before place. That is also why Step 1 changed
nothing, since the prompt is text.

Two placements are worth a second look. The run's text was passed to
`run_sync`, last, yet it lands before the function: it is a `str`, so it
is static. And the toolset's sentence is static too: the workspace
toolset was given a string (`instructions=f"Your workspace holds ..."`),
and a string is static wherever it comes from. The
[Agents](https://pydantic.dev/docs/ai/core-concepts/agent/) page, as read
on 2026-10-20, lists "toolset instructions" among the dynamic ones; at
2.56.0 only a function is.

## Step 3: Change the prompt in a real chat

> **Goal:** see a prompt edit change a real model's behaviour.

This step uses your API key. Ask a question, then edit the file, then ask it again:

```bash
uv run python -m pydantic_tutor
```

```text
What does tinyjobs do?
```

End the chat with an empty line. Edit `src/pydantic_tutor/prompts/chat.md` to read `You are a helpful assistant. Answer in one short sentence.`, start the chat again, and ask the same question.

> [!TIP]
> **Check:** the second answer should be noticeably shorter. It's the model's wording, so yours will differ. Put the file back when you're done.

**What it means**

You changed the chat's behaviour without touching Python. `PromptFile` reads the file when the agent is built, which is why the edit needed a restart.

## What you can now do

- **A capability can bring instructions and nothing else.** The chat's prompt is now a file that a capability reads.
- **Static text goes before dynamic, each in the order of places.** You can work out the order the model gets instructions in, from the table in Concepts.

## Next

`PromptFile` brings instructions and nothing else. Next, the chat gets a
capability that brings tools and instructions together: a notebook it keeps
between sessions.

---

*Tested with `pydantic-ai` 2.56.0 and `pydantic-ai-harness` 0.56.0.*
