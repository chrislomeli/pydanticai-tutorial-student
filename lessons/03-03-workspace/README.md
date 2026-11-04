---
title: "A workspace"
description: "Give the chat a workspace, a folder its tools will work in, built once at startup from a setting, and see what a local workspace gives you and what it does not."
---

# A workspace

## Welcome

Welcome back. We now have one place to choose the model, hand it its key,
and build the chat before it opens. Next we'll give that chat somewhere
to work.

Imagine asking it to inspect a small Python project. Before it can read
a file or run tests, we need to answer a mundane question: **which
project?** A workspace is the place its tools will start from. In this
lesson we'll choose a folder through `Settings` and attach it while
building the application.

We're deliberately **not** adding file or shell tools yet. The chat
will still only talk. That separation helps us see an important
distinction: a workspace tells tools *where to begin*, but doesn't
itself grant the model permission to do anything—or keep tools from
reaching beyond that folder.

By the end, you'll be able to:

1. **Attach a workspace to the chat:** name its folder in `Settings`, build
   it at the composition root, and have the app refuse to start without
   the folder.
2. **Say what a local workspace does not give you:** it gives the agent
   no tools, and it does not keep anything inside its folder.

About 15 minutes. No API key is needed.

## A folder for our existing chat

In 03-02, `Settings` chose a model and `build_app` built it before the conversation opened. Now suppose you ask the chat to inspect a Python project. Which project should it inspect?

A **workspace** gives the tools of an agent a starting place for file operations. We'll use a disposable working copy of a small project called `tinyjobs`. This lesson attaches the workspace but **does not add file tools to the chat**. That distinction matters: naming a folder doesn't give a model the ability to read it.

We'll also see why a local workspace is **not a sandbox**. It starts paths at a folder, but it does not automatically keep them inside that folder. We'll demonstrate this using only a supplied sample project, never your private files.

## What you'll build

```text
pydantic-tutor/
├── sample_code/tinyjobs/       given: original sample project
├── demos/
│   ├── fresh_copy.py          given: creates disposable work/tinyjobs
│   └── workspace_reach.py     new: explore what paths a tool can read
├── work/tinyjobs/             generated: disposable copy, not committed
└── src/pydantic_tutor/
    ├── settings.py            edit: add workspace_root
    ├── chat/agent.py          edit: attach LocalWorkspace
    └── composition.py         edit: check the folder and pass the setting
```

We'll name and create the folder, attach it when the app starts, then observe what it does and does not allow. Stop at any check that doesn't match before moving on.

## Before you start

Continue in your existing `pydantic-tutor` directory from 03-02. From the project root, check that the keyless chat still works:

```bash
PYDANTIC_TUTOR_CHAT_MODEL=test PYDANTIC_TUTOR_WEB_SEARCH=false uv run python -m pydantic_tutor
```

> [!TIP]
> **Check:** type `hello`. You should see `success (no tool calls)`; then finish with an empty line. No API key or paid model request is needed.

> [!WARNING]
> **Already have `pydantic-tutor`? Do not run these clone commands.**
>
> Only if you're entering directly at this lesson, or starting again without an existing project, run them from a parent directory where **no `pydantic-tutor` folder already exists**. Do not delete or replace your existing project: starting again would lose the work and settings you have already added.
>
> ```bash
> git clone --branch 03-02-models/end --single-branch https://github.com/chrislomeli/pydanticai-tutorial-student.git pydantic-tutor
> cd pydantic-tutor
> uv sync
> ```

### Fetch the starter files for this lesson

The next steps use a sample Python project and some helpers that weren't in 03-02. We can add them to **your existing project**, without starting another clone.

From the root of `pydantic-tutor` (where `pyproject.toml` lives), run:

```bash
uv run https://raw.githubusercontent.com/chrislomeli/pydanticai-tutorial-student/main/starter/get_starter.py workspace
```

This fetches `sample_code/tinyjobs/`, `demos/fresh_copy.py`, `demos/show_chat_request.py`, `demos/show_capabilities.py`, and `src/tutor_given/`. The supplied `tutor_given` code and the two `show_` scripts also support later lessons; you won't need to recreate them.

> [!TIP]
> **Check:** the utility lists the files it installed as `wrote`, or `same` if an identical file was already there. It then reminds you about the configuration edits below. If a file already exists with different content, the utility stops without replacing it. Review such a conflict before taking any further action; don't use `--force` without understanding what would be overwritten.

### Keep the disposable copy out of git

Later we'll make `work/tinyjobs/`, a disposable copy where tools can work. It isn't project source and shouldn't be committed.

Open the existing `.gitignore`. Keep the other entries, **especially `.env`**, and add at the bottom:

```gitignore
/work/
```

**Check:** the existing `.env` rule is unchanged; the new rule excludes the `work/` folder at the project root.

### Keep the example project out of our routine checks

The fetched sample has its own code and tests. We want Ruff, mypy and pytest to check our tutorial project without treating the sample and its disposable copy as part of our application.

Open `pyproject.toml`. First, before the existing `[tool.ruff.lint]` section, add this new section:

```toml
[tool.ruff]
extend-exclude = ["sample_code", "work"]
```

**Check:** `[tool.ruff.lint]` and its existing rules are still there. The new section is separate from them.

Inside `[tool.mypy]`, replace **only** the existing `exclude = ...` line with these entries:

```toml
exclude = [
    "^tests/",
    "^sample_code/",
    "^work/",
]
```

**Check:** `strict = true` and the Pydantic mypy plugin remain intact.

Finally, after the `[tool.mypy]` section, add a section for pytest, the test runner:

```toml
[tool.pytest]
addopts = ["--ignore=sample_code", "--ignore=work"]
```

**Check:** none of the project dependencies or build settings were changed. These exclusions protect the example project from *our* routine checks; they do not remove the files.

### Package the given code with yours

`src/tutor_given/` is a second package beside `pydantic_tutor`. Unless told otherwise, uv's build backend packages only the one named after the project, so a wheel built from it would leave `tutor_given` out. Your own runs still work without this, because `uv sync` installs the project in place, but the project should build the same way it runs.

In `pyproject.toml`, after the `[build-system]` section, add:

```toml
[tool.uv.build-backend]
module-name = ["pydantic_tutor", "tutor_given"]
```

**Check:** the section matches the one in the finished file below.

Use the complete result below to compare placement. **Do not paste it over your own file**; the edits above are the exercise.

<details>
<summary>Show the finished <code>pyproject.toml</code> (reference only)</summary>

```toml
[project]
name = "pydantic-tutor"
version = "0.1.0"
description = "Add your description here"
readme = "README.md"
authors = [
    { name = "Ada Student", email = "ada@example.com" }
]
requires-python = ">=3.14"
dependencies = [
    "pydantic==2.14.1",
    "pydantic-ai==2.56.0",
    "pydantic-ai-harness==0.56.0",
    "pydantic-settings==2.16.0",
]

[project.scripts]
pydantic-tutor = "pydantic_tutor:main"

[build-system]
requires = ["uv_build>=0.13.1,<0.14.0"]
build-backend = "uv_build"

[tool.uv.build-backend]
module-name = ["pydantic_tutor", "tutor_given"]

[dependency-groups]
dev = [
    "mypy==2.4.0",
    "pytest==9.1.1",
    "ruff==0.16.10",
]

[tool.ruff]
extend-exclude = ["sample_code", "work"]  # code the agent reads, not code you write

[tool.ruff.lint]
select = [
    "E", "W",  # style: pycodestyle
    "F",       # unused imports and names, undefined names
    "I",       # import order
    "N",       # PEP 8 names
    "UP",      # modern syntax for this Python
    "B",       # likely bugs
    "ANN",     # every parameter and return annotated; ANN401 flags Any
    "FBT",     # no booleans by position, in a call or a def
    "SIM",     # simpler equivalents
    "RUF",     # Ruff's own rules
]
ignore = ["E501"]  # line length is the formatter's job

[tool.mypy]
strict = true
plugins = ["pydantic.mypy"]
exclude = [
    "^tests/",        # the series' lesson tests refer to code you have not written yet
    "^sample_code/",  # code the agent reads, not code you write
    "^work/",         # the agent's working copy of it
]

[tool.pytest]
addopts = ["--ignore=sample_code", "--ignore=work"]  # the sample's tests are not ours
```

</details>

The starter files and these three small configuration updates are now in place. You can continue to Step 1 without changing projects.

## Step 1: Name and create the workspace folder

In 03-01 you gave the model choice a home in `Settings`. We'll do the same for the folder. We want the app to know the project location before it opens.

### Add a path to Settings

Python's `Path` represents a filesystem path. Unlike a plain string, it also has convenient operations, such as `is_dir()` to check whether a folder exists. A **relative** path such as `Path("work/tinyjobs")` begins at the directory where we launch the application.

Open `src/pydantic_tutor/settings.py`. Add this import with the other imports near the top:

```python
from pathlib import Path
```

**Check:** the import is outside `class Settings`, above the existing third-party imports.

The app's paths shouldn't depend on where a program is started, or a run from your IDE would look for them in the wrong folder. `project_root()`, a given function in the `tutor_given` you just fetched, finds the folder holding `pyproject.toml`. Import it after the `pydantic_tutor.model` import:

```python
from tutor_given.paths import project_root
```

In `model_config`, change `env_file` to use it:

```python
        env_file=project_root() / ".env",
```

Inside `Settings`, add the new field **just below `web_search` and above the API-key fields**:

```python
    # The folder the chat works in. `demos/fresh_copy.py` makes the default.
    workspace_root: Path = project_root() / "work" / "tinyjobs"
```

**Check:** `chat_model`, `web_search`, and the provider-key fields still appear as before. The new field sets a default; a shell variable named `PYDANTIC_TUTOR_WORKSPACE_ROOT` can override it.

For comparison, here is the resulting file. **Don't replace your file**; use this only if you want to confirm the placement of the two edits.

<details>
<summary>Show the finished <code>src/pydantic_tutor/settings.py</code> (reference only)</summary>

```python
"""Every value the app is configured with, typed and checked at startup.

Values only: strings and flags read from the environment. The objects built
from them are in `composition.py`.
"""

from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

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
    # One key per provider the composition root can build. SecretStr:
    # printing the settings shows `**********`, never the key.
    openai_api_key: SecretStr | None = Field(
        default=None, validation_alias="OPENAI_API_KEY"
    )
    anthropic_api_key: SecretStr | None = Field(
        default=None, validation_alias="ANTHROPIC_API_KEY"
    )
```

</details>

### Create a disposable working copy

The default names `work/tinyjobs`, but naming a folder doesn't create it. The given helper copies the small project from `sample_code/tinyjobs` into that working directory. **Running it again replaces the disposable copy.** Keep your own work elsewhere.

```bash
uv run demos/fresh_copy.py
```

> [!TIP]
> **Check:** the script reports:
>
> ```text
> work/tinyjobs: a fresh copy of sample_code/tinyjobs, with a fake .env
> ```

That fake `.env` belongs to the sample project, not your own project credentials. Leave your actual `.env` untouched.

### Read the configured location

Now ask `Settings` what it will use:

```bash
uv run python -c "from pydantic_tutor.settings import Settings; print(Settings().workspace_root)"
```

> [!TIP]
> **Check:** it prints the full path of `work/tinyjobs` inside your project, ending in `pydantic-tutor/work/tinyjobs`.

We have chosen a location and made it exist. Nothing has yet connected it to the agent.

## Step 2: Attach the workspace to the existing agent

In 03-02, `build_app` assembled the model from settings before the chat opened. We want the same startup process to attach our workspace.

Pydantic AI calls this sort of agent feature a **capability**. You've already seen `WebSearch` in the chat's capabilities. `LocalWorkspace` is a different capability: it tells tools where to work on this machine. **It does not itself give the model a file tool.**

### Add LocalWorkspace to the agent builder

This adds a `workspace_root` parameter to `build_chat_agent`, and `LocalWorkspace` as the first capability. Replace `src/pydantic_tutor/chat/agent.py` with:

```python
"""The agent the chat talks to.

No model here, as for every agent in this project: the chat passes one on
each run.

Built from these pieces:
- Workspace: the folder the tools work in.
- Web search: switched on by a setting.
"""

from pathlib import Path

from pydantic_ai import Agent
from pydantic_ai.capabilities import AbstractCapability, LocalWorkspace, WebSearch

from tutor_given.paths import project_root


def build_chat_agent(*, web_search: bool, workspace_root: Path) -> Agent[None, str]:
    """Build the chat's agent."""
    capabilities: list[AbstractCapability[None]] = [
        # Workspace: a place for tools to work in. It adds no tool: they use
        # `ctx.workspace`.
        LocalWorkspace(working_dir=workspace_root),
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

`working_dir` is the folder relative paths will start from. The `web_search` block is unchanged: we added a capability and removed none, so the workspace is present whether or not web search is on.

The ready-made `chat_agent` at the bottom needs a default workspace too, because earlier scripts still import it. This doesn't create the folder; Step 1 already did that.

### Check the folder when the app starts

We could attach `LocalWorkspace` without checking the folder. But then the chat might open and only report a missing folder when a future tool tries to read it. Let's instead give the learner a useful startup error.

Open `src/pydantic_tutor/composition.py`. Add the `UserError` import beside the existing Pydantic AI imports:

```python
from pydantic_ai.exceptions import UserError
```

**Check:** the existing `Agent` and model imports remain unchanged.

Within `build_app`, **before `return App(...)`**, insert the folder check:

```python
    if not settings.workspace_root.is_dir():
        raise UserError(
            f"PYDANTIC_TUTOR_WORKSPACE_ROOT is {str(settings.workspace_root)!r}, "
            "which is not a folder. Make the default with "
            "`uv run demos/fresh_copy.py`, or set the variable to another folder."
        )
```

The `is_dir()` method checks that a folder exists. `UserError` is an error the existing chat loop already catches at startup; we don't need to modify the loop.

**Check:** the new `if` block is indented inside `build_app`, before its `return`.

One more focused change in the same `return App(...)`: add `workspace_root` to the call that builds the agent, leaving model construction as it was:

```python
        agent=build_chat_agent(
            web_search=settings.web_search, workspace_root=settings.workspace_root
        ),
```

**Check:** `model=build_model(settings=settings)` still appears underneath. The chat loop remains untouched.

Here is the finished module for comparison only:

<details>
<summary>Show the finished <code>src/pydantic_tutor/composition.py</code> (reference only)</summary>

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
        # Checked here so the chat stops now, by the setting's name, instead
        # of at the first tool that uses the workspace.
        raise UserError(
            f"PYDANTIC_TUTOR_WORKSPACE_ROOT is {str(settings.workspace_root)!r}, "
            "which is not a folder. Make the default with "
            "`uv run demos/fresh_copy.py`, or set the variable to another folder."
        )
    return App(
        agent=build_chat_agent(
            web_search=settings.web_search, workspace_root=settings.workspace_root
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

</details>

### Confirm the chat still works

With the sample folder in place, run the same keyless chat command from Before you start:

```bash
PYDANTIC_TUTOR_CHAT_MODEL=test PYDANTIC_TUTOR_WEB_SEARCH=false uv run python -m pydantic_tutor
```

> [!TIP]
> **Check:** you can still send `hello` and see `success (no tool calls)`. End with an empty line.

The agent now has a local workspace, but the conversation behaves just as before. We haven't given the model anything new to call.

The folder check also means a missing folder will stop the app **before** the first `you>` prompt, with a message naming `PYDANTIC_TUTOR_WORKSPACE_ROOT`. We don't need to break a working setup to demonstrate that in this lesson.

## Step 3: Observe what the workspace allows

We've attached a workspace without adding tools to our chat. There are two separate questions now: **does the model have a file-reading tool?** And **if a tool is provided, does the workspace constrain its paths?**

A tool that reads through the workspace uses `ctx.workspace`, the workspace available during a run. Its `read_text` method is asynchronous: the tool uses `async def` and `await` while it waits for a read. You'll see those details in this demonstration; the next lessons will work with file tools directly.

The demo uses `TestModel` to count the tools offered to the existing agent, then a given stand-in model to request a temporary file-reading tool. `FunctionToolset` supplies that tool **for these runs**, without permanently adding it to the chat. The path is chosen by the stand-in model, not by the workspace itself.

We're going to read just the first lines of **known, non-secret files**: one within the disposable copy and one in our tutorial project. No real credential files are involved.

Create the **new** file `demos/workspace_reach.py`:

```python
"""Show what the chat's workspace is, and what it is not.

Needs no API key. Run `demos/fresh_copy.py`
first.
"""

from pydantic_ai import RunContext
from pydantic_ai.models.test import TestModel
from pydantic_ai.toolsets import FunctionToolset

from pydantic_tutor.chat.agent import build_chat_agent
from pydantic_tutor.settings import Settings
from tutor_given.stand_ins import calls_tool_with_the_prompt

root = Settings().workspace_root
agent = build_chat_agent(web_search=False, workspace_root=root)

# Offer the tools without calling them: this run only counts them.
test_model = TestModel(call_tools=[])
agent.run_sync(user_prompt="hello", model=test_model)
parameters = test_model.last_model_request_parameters
assert parameters is not None  # the run made a request
print(f"tools the model is offered: {len(parameters.function_tools)}")


async def first_line(ctx: RunContext[None], *, path: str) -> str:
    """Read the first line of a file, by a path the model gives."""
    text = await ctx.workspace.read_text(path=path)
    return text.splitlines()[0]


# The folder is `work/tinyjobs`: `..` is `work/`, and `../..` is the project.
outside = "../../pyproject.toml"
paths = {
    "README.md": "README.md",
    outside: outside,
    "absolute path": str((root / outside).resolve()),
}
for label, path in paths.items():
    result = agent.run_sync(
        user_prompt=path,
        model=calls_tool_with_the_prompt(tool_name="first_line", argument="path"),
        # A tool for this run only: the chat has none of its own yet.
        toolsets=[FunctionToolset(tools=[first_line])],
    )
    print(f"{label}: {result.output}")
```

The path `../../pyproject.toml` starts from `work/tinyjobs`, moves up twice, and reaches the tutorial's own project file. The absolute path points at that same file without using the workspace root as a starting point.

Run the demo:

```bash
uv run demos/workspace_reach.py
```

> [!TIP]
> **Check:** the important four lines are:
>
> ```text
> tools the model is offered: 0
> README.md: # tinyjobs
> ../../pyproject.toml: [project]
> absolute path: [project]
> ```
>
> If the disposable working copy is missing, rerun `uv run demos/fresh_copy.py`, then try again.

**What happened?** The configured chat offered the model **zero** function tools; its workspace alone adds none. In the demonstration runs, we temporarily supplied `first_line`. That tool could read a relative file *inside* the workspace, then `../../pyproject.toml` *outside* it, and finally the same file by an absolute path.

That's the key limitation: **a local workspace is a starting directory, not a permission boundary**. A tool must validate paths and permissions independently. `LocalWorkspace` works with the permissions of your process on this machine; it isn't a sandbox. Never try the same experiment with a real `.env` file, because the demo prints what it reads.

## What you can now do

You can **configure a workspace for the same chat**: `Settings` names the folder, `build_app` checks it, and `build_chat_agent` attaches a `LocalWorkspace` before the conversation starts. The chat still streams and remembers messages, and no file tool has been added to it.

You can also explain what the workspace **does not guarantee**: tools can resolve paths outside its starting folder unless the tools enforce a boundary. This distinction matters in the next lesson, where we finally give the chat file tools.

## Next

The chat has a place to work. Next we'll give it file tools that can use the workspace—and must decide which paths they will accept.

---

*Tested with `pydantic-ai` 2.56.0 and `pydantic-ai-harness` 0.56.0.*
