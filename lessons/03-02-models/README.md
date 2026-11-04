---
title: "Models and providers"
description: "Give the existing chat an explicit, secret-safe way to construct OpenAI and Anthropic models."
---

# Models and providers

## Welcome

Welcome back. In 03-01, you moved the chat's choices into `Settings` and gave `build_app` the job of assembling the agent and model before the conversation opens. The chat itself still streams answers and remembers its messages.

You can now choose a model by changing a setting. But there's one part of model construction we haven't taken control of: **how the model service gets its API key**. Until now, Pydantic AI has been free to look for that key in the environment.

In this lesson, we'll prepare our chat to work with two model providers: OpenAI and Anthropic. We'll teach our application where to find each provider's API key and how to pass it along when creating a model, without changing the conversation loop.

**We won't send a request to either provider yet.** Instead, we'll use made-up API keys and run small checks to confirm that our application reads the right settings, selects the right provider, and passes the key where it belongs. That way, we can understand how the pieces fit together without needing real credentials or paying for model requests.

By the end, you'll be able to:

1. **Read API keys through `Settings` without printing their values**, and pass the selected key to a provider.
2. **Build either an OpenAI or an Anthropic model from configuration**, leaving the chat loop alone.

## Where the key goes

The **model** is the object the agent uses to request an answer. A **provider** connects that model to a service that hosts it. For an OpenAI model, an `OpenAIProvider` supplies the connection details, including an API key. An `AnthropicProvider` plays the same role for Anthropic.

Think about the sequence *before* a chat message is sent:

1. `Settings` reads the model choice and any configured keys.
2. Our code creates the provider for that choice and hands it the right key.
3. Pydantic AI constructs the model using that provider.
4. `build_app` gives the model to the same chat you've already built.

Our code will choose a service and hand it the corresponding key. Pydantic AI still handles requests and responses. We are changing how the application starts, not how the chat converses.

We will make those changes in three files:

```text
src/pydantic_tutor/
├── settings.py       edit: read keys and .env
├── providers.py      new: select a provider and pass its key
├── composition.py    edit: build models using our provider selection
└── chat/loop.py      unchanged
```

The last line matters: we're improving the existing application, not building a separate example.

## Before you start

Continue in your existing `pydantic-tutor` directory from 03-01. From the project root, open the keyless test chat:

```bash
PYDANTIC_TUTOR_CHAT_MODEL=test PYDANTIC_TUTOR_WEB_SEARCH=false uv run python -m pydantic_tutor
```

> [!TIP]
> **Check:** enter `hello`, see the test model answer, and end with an empty line. If that doesn't work, resolve it before editing; this lesson builds on that application.

> [!WARNING]
> **Already have the project? Do not run these clone commands.**
>
> Only if you're joining at 03-02 or starting again without an existing project, run these from a parent directory where **no `pydantic-tutor` folder exists**. Every lesson uses that same project folder name. If you already have `pydantic-tutor`, keep using it; do not delete or replace it. Starting over would lose the work and settings in your existing project:
>
> ```bash
> git clone --branch 03-01-settings/end --single-branch https://github.com/chrislomeli/pydanticai-tutorial-student.git pydantic-tutor
> cd pydantic-tutor
> uv sync
> ```

## Step 1: Add provider keys to Settings

In 03-01, `Settings` held two choices: `chat_model` and `web_search`. We're going to keep them, add one optional key for each provider, and teach `Settings` to read the same `.env` file you've been using. None of this contacts a model service.

### Import the two Pydantic helpers

An API key is sensitive text. Pydantic's `SecretStr` stores it as a string that is **masked when displayed**; it is not encryption, and your code can still retrieve the value deliberately. `Field` lets us configure how `Settings` reads a field.

Open `src/pydantic_tutor/settings.py`. Beneath the module docstring, add this import beside the existing `pydantic_settings` import:

```python
from pydantic import Field, SecretStr
```

**Check:** the new import is at the top of the module, not inside `Settings`. Keep the existing imports.

### Add optional keys with their usual environment names

Normally, `Settings` looks for variables beginning with `PYDANTIC_TUTOR_`. Provider keys already have familiar names: `OPENAI_API_KEY` and `ANTHROPIC_API_KEY`. A `validation_alias` tells Pydantic Settings to read that exact name instead of deriving one from the field name.

The annotation `SecretStr | None` means a key can be missing. That matters for our no-key test model: it should still start without real credentials. `default=None` gives the field that starting value.

Inside `class Settings`, immediately after the existing `web_search` field, add:

```python
    # One key per provider the composition root can build. SecretStr:
    # printing the settings shows `**********`, never the key.
    openai_api_key: SecretStr | None = Field(
        default=None, validation_alias="OPENAI_API_KEY"
    )
    anthropic_api_key: SecretStr | None = Field(
        default=None, validation_alias="ANTHROPIC_API_KEY"
    )
```

**Check:** both fields are indented inside `Settings`; `chat_model` and `web_search` are unchanged. At this point these fields *read* keys, but nothing passes a key to a provider yet.

### Tell Settings to read .env automatically

In 03-01, we passed `_env_file` explicitly when we wanted to read a practice file. Now we'll set a default. `env_file=".env"` means look for `.env` in the working directory when `Settings()` is created. A shell variable takes priority over the corresponding entry in that file.

There are two details that matter when reading an existing `.env`:

- `env_ignore_empty=True` means an empty environment value is treated as absent instead of as an actual key.
- `extra="ignore"` means unrelated names in the file won't cause a validation error. We shouldn't require every entry in your `.env` to belong to this app.

Locate the existing single-line `model_config = SettingsConfigDict(...)` inside `Settings`. Keep its **existing** `env_prefix` and `frozen` arguments, but expand it into this block:

```python
    model_config = SettingsConfigDict(
        env_prefix="PYDANTIC_TUTOR_",
        frozen=True,
        # Read here, into these fields only: a key in `.env` never enters
        # the process environment. A shell variable wins over `.env`.
        env_file=".env",
        # `OPENAI_API_KEY=` with nothing after it counts as unset.
        env_ignore_empty=True,
        # `.env` may hold variables for other tools; skip them.
        extra="ignore",
    )
```

**Check:** `model_config` remains inside `Settings`, and all five arguments are present. This changes where settings can come from, not their original defaults.

**Important:** don't delete, clear, or replace your real `.env`. It may hold credentials or settings you still need.

### Compare your finished Settings file

These were incremental changes to an existing file. Compare them with this **optional finished-file reference**. Do not paste it over your file; the small edits above are the exercise.

<details>
<summary>Show the finished <code>src/pydantic_tutor/settings.py</code></summary>

```python
"""Every value the app is configured with, typed and checked at startup.

Values only: strings and flags read from the environment. The objects built
from them are in `composition.py`.
"""

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from pydantic_tutor.model import CHAT_MODEL


class Settings(BaseSettings):
    """Read from variables named `PYDANTIC_TUTOR_<FIELD>`; frozen once created."""

    model_config = SettingsConfigDict(
        env_prefix="PYDANTIC_TUTOR_",
        frozen=True,
        # Read here, into these fields only: a key in `.env` never enters
        # the process environment. A shell variable wins over `.env`.
        env_file=".env",
        # `OPENAI_API_KEY=` with nothing after it counts as unset.
        env_ignore_empty=True,
        # `.env` may hold variables for other tools; skip them.
        extra="ignore",
    )

    # A `<provider>:<model>` string, the name of the model: not the model.
    chat_model: str = CHAT_MODEL
    # Whether the chat may search the web. Each search is billed.
    web_search: bool = True
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

### Verify that the key is read and masked

We'll test one field using a **disposable** file with a made-up key. This is deliberately separate from your real `.env`. From the project root, create it:

```bash
printf 'OPENAI_API_KEY=sk-fake-for-the-lesson\n' > lesson-provider.env
```

**Check:** open `lesson-provider.env`; it should contain exactly the single line above. Don't put a real key in it.

Now read it through `Settings`. The `_env_file` argument overrides the usual `.env` **for this instance only**. `env -u OPENAI_API_KEY` temporarily removes a shell-level OpenAI key for this command so it cannot override our placeholder:

```bash
env -u OPENAI_API_KEY uv run python -c "from pydantic_tutor.settings import Settings; print(Settings(_env_file='lesson-provider.env').openai_api_key)"
```

> [!TIP]
> **Check:** it should print `**********`, not `sk-fake-for-the-lesson`. If it does not, stop here and inspect the field name, alias, and file path.

**What it means:** Pydantic Settings read a key from the practice file and `SecretStr` masked it when printed. This check does **not** authenticate a key, nor does masking prevent code from retrieving the real string with `get_secret_value()`. We'll pass that string to a provider in Step 2.

## Step 2: Give provider construction one home

We can read keys, but `build_app` still doesn't pass them to providers. We'll add a **new** file, `src/pydantic_tutor/providers.py`, for that responsibility.

A **factory function** is simply a function whose job is to create and return an object. Our `build_provider` factory will receive a provider name and the current `Settings`, and return the matching provider with its key.

Before writing the factory, there are two small pieces to recognize:

- `require_key` will check for a missing key before we try to create a provider. If a key exists, it will retrieve its actual string value.
- `ProviderFactory` will describe the *shape of a function* Pydantic AI can call: given a provider name, return a provider. This is a Python **type alias**, not a registration system. `Provider[Any]` accommodates the differing client types wrapped by OpenAI and Anthropic providers; it doesn't authorize every possible provider.

Because `providers.py` is new, we can show the full file. The two decisions to follow are whether the chosen key is present and which provider should receive it. The `match` statement selects one `case` according to the provider name.

Create `src/pydantic_tutor/providers.py`:

```python
"""Build only the model providers this chat supports."""

from collections.abc import Callable
from typing import Any

from pydantic import SecretStr
from pydantic_ai.exceptions import UserError
from pydantic_ai.providers import Provider
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.providers.openai import OpenAIProvider

from pydantic_tutor.settings import Settings

# The function shape expected by infer_model's provider_factory argument.
ProviderFactory = Callable[[str], Provider[Any]]


def require_key(*, key: SecretStr | None, variable: str) -> str:
    """Return a configured key or name the missing setting."""
    if key is None:
        raise UserError(f"Set {variable}, in .env or your shell, to use this model.")
    return key.get_secret_value()


def build_provider(*, name: str, settings: Settings) -> Provider[Any]:
    """Choose a supported provider and give it its configured key."""
    match name:
        case "openai" | "openai-chat" | "openai-responses":
            return OpenAIProvider(
                api_key=require_key(
                    key=settings.openai_api_key, variable="OPENAI_API_KEY"
                )
            )
        case "anthropic":
            return AnthropicProvider(
                api_key=require_key(
                    key=settings.anthropic_api_key, variable="ANTHROPIC_API_KEY"
                )
            )
        case _:
            raise UserError(
                f"Unknown provider {name!r} in {settings.chat_model!r}: "
                "pydantic-tutor builds openai and anthropic models only."
            )
```

Follow the OpenAI case. The `match` statement chooses a branch by name; three OpenAI-related names use the same OpenAI provider. `require_key` checks our setting and returns the underlying string; `OpenAIProvider(api_key=...)` receives it explicitly. The Anthropic case does the same with the Anthropic field.

The final case refuses unsupported providers rather than quietly letting some other provider obtain a key on its own. This function defines our policy about what the application can build. It does not, by itself, decide who is authorized to use a credential.

**Check the new file:** OpenAI cases read `settings.openai_api_key`; the Anthropic case reads `settings.anthropic_api_key`. Both pass the result of `require_key` as `api_key`. The fallback case rejects unsupported names.

**What changed?** We've written the provider factory, but no application code calls it yet. That is the next edit.

## Step 3: Connect the provider factory to model construction

Recall `composition.py` from 03-01: `build_app` assembles the chat's agent and model before the user sees a prompt. We will keep that function and change **only how it gets a model**.

### Import the model type and provider factory

We need the model's type annotation and our new provider-building function before we can use them.

In `src/pydantic_tutor/composition.py`, keep the existing imports, including `from pydantic_ai.models import Model, infer_model`, and add the import from our new module below it:

```python
from pydantic_ai.models import Model, infer_model
from pydantic_tutor.providers import ProviderFactory, build_provider
```

`Model` describes the type the new function returns; it doesn't construct an object.

**Check:** `infer_model` is still imported, along with `Model`, and `build_provider` comes from your new module.

### Add build_model

A model name such as `openai:gpt-4o-mini` identifies both the provider and the model. `infer_model` already understands that name; its `provider_factory` option lets us decide how its provider is created.

The `lambda` below is a small unnamed function. When Pydantic AI asks it for a provider name, it forwards that name and the current `Settings` to `build_provider`. The optional `provider_factory` argument allows a different function to be supplied later; we won't use that option here.

At the **end of `composition.py`**, after `build_app`, add:

```python
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

**Check:** `build_model` is at the same indentation level as `build_app` and returns the result of `infer_model`. At this point you've defined it, but `build_app` hasn't yet called it.

### Use build_model in build_app

Find the existing `return App(` inside `build_app`. **Leave its agent line untouched.** Replace only the expression that constructs its model:

```python
        model=build_model(settings=settings),
```

**Check:** `build_app` still constructs the agent in the same way; only its `model=` expression changed. The model is still built before the chat opens. No edit to `chat/loop.py` is needed.

### Check your finished composition.py

You've made several focused edits to an existing file. To check their placement, compare your result with this **complete reference version**. This is not another editing instruction: keep the edits you've just made rather than copying over your file.

<details>
<summary>Show the finished <code>src/pydantic_tutor/composition.py</code></summary>

```python
"""The composition root: the one place the app's objects are built.

`build_app` turns `Settings` (values) into the agent and the model (objects).
The model is built here, at startup, so an unknown provider or a missing API key
stops the app before the chat opens, not when the first message is sent.
"""
from dataclasses import dataclass

from pydantic_ai import Agent
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
    return App(
        agent=build_chat_agent(web_search=settings.web_search),
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

**Check:** you should have the two imports, the updated `model=` line in `build_app`, and the new `build_model` function after it. The `App` class and the agent line should be unchanged.

### Check the OpenAI handoff without contacting OpenAI

We can check construction with the placeholder in `lesson-provider.env`. This one command also chooses the OpenAI model explicitly, rather than relying on the current default. It compares the key in the created model's client with the key in `Settings`, but **prints only the result of the comparison**, never the key.

```bash
env -u OPENAI_API_KEY PYDANTIC_TUTOR_CHAT_MODEL=openai:gpt-4o-mini uv run python -c "
from pydantic_tutor.composition import build_model
from pydantic_tutor.settings import Settings
settings = Settings(_env_file='lesson-provider.env')
model = build_model(settings=settings)
print(type(model).__name__, model.model_name)
print(model.client.api_key == settings.openai_api_key.get_secret_value())
"
```

> [!TIP]
> **Check:** expect:
>
> ```text
> OpenAIResponsesModel gpt-4o-mini
> True
> ```

`True` tells us the provider's client received the same value that `Settings` read. It does **not** mean the made-up key is valid. We built an object; we didn't send a request.

## Step 4: Use the same function for Anthropic

The OpenAI check established that our placeholder reached the client's configuration; it did not authenticate. Now change only the provider choice and the disposable credential to see the same construction path select Anthropic.

The second goal is to select a different provider without rewriting our chat. This time a one-command environment variable supplies a disposable Anthropic key. The `chat_model` argument chooses the model, and `_env_file=None` skips the local file for this check.

```bash
ANTHROPIC_API_KEY=sk-ant-placeholder uv run python -c "
from pydantic_tutor.composition import build_model
from pydantic_tutor.settings import Settings
settings = Settings(_env_file=None, chat_model='anthropic:claude-sonnet-5-5')
model = build_model(settings=settings)
print(type(model).__name__, model.model_name)
"
```

> [!TIP]
> **Check:** the output should name an Anthropic model and `claude-sonnet-5-5`. No model request is made.

The same `build_model` function routed the model name to the Anthropic provider and passed it the Anthropic setting. The chat still receives a model from `build_app`; it doesn't need to know which company serves it.

There's a boundary worth keeping: at startup we can tell whether a key was *supplied*. Only the remote service can confirm whether a real key is *valid* when we make a request.

### Remove the practice file

When you've finished checking, remove only the disposable file we created:

```bash
rm lesson-provider.env
```

**Check:** the disposable `lesson-provider.env` is gone. Your actual `.env` and any real credentials remain untouched.

## What you can now do

You wanted to make the model's credentials an explicit part of application configuration. `Settings` now reads and masks provider keys; `build_provider` chooses the supported service and hands it the matching key.

You also wanted one chat that can use more than one provider. `build_model` now builds either OpenAI or Anthropic models, while `build_app` and the existing conversation loop keep their roles. The checks demonstrated local construction and handoff—not remote authentication. A real request with a real credential remains outside this lesson.

## Next

In the next lesson, we'll give this same chat a workspace: a folder its tools can read from and write to.

---

*Tested with `pydantic-ai` 2.56.0 and `pydantic-ai-harness` 0.56.0.*
