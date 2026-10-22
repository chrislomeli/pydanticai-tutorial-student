"""The composition root: the one place that builds this application's objects.

From the settings lesson (03-01); since the models lesson (03-02) it
chooses the model's provider and hands it its key. Kept: a later lesson
that needs a new object builds it here, from `Settings`, and passes it to
what uses it.
"""

from dataclasses import dataclass
from typing import Any

import logfire
from pydantic import SecretStr
from pydantic_ai import Agent
from pydantic_ai.exceptions import UserError
from pydantic_ai.models import Model, infer_model
from pydantic_ai.providers import Provider
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.providers.openai import OpenAIProvider

from pydantic_tutor.agent import agent
from pydantic_tutor.conversations import ConversationStore
from pydantic_tutor.repo_tools import RepoDeps
from pydantic_tutor.settings import Settings


@dataclass(frozen=True)
class App:
    """Everything the chat needs, built and ready to run."""

    agent: Agent[RepoDeps, str]
    model: Model
    deps: RepoDeps
    # Where the chat saves its conversations. None: nothing is saved, and
    # every chat starts a new one, as the scripts' stand-in chats do.
    store: ConversationStore | None = None


def build_app(*, settings: Settings) -> App:
    """Build the application's objects from `settings`, once, at startup.

    The model is built here, not on the first prompt, so a model string
    that names no known model, or a provider whose API key is not set,
    fails now, before the chat opens. So does a repository folder that is
    not there.

    Raises:
        UserError: from `build_model`, or `repo_root` is not a folder.
    """
    if settings.trace:
        configure_tracing()
    model = build_model(settings=settings)
    if not settings.repo_root.is_dir():
        # Otherwise the chat opens, and the first question that reaches the
        # workspace fails instead.
        raise UserError(
            message=f"The repository {str(settings.repo_root)!r} is not a "
            "folder. Set PYDANTIC_TUTOR_REPO_ROOT to one, or make the default "
            "with `uv run scripts/fresh_copy.py`."
        )
    return App(
        agent=agent,
        model=model,
        deps=RepoDeps(root=settings.repo_root),
        store=ConversationStore(path=settings.database),
    )


def build_model(*, settings: Settings) -> Model:
    """Build the model `settings.chat_model` names, its provider holding its key.

    `infer_model` picks the model class from the string, and asks
    `build_provider` for the provider, by the part before the colon.

    Raises:
        UserError: the model string is unknown, names a provider this app
            does not build, or that provider's key is not set.
    """
    return infer_model(
        model=settings.chat_model,
        # `infer_model` calls this with one argument, the provider's name.
        provider_factory=lambda name: build_provider(name=name, settings=settings),
    )


# `Provider[Any]`: each provider wraps a different SDK's client, and
# `infer_model` accepts any of them.
def build_provider(*, name: str, settings: Settings) -> Provider[Any]:
    """Build the provider `name`, passing it its key from `settings`.

    The key is passed in, so the provider never reads the environment for
    it. Only the providers listed here can be built: another name is
    refused, rather than left to find its key on its own.
    """
    # Each `variable` is the alias its field is read from in settings.py:
    # rename one, rename both.
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


def require_key(*, key: SecretStr | None, variable: str) -> str:
    """The key's value, or a `UserError` naming the variable that sets it.

    A `UserError`, as the provider itself raises for a missing key, so the
    entry point treats both alike.
    """
    if key is None:
        raise UserError(f"Set {variable}, in .env or your shell, to use this model.")
    return key.get_secret_value()


def configure_tracing() -> None:
    """Print a span for each run, model request and tool call, to this terminal.

    `send_to_logfire=False`: nothing leaves the machine, and no Logfire
    account or OpenTelemetry collector is needed.
    """
    logfire.configure(
        send_to_logfire=False, console=logfire.ConsoleOptions(colors="auto")
    )
    logfire.instrument_pydantic_ai()
    print("tracing on (PYDANTIC_TUTOR_TRACE is true); spans print to this terminal")
