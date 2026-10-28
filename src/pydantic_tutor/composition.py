"""The composition root: the one place the app's objects are built.

From the settings lesson (03-01); since the models lesson (03-02) it
chooses the model's provider and hands it its key. `build_app` turns
`Settings` (values) into the agent and the model (objects), at startup, so
a bad model name or a missing API key stops the app before the chat opens.
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

    Raises `pydantic_ai.exceptions.UserError` if the model is unknown, names
    a provider this app does not build, or its provider's key is not set.
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
