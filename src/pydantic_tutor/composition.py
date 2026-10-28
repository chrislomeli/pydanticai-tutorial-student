"""The composition root: the one place the app's objects are built.

From the settings lesson (03-01). `build_app` turns `Settings` (values)
into the agent and the model (objects). The model is built here, at
startup, so a bad model name or a missing API key stops the app before the
chat opens, not when the first message is sent.
"""

from dataclasses import dataclass

from pydantic_ai import Agent
from pydantic_ai.models import Model, infer_model

from pydantic_tutor.chat.agent import build_chat_agent
from pydantic_tutor.settings import Settings


@dataclass(frozen=True)
class App:
    """What the chat needs to run, built once from the settings."""

    agent: Agent[None, str]
    model: Model


def build_app(*, settings: Settings) -> App:
    """Build the agent and the model from `settings`.

    Raises `pydantic_ai.exceptions.UserError` if the model name is unknown
    or its provider's API key is not set.
    """
    return App(
        agent=build_chat_agent(web_search=settings.web_search),
        model=infer_model(settings.chat_model),
    )
