"""The composition root: the one place that builds this application's objects.

From the settings lesson (03-01). Kept: a later lesson that needs a new
object builds it here, from `Settings`, and passes it to what uses it.
"""

from dataclasses import dataclass

import logfire
from pydantic_ai import Agent
from pydantic_ai.models import Model, infer_model

from pydantic_tutor.agent import agent
from pydantic_tutor.codebase_tools import CodebaseDeps
from pydantic_tutor.settings import Settings


@dataclass(frozen=True)
class App:
    """Everything the chat needs, built and ready to run."""

    agent: Agent[CodebaseDeps, str]
    model: Model
    deps: CodebaseDeps


def build_app(*, settings: Settings) -> App:
    """Build the application's objects from `settings`, once, at startup.

    The model is built here, not on the first prompt, so a model string
    that names no known model, or a provider whose API key is not set,
    fails now, before the chat opens.

    Raises:
        UserError: the model string is unknown, or its provider has no key.
    """
    if settings.trace:
        configure_tracing()
    return App(
        agent=agent,
        model=infer_model(model=settings.chat_model),
        deps=CodebaseDeps(root=settings.codebase_root),
    )


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
