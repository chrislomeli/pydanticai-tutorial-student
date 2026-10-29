"""The composition root: the one place the app's objects are built.

From the settings lesson (03-01); since the models lesson (03-02) it
chooses the model's provider and hands it its key; since the workspace
lesson (03-03) it gives the agent its workspace. `build_app` turns
`Settings` (values) into the agent and the model (objects), at startup, so
a bad model name, a missing API key or a missing workspace folder stops the
app before the chat opens.
"""

from dataclasses import dataclass

from pydantic_ai import Agent
from pydantic_ai.exceptions import UserError
from pydantic_ai.models import Model, infer_model
from pydantic_ai_harness.step_persistence import SqliteStepStore, StepStore

from pydantic_tutor.chat.agent import build_chat_agent
from pydantic_tutor.providers import ProviderFactory, build_provider
from pydantic_tutor.settings import Settings
from pydantic_tutor.stores.conversations import ConversationStore
from pydantic_tutor.stores.sqlite_conversations import SyncSqliteConversationStore


@dataclass(frozen=True)
class App:
    """What the chat needs to run, built once from the settings."""

    agent: Agent[None, str]
    model: Model
    store: ConversationStore | None = None  # None: nothing is saved
    steps: StepStore | None = None  # None: nothing is recorded


def build_app(*, settings: Settings) -> App:
    """Build the agent and the model; `UserError` if anything is missing."""
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
        store=SyncSqliteConversationStore(path=settings.database),
        steps=SqliteStepStore(database=settings.database),
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
