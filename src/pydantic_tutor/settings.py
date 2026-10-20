"""The values this application is configured with, typed and checked.

From the settings lesson (03-01). Kept: a later lesson that needs a new
value adds a field here, and reads it nowhere else.
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Values only: strings, flags, paths. Nothing here builds an object.

    `Settings()` reads each field from the environment variable named
    `PYDANTIC_TUTOR_` plus the field's name, in any case: `chat_model`
    from `PYDANTIC_TUTOR_CHAT_MODEL`. A field with no variable keeps its
    default. A value that does not fit the field's type raises a
    `ValidationError` naming the field, before anything else runs.
    """

    # Frozen: settings are read once, at startup, and never changed after.
    model_config = SettingsConfigDict(env_prefix="PYDANTIC_TUTOR_", frozen=True)

    # The model the chat talks to, as "<provider>:<model name>".
    chat_model: str = "openai:gpt-5-mini"
    # Console tracing. Off unless set to a true value: 1, true, yes, on, t or y.
    trace: bool = False
    # The repository the agent answers questions about. A relative path is
    # read from the folder the app runs in: run it from the project root.
    repo_root: Path = Path("sample_repos/tinyjobs")
