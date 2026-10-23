"""The values this application is configured with, typed and checked.

From the settings lesson (03-01); since the models lesson (03-02) it
holds the API keys too, and reads `.env` itself. Kept: a later lesson
that needs a new value adds a field here, and reads it nowhere else.
"""

from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Values only: strings, flags, paths, keys. Nothing here builds an object.

    `Settings()` reads each field from the environment variable named
    `PYDANTIC_TUTOR_` plus the field's name, in any case: `chat_model`
    from `PYDANTIC_TUTOR_CHAT_MODEL`. The API keys are the exception: they
    keep the names every tool uses, such as `OPENAI_API_KEY`. Each is read
    from the shell first, then from `.env` in the folder the app runs in.
    A field with neither keeps its default. A value that does not fit the
    field's type raises a `ValidationError` naming the field, before
    anything else runs.
    """

    model_config = SettingsConfigDict(
        env_prefix="PYDANTIC_TUTOR_",
        # Read here, into these fields only: a key in `.env` never enters
        # the process environment.
        env_file=".env",
        # `OPENAI_API_KEY=` with nothing after it counts as unset.
        env_ignore_empty=True,
        # `.env` may hold variables for other tools; ignore them, as the
        # shell's are.
        extra="ignore",
        # Frozen: settings are read once, at startup, and never changed after.
        frozen=True,
    )

    # The model the chat talks to, as "<provider>:<model name>".
    chat_model: str = "openai:gpt-5-mini"
    # Console tracing. Off unless set to a true value: 1, true, yes, on, t or y.
    trace: bool = False
    # The codebase the agent works on: its tools read it, and may change
    # it. A relative path is read from the folder the app runs in: run it
    # from the project root. The default is the working copy that
    # scripts/fresh_copy.py makes, never the sample codebase itself.
    codebase_root: Path = Path("work/tinyjobs")
    # One key per provider the composition root can build. SecretStr:
    # printing the settings shows `**********`, never the key.
    openai_api_key: SecretStr | None = Field(
        default=None, validation_alias="OPENAI_API_KEY"
    )
    anthropic_api_key: SecretStr | None = Field(
        default=None, validation_alias="ANTHROPIC_API_KEY"
    )
