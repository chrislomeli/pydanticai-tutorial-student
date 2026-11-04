"""Every value the app is configured with, typed and checked at startup.

Values only: strings and flags read from the environment. The objects built
from them are in `composition.py`.
"""

from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from pydantic_tutor.capabilities.activity_log import HookName
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
    # The activity log's file. Beside the workspace and not in it, so the file
    # tools can't reach it. `work/` is already ignored by git.
    activity_log_path: Path = project_root() / "work" / "activity.jsonl"
    # Which hooks the activity log records. An empty list turns it off.
    activity_log: frozenset[HookName] = frozenset(HookName)
    # One key per provider the composition root can build. SecretStr:
    # printing the settings shows `**********`, never the key.
    openai_api_key: SecretStr | None = Field(
        default=None, validation_alias="OPENAI_API_KEY"
    )
    anthropic_api_key: SecretStr | None = Field(
        default=None, validation_alias="ANTHROPIC_API_KEY"
    )
