"""Every value the app is configured with, typed and checked at startup.

From the settings lesson (03-01); since the models lesson (03-02) it holds
the API keys too, and reads `.env` itself; since the workspace lesson
(03-03) it names the workspace's folder. Values only: the objects built
from them are in `composition.py`.
"""

from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from pydantic_tutor.model import CHAT_MODEL


class Settings(BaseSettings):
    """Read from variables named `PYDANTIC_TUTOR_<FIELD>`; frozen once created.

    The API keys are the exception: they keep the names every tool uses.
    Each field is read from the shell first, then from `.env` in the folder
    the app runs in, then its default.
    """

    model_config = SettingsConfigDict(
        env_prefix="PYDANTIC_TUTOR_",
        # Read here, into these fields only: a key in `.env` never enters
        # the process environment.
        env_file=".env",
        # `OPENAI_API_KEY=` with nothing after it counts as unset.
        env_ignore_empty=True,
        # `.env` may hold variables for other tools; skip them.
        extra="ignore",
        frozen=True,
    )

    # A `<provider>:<model>` string, the name of the model: not the model.
    chat_model: str = CHAT_MODEL
    # Whether the chat may search the web. Each search is billed.
    web_search: bool = True
    # The folder the chat works in. Relative to the folder the app runs in;
    # `demos/fresh_copy.py` makes the default.
    workspace_root: Path = Path("work/tinyjobs")
    # The SQLite file the app keeps its state in; made on first use.
    database: Path = Path("pydantic-tutor.db")
    # One key per provider the composition root can build. SecretStr:
    # printing the settings shows `**********`, never the key.
    openai_api_key: SecretStr | None = Field(
        default=None, validation_alias="OPENAI_API_KEY"
    )
    anthropic_api_key: SecretStr | None = Field(
        default=None, validation_alias="ANTHROPIC_API_KEY"
    )
