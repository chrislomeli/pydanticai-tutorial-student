"""Every value the app is configured with, typed and checked at startup.

From the settings lesson (03-01). Values only: strings and flags read from
the environment. The objects built from them are in `composition.py`.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict

from pydantic_tutor.model import CHAT_MODEL


class Settings(BaseSettings):
    """Read from variables named `PYDANTIC_TUTOR_<FIELD>`; frozen once created."""

    model_config = SettingsConfigDict(env_prefix="PYDANTIC_TUTOR_", frozen=True)

    # A `<provider>:<model>` string, the name of the model: not the model.
    chat_model: str = CHAT_MODEL
    # Whether the chat may search the web. Each search is billed.
    web_search: bool = True
