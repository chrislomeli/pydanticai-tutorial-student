"""Build only the model providers this chat supports."""

from collections.abc import Callable
from typing import Any

from pydantic import SecretStr
from pydantic_ai.exceptions import UserError
from pydantic_ai.providers import Provider
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.providers.openai import OpenAIProvider

from pydantic_tutor.settings import Settings

# The function shape expected by infer_model's provider_factory argument.
ProviderFactory = Callable[[str], Provider[Any]]


def require_key(*, key: SecretStr | None, variable: str) -> str:
    """Return a configured key or name the missing setting."""
    if key is None:
        raise UserError(f"Set {variable}, in .env or your shell, to use this model.")
    return key.get_secret_value()


def build_provider(*, name: str, settings: Settings) -> Provider[Any]:
    """Choose a supported provider and give it its configured key."""
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
