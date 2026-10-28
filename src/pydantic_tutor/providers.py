"""The providers the app can build, and the seam to replace them.

From the models lesson (03-02). `infer_model` asks a function for the
provider of a model string; `ProviderFactory` is that function's type, and
`build_provider` is ours: it hands each provider its key from `Settings`.
Replacing it is passing another function to `build_model`; there is no
registry to register with.
"""

from collections.abc import Callable
from typing import Any

from pydantic import SecretStr
from pydantic_ai.exceptions import UserError
from pydantic_ai.providers import Provider
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.providers.openai import OpenAIProvider

from pydantic_tutor.settings import Settings

# What `infer_model` calls, with the part of the model string before the colon.
# `Provider[Any]`: each provider wraps a different SDK's client, and
# `infer_model` accepts any of them.
ProviderFactory = Callable[[str], Provider[Any]]


def build_provider(*, name: str, settings: Settings) -> Provider[Any]:
    """Build the provider `name`, passing it its key from `settings`.

    The key is passed in, so the provider never reads the environment for
    it. Only the providers listed here can be built: another name is
    refused, rather than left to find its key on its own.
    """
    # Each `variable` is the alias its field is read from in settings.py:
    # rename one, rename both.
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


def require_key(*, key: SecretStr | None, variable: str) -> str:
    """The key's value, or a `UserError` naming the variable that sets it.

    A `UserError`, as the provider itself raises for a missing key, so the
    chat treats both alike.
    """
    if key is None:
        raise UserError(f"Set {variable}, in .env or your shell, to use this model.")
    return key.get_secret_value()
