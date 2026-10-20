"""Checks that the project from the models lesson is in place.

Our checks for the models lesson: they verify the lesson, and the student
is not asked to run them. Run them with `uv run pytest`. They fail until
the lesson is done. None of them needs an API key or makes a request, and
none opens the interactive chat: `to_cli_sync` is replaced where `main()`
would reach it.

Each test runs in an empty folder of its own, so the project's own `.env`,
if it has one, is never read; a test that needs a `.env` writes one there.
A key written only to that file, and never to the process environment,
can reach a provider only by being passed in.

Every import of the project is inside the test that needs it, so that
before the lesson each test fails on its own, instead of the whole file
failing to load.

One test was moved by the file tools lesson (03-04), whose `build_app`
refuses a repository folder that is not there, as this file's empty
folders are: that the entry point opens the chat with the built app, its
key from `.env`, is now proved in `tests/03_04_files/test_files.py`, by
`test_the_entry_point_opens_the_chat_with_the_built_app`.
"""

import os
from pathlib import Path

import pytest
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
from pydantic import SecretStr
from pydantic_ai import models
from pydantic_ai.exceptions import UserError
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel
from pydantic_ai.models.test import TestModel

PREFIX = "PYDANTIC_TUTOR_"
KEY_VARIABLES = ("OPENAI_API_KEY", "ANTHROPIC_API_KEY")


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # Each test starts in an empty folder (no `.env`), with none of this
    # project's variables and no key set, whatever the shell running
    # pytest has, and no real model call.
    for name in list(os.environ):
        if name.upper().startswith(PREFIX) or name.upper() in KEY_VARIABLES:
            monkeypatch.delenv(name)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(target=models, name="ALLOW_MODEL_REQUESTS", value=False)


def write_dotenv(*lines: str) -> None:
    Path(".env").write_text("\n".join(lines) + "\n")


def api_key_of(model: models.Model) -> str:
    # The key the model's SDK client will send: each model class here
    # exposes its client as `.client`, and each client its `api_key`.
    assert isinstance(model, OpenAIResponsesModel | OpenAIChatModel | AnthropicModel)
    client = model.client
    assert isinstance(client, AsyncOpenAI | AsyncAnthropic)
    key = client.api_key
    assert isinstance(key, str)
    return key


# Goal 1: the keys are settings, read from `.env` without entering the
# process environment.


def test_keys_are_unset_by_default() -> None:
    from pydantic_tutor.settings import Settings

    settings = Settings()
    assert settings.openai_api_key is None
    assert settings.anthropic_api_key is None


def test_a_key_in_dotenv_is_read_and_stays_out_of_the_environment() -> None:
    from pydantic_tutor.settings import Settings

    write_dotenv("OPENAI_API_KEY=sk-from-dotenv")
    key = Settings().openai_api_key
    assert key is not None
    assert key.get_secret_value() == "sk-from-dotenv"
    assert "OPENAI_API_KEY" not in os.environ  # read into Settings only


def test_the_shell_wins_over_dotenv(monkeypatch: pytest.MonkeyPatch) -> None:
    from pydantic_tutor.settings import Settings

    write_dotenv("OPENAI_API_KEY=sk-from-dotenv")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-shell")
    key = Settings().openai_api_key
    assert key is not None
    assert key.get_secret_value() == "sk-from-shell"


def test_a_key_never_prints() -> None:
    from pydantic_tutor.settings import Settings

    write_dotenv("OPENAI_API_KEY=sk-secret-value")
    settings = Settings()
    assert settings.openai_api_key is not None  # read, and still not shown
    assert "sk-secret-value" not in str(settings)
    assert "sk-secret-value" not in repr(settings)


def test_an_empty_key_counts_as_unset() -> None:
    from pydantic_tutor.settings import Settings

    write_dotenv("OPENAI_API_KEY=")
    assert Settings().openai_api_key is None


def test_other_tools_lines_in_dotenv_are_ignored() -> None:
    from pydantic_tutor.settings import Settings

    write_dotenv("OPENAI_API_KEY=sk-a", "LOGFIRE_TOKEN=for-another-tool")
    assert Settings().openai_api_key is not None  # no "extra inputs" error


def test_settings_hold_values_not_objects() -> None:
    # Moved from 03-01, widened: a key is a value too, and unset is None.
    from pydantic_tutor.settings import Settings

    write_dotenv("OPENAI_API_KEY=sk-a")
    values = Settings().model_dump()
    assert {"openai_api_key", "anthropic_api_key"} <= values.keys()
    for value in values.values():
        assert isinstance(value, str | bool | int | float | Path | SecretStr | None)


# Goal 2: the composition root chooses the provider from the model string
# and passes it its key.


def test_the_key_from_settings_reaches_the_provider() -> None:
    # The key is only in `.env`, never in the environment: if the client
    # holds it, `build_model` passed it in.
    from pydantic_tutor.composition import build_model
    from pydantic_tutor.settings import Settings

    write_dotenv("OPENAI_API_KEY=sk-passed-in")
    model = build_model(settings=Settings())
    assert isinstance(model, OpenAIResponsesModel)
    assert api_key_of(model) == "sk-passed-in"


def test_a_second_provider_is_a_setting() -> None:
    from pydantic_tutor.composition import build_model
    from pydantic_tutor.settings import Settings

    write_dotenv(
        "PYDANTIC_TUTOR_CHAT_MODEL=anthropic:claude-sonnet-5-5",
        "ANTHROPIC_API_KEY=sk-ant-passed-in",
    )
    model = build_model(settings=Settings())
    assert isinstance(model, AnthropicModel)
    assert model.model_name == "claude-sonnet-5-5"
    assert api_key_of(model) == "sk-ant-passed-in"


def test_the_openai_chat_variant_takes_the_same_key() -> None:
    from pydantic_tutor.composition import build_model
    from pydantic_tutor.settings import Settings

    write_dotenv(
        "PYDANTIC_TUTOR_CHAT_MODEL=openai-chat:gpt-5-mini", "OPENAI_API_KEY=sk-o"
    )
    model = build_model(settings=Settings())
    assert isinstance(model, OpenAIChatModel)
    assert api_key_of(model) == "sk-o"


def test_the_test_model_needs_no_key() -> None:
    from pydantic_tutor.composition import build_model
    from pydantic_tutor.settings import Settings

    write_dotenv("PYDANTIC_TUTOR_CHAT_MODEL=test")
    assert isinstance(build_model(settings=Settings()), TestModel)


@pytest.mark.parametrize(
    argnames=("chat_model", "variable"),
    argvalues=[
        ("openai:gpt-5-mini", "OPENAI_API_KEY"),
        ("anthropic:claude-sonnet-5-5", "ANTHROPIC_API_KEY"),
    ],
)
def test_a_missing_key_fails_by_name_before_any_request(
    chat_model: str, variable: str
) -> None:
    # Moved from 03-01, now for each provider, and raised by the app.
    from pydantic_tutor.composition import build_model
    from pydantic_tutor.settings import Settings

    write_dotenv(f"PYDANTIC_TUTOR_CHAT_MODEL={chat_model}")
    with pytest.raises(UserError) as excinfo:
        build_model(settings=Settings())
    assert variable in str(excinfo.value)


def test_an_empty_key_fails_like_a_missing_one() -> None:
    # In 03-01 this was the OpenAI SDK's own error, with a traceback.
    from pydantic_tutor.composition import build_model
    from pydantic_tutor.settings import Settings

    write_dotenv("OPENAI_API_KEY=")
    with pytest.raises(UserError) as excinfo:
        build_model(settings=Settings())
    assert "OPENAI_API_KEY" in str(excinfo.value)


def test_a_provider_with_no_key_setting_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Even with that provider's key in the environment: the app builds
    # only the providers it passes a key to.
    from pydantic_tutor.composition import build_model
    from pydantic_tutor.settings import Settings

    monkeypatch.setenv("GROQ_API_KEY", "gsk-in-the-environment")
    write_dotenv("PYDANTIC_TUTOR_CHAT_MODEL=groq:llama-3.3-70b-versatile")
    with pytest.raises(UserError) as excinfo:
        build_model(settings=Settings())
    assert "groq:llama-3.3-70b-versatile" in str(excinfo.value)


# The entry point reads `.env` through `Settings`, and nothing else.


@pytest.mark.parametrize(
    argnames=("dotenv", "named"),
    argvalues=[
        ("PYDANTIC_TUTOR_TRACE=maybe", "trace"),
        ("PYDANTIC_TUTOR_CHAT_MODEL=nonsense:x", "nonsense:x"),
        ("PYDANTIC_TUTOR_CHAT_MODEL=openai:gpt-5-mini", "OPENAI_API_KEY"),
        ("OPENAI_API_KEY=", "OPENAI_API_KEY"),
        ("PYDANTIC_TUTOR_CHAT_MODEL=anthropic:claude-sonnet-5-5", "ANTHROPIC_API_KEY"),
    ],
)
def test_the_entry_point_stops_by_name_before_the_chat_opens(
    monkeypatch: pytest.MonkeyPatch, dotenv: str, named: str
) -> None:
    # Moved from 03-01, with an empty key and the second provider added.
    import pydantic_tutor.__main__ as entry
    from pydantic_tutor import agent as agent_module

    def chat_must_not_open(**kwargs: object) -> None:
        raise AssertionError("the chat opened")

    monkeypatch.setattr(
        target=agent_module.agent, name="to_cli_sync", value=chat_must_not_open
    )
    write_dotenv(dotenv)
    with pytest.raises(SystemExit) as excinfo:
        entry.main()
    message = str(excinfo.value.code)
    assert message.startswith("pydantic-tutor cannot start:")
    assert named in message
