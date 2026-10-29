"""Checks that the project from the models lesson is in place.

Our checks for the models lesson: they verify the lesson, and the student
is not asked to run them. Run them with `uv run pytest`. They fail until
the lesson is done. None of them needs an API key or makes a request:
building a provider with a key makes no network call.

Each test runs in an empty folder of its own, so the project's own `.env`,
if it has one, is never read; a test that needs a `.env` writes one there.
A key written only to that file, and never to the process environment,
can reach a provider only by being passed in.

One check moved here from `tests/03_01_settings/`, because this lesson
changes what it pinned: `test_settings_hold_values_not_objects` (a key is
a value too, and an unset one is `None`).

Every import of the project is inside the test that needs it, so that
before the lesson each test fails on its own, instead of the whole file
failing to load.
"""

import os
import subprocess
import sys
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

# This file is tests/03_02_models/test_models.py: two folders below the root.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
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
    # The workspace folder (03-03) the app now needs at startup.
    (tmp_path / "work" / "tinyjobs").mkdir(parents=True)
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


# Point 1: the composition root builds the provider and hands it its key
# from Settings.


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


def test_the_key_from_settings_reaches_the_provider() -> None:
    # The key is only in `.env`, never in the environment: if the client
    # holds it, `build_model` passed it in.
    from pydantic_tutor.composition import build_model
    from pydantic_tutor.settings import Settings

    write_dotenv("OPENAI_API_KEY=sk-passed-in")
    model = build_model(settings=Settings())
    assert isinstance(model, OpenAIResponsesModel)
    assert api_key_of(model) == "sk-passed-in"


def test_build_app_uses_the_model_built_with_the_key() -> None:
    from pydantic_tutor.composition import build_app
    from pydantic_tutor.settings import Settings

    write_dotenv("OPENAI_API_KEY=sk-passed-in")
    assert api_key_of(build_app(settings=Settings()).model) == "sk-passed-in"


def test_the_test_model_needs_no_key() -> None:
    from pydantic_tutor.composition import build_model
    from pydantic_tutor.settings import Settings

    write_dotenv("PYDANTIC_TUTOR_CHAT_MODEL=test")
    assert isinstance(build_model(settings=Settings()), TestModel)


@pytest.mark.parametrize(
    argnames=("chat_model", "variable"),
    argvalues=[
        ("openai:gpt-4o-mini", "OPENAI_API_KEY"),
        ("anthropic:claude-sonnet-5-5", "ANTHROPIC_API_KEY"),
    ],
)
def test_a_missing_key_fails_by_name_before_any_request(
    chat_model: str, variable: str
) -> None:
    from pydantic_tutor.composition import build_model
    from pydantic_tutor.settings import Settings

    write_dotenv(f"PYDANTIC_TUTOR_CHAT_MODEL={chat_model}")
    with pytest.raises(UserError) as excinfo:
        build_model(settings=Settings())
    assert variable in str(excinfo.value)


def test_an_empty_key_fails_like_a_missing_one() -> None:
    from pydantic_tutor.composition import build_model
    from pydantic_tutor.settings import Settings

    write_dotenv("OPENAI_API_KEY=")
    with pytest.raises(UserError) as excinfo:
        build_model(settings=Settings())
    assert "OPENAI_API_KEY" in str(excinfo.value)


# Point 2: only the listed providers can be built; another provider's model
# is a setting.


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

    write_dotenv("PYDANTIC_TUTOR_CHAT_MODEL=openai-chat:gpt-4o", "OPENAI_API_KEY=sk-o")
    model = build_model(settings=Settings())
    assert isinstance(model, OpenAIChatModel)
    assert api_key_of(model) == "sk-o"


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


# The chat stops by name, from `.env`, before it opens.


@pytest.mark.parametrize(
    argnames=("dotenv", "named"),
    argvalues=[
        ("PYDANTIC_TUTOR_WEB_SEARCH=maybe", "web_search"),
        ("PYDANTIC_TUTOR_CHAT_MODEL=groq:x", "groq"),
        ("OPENAI_API_KEY=", "OPENAI_API_KEY"),
        ("PYDANTIC_TUTOR_CHAT_MODEL=anthropic:claude-sonnet-5-5", "ANTHROPIC_API_KEY"),
    ],
)
def test_the_chat_stops_by_name_before_it_opens(dotenv: str, named: str) -> None:
    # Moved from 03-01's startup check, with an empty key and a second
    # provider added. A subprocess in the empty folder, so `.env` is read
    # from there; the script path is the project's.
    write_dotenv(dotenv)
    env = {k: v for k, v in os.environ.items() if k not in KEY_VARIABLES}
    result = subprocess.run(
        [sys.executable, "-m", "pydantic_tutor"],
        input="",
        capture_output=True,
        text=True,
        check=False,
        env={**env, "PYTHONPATH": str(PROJECT_ROOT / "src")},
    )
    assert result.returncode != 0
    assert result.stderr.startswith("pydantic-tutor cannot start:")
    assert named in result.stderr
    assert "Traceback" not in result.stderr


def test_the_chat_reads_a_key_from_dotenv_and_opens() -> None:
    write_dotenv("OPENAI_API_KEY=sk-from-dotenv", "PYDANTIC_TUTOR_WEB_SEARCH=true")
    env = {k: v for k, v in os.environ.items() if k not in KEY_VARIABLES}
    result = subprocess.run(
        [sys.executable, "-m", "pydantic_tutor"],
        input="",
        capture_output=True,
        text=True,
        check=False,
        env={**env, "PYTHONPATH": str(PROJECT_ROOT / "src")},
    )
    assert result.returncode == 0
    assert "Chat with the agent" in result.stdout


def test_build_model_takes_its_providers_from_any_function() -> None:
    from pydantic_ai.providers.openai import OpenAIProvider

    from pydantic_tutor.composition import build_model
    from pydantic_tutor.settings import Settings

    asked: list[str] = []

    def acme(name: str) -> OpenAIProvider:
        asked.append(name)
        return OpenAIProvider(api_key="acme-key", base_url="https://acme.invalid/v1")

    model = build_model(settings=Settings(), provider_factory=acme)
    assert asked == [Settings().chat_model.split(":")[0]]
    assert "acme.invalid" in str(model.base_url)
