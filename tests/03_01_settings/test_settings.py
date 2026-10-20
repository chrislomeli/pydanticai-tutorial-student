"""Checks that the project from the settings lesson is in place.

Our checks for the settings lesson: they verify the lesson, and the
student is not asked to run them. Run them with `uv run pytest`. They fail
until the lesson is done. None of them needs an API key, and none opens
the interactive chat: `to_cli_sync` is replaced where `main()` would
reach it.

Every import of the project is inside the test that needs it, so that
before the lesson each test fails on its own, instead of the whole file
failing to load.

Four tests were moved by the models lesson (03-02), which moves the API
keys into `Settings`, has `Settings` read `.env` itself, and takes
`load_dotenv` out of `__main__.py`. Their points are now proved in
`tests/03_02_models/test_models.py`:

- settings hold values, not objects (now keys too, unset by default):
  `test_settings_hold_values_not_objects`;
- a missing key fails when the app is built, naming its variable:
  `test_a_missing_key_fails_by_name_before_any_request`;
- the entry point opens the chat with the built app, and stops by name
  before it opens: `test_the_entry_point_opens_the_chat_with_the_built_app`
  and `test_the_entry_point_stops_by_name_before_the_chat_opens`.
"""

import os
from pathlib import Path

import pytest
from pydantic import ValidationError
from pydantic_ai import models
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel

PREFIX = "PYDANTIC_TUTOR_"


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    # Each test starts with none of this project's variables set and no
    # key, whatever the shell running pytest has, and no real model call.
    for name in list(os.environ):
        if name.upper().startswith(PREFIX):
            monkeypatch.delenv(name)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(target=models, name="ALLOW_MODEL_REQUESTS", value=False)


def scripted_model() -> FunctionModel:
    def reply(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart(content="hi there")])

    return FunctionModel(function=reply)


# Goal 1: settings hand out values, typed and checked.


def test_settings_have_the_defaults_the_chat_used() -> None:
    from pydantic_tutor.settings import Settings

    settings = Settings()
    assert settings.chat_model == "openai:gpt-5-mini"
    assert settings.trace is False
    assert settings.repo_root == Path("sample_repos/tinyjobs")


def test_settings_read_the_prefixed_variables(monkeypatch: pytest.MonkeyPatch) -> None:
    from pydantic_tutor.settings import Settings

    monkeypatch.setenv(PREFIX + "CHAT_MODEL", "test")
    monkeypatch.setenv(PREFIX + "TRACE", "1")
    monkeypatch.setenv(PREFIX + "REPO_ROOT", "elsewhere")
    settings = Settings()
    assert settings.chat_model == "test"
    assert settings.trace is True  # "1" became a bool, not the string "1"
    assert settings.repo_root == Path("elsewhere")


def test_settings_cannot_be_changed_after_startup() -> None:
    from pydantic_tutor.settings import Settings

    settings = Settings()
    with pytest.raises(ValidationError):
        settings.trace = True  # type: ignore[misc]


def test_a_bad_value_fails_naming_the_setting(monkeypatch: pytest.MonkeyPatch) -> None:
    from pydantic_tutor.settings import Settings

    monkeypatch.setenv(PREFIX + "TRACE", "maybe")
    with pytest.raises(ValidationError) as excinfo:
        Settings()
    assert [error["loc"] for error in excinfo.value.errors()] == [("trace",)]


# Goal 2: one place builds the objects, at startup, so what is missing
# fails before the chat opens.


def test_build_app_builds_the_objects_from_the_settings() -> None:
    # All from `pydantic_tutor`, so Ruff sorts them the same way before
    # the lesson (no `composition` yet) and after it.
    from pydantic_tutor import agent as agent_module
    from pydantic_tutor import composition
    from pydantic_tutor import settings as settings_module

    app = composition.build_app(
        settings=settings_module.Settings(
            chat_model="test", repo_root=Path("elsewhere")
        )
    )
    assert app.agent is agent_module.agent  # the project's one agent, not a second
    assert isinstance(app.model, TestModel)  # an object, built from the string
    assert app.deps.root == Path("elsewhere")


def test_an_unknown_model_fails_when_the_app_is_built() -> None:
    from pydantic_tutor.composition import build_app
    from pydantic_tutor.settings import Settings

    with pytest.raises(UserError) as excinfo:
        build_app(settings=Settings(chat_model="nonsense:x"))
    assert "nonsense:x" in str(excinfo.value)


def test_tracing_stays_off_without_the_setting(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Moved from 02-04: tracing is now a setting, wired in the composition root.
    from pydantic_tutor.composition import build_app
    from pydantic_tutor.settings import Settings

    build_app(settings=Settings(chat_model="test"))
    assert capsys.readouterr().out == ""


def test_tracing_prints_spans_for_a_scripted_turn(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Moved from 02-04: with the setting on, a scripted turn's spans print
    # to this terminal -- no key, no network call, nothing sent anywhere.
    from pydantic_tutor.composition import build_app
    from pydantic_tutor.settings import Settings

    app = build_app(settings=Settings(chat_model="test", trace=True))
    capsys.readouterr()  # discard the "tracing on" line
    app.agent.run_sync(user_prompt="hello", model=scripted_model(), deps=app.deps)
    assert "repo_assistant run" in capsys.readouterr().out
