"""Checks that the project from the settings lesson is in place.

Our checks for the settings lesson: they verify the lesson, and the
student is not asked to run them. Run them with `uv run pytest`. They fail
until the lesson is done. None of them needs an API key, and none calls a
real model: the chat's own loop runs with its model swapped for a stand-in.

The history and streaming checks from the last two Part 2 lessons live
here too: they used to swap `CHAT_MODEL` inside `pydantic_tutor/chat/loop.py`, and
this lesson moves the model into `build_app`.

Every import of the project is inside the test that needs it, so that
before the lesson each test fails on its own, instead of the whole file
failing to load.
"""

import importlib
import os
import subprocess
import sys
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from types import ModuleType

import pytest
from pydantic import ValidationError
from pydantic_ai import models
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, UserPromptPart
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.native_tools import WebSearchTool

# This file is tests/03_01_settings/test_settings.py: two folders below the root.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
PREFIX = "PYDANTIC_TUTOR_"


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # Each test starts in an empty folder (so the project's own `.env`, read
    # from 03-02 on, is never seen), with none of this project's variables
    # set and no key, whatever the shell running pytest has, and no real
    # model call.
    monkeypatch.chdir(tmp_path)
    for name in list(os.environ):
        if name.upper().startswith(PREFIX):
            monkeypatch.delenv(name)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(target=models, name="ALLOW_MODEL_REQUESTS", value=False)


def load_chat() -> ModuleType:
    """The chat loop, as a module, without running its main()."""
    return importlib.import_module("pydantic_tutor.chat.loop")


def user_prompts(messages: list[ModelMessage]) -> list[str]:
    return [
        str(part.content)
        for message in messages
        for part in message.parts
        if isinstance(part, UserPromptPart)
    ]


# Goal 1: settings are values, typed and checked; objects are built from them.


def test_settings_have_the_defaults_the_chat_used() -> None:
    from pydantic_tutor.model import CHAT_MODEL
    from pydantic_tutor.settings import Settings

    settings = Settings()
    assert settings.chat_model == CHAT_MODEL
    assert settings.web_search is True


def test_settings_read_the_prefixed_variables(monkeypatch: pytest.MonkeyPatch) -> None:
    from pydantic_tutor.settings import Settings

    monkeypatch.setenv(PREFIX + "CHAT_MODEL", "test")
    monkeypatch.setenv(PREFIX + "WEB_SEARCH", "false")
    settings = Settings()
    assert settings.chat_model == "test"
    assert settings.web_search is False  # the text "false" became a bool


def test_settings_cannot_be_changed_after_startup() -> None:
    from pydantic_tutor.settings import Settings

    settings = Settings()
    with pytest.raises(ValidationError):
        settings.web_search = False  # type: ignore[misc]


def test_a_bad_value_fails_naming_the_setting(monkeypatch: pytest.MonkeyPatch) -> None:
    from pydantic_tutor.settings import Settings

    monkeypatch.setenv(PREFIX + "WEB_SEARCH", "maybe")
    with pytest.raises(ValidationError) as excinfo:
        Settings()
    assert [error["loc"] for error in excinfo.value.errors()] == [("web_search",)]


def test_build_app_builds_the_objects_from_the_settings() -> None:
    from pydantic_tutor.composition import build_app
    from pydantic_tutor.settings import Settings

    app = build_app(settings=Settings(chat_model="test"))
    assert isinstance(app.model, TestModel)  # an object, built from the string


@pytest.mark.parametrize("web_search", [True, False])
def test_the_web_search_setting_decides_the_agent_s_capabilities(
    web_search: bool,  # noqa: FBT001
) -> None:
    from pydantic_tutor.composition import build_app
    from pydantic_tutor.settings import Settings

    sent: list[ModelRequestParameters] = []

    def reply(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        sent.append(info.model_request_parameters)
        return ModelResponse(parts=[TextPart(content="ok")])

    app = build_app(settings=Settings(chat_model="test", web_search=web_search))
    app.agent.run_sync(user_prompt="hi", model=FunctionModel(function=reply))

    offered = any(isinstance(tool, WebSearchTool) for tool in sent[0].native_tools)
    assert offered is web_search


# Goal 2: what is missing or wrong fails at startup, by name.


def test_a_missing_key_fails_when_the_app_is_built() -> None:
    # In Part 2 this failed on the first prompt; built at startup, it fails
    # before any run, naming the variable.
    from pydantic_tutor.composition import build_app
    from pydantic_tutor.settings import Settings

    with pytest.raises(UserError) as excinfo:
        build_app(settings=Settings())
    assert "OPENAI_API_KEY" in str(excinfo.value)


def test_an_unknown_model_fails_when_the_app_is_built() -> None:
    from pydantic_tutor.composition import build_app
    from pydantic_tutor.settings import Settings

    with pytest.raises(UserError) as excinfo:
        build_app(settings=Settings(chat_model="nonsense:x"))
    assert "nonsense:x" in str(excinfo.value)


@pytest.mark.parametrize(
    argnames=("variable", "value", "named"),
    argvalues=[
        (PREFIX + "WEB_SEARCH", "maybe", "web_search"),
        (PREFIX + "CHAT_MODEL", "nonsense:x", "nonsense:x"),
        (PREFIX + "CHAT_MODEL", "openai:gpt-4o-mini", "OPENAI_API_KEY"),
    ],
)
def test_the_chat_stops_by_name_before_it_opens(
    monkeypatch: pytest.MonkeyPatch, variable: str, value: str, named: str
) -> None:
    def input_must_not_be_called(prompt: str) -> str:
        raise AssertionError("the chat opened")

    chat = load_chat()
    monkeypatch.setenv(variable, value)
    monkeypatch.setattr("builtins.input", input_must_not_be_called)
    with pytest.raises(SystemExit) as excinfo:
        chat.main()
    message = str(excinfo.value.code)
    assert message.startswith("pydantic-tutor cannot start:")
    assert named in message


def test_the_script_prints_no_traceback_without_a_key() -> None:
    env = {k: v for k, v in os.environ.items() if k != "OPENAI_API_KEY"}
    result = subprocess.run(
        [sys.executable, "-m", "pydantic_tutor"],
        input="",
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert result.returncode != 0
    assert result.stderr.startswith("pydantic-tutor cannot start:")
    assert "OPENAI_API_KEY" in result.stderr
    assert "Traceback" not in result.stderr


# The chat loop is unchanged: it uses the built app. These two moved here
# from the history and streaming lessons.


def test_each_run_is_sent_the_chat_so_far(monkeypatch: pytest.MonkeyPatch) -> None:
    from pydantic_tutor.chat.agent import chat_agent
    from pydantic_tutor.composition import App

    sent: list[list[str]] = []

    def reply(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        sent.append(user_prompts(messages))
        return ModelResponse(parts=[TextPart(content="ok")])

    async def stream(
        messages: list[ModelMessage], info: AgentInfo
    ) -> AsyncIterator[str]:
        sent.append(user_prompts(messages))
        yield "ok"

    chat = load_chat()
    lines: Iterator[str] = iter(["My name is Sam.", "What is my name?", ""])
    # It answers whole or streamed, so it holds before and after streaming.
    model = FunctionModel(function=reply, stream_function=stream)
    app = App(agent=chat_agent, model=model)
    monkeypatch.setattr(chat, "build_app", lambda *, settings: app)
    monkeypatch.setattr("builtins.input", lambda prompt: next(lines))
    chat.main()

    assert sent == [["My name is Sam."], ["My name is Sam.", "What is my name?"]]


def test_the_chat_streams_and_still_remembers(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from pydantic_tutor.chat.agent import chat_agent
    from pydantic_tutor.composition import App

    sent: list[int] = []

    async def stream(
        messages: list[ModelMessage], info: AgentInfo
    ) -> AsyncIterator[str]:
        sent.append(len(user_prompts(messages)))
        for piece in ["Hel", "lo ", "there."]:
            yield piece

    chat = load_chat()
    lines: Iterator[str] = iter(["Hi", "Again", ""])
    # A model that can only stream: a chat that doesn't stream fails on it.
    app = App(agent=chat_agent, model=FunctionModel(stream_function=stream))
    monkeypatch.setattr(chat, "build_app", lambda *, settings: app)
    monkeypatch.setattr("builtins.input", lambda prompt: next(lines))
    chat.main()

    assert "agent> Hello there." in capsys.readouterr().out
    assert sent == [1, 2]  # the second run was sent both questions
