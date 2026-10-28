"""Checks that the project from the streaming lesson is in place.

Our checks for the streaming lesson: they verify the lesson, and the
student is not asked to run them. Run them with `uv run pytest`. They fail
until the lesson is done. None of them calls a real model: they run the
chat's own loop, with its model swapped for one that can only stream.
"""

import importlib.util
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from types import ModuleType

import pytest
from pydantic_ai import models
from pydantic_ai.messages import ModelMessage, UserPromptPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

# This file is tests/02_04_streaming/test_streaming.py: two folders below the root.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def no_real_models(monkeypatch: pytest.MonkeyPatch) -> None:
    # Any request to a real model raises instead of reaching a provider.
    monkeypatch.setattr(models, "ALLOW_MODEL_REQUESTS", False)


def load_chat() -> ModuleType:
    """Load scripts/chat.py as a module, without running its main()."""
    spec = importlib.util.spec_from_file_location(
        "chat_script", PROJECT_ROOT / "scripts" / "chat.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_chat_streams_and_still_remembers(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    sent: list[int] = []

    async def stream(
        messages: list[ModelMessage], info: AgentInfo
    ) -> AsyncIterator[str]:
        sent.append(
            sum(
                isinstance(part, UserPromptPart)
                for message in messages
                for part in message.parts
            )
        )
        for piece in ["Hel", "lo ", "there."]:
            yield piece

    chat = load_chat()
    lines: Iterator[str] = iter(["Hi", "Again", ""])
    # A model that can only stream: a chat that doesn't stream fails on it.
    monkeypatch.setattr(chat, "CHAT_MODEL", FunctionModel(stream_function=stream))
    monkeypatch.setattr("builtins.input", lambda prompt: next(lines))
    chat.main()

    assert "agent> Hello there." in capsys.readouterr().out
    assert sent == [1, 2]  # the second run was sent both questions
