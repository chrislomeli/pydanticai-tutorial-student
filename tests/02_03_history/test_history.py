"""Checks that the project from the history lesson is in place.

Our checks for the history lesson: they verify the lesson, and the student
is not asked to run them. Run them with `uv run pytest`. They fail until
the lesson is done. None of them calls a real model: they run the chat's
own loop, with its model swapped for a stand-in.
"""

import importlib.util
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from types import ModuleType

import pytest
from pydantic_ai import models
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

# This file is tests/02_03_history/test_history.py: two folders below the root.
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


def test_each_run_is_sent_the_chat_so_far(monkeypatch: pytest.MonkeyPatch) -> None:
    sent: list[list[str]] = []

    def prompts(messages: list[ModelMessage]) -> list[str]:
        return [
            str(part.content)
            for message in messages
            for part in message.parts
            if isinstance(part, UserPromptPart)
        ]

    def reply(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        sent.append(prompts(messages))
        return ModelResponse(parts=[TextPart(content="ok")])

    async def stream(
        messages: list[ModelMessage], info: AgentInfo
    ) -> AsyncIterator[str]:
        sent.append(prompts(messages))
        yield "ok"

    chat = load_chat()
    lines: Iterator[str] = iter(["My name is Sam.", "What is my name?", ""])
    # It answers whole or streamed, so it holds after the streaming lesson too.
    model = FunctionModel(function=reply, stream_function=stream)
    monkeypatch.setattr(chat, "CHAT_MODEL", model)
    monkeypatch.setattr("builtins.input", lambda prompt: next(lines))
    chat.main()

    assert sent == [["My name is Sam."], ["My name is Sam.", "What is my name?"]]
