"""Checks that the project from the console-chat lesson is in place.

Our checks for the console-chat lesson: they verify the lesson, and the
student is not asked to run them. Run them with `uv run pytest`. They fail
until the lesson is done. None of them calls a real model.

Every import of the project is inside the test that needs it, so that
before the lesson each test fails on its own, instead of the whole file
failing to load.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic_ai import models
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

# This file is tests/02_02_console_chat/test_console_chat.py: two folders
# below the root.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def no_real_models(monkeypatch: pytest.MonkeyPatch) -> None:
    # Any request to a real model raises instead of reaching a provider.
    monkeypatch.setattr(models, "ALLOW_MODEL_REQUESTS", False)


@pytest.mark.parametrize("typed", ["\n", ""], ids=["empty line", "end of input"])
def test_the_chat_ends_without_calling_the_model(typed: str, tmp_path: Path) -> None:
    # A fake key, so the chat can start whatever the lesson (it never calls
    # the model): if it did, the request would fail. Since the workspace
    # lesson (03-03) the chat also needs its workspace folder to exist.
    env = {
        **os.environ,
        "OPENAI_API_KEY": "sk-not-a-real-key",
        "PYDANTIC_TUTOR_WORKSPACE_ROOT": str(tmp_path),
    }
    result = subprocess.run(
        [sys.executable, "-m", "pydantic_tutor"],
        cwd=PROJECT_ROOT,
        input=typed,
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.startswith("Chat with the agent.")


def test_each_run_is_sent_only_its_own_message() -> None:
    from pydantic_tutor.chat.agent import chat_agent

    prompts: list[list[str]] = []

    def reply(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        prompts.append(
            [
                str(part.content)
                for message in messages
                for part in message.parts
                if isinstance(part, UserPromptPart)
            ]
        )
        return ModelResponse(parts=[TextPart(content="ok")])

    model = FunctionModel(function=reply)
    chat_agent.run_sync(user_prompt="My name is Sam.", model=model)
    chat_agent.run_sync(user_prompt="What is my name?", model=model)

    assert prompts == [["My name is Sam."], ["What is my name?"]]
