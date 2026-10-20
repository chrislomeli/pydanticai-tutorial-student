"""Checks that the project from the history lesson is in place.

Our checks for the history lesson: they verify the lesson, and the student
is not asked to run them. Run them with `uv run pytest`. They fail until
the lesson is done.

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

# This file is tests/02_01_history/test_history.py: two folders below the
# root.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def no_real_model(monkeypatch: pytest.MonkeyPatch) -> None:
    # Any request to a real model raises instead of reaching a provider.
    monkeypatch.setattr(target=models, name="ALLOW_MODEL_REQUESTS", value=False)


def run_script(name: str) -> str:
    """Run one of the lesson's scripts, by its path from the project root; return its stdout."""
    result = subprocess.run(
        args=[sys.executable, name],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
        # The banner goes to stderr; turning it off keeps a failure readable.
        env={**os.environ, "PYDANTIC_AI_NO_BANNER": "1"},
    )
    assert result.returncode == 0, f"{name} failed:\n{result.stderr}"
    return result.stdout


def test_the_second_turn_only_resolves_it_with_the_history() -> None:
    # Goal 1: passing message_history lets the second turn use what the
    # first turn said; without it, the model sees only its own prompt and
    # answers turn 1's question again.
    printed = run_script("scripts/chat_turns.py")
    lines = printed.splitlines()

    assert lines[0] == "Turn 1:"
    assert lines[1] == "  jobs.py: The job record and the queue that holds jobs."
    assert lines[2] == "Turn 2, with the first turn's history:"
    assert lines[3] == "  run_next calls it, from the CLI's loop."
    assert lines[4] == "Turn 2 again, without the history:"
    # Without the history, "And who calls it?" gets turn 1's answer again:
    # nothing in that prompt names a module on its own.
    assert lines[5] == lines[1]


def test_the_conversation_is_requests_and_responses_made_of_parts() -> None:
    # Goal 2: the parts model. A tool-using first turn, then a plain-text
    # second turn, thread into six messages: request, response, request,
    # response, request, response.
    printed = run_script("scripts/chat_turns.py")
    kind_lines = [
        line.strip()
        for line in printed.splitlines()
        if line.strip().startswith(("Model",))
    ]

    assert kind_lines == [
        "ModelRequest: ['UserPromptPart']",
        "ModelResponse: ['ToolCallPart']",
        "ModelRequest: ['ToolReturnPart']",
        "ModelResponse: ['TextPart']",
        "ModelRequest: ['UserPromptPart']",
        "ModelResponse: ['TextPart']",
    ]


def test_a_saved_conversation_loads_back_the_same() -> None:
    # Goal 3: ModelMessagesTypeAdapter round-trips a conversation through
    # JSON, and a run can continue from what it loads.
    printed = run_script("scripts/save_and_resume.py")
    lines = printed.splitlines()

    assert lines[0] == "turn 1: this is turn 1"
    assert lines[1].startswith("saved ") and lines[1].endswith(
        "bytes to conversation.json"
    )
    assert lines[2] == "loaded: 2 messages, same as before: True"
    assert lines[3] == "turn 2: this is turn 2"
    # The script cleans up after itself: nothing it wrote is left behind.
    assert not (PROJECT_ROOT / "conversation.json").exists()
