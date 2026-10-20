"""Checks that the project from the Conversation lesson is in place.

Our checks for the Conversation lesson: they verify the lesson, and the
student is not asked to run them. Run them with `uv run pytest`. They fail
until the lesson is done.

Every import of the project is inside the test that needs it, so that
before the lesson each test fails on its own, instead of the whole file
failing to load.
"""

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic_ai import models

# This file is tests/02_02_conversation/test_conversation.py: two folders
# below the root.
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


def test_the_budget_holds_with_conversation_and_leaks_with_message_history() -> None:
    # Goal 2: a 2-request UsageLimits budget refuses turns 3 and 4 when
    # threaded with conversation=, and does not when re-threaded with
    # message_history= alone.
    printed = run_script("scripts/conversation_budget.py")

    assert "turns 1-4: ['ok', 'ok', 'LIMIT', 'LIMIT']" in printed
    assert "turns 1-4: ['ok', 'ok', 'ok', 'ok']" in printed


def test_the_conversation_carries_a_stable_id_and_a_running_usage() -> None:
    # Goal 1: result.conversation carries a conversation_id unchanged
    # across turns, and a usage that accumulates across them — neither of
    # which a bare message list carries.
    printed = run_script("scripts/conversation_budget.py")

    ids = re.findall(r"conversation_id=(\S+)", printed)
    assert len(ids) == 2  # turns 1 and 2; turns 3-4 hit the limit first
    assert ids[0] == ids[1]

    requests = [int(n) for n in re.findall(r"usage\.requests=(\d+)", printed)]
    # The first two numbers are the conversation=-threaded run's running
    # total (1, then 2); the next four are message_history='s own-call
    # usage, which never climbs past 1.
    assert requests[:2] == [1, 2]
    assert requests[2:] == [1, 1, 1, 1]
